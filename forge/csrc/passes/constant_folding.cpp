// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/constant_folding.hpp"

#include <pybind11/pybind11.h>

#include <functional>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "ops/op.hpp"
#include "passes/commute_utils.hpp"
#include "passes/passes_utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{
using FoldFn = bool(graphlib::Graph *, graphlib::OpNode *, graphlib::OpNode *);

static graphlib::InputNode *get_constant_input(graphlib::Graph *graph, graphlib::OpNode *binary)
{
    auto operands = graph->data_operands(binary);
    TT_ASSERT(operands.size() == 2);
    graphlib::Node *input0 = operands[0];
    graphlib::Node *input1 = operands[1];

    graphlib::InputNode *constant = dynamic_cast<graphlib::InputNode *>(input0);
    if (not constant or not constant->is_constant())
        std::swap(input0, input1);

    return dynamic_cast<graphlib::InputNode *>(input0);
}

static graphlib::OpNode *get_producer_input(graphlib::Graph *graph, graphlib::OpNode *binary)
{
    auto operands = graph->data_operands(binary);
    TT_ASSERT(operands.size() == 2);
    graphlib::Node *input0 = operands[0];
    graphlib::Node *input1 = operands[1];

    graphlib::OpNode *producer = dynamic_cast<graphlib::OpNode *>(input0);
    if (not producer)
        std::swap(input0, input1);

    return dynamic_cast<graphlib::OpNode *>(input0);
}

bool is_constant_eltwise_binary(graphlib::Graph *graph, graphlib::OpNode *binary)
{
    if (not binary->is_eltwise_binary())
        return false;

    return bool(get_constant_input(graph, binary)) and bool(get_producer_input(graph, binary));
}

template <typename CommutableFn, typename SinkOpFn>
std::vector<graphlib::Node *> find_operands_commute_through(
    graphlib::Graph *graph, graphlib::Node *root, CommutableFn commutable_fn, SinkOpFn sink_op_fn)
{
    std::vector<graphlib::Node *> sinks;
    std::vector<graphlib::Node *> needs_visit = {root};
    while (not needs_visit.empty())
    {
        graphlib::Node *node = needs_visit.back();
        needs_visit.pop_back();

        if (sink_op_fn(node))
        {
            sinks.push_back(node);
        }
        else if (commutable_fn(node))
        {
            for (auto *operand : graph->data_operands(node))
            {
                needs_visit.push_back(operand);
            }
        }
        else
        {
            // If any operands, through any path isn't a sink or commutable, give up
            return {};
        }
    }
    return sinks;
}

static bool try_fold_constant_multiply_into_matmul_rhs(
    graphlib::Graph *graph, graphlib::OpNode *operand, graphlib::OpNode *multiply)
{
    // Hoists and returns true if:
    //  - op is eltwise multiply
    //  - 1 argument is a 1 dimensional constant tensor
    //  - 1 argument is a matmul with RHS parameters
    if (multiply->op_type() != ops::OpType::Multiply)
        return false;

    std::vector<graphlib::Node *> matmuls = find_operands_commute_through(
        graph,
        operand,
        [](graphlib::Node *commutable)
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(commutable);
            return op and (op->op_type() == ops::OpType::Add or op->op_type() == ops::OpType::Nop);
        },
        [](graphlib::Node *matmul)
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(matmul);
            return op and op->is_matmul();
        });

    if (matmuls.empty())
        return false;

    graphlib::InputNode *constant = get_constant_input(graph, multiply);
    TT_ASSERT(constant);

    auto shape = constant->shape();
    if (shape.volume() != shape[-1])
        return false;

    // Check all matmul weights can legally fold this constant
    for (graphlib::Node *matmul : matmuls)
    {
        auto matmul_operands = graph->operand_data_edges(matmul);
        TT_ASSERT(matmul_operands.size() >= 2);
        graphlib::InputNode *matmul_rhs =
            dynamic_cast<graphlib::InputNode *>(graph->node_by_id(matmul_operands[1].producer_node_id));

        if (not matmul_rhs or
            not(matmul_rhs->is_parameter() or matmul_rhs->is_constant() or matmul_rhs->is_optimizer_parameter()))
            return false;

        if (graph->training() and matmul_rhs->is_parameter())
            return false;
    }

    auto constant_edges = graph->get_edges(constant, multiply);
    TT_ASSERT(constant_edges.size() == 1);
    auto constant_edge = constant_edges.front();
    auto constant_attr = graph->remove_edge(constant_edge);

    // Fold
    for (graphlib::Node *matmul : matmuls)
    {
        auto matmul_operands = graph->operand_data_edges(matmul);
        TT_ASSERT(matmul_operands.size() >= 2);
        graphlib::InputNode *matmul_rhs =
            dynamic_cast<graphlib::InputNode *>(graph->node_by_id(matmul_operands[1].producer_node_id));

        log_trace(
            LogGraphCompiler, "Fold multiply into matmul weights: {} -> {}", multiply->name(), matmul_rhs->name());

        // Fixup broadcast
        for (auto &tm : constant_attr->get_tms())
        {
            if (tm.type() == ops::OpType::Broadcast)
            {
                int tm_dim = multiply->shape().negative_index(tm.attr_as<int>("dim"));
                if (tm_dim == -2)
                    tm.set_attr("size", static_cast<int>(matmul_rhs->shape()[-2]));
            }
        }

        auto *multiply_clone = graph->add_node(
            multiply->clone(multiply->name() + "_" + matmul_rhs->name()),
            graph->get_subgraph_id_for_node(matmul->id()));
        multiply_clone->set_shape(matmul_rhs->shape());

        auto *constant_clone = graph->add_node(
            constant->clone(constant->name() + "_" + multiply->name()), graph->get_subgraph_id_for_node(matmul->id()));

        // Connect matmul rhs to multiply LHS
        graphlib::insert_node_on_edge(graph, matmul_operands[1], multiply_clone);

        // Connect constant to multiply RHS
        constant_edge.producer_node_id = constant_clone->id();
        constant_edge.consumer_input_port_id = 1;
        constant_edge.consumer_node_id = multiply_clone->id();
        graph->add_edge(constant_edge, constant_attr);

        graphlib::try_consteval_op(graph, multiply_clone);
    }

    // Remove multiply from the graph, but check if constant has other consumers before removing constant
    graphlib::bypass_node(graph, multiply, true);
    if (graph->user_edges(constant).size() == 0)
    {
        graph->remove_node(constant);
    }

    return true;
}

static bool try_fold_constant_associative(graphlib::Graph *graph, graphlib::OpNode *a, graphlib::OpNode *b)
{
    if (a->op_type() != b->op_type())
        return false;

    if (a->op_type() != ops::OpType::Multiply and a->op_type() != ops::OpType::Add)
        return false;

    graphlib::InputNode *a_constant = get_constant_input(graph, a);
    if (not a_constant or not a_constant->is_constant())
        return false;

    if (graph->user_data_edges(a).size() > 1)
        return false;

    log_trace(LogGraphCompiler, "Fold constant associative: {} {}", a->name(), b->name());

    graphlib::InputNode *b_constant = get_constant_input(graph, b);
    TT_ASSERT(b_constant);
    auto a_edges = graph->get_edges(a_constant, a);
    auto b_edges = graph->get_edges(b_constant, b);
    TT_ASSERT(a_edges.size() == 1);
    TT_ASSERT(b_edges.size() == 1);

    auto b_attr = graph->get_edge_attributes(b_edges.front());
    graph->remove_edge(b_edges.front());
    auto b_subgraph_id = graph->get_subgraph_id_for_node(b->id());
    b = graph->add_node(graphlib::bypass_node(graph, b, true), b_subgraph_id)->as<graphlib::OpNode>();
    insert_node_on_edge(graph, a_edges.front(), b);
    b_edges.front().consumer_node_id = b->id();
    b_edges.front().consumer_input_port_id = 1;
    graph->add_edge(b_edges.front(), b_attr);
    graphlib::calculate_and_set_node_shape(graph, b);
    graphlib::try_consteval_op(graph, b);

    return true;
}

static std::vector<FoldFn *> fold_fns = {
    try_fold_constant_multiply_into_matmul_rhs,
    try_fold_constant_associative,
};

static bool try_fold_constant_binary_op(graphlib::Graph *graph, graphlib::OpNode *binary)
{
    if (not is_constant_eltwise_binary(graph, binary))
        return false;

    for (FoldFn *fn : fold_fns)
    {
        auto *producer = get_producer_input(graph, binary);
        TT_ASSERT(producer);
        if (fn(graph, producer, binary))
            return true;
    }

    return false;
}

void constant_folding(graphlib::Graph *graph)
{
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op)
                continue;

            if (try_fold_constant_binary_op(graph, op))
            {
                updated = true;
                break;
            }
        }
    }
}
}  // namespace tt::passes
