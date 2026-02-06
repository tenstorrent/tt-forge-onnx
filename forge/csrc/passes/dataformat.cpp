// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/dataformat.hpp"

#include "backend_api/device_config.hpp"
#include "forge_passes.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "lower_to_forge/common.hpp"
#include "ops/op.hpp"
#include "passes/amp.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{
using namespace graphlib;

// Inserts a cast node after every non-integer constant input node in the graph,
// placing it on the first user edge and re-routing all other edges.
// The cast node will convert outputs to the given DataFormat.
static void insert_cast_on_const_input_nodes(graphlib::Graph *graph, DataFormat df_override)
{
    for (Node *node : graph->nodes_by_type(graphlib::NodeType::kInput))
    {
        InputNode *input_node = node->as<graphlib::InputNode>();

        std::vector<Edge> user_edges = graph->user_data_edges(node);
        if (user_edges.empty())
            continue;

        if (input_node->input_type() != graphlib::InputNodeType::Constant || node->output_df() == df_override ||
            is_integer_data_format(node->output_df()))
            continue;

        ops::Op op_type = ops::Op(ops::OpType::Cast, {{"dtype", static_cast<int>(df_override)}});

        Node *cast_node = graph->add_node(
            graphlib::create_node<graphlib::PyOpNode>("cast_input_" + node->name(), op_type),
            graph->get_subgraph_id_for_node(node->id()));

        cast_node->set_shape(node->shape());

        // First edge: insert cast node on it.
        graphlib::insert_node_on_edge(graph, user_edges[0], cast_node, true, true, 0, true);
        cast_node->set_output_df(df_override);
        TT_ASSERT(cast_node != nullptr, "Cast node should not be null");

        // Remaining edges: reconnect edges to cast node.
        for (size_t i = 1; i < user_edges.size(); ++i)
        {
            Edge &edge = user_edges[i];

            Edge new_edge = Edge(
                cast_node->id(),
                0,  // output port
                edge.consumer_node_id,
                edge.consumer_input_port_id,
                edge.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(edge, new_edge);
            graph->remove_edge(edge);
        }
    }
}

// Sets output data formats of all nodes to user specified data format.
void configure_output_data_formats(graphlib::Graph *graph, std::optional<DataFormat> default_df_override)
{
    for (Node *node : graph->nodes())
    {
        bool node_is_int = is_integer_data_format(node->output_df());
        bool node_is_input = node->node_type() == graphlib::NodeType::kInput;
        bool node_input_parameter = node->node_type() == graphlib::NodeType::kInput &&
                                    node->as<graphlib::InputNode>()->input_type() == graphlib::InputNodeType::Parameter;
        // We should override output data format for input parameters because TVM casts them to the default data format
        // float32. Other types of input nodes like activations and constants are accurately represented in Forge graph
        // (dataformat corresponding to what will come from the host).

        if (node_is_input)
        {
            bool node_input_constant =
                node->as<graphlib::InputNode>()->input_type() == graphlib::InputNodeType::Constant;
            bool node_input_activation =
                node->as<graphlib::InputNode>()->input_type() == graphlib::InputNodeType::Activation;
            if (node_input_activation)
            {
                TT_ASSERT(
                    node->output_df() == default_df_override || node_is_int,
                    "Non integer Input activations should have output_df same as dataformat override which is: {}, but "
                    "node output_df is: {}",
                    default_df_override.value(),
                    node->output_df());
            }
            if (node_input_constant && (node->output_df() != default_df_override.value()))
            {
                std::vector<Node *> consumers = graph->data_users(node);
                TT_ASSERT(
                    consumers.size() == 1 || node_is_int,
                    "By this point, non integer constant input node should have only one consumer, but it has {} "
                    "consumers",
                    consumers.size());
                // check if consumer is cast node
                bool consumer_is_cast = consumers[0]->node_type() == graphlib::NodeType::kPyOp &&
                                        consumers[0]->as<graphlib::PyOpNode>()->op_type() == ops::OpType::Cast;
                TT_ASSERT(
                    (consumer_is_cast && consumers[0]->output_df() == default_df_override.value()) || node_is_int,
                    "Non integer constant input node that doesn't have data format same as default_df_override {}, "
                    "should have cast op as consumer with output_df same as default_df_override",
                    default_df_override.value());
            }
        }

        if (default_df_override && !node_is_int && (!node_is_input || node_input_parameter))
        {
            node->set_output_df(preserve_lower_precision_cast(node->output_df(), *default_df_override));
        }
    }
}

// Inserts cast nodes (user specified data format) on all input nodes in the graph,
// and sets output data formats of all nodes to user specified data format.
void apply_user_data_format_override(graphlib::Graph *graph, py::object compiler_cfg_object)
{
    // Extract optional default_df_override from Python
    std::optional<DataFormat> default_df_override = std::nullopt;
    py::object attr = compiler_cfg_object.attr("default_df_override");
    if (!attr.is_none())
        default_df_override = attr.cast<DataFormat>();

    // Skip everything if no override is provided
    if (!default_df_override)
        return;

    insert_cast_on_const_input_nodes(graph, *default_df_override);
    configure_output_data_formats(graph, default_df_override);
}

static std::vector<ExpPrecision> get_exponent_conversion_preference(
    const std::vector<DataFormat> &operand_dfs, const std::optional<DataFormat> default_df_override)
{
    int num_a_formats = std::count_if(operand_dfs.begin(), operand_dfs.end(), is_a_data_format);
    int num_b_formats = std::count_if(operand_dfs.begin(), operand_dfs.end(), is_b_data_format);

    // if equal, use the default_df to break the tie.
    if (num_a_formats == num_b_formats)
    {
        return (default_df_override and is_a_data_format(*default_df_override))
                   ? std::vector<ExpPrecision>{ExpPrecision::A, ExpPrecision::B}
                   : std::vector<ExpPrecision>{ExpPrecision::B, ExpPrecision::A};
    }
    else if (num_a_formats > num_b_formats)
    {
        return {ExpPrecision::A, ExpPrecision::B};
    }
    else
    {
        return {ExpPrecision::B, ExpPrecision::A};
    }
}

static bool is_match_precision_data_format(DataFormat df, ExpPrecision precision)
{
    return (precision == ExpPrecision::A and is_a_data_format(df)) or
           (precision == ExpPrecision::B and is_b_data_format(df));
}

void cast_input_data_formats(graphlib::Graph *graph, Node *node, ExpPrecision to_precision)
{
    const std::unordered_map<ExpPrecision, std::function<DataFormat(DataFormat)>> conversion_function = {
        {ExpPrecision::B, to_a_data_format}, {ExpPrecision::A, to_b_data_format}};
    ExpPrecision from_precision = to_precision == ExpPrecision::A ? ExpPrecision::B : ExpPrecision::A;
    const std::function<DataFormat(DataFormat)> convert_df_function = conversion_function.at(from_precision);

    for (Node *operand : graph->data_operands(node))
    {
        if (is_match_precision_data_format(operand->output_df(), from_precision))
        {
            operand->set_output_df(convert_df_function(operand->output_df()));
        }
    }
}

void cast_and_resolve_input_data_formats(
    graphlib::Graph *graph,
    Node *node,
    const std::vector<DataFormat> &input_data_formats,
    const std::optional<DataFormat> default_df_override)
{
    std::vector<ExpPrecision> conversion_preference =
        get_exponent_conversion_preference(input_data_formats, default_df_override);

    // Try to cast to the preferred exponent size first. We can always fallback to
    // casting to FP32 if we can't cast to the preferred exponent size.
    // For now, let's keep it simple and select first conversion preference.
    log_debug(LogGraphCompiler, "{} contains inputs with mixed a/b data formats: {}", node->name(), input_data_formats);

    ExpPrecision preferred_precision = conversion_preference.at(0);
    cast_input_data_formats(graph, node, preferred_precision);

    std::vector<DataFormat> updated_input_data_formats;
    for (const auto &operand : graph->data_operands(node))
    {
        updated_input_data_formats.push_back(operand->output_df());
    }
    log_debug(LogGraphCompiler, "{} updated input_data_formats: {}", node->name(), updated_input_data_formats);
}

//
// Fix illegal situations
//
// Current rules:
// 1. On Wormhole, BFP* formats can convert from a to b when packing.
// 2. Intermediate format is currently only used for matmul, but for non-matmul we should follow the rule that
//    intermed df == output df
// 3. Matmul is a special case where operand 1, intermed df, and output df must all match
// 4. Acc_df can be either math format or FP32 on Wormhole
// 5. Untilize op can't be in Bfp* formats
//
// When fixing, try not to change formats that were specifically overriden by the user
void fix_data_formats(graphlib::Graph *graph, bool fp32_acc_supported)
{
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kQueue)
        {
            // The producer may have had its output_df modified. We need to update the output_df
            // of user-defined queues so that queue->output_df() reflects the producer output_df.
            auto producer = graph->data_operands(node)[0];
            node->set_output_df(producer->output_df());
        }
        else if (node->node_type() == graphlib::NodeType::kOutput)
        {
            auto output_op = graph->data_operands(node)[0];
            auto is_partial_datacopy_edge = [](Edge e)
            { return (e.edge_type == graphlib::EdgeType::kPartialDataCopy); };
            if (node->as<graphlib::OutputNode>()->untilize())
            {
                if ((output_op->output_df() == DataFormat::Bfp8_b) || (output_op->output_df() == DataFormat::Bfp4_b) ||
                    (output_op->output_df() == DataFormat::Bfp2_b) || (output_op->output_df() == DataFormat::Float16_b))
                {
                    output_op->set_output_df(DataFormat::Float16_b);
                }
                else if (
                    (output_op->output_df() == DataFormat::Bfp8) || (output_op->output_df() == DataFormat::Bfp4) ||
                    (output_op->output_df() == DataFormat::Bfp2) || (output_op->output_df() == DataFormat::Float16))
                {
                    output_op->set_output_df(DataFormat::Float16);
                }
            }

            std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_edges(node, is_partial_datacopy_edge);
            if (not partial_datacopy_edges.empty())
            {
                // Current queue is aliased to an existing queue. Impose constraint on write-back producer
                auto consumer_node_id = partial_datacopy_edges.front().consumer_node_id;
                auto aliased_queue = graph->node_by_id(consumer_node_id);
                if (output_op->output_df() != aliased_queue->output_df())
                {
                    log_warning(
                        "Op ({}) writing to aliased queue ({}) must have matching data-formats."
                        "Overriding {} output_df to {}.",
                        output_op->name(),
                        aliased_queue->name(),
                        output_op->output_df(),
                        aliased_queue->output_df());
                    output_op->set_output_df(aliased_queue->output_df());
                }
            }

            node->set_output_df(output_op->output_df());
        }
    }
}

void validate_data_formats(const graphlib::Graph *graph, const DeviceConfig &device_config)
{
    for (Node *node : graph->nodes())
    {
        if (node->node_type() == graphlib::NodeType::kQueue)
        {
            auto producer = graph->data_operands(node).at(0);
            TT_LOG_ASSERT(
                producer->output_df() == node->output_df(),
                "Queue: {} is configured for data format: {}, but producer: {} is configured for data format: {}",
                node->name(),
                node->output_df(),
                producer->name(),
                producer->output_df());
        }
        else if (node->node_type() == graphlib::NodeType::kOutput)
        {
            auto producer = graph->data_operands(node).at(0);
            auto is_partial_datacopy_edge = [](Edge e)
            { return (e.edge_type == graphlib::EdgeType::kPartialDataCopy); };
            std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_edges(node, is_partial_datacopy_edge);
            if (not partial_datacopy_edges.empty())
            {
                // Current queue is aliased to an existing queue. Impose constraint on write-back producer
                auto consumer_node_id = partial_datacopy_edges.front().consumer_node_id;
                auto aliased_queue = graph->node_by_id(consumer_node_id);

                TT_LOG_ASSERT(
                    producer->output_df() == aliased_queue->output_df(),
                    "Producer Op ({}) output df ({}) must have matching data-formats as ({}) aliased queue ({}).",
                    producer->name(),
                    producer->output_df(),
                    aliased_queue->name(),
                    aliased_queue->output_df());
            }
        }
    }
}

void satisfy_data_format_constraints(graphlib::Graph *graph, bool fp32_acc_supported)
{
    fix_data_formats(graph, fp32_acc_supported);
}

void run_dataformat_passes(
    graphlib::Graph *graph,
    const DeviceConfig &device_config,
    const std::optional<DataFormat> default_df_override,
    const std::optional<DataFormat> default_accumulate_df,
    const DataFormat fp32_fallback,
    const MathFidelity default_math_fidelity,
    const int amp_level,
    const std::vector<AMPNodeProperties> &amp_properties)
{
    // Apply user overrides
    configure_output_data_formats(graph, default_df_override);

    // Fix illegal situations
    satisfy_data_format_constraints(graph, device_config.supports_fp32_accumulation());

    // Apply automatic mixed precision based on user-defined levels
    run_automatic_mixed_precision(graph, device_config, default_df_override, amp_level, amp_properties);

    validate_data_formats(graph, device_config);
}

}  // namespace tt::passes
