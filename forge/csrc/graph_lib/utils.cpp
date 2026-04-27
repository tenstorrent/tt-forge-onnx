// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "utils.hpp"

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <functional>
#include <map>
#include <queue>
#include <unordered_set>
#include <vector>

#include "autograd/binding.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "ops/op.hpp"
#include "reportify/reportify.hpp"
#include "utils/logger.hpp"

namespace tt
{

namespace graphlib
{
bool default_node_filter(Node *) { return true; }

static bool requires_visit(const std::unordered_map<NodeId, bool> &visited, NodeId node_id)
{
    return visited.find(node_id) == visited.end() or visited.at(node_id) == false;
}

int get_row_size_from_tile_size(TileDim tile_dim)
{
    int ret = 32;
    switch (tile_dim)
    {
        case TileDim::Dim32x32: ret = 32; break;
        case TileDim::Dim16x32: ret = 16; break;
        case TileDim::Dim32x16: ret = 32; break;
        case TileDim::Dim8x32: ret = 8; break;
        case TileDim::Dim4x32: ret = 4; break;
        case TileDim::Dim2x32: ret = 2; break;
        case TileDim::Dim1x32: ret = 1; break;
        default: TT_ASSERT(false, "Invalid tile dim");
    }
    return ret;
}

int get_col_size_from_tile_size(TileDim tile_dim)
{
    int ret = 32;
    switch (tile_dim)
    {
        case TileDim::Dim32x32: ret = 32; break;
        case TileDim::Dim16x32: ret = 32; break;
        case TileDim::Dim32x16: ret = 16; break;
        case TileDim::Dim8x32: ret = 32; break;
        case TileDim::Dim4x32: ret = 32; break;
        case TileDim::Dim2x32: ret = 32; break;
        case TileDim::Dim1x32: ret = 32; break;
        default: TT_ASSERT(false, "Invalid tile dim");
    }
    return ret;
}

TileDim get_tile_dim_from_height_width(int tile_height, int tile_width)
{
    TileDim ret = TileDim::Dim32x32;

    switch (tile_height)
    {
        case 32:
            if (tile_width == 16)
            {
                ret = TileDim::Dim32x16;
            }
            else if (tile_width == 32)
            {
                ret = TileDim::Dim32x32;
            }
            else
            {
                TT_ASSERT(false, "Invalid tile dim");
            }
            break;
        case 16: ret = TileDim::Dim16x32; break;
        case 8: ret = TileDim::Dim8x32; break;
        case 4: ret = TileDim::Dim4x32; break;
        case 2: ret = TileDim::Dim2x32; break;
        case 1: ret = TileDim::Dim1x32; break;
        default: TT_ASSERT(false, "Invalid tile dim");
    }
    return ret;
}

void validate_tile_dims(Graph *graph, const graphlib::OpNode *op_node)
{
    if (op_node->is_eltwise_binary())
    {
        auto srcA_tile_dim = graph->operands(op_node)[0]->shape().get_tile_dim();
        auto srcB_tile_dim = graph->operands(op_node)[1]->shape().get_tile_dim();
        if (srcA_tile_dim == srcB_tile_dim)
        {
            return;
        }

        // Canonicalize tile dim for binary op
        auto srcA_tile_volume = graph->operands(op_node)[0]->shape().get_tile_height() *
                                graph->operands(op_node)[0]->shape().get_tile_width();
        auto srcB_tile_volume = graph->operands(op_node)[1]->shape().get_tile_height() *
                                graph->operands(op_node)[1]->shape().get_tile_width();

        auto srcA_shape = graph->operands(op_node)[0]->shape();
        auto srcB_shape = graph->operands(op_node)[1]->shape();

        if (srcA_tile_volume > srcB_tile_volume)
        {
            graphlib::Shape trans_shape(true, Shape::Type::FREE, srcB_shape.as_vector());
            trans_shape.set_tile_dim(srcA_tile_dim);
            auto padded_srcB_shape = graphlib::Shape::to_forge(trans_shape);
            graph->operands(op_node)[1]->set_shape(padded_srcB_shape);
        }
        else if (srcA_tile_volume < srcB_tile_volume)
        {
            graphlib::Shape trans_shape(true, Shape::Type::FREE, srcA_shape.as_vector());
            trans_shape.set_tile_dim(srcB_tile_dim);
            auto padded_srcA_shape = graphlib::Shape::to_forge(trans_shape);
            graph->operands(op_node)[0]->set_shape(padded_srcA_shape);
        }
        else
        {
            // Volume match iff 32x16 and 16x32
            // Insert NOP to make sure both inputs are padded to 32x32
            TT_ASSERT(false, "Volume match but tile dims don't match");
        }

        TT_ASSERT(
            graph->operands(op_node)[0]->shape().get_tile_dim() == graph->operands(op_node)[1]->shape().get_tile_dim());
    }
    else if (op_node->is_matmul())
    {
        // check RHS matmul, set to full tile
        auto rhs = graph->operands(op_node)[1];

        if (rhs->shape().get_tile_dim() != TileDim::Dim32x32)
        {
            graphlib::Shape trans_shape(true, Shape::Type::FREE, rhs->shape().as_vector());
            trans_shape.set_tile_dim(TileDim::Dim32x32);
            auto padded_rhs_shape = graphlib::Shape::to_forge(trans_shape);
            rhs->set_shape(padded_rhs_shape);
        }
    }
    else if (op_node->is_reduce())
    {
        auto operand = graph->operands(op_node)[0];
        if (operand->shape().get_tile_dim() != TileDim::Dim32x32)
        {
            graphlib::Shape trans_shape(true, Shape::Type::FREE, operand->shape().as_vector());
            trans_shape.set_tile_dim(TileDim::Dim32x32);
            auto padded_shape = graphlib::Shape::to_forge(trans_shape);
            operand->set_shape(padded_shape);
        }
    }
    else if (op_node->is_embedding())
    {
        for (auto operand : graph->operands(op_node))
        {
            if (operand->shape().get_tile_dim() != TileDim::Dim32x32)
            {
                graphlib::Shape trans_shape(true, Shape::Type::FREE, operand->shape().as_vector());
                trans_shape.set_tile_dim(TileDim::Dim32x32);
                auto padded_shape = graphlib::Shape::to_forge(trans_shape);
                operand->set_shape(padded_shape);
            }
        }
    }

    return;
}

std::vector<std::vector<Node *>> topological_generations(const Graph &graph)
{
    std::vector<std::vector<Node *>> generations;

    // the first step is to discover top level nodes in the graph
    // queue up all visible nodes
    std::vector<Node *> nodes = graph.nodes();
    std::queue<Node *> node_queue;
    for (Node *node : nodes)
    {
        if (graph.is_node_visible(node))
        {
            node_queue.push(node);
        }
    }
    // vector to store top level nodes
    std::vector<Node *> top_level_nodes;
    std::unordered_map<NodeId, bool> visited{};

    std::function<void(Node *)> VisitNode = [&](Node *node)
    {
        visited[node->id()] = true;

        // count the number of operands of the node
        int num_operands = 0;
        for (const Edge &operand_edge : graph.operand_edges(node))
        {
            if (operand_edge.edge_type == EdgeType::kDataLoopback or
                operand_edge.edge_type == EdgeType::kPartialDataCopy)
            {
                continue;
            }
            else if (operand_edge.edge_type == EdgeType::kControlLoop)
            {
                continue;  // not unrolling loops, just terminate
            }
            num_operands++;

            NodeId predecessor_id = operand_edge.producer_node_id;
            Node *predecessor_node = graph.node_by_id(predecessor_id);
            if (requires_visit(visited, predecessor_id))
            {
                VisitNode(predecessor_node);
            }
        }
        if (num_operands == 0)
        {
            top_level_nodes.push_back(node);
        }
    };

    // recurse through node operands until top, then stop, and add to result
    while (not node_queue.empty())
    {
        Node *node = node_queue.front();

        if (requires_visit(visited, node->id()))
        {
            VisitNode(node);
        }
        node_queue.pop();
    }

    // now do a BFS through nodes
    std::queue<Node *> bfs_queue;

    // also store a mapping of each node to its level (or generation)
    std::unordered_map<NodeId, unsigned> node_to_level;

    // add top level nodes to the queue
    for (Node *node : top_level_nodes)
    {
        bfs_queue.push(node);
        node_to_level[node->id()] = 0;
    }

    // iterate through the queue
    // store processed nodes in a set
    std::unordered_set<NodeId> processed_nodes;
    while (not bfs_queue.empty())
    {
        Node *node = bfs_queue.front();
        bfs_queue.pop();

        // queue eligible children of this node
        for (const Edge &user_edge : graph.user_edges(node))
        {
            if (user_edge.edge_type == EdgeType::kControlLoop)
            {
                continue;  // not unrolling loops, just terminate
            }
            if (user_edge.edge_type == EdgeType::kDataLoopback or user_edge.edge_type == EdgeType::kPartialDataCopy)
            {
                continue;
            }
            NodeId user_id = user_edge.consumer_node_id;
            Node *user_node = graph.node_by_id(user_id);

            // if this node has already been processed, then skip it
            if (processed_nodes.find(user_id) != processed_nodes.end())
            {
                continue;
            }

            // if all the operands of this node already have levels, then this node will be inserted into the queue
            bool all_operands_have_levels = true;
            unsigned level = 0;
            for (const Edge &operand_edge : graph.operand_edges(user_node))
            {
                if (operand_edge.edge_type == EdgeType::kDataLoopback or
                    operand_edge.edge_type == EdgeType::kPartialDataCopy)
                {
                    continue;
                }
                else if (operand_edge.edge_type == EdgeType::kControlLoop)
                {
                    continue;  // not unrolling loops, just terminate
                }
                NodeId operand_id = operand_edge.producer_node_id;
                if (node_to_level.find(operand_id) == node_to_level.end())
                {
                    all_operands_have_levels = false;
                    break;
                }
                else
                {
                    level = std::max(level, node_to_level[operand_id]);
                }
            }
            // insert node into queue if all operands have levels
            if (all_operands_have_levels)
            {
                bfs_queue.push(user_node);
                node_to_level[user_id] = level + 1;
                // mark node as processed
                processed_nodes.insert(user_id);
            }
        }
    }

    // now that we have the levels, we can create the generations
    for (auto const &[node_id, level] : node_to_level)
    {
        if (generations.size() <= level)
        {
            generations.resize(level + 1);
        }
        generations[level].push_back(graph.node_by_id(node_id));
    }

    return generations;
}

std::vector<Node *> top_row(graphlib::Graph const *graph, std::vector<Node *> const &nodes)
{
    std::vector<Node *> sorted_nodes;

    // find the first generation that contains at least one of the nodes
    // iterate over each generation in topological_generations
    for (auto const &generation : topological_generations(*graph))
    {
        // iterate over each node to check if it belongs to this generation
        for (auto *n : nodes)
        {
            if (std::find(generation.begin(), generation.end(), n) != generation.end())
            {
                sorted_nodes.push_back(n);
            }
        }
        // if sorted_nodes is not empty, then we have found the first generation that contains at least one of the nodes
        if (sorted_nodes.size() > 0)
        {
            return sorted_nodes;
        }
    }
    return sorted_nodes;
}

std::vector<Node *> bot_row(graphlib::Graph const *graph, std::vector<Node *> const &nodes)
{
    std::vector<Node *> sorted_nodes;

    // find the last generation that contains at least one of the nodes
    // iterate over each generation in topological_generations in reverse order
    auto generations = topological_generations(*graph);
    // number of generations
    int num_generations = generations.size();

    // iterate over each generation in reverse order
    for (auto g = 0; g < num_generations; g++)
    {
        auto generation = generations[num_generations - g - 1];

        // iterate over each node to check if it belongs to this generation
        for (auto *n : nodes)
        {
            if (std::find(generation.begin(), generation.end(), n) != generation.end())
            {
                sorted_nodes.push_back(n);
            }
        }
        // if sorted_nodes is not empty, then we have found the last generation that contains at least one of the nodes
        if (sorted_nodes.size() > 0)
        {
            return sorted_nodes;
        }
    }
    return sorted_nodes;
}

std::vector<Node *> topological_sort(const Graph &graph, std::function<bool(Node *)> node_filter, bool unroll_loops)
{
    std::vector<Node *> result;
    std::unordered_map<NodeId, bool> visited{};
    std::unordered_map<Edge, int> control_loop_edge_to_iteration;

    std::vector<Node *> nodes = graph.nodes();
    std::queue<Node *> node_queue;
    for (Node *node : nodes)
    {
        if (graph.is_node_visible(node))
        {
            node_queue.push(node);
        }
    }

    std::function<void(Node *)> VisitNode = [&](Node *node)
    {
        visited[node->id()] = true;

        for (const Edge &operand_edge : graph.operand_edges(node))
        {
            if (operand_edge.edge_type == EdgeType::kDataLoopback or
                operand_edge.edge_type == EdgeType::kPartialDataCopy)
            {
                continue;
            }
            else if (operand_edge.edge_type == EdgeType::kControlLoop)
            {
                continue;  // not unrolling loops, just terminate
            }

            NodeId predecessor_id = operand_edge.producer_node_id;
            Node *predecessor_node = graph.node_by_id(predecessor_id);
            if (requires_visit(visited, predecessor_id))
            {
                VisitNode(predecessor_node);
            }
        }
        if (node_filter(node))
        {
            result.push_back(node);
        }

        if (unroll_loops)
        {
            for (const Edge &user_edge : graph.user_edges(node))
            {
                if (user_edge.edge_type == EdgeType::kControlLoop)
                {
                    auto loop_attributes = EdgeAttributes::as<LoopEdgeAttributes>(graph.get_edge_attributes(user_edge));
                    if (control_loop_edge_to_iteration.find(user_edge) == control_loop_edge_to_iteration.end())
                    {
                        control_loop_edge_to_iteration[user_edge] = 1;  // initialize loop count
                    }
                    if (control_loop_edge_to_iteration[user_edge] < loop_attributes->loop_iterations())
                    {
                        // Re-enqueue nodes in the same order they were originally intended to be processed
                        for (Node *node : nodes)
                        {
                            if (loop_attributes->is_processed_in_loop(node->id()))
                            {
                                visited[node->id()] = false;
                                node_queue.push(node);
                            }
                        }
                    }
                    control_loop_edge_to_iteration[user_edge] += 1;
                }
            }
        }
    };

    while (not node_queue.empty())
    {
        Node *node = node_queue.front();

        if (requires_visit(visited, node->id()))
        {
            VisitNode(node);
        }
        node_queue.pop();
    }
    return result;
}

std::vector<Node *> visible_nodes(Graph const &graph, std::function<bool(Node *)> node_filter)
{
    std::vector<Node *> result;

    for (Node *node : graph.nodes())
    {
        if (graph.is_node_visible(node) and node_filter(node))
        {
            result.push_back(node);
        }
    }

    return result;
}

std::vector<Node *> reachable_nodes(
    const Graph *graph, Node *start, std::function<bool(Node *)> node_filter, bool ancenstors_only)
{
    std::vector<Node *> result;
    std::unordered_map<NodeId, bool> visited{};

    std::function<void(Node *)> VisitNode = [&](Node *node)
    {
        visited[node->id()] = true;

        for (auto operand : graph->data_operands(node))
        {
            if (requires_visit(visited, operand->id()))
            {
                VisitNode(operand);
            }
        }
        if (node->node_type() != NodeType::kInput and not ancenstors_only)
        {
            for (auto user : graph->data_users(node))
            {
                if (requires_visit(visited, user->id()))
                {
                    VisitNode(user);
                }
            }
        }
        if (node_filter(node))
        {
            result.push_back(node);
        }
    };

    VisitNode(start);

    return result;
}

// Check if there exists a data edge between the two nodes(producer, consumer )
bool check_producer_consumer(Graph *graph, Node *producer, Node *consumer, std::function<bool(Node *)> node_filter)
{
    std::vector<graphlib::Node *> rc_nodes = reachable_nodes(graph, producer, node_filter, true);

    // if there exists a dependency between the two given nodes, return true
    return (std::find(rc_nodes.begin(), rc_nodes.end(), consumer) != rc_nodes.end());
}

// Find the longest path from the graph. Optionally look for paths that don't start from ordered inputs.
// TODO: write a few unit tests
std::vector<Node *> get_longest_path(const Graph *graph, bool from_inputs_only)
{
    std::unordered_map<Node *, int> cost;
    std::unordered_map<Node *, Node *> parent_map;

    if (from_inputs_only)
    {
        // set big negative numbers on all other inputs
        for (Node *node : graph->nodes()) cost.emplace(std::make_pair(node, std::numeric_limits<int>::lowest()));
        for (Node *node : graph->ordered_module_inputs()) cost[node] = 0;
    }

    int max_distance = std::numeric_limits<int>::lowest();
    Node *max_path_output = NULL;
    for (Node *node : topological_sort(*graph))
    {
        for (Node *user : graph->data_users(node))
        {
            if (cost[user] < cost[node] + 1)
            {
                cost[user] = cost[node] + 1;
                parent_map[user] = node;
            }
            if (cost[node] > max_distance)
            {
                max_distance = cost[node];
                max_path_output = node;
            }
        }
    }

    std::vector<Node *> max_path = {max_path_output};
    while (parent_map.find(max_path_output) != parent_map.end())
    {
        max_path_output = parent_map.at(max_path_output);
        max_path.push_back(max_path_output);
    }

    std::reverse(max_path.begin(), max_path.end());

    return max_path;
}

std::vector<Node *> get_nodes_with_indegree_zero(Graph *graph)
{
    std::vector<Node *> indegree_zero_nodes;
    for (Node *node : graph->nodes())
    {
        int num_operands = 0;
        for (auto operand : graph->operands(node))
        {
            if (operand->node_type() != NodeType::kInput)
            {
                num_operands++;
            }
        }
        if (num_operands == 0)
        {
            if (node->node_type() != NodeType::kInput)
            {
                indegree_zero_nodes.push_back(node);
            }
        }
    }
    return indegree_zero_nodes;
}

std::vector<Node *> get_nodes_with_outdegree_zero(Graph *graph)
{
    std::vector<Node *> outdegree_zero_nodes;
    for (Node *node : graph->nodes())
    {
        if (graph->users(node).size() == 0)
        {
            if (node->node_type() != NodeType::kInput)
            {
                outdegree_zero_nodes.push_back(node);
            }
        }
    }
    return outdegree_zero_nodes;
}

std::vector<Node *> get_nodes_with_data_outdegree_zero(Graph *graph)
{
    std::vector<Node *> outdegree_zero_nodes;
    for (Node *node : graph->nodes())
    {
        if (graph->user_data_edges(node).size() == 0)
        {
            if (node->node_type() != NodeType::kInput)
            {
                outdegree_zero_nodes.push_back(node);
            }
        }
    }
    return outdegree_zero_nodes;
}

DataFormat infer_data_format_from_py_tensor(const py::object &py_tensor)
{
    py::module_ tensor_module = py::module_::import("forge.tensor");
    py::object tensor_dtype = py_tensor.attr("dtype");

    try
    {
        return py::cast<DataFormat>(tensor_module.attr("pytorch_dtype_to_forge_dataformat")(tensor_dtype));
    }
    catch (const py::error_already_set &e)
    {
        throw std::runtime_error(
            "Encountered Python error while dtype to DataFormat mapping: " + std::string(e.what()));
    }
}

/**
 * Converts pytorch scalar type to data format.
 * C++ equivalent of pytorch_dtype_to_forge_dataformat in forge/forge/tensor.py.
 */
DataFormat scalar_type_to_data_format(const at::ScalarType scalar_type)
{
    switch (scalar_type)
    {
        case at::ScalarType::Float: return DataFormat::Float32;
        case at::ScalarType::Half: return DataFormat::Float16;
        case at::ScalarType::BFloat16: return DataFormat::Float16_b;
        case at::ScalarType::Byte:  // uint8
            log_warning("Parameter is uint8. Setting to Int32, since uint8 is not supported.");
            return DataFormat::Int32;
        case at::ScalarType::Char:  // int8
            log_warning("Parameter is int8. Setting to Int32, since int8 is not supported.");
            return DataFormat::Int32;
        case at::ScalarType::Bool:
            log_warning("Parameter is bool. Setting to Int32, since bool is not supported.");
            return DataFormat::Int32;
        case at::ScalarType::Int:  // int32
            return DataFormat::Int32;
        case at::ScalarType::Long:  // int64
            log_warning("Parameter is int64. Setting to int32, since int64 is not supported.");
            return DataFormat::Int32;
        default: throw std::runtime_error("Unsupported torch ScalarType: " + std::string(c10::toString(scalar_type)));
    }
}

/**
 * Converts data format to pytorch scalar type.
 * C++ equivalent of forge_dataformat_to_pytorch_dtype in forge/forge/tensor.py.
 */
at::ScalarType data_format_to_scalar_type(const DataFormat data_format)
{
    switch (data_format)
    {
        case DataFormat::Float32: return at::ScalarType::Float;
        case DataFormat::Float16:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2: return at::ScalarType::Half;
        case DataFormat::Float16_b:
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b: return at::ScalarType::BFloat16;
        case DataFormat::RawUInt8: return at::ScalarType::Byte;
        case DataFormat::RawUInt32:
        case DataFormat::UInt16:
        case DataFormat::Int32: return at::ScalarType::Int;
        default:
            TT_THROW("Unsupported DataFormat for casting: " + std::to_string(static_cast<int>(data_format)));
            unreachable();
    }
}

// Insert new node on the given edge. Node attributes will be picked up from consumer node.
std::pair<Edge, Edge> insert_node_on_edge(
    Graph *graph,
    Edge &edge,
    Node *node,
    bool inherit_consumer_attrs,
    bool remove_edge,
    std::uint32_t consumer_index,
    bool place_tms_on_outgoing)
{
    Node *consumer = graph->node_by_id(edge.consumer_node_id);
    Node *producer = graph->node_by_id(edge.producer_node_id);

    graph->copy_node_attributes(inherit_consumer_attrs ? consumer : producer, node);

    // Don't copy "gradient op" flag, since the last node is still the one accumulating
    if (node->node_type() == NodeType::kPyOp)
        node->as<graphlib::OpNode>()->set_gradient_op(false);

    // Create new edges
    Edge new_edge0 =
        Edge(edge.producer_node_id, edge.producer_output_port_id, node->id(), consumer_index, edge.edge_type);

    Edge new_edge1 = Edge(node->id(), 0, edge.consumer_node_id, edge.consumer_input_port_id, edge.edge_type);

    graph->add_edge(new_edge0);
    graph->add_edge(new_edge1);

    graph->copy_edge_attributes(edge, new_edge0);
    graph->copy_edge_attributes(edge, new_edge1);

    // TMs should be placed only on one of the edges.
    // Since we've copied all edge attributes (including TMs) to both edges,
    // we need to remove TMs from one of them.
    if (not place_tms_on_outgoing)
    {
        graph->get_edge_attributes(new_edge1)->set_tms({});
    }
    else
    {
        graph->get_edge_attributes(new_edge0)->set_tms({});
    }

    bool edges_added = false;
    for (Edge &e : graph->operand_edges(consumer))
    {
        // Adjust control & autograd edges
        if ((e.edge_type != EdgeType::kData) && (e.edge_type != EdgeType::kAutogradOutputToLoss) &&
            (e.edge_type != EdgeType::kAutogradInputToGradientOut) &&
            (e.edge_type != EdgeType::kAutogradFwdToGradient) && (e.edge_type != EdgeType::kAutogradFwdToRecompute)

        )
        {
            edges_added = true;
            graph->add_edge(graph->node_by_id(e.producer_node_id), node, e.producer_output_port_id, 0, e.edge_type);
        }
    }

    // If the producer was in backward (or optimizer) epoch, and there are fwd->bwd edges going to it,
    // the need to go to the new op, too
    if (not edges_added and producer->get_epoch_type() != graphlib::NodeEpochType::Forward)
    {
        for (Edge &e : graph->operand_edges(producer))
        {
            // Adjust control & autograd edges
            if ((e.edge_type == EdgeType::kAutogradFwdToBwd) || (e.edge_type == EdgeType::kAutogradFwdToOptimizer) ||
                (e.edge_type == EdgeType::kAutogradFwdToGradient))
            {
                graph->add_edge(graph->node_by_id(e.producer_node_id), node, e.producer_output_port_id, 0, e.edge_type);
            }
            // Move the kAutogradFwdToGradient edge, since we can only have one
            if (e.edge_type == EdgeType::kAutogradFwdToGradient)
            {
                graph->remove_edge(e);
            }
        }
    }
    // If the consumer of the edge we're trying to add a node on is a "recompute-node",
    // we need to also create replicated fwd->recompute edges on the newly added node.
    // this is to keep track of which nodes are considered to be "recompute".
    for (Edge &e : graph->operand_edges(consumer))
    {
        if (e.edge_type == EdgeType::kAutogradFwdToRecompute)
        {
            Node *fwd_node_being_recompute = graph->node_by_id(e.producer_node_id);
            graph->add_edge(fwd_node_being_recompute, node, e.producer_output_port_id, 0, e.edge_type);
        }
    }

    if (remove_edge)
    {
        graph->remove_edge(edge);
    }

    return std::make_pair(new_edge0, new_edge1);
}

// Copy non-data edges from old dest to new
void copy_control_edges(Graph *graph, Node *old_dest, Node *new_dest)
{
    std::vector<Node *> data_operands = graph->data_operands(old_dest);
    Node *data_operand = data_operands.at(0);

    for (Edge &e : graph->user_edges(old_dest))
    {
        if (e.edge_type == EdgeType::kData)
        {
            continue;
        }

        // Copy control & autograd edges
        if (e.edge_type == EdgeType::kControl)
        {
            graph->add_edge(new_dest, graph->node_by_id(e.consumer_node_id), 0, 0, e.edge_type);
        }
        else
        {
            // if it's an autograd-edge between <NODE_TO_DELETE> -> consumer, we'll reassign
            // the edge to the producer node since `new_dest` may be an output node
            graph->add_edge(data_operand, graph->node_by_id(e.consumer_node_id), 0, 0, e.edge_type);
        }
    }
}

// Copy non-data edges when removing a node
void handle_control_edges_when_removing_node(Graph *graph, Node *node_being_removed)
{
    std::vector<Edge> operand_data_edges = graph->operand_data_edges(node_being_removed);
    TT_ASSERT(
        operand_data_edges.size() == 1,
        "Tried to handle control edges, but node being removed has more than 1 operand!");

    Edge &producer_to_nbr_edge = operand_data_edges.front();
    Node *producer = graph->node_by_id(producer_to_nbr_edge.producer_node_id);

    auto is_not_data_edge = [](Edge e) { return (e.edge_type != EdgeType::kData); };
    std::vector<Edge> operand_edges = graph->operand_edges(node_being_removed, is_not_data_edge);
    std::vector<Edge> user_edges = graph->user_edges(node_being_removed, is_not_data_edge);

    // Handle operand edges
    for (Edge &o_e : operand_edges)
    {
        if (node_being_removed->is_forward())
        {
            if (o_e.edge_type == EdgeType::kControl)
            {
                for (Edge &user : graph->user_data_edges(node_being_removed))
                {
                    Edge new_edge(
                        o_e.producer_node_id,
                        o_e.producer_output_port_id,
                        user.consumer_node_id,
                        user.consumer_input_port_id,
                        o_e.edge_type);
                    graph->add_edge(new_edge);
                }
            }
            else
            {
                TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(o_e.edge_type));
            }
        }
        else if (node_being_removed->is_backward())
        {
            if (o_e.edge_type == EdgeType::kAutogradFwdToBwd)
            {
                // We can just "delete" this edge, i.e. not copy it
                continue;
            }
            if (o_e.edge_type == EdgeType::kAutogradFwdToGradient)
            {
                graph->add_edge(graph->node_by_id(o_e.producer_node_id), producer, o_e.edge_type);
                continue;
            }
            if (o_e.edge_type == EdgeType::kAutogradFwdToRecompute)
            {
                // We can just "delete" this edge, i.e. not copy it
                continue;
            }

            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(o_e.edge_type));
        }
        else if (node_being_removed->is_optimizer())
        {
            if (o_e.edge_type == EdgeType::kAutogradFwdToOptimizer)
            {
                // We can just "delete" this edge, i.e. not copy it
                continue;
            }

            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(o_e.edge_type));
        }

        // TODO: Other control edges
    }

    // Handle user edges
    for (Edge &u_e : user_edges)
    {
        if (node_being_removed->is_forward())
        {
            if (u_e.edge_type == EdgeType::kAutogradFwdToBwd)
            {
                // Push the edge to parent of node being removed
                graph->add_edge(producer, graph->node_by_id(u_e.consumer_node_id), u_e.edge_type);
                continue;
            }
            if (u_e.edge_type == EdgeType::kAutogradFwdToOptimizer)
            {
                graph->add_edge(producer, graph->node_by_id(u_e.consumer_node_id), u_e.edge_type);
                continue;
            }
            if (u_e.edge_type == EdgeType::kAutogradFwdToGradient)
            {
                // Since there will be no fwd node anymore, we can just delete this edge
                continue;
            }
            if (u_e.edge_type == EdgeType::kAutogradFwdToRecompute)
            {
                // Moving this edge from nbr(fwd)->recompute(bwd) to nbr's_parent(fwd)->recompute(bwd)
                // Not sure this makes sense though, depends what the edge is used for later on
                graph->add_edge(producer, graph->node_by_id(u_e.consumer_node_id), u_e.edge_type);
                continue;
            }

            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(u_e.edge_type));
        }
        else if (node_being_removed->is_backward())
        {
            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(u_e.edge_type));
        }
        else if (node_being_removed->is_optimizer())
        {
            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(u_e.edge_type));
        }
        // TODO: Other control edges
    }
}

// Creates buffering queue and adds it to the graph. Returns pointer to created queue node.
// Queue inherits shape output_df, and epoch_type from producer node.
graphlib::QueueNode *create_buffering_queue(
    Graph *graph, const graphlib::Node *producer_node, const std::string name, int num_entries)
{
    TT_ASSERT(num_entries > 0, "Number of entries in queue has to be greater than 0");
    if (num_entries > graph->get_microbatch())
    {
        log_warning(
            "Wasting DRAM. Number of entries in queue is greater than microbatch size. For buffering queue the "
            "theoretical maximum number of entries is equal to microbatch size.");
    }

    // Create new queue
    std::unique_ptr<graphlib::BufferingQueueNode> queue_node_unique =
        graphlib::create_node<graphlib::BufferingQueueNode>(name, num_entries);
    queue_node_unique->set_shape(producer_node->shape());
    queue_node_unique->set_output_df(producer_node->output_df());
    queue_node_unique->set_epoch_type(producer_node->get_epoch_type());

    graphlib::QueueNode *queue =
        graph->add_node(std::move(queue_node_unique), graph->get_subgraph_id_for_node(producer_node->id()));
    return queue;
}

// Bypass queue, connecting its source to its destination. There has to be only one source for queue, and user is
// defined by user_edge.
std::unique_ptr<Node> connect_queue_src_to_queue_user(Graph *graph, Node *queue, Edge &user_edge, bool remove_queue)
{
    TT_ASSERT(queue->node_type() == NodeType::kQueue, " provided node has to be NodeType::kQueue");
    std::vector<Edge> op_edges = graph->operand_data_edges(queue);
    TT_ASSERT(op_edges.size() == 1, "connect_queue_src_to_queue_user can only be called on nodes with one operand");

    Edge src_edge = op_edges[0];
    std::vector<ops::Op> operand_tms = graph->get_edge_attributes(src_edge)->get_tms();

    // if we want to remove queue at the end, we won't remove user_edge now since it will be done in
    // graph->remove_node() if we only wan't to connect queue src to its dest (determined by user_edge), we will delete
    // user_edge.
    std::shared_ptr<EdgeAttributes> user_edge_attrs =
        remove_queue ? graph->get_edge_attributes(user_edge) : graph->remove_edge(user_edge);
    std::vector<ops::Op> user_tms = user_edge_attrs->get_tms();

    Edge new_edge(
        src_edge.producer_node_id,
        src_edge.producer_output_port_id,
        user_edge.consumer_node_id,
        user_edge.consumer_input_port_id,
        user_edge.edge_type);
    graph->add_edge(new_edge);

    std::vector<ops::Op> new_edge_tms;
    new_edge_tms.insert(new_edge_tms.end(), operand_tms.begin(), operand_tms.end());
    new_edge_tms.insert(new_edge_tms.end(), user_tms.begin(), user_tms.end());

    auto new_edge_attributes = graph->get_edge_attributes(new_edge);
    new_edge_attributes->set_tms(new_edge_tms);
    new_edge_attributes->set_ublock_order(user_edge_attrs->get_ublock_order());

    return remove_queue ? graph->remove_node(queue) : nullptr;
}

// Bypass node, connecting its source to its destination(s). The node must only have one input operand.
// Optionally, user can provide callback on each of the newly created edges, and original edge.
std::unique_ptr<Node> bypass_node(Graph *graph, Node *node, bool remove_node, std::function<void(Edge, Edge)> callback)
{
    std::vector<Edge> op_edges = graph->operand_data_edges(node);
    TT_ASSERT(op_edges.size() == 1, "bypass_node can only be called on nodes with one operand");

    Edge src_edge = op_edges[0];
    std::vector<ops::Op> operand_tms = graph->get_edge_attributes(src_edge)->get_tms();

    for (Edge &user : graph->user_data_edges(node))
    {
        std::vector<ops::Op> user_tms = graph->get_edge_attributes(user)->get_tms();

        Edge new_edge(
            src_edge.producer_node_id,
            src_edge.producer_output_port_id,
            user.consumer_node_id,
            user.consumer_input_port_id,
            user.edge_type);
        graph->add_edge(new_edge);

        std::vector<ops::Op> new_edge_tms;
        new_edge_tms.insert(new_edge_tms.end(), operand_tms.begin(), operand_tms.end());
        new_edge_tms.insert(new_edge_tms.end(), user_tms.begin(), user_tms.end());

        auto new_edge_attributes = graph->get_edge_attributes(new_edge);
        new_edge_attributes->set_tms(new_edge_tms);
        new_edge_attributes->set_ublock_order(graph->get_edge_attributes(user)->get_ublock_order());

        callback(new_edge, user);
    }

    handle_control_edges_when_removing_node(graph, node);

    OpNode *op_node = dynamic_cast<OpNode *>(node);
    if (op_node and op_node->is_gradient_op())
    {
        OpNode *producer_op_node = dynamic_cast<OpNode *>(graph->node_by_id(src_edge.producer_node_id));
        if (producer_op_node)
            producer_op_node->set_gradient_op();
    }

    return remove_node ? graph->remove_node(node) : nullptr;
}

// Replace node with a new one, removing the old one and reconnecting all edges as before.
// The new node must have the same number of operands, or skip_operands must be set.
void replace_node(Graph *graph, Node *original_node, Node *new_node, bool skip_operands)
{
    if (!skip_operands)
    {
        for (Edge &operand : graph->operand_data_edges(original_node))
        {
            Edge new_edge = Edge(
                operand.producer_node_id,
                operand.producer_output_port_id,
                new_node->id(),
                operand.consumer_input_port_id,
                operand.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(operand, new_edge);
        }
    }

    for (Edge &user : graph->user_edges(original_node))
    {
        if (user.edge_type == graphlib::EdgeType::kData)
        {
            Edge new_edge = Edge(
                new_node->id(),
                (graphlib::PortId)0,
                user.consumer_node_id,
                user.consumer_input_port_id,
                user.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(user, new_edge);
        }
    }

    copy_control_edges(graph, original_node, new_node);
    graph->copy_node_attributes(original_node, new_node);
    graph->remove_node(original_node);
}

Edge swap(Graph *graph, Edge edge, std::function<void(Edge)> operand_callback, std::function<void(Edge)> user_callback)
{
    auto replace_edge = [graph](Edge orig, Edge new_edge)
    {
        auto attr = graph->get_edge_attributes(orig);
        graph->remove_edge(orig);
        graph->add_edge(new_edge, attr);
    };

    Node *producer = graph->node_by_id(edge.producer_node_id);
    Node *consumer = graph->node_by_id(edge.consumer_node_id);
    auto producer_operands = graph->operand_data_edges(producer);
    auto consumer_users = graph->user_data_edges(consumer);

    TT_ASSERT(producer_operands.size() == 1, "swap is only compatible with unary producers");

    // Swap the orientation of the original edge
    auto swapped_edge = edge;
    std::swap(swapped_edge.producer_node_id, swapped_edge.consumer_node_id);
    swapped_edge.consumer_input_port_id = 0;
    replace_edge(edge, swapped_edge);

    // Producer operand point to consumer
    auto producer_operand = producer_operands.front();
    auto remap_producer = producer_operand;
    remap_producer.consumer_node_id = consumer->id();
    remap_producer.consumer_input_port_id = edge.consumer_input_port_id;
    replace_edge(producer_operand, remap_producer);

    for (auto const &operand : graph->operand_data_edges(consumer))
    {
        operand_callback(operand);
    }

    // Consumer users map to producer
    for (auto const &user : consumer_users)
    {
        auto new_user = user;
        new_user.producer_node_id = producer->id();
        replace_edge(user, new_user);
        user_callback(new_user);
    }

    return swapped_edge;
}

std::vector<Node *> subgraph(const Graph *graph, Node *producer, Node *consumer)
{
    bool found = false;
    std::unordered_map<Node *, std::vector<Node *>> deps;
    std::unordered_set<Node *> visited;
    std::vector<Node *> visit = {producer};
    while (not visit.empty())
    {
        Node *node = visit.back();
        visit.pop_back();
        for (Node *user : graph->data_users(node))
        {
            deps[user].push_back(node);

            if (user == consumer)
            {
                // We can stop visiting this path since we hit the consumer
                found = true;
            }
            else if (visited.find(user) == visited.end())
            {
                // Only continue to visit nodes that haven't been visited yet
                visit.push_back(user);
            }

            visited.insert(user);
        }
    }

    if (not found)
        return {};

    std::vector<Node *> sub;

    visit = deps[consumer];
    while (not visit.empty())
    {
        std::vector<Node *> next;
        for (Node *node : visit)
        {
            if (node == producer)
                continue;

            sub.push_back(node);
            auto const &d = deps.at(node);
            next.insert(next.end(), d.begin(), d.end());
        }
        std::swap(visit, next);
    }

    return sub;
}

void convert_implicit_to_explicit_bcasts(Graph *graph, Edge edge)
{
    auto edge_attr = graph->get_edge_attributes(edge);
    for (ops::Op &op_type : graph->get_edge_attributes(edge)->get_tms())
    {
        if (op_type.type() == ops::OpType::Broadcast)
            op_type.set_attr("explicit_bcast", true);
    }
}

graphlib::Node *cascade_nary_to_binary_op(graphlib::Graph *graph, graphlib::Node *nary_op)
{
    auto operands = graph->operand_data_edges(nary_op);
    TT_ASSERT(operands.size() >= 2, nary_op->name(), operands.size());
    if (operands.size() == 2)
        return nary_op;

    graphlib::Node *sink = graph->add_node(
        nary_op->clone(nary_op->name() + "_cascade_sink"), graph->get_subgraph_id_for_node(nary_op->id()));
    for (int i = 0; i < ((int)operands.size() / 2); ++i)
    {
        graphlib::Edge operand_a = operands[i * 2];
        graphlib::Edge operand_b = operands[i * 2 + 1];
        auto attrs_a = graph->get_edge_attributes(operand_a);
        auto attrs_b = graph->get_edge_attributes(operand_b);
        graphlib::Node *add = graph->add_node(
            nary_op->clone(nary_op->name() + "_cascade_" + std::to_string(i)),
            graph->get_subgraph_id_for_node(nary_op->id()));
        operand_a.consumer_input_port_id = 0;
        operand_a.consumer_node_id = add->id();
        operand_b.consumer_input_port_id = 1;
        operand_b.consumer_node_id = add->id();
        graph->add_edge(operand_a, attrs_a);
        graph->add_edge(operand_b, attrs_b);

        graphlib::Edge sink_edge(add->id(), 0, sink->id(), i, graphlib::EdgeType::kData);
        graph->add_edge(sink_edge);
    }

    if ((operands.size() % 2) != 0)
    {
        graphlib::Edge back = operands.back();
        graphlib::Edge sink_edge(back.producer_node_id, 0, sink->id(), operands.size() - 1, graphlib::EdgeType::kData);
        graph->add_edge(sink_edge);
    }

    for (graphlib::Edge user : graph->user_data_edges(nary_op))
    {
        user.producer_node_id = sink->id();
        graph->add_edge(user);
    }

    graph->remove_node(nary_op);
    return cascade_nary_to_binary_op(graph, sink);
}

bool swap_broadcast_dims(graphlib::Graph *graph, graphlib::Edge edge, int old_dim, int new_dim)
{
    bool swapped = false;
    auto tms = graph->get_edge_attributes(edge)->get_tms();
    std::vector<ops::Op> new_tms;
    for (ops::Op &op_type : tms)
    {
        if (op_type.type() == ops::OpType::Broadcast)
        {
            int dim = op_type.attr_as<int>("dim");
            int size = op_type.attr_as<int>("size");
            bool explicit_bcast = op_type.attr_as<bool>("explicit_bcast");
            if (dim == old_dim)
            {
                new_tms.push_back(ops::Op(
                    ops::OpType::Broadcast, {{"dim", new_dim}, {"size", size}, {"explicit_bcast", explicit_bcast}}));
                swapped = true;
            }
            else
            {
                new_tms.push_back(op_type);
            }
        }
        else
        {
            new_tms.push_back(op_type);
        }
    }
    graph->get_edge_attributes(edge)->set_tms(new_tms);
    return swapped;
}

void handle_change_rank(graphlib::Graph *graph, graphlib::Edge edge)
{
    auto get_consumer_size = [](std::uint32_t producer_size, graphlib::Node *node)
    {
        std::uint32_t consumer_size = node->shape().size();
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            return consumer_size;
        if (op->op_type() == ops::OpType::Reshape)
            return producer_size;
        if (op->op_type() == ops::OpType::Squeeze)
            return (consumer_size + 1);
        if (op->op_type() == ops::OpType::Unsqueeze)
            return (consumer_size - 1);
        // For reduction ops with keep_dim=False, the input rank equals output rank + number of reduced dims.
        // Without this, handle_change_rank would incorrectly insert squeezes before the reduction op
        // when a producer's rank increases after decomposition (e.g. after Layernorm decomposition).
        if (op->is_reduce() && !op->op().attr_as<bool>("keep_dim"))
        {
            auto dims = op->op().attr_as<std::vector<int>>("dim_arg");
            return consumer_size + static_cast<std::uint32_t>(dims.size());
        }
        return consumer_size;
    };

    auto producer_size = graph->node_by_id(edge.producer_node_id)->shape().size();
    auto consumer_size = get_consumer_size(producer_size, graph->node_by_id(edge.consumer_node_id));

    if (producer_size == consumer_size)
        return;

    graphlib::OpNode *consumer = dynamic_cast<graphlib::OpNode *>(graph->node_by_id(edge.consumer_node_id));
    if (consumer and consumer->op_type() == ops::OpType::Embedding)
        return;

    // This is one of the few cases where we actually want to move tms downstream
    auto tms = graph->get_edge_attributes(edge)->get_tms();
    graph->get_edge_attributes(edge)->set_tms({});

    auto insert = [graph](graphlib::Edge edge, ops::OpType op_type, std::uint32_t rank) -> graphlib::Edge
    {
        graphlib::Node *producer = graph->node_by_id(edge.producer_node_id);
        graphlib::Node *consumer = graph->node_by_id(edge.consumer_node_id);
        graphlib::OpNode *inherit = dynamic_cast<graphlib::OpNode *>(consumer)
                                        ? dynamic_cast<graphlib::OpNode *>(consumer)
                                        : dynamic_cast<graphlib::OpNode *>(producer);
        TT_ASSERT(inherit);
        // If there are 2 edges from the same producer to the same consumer (eg. eltwise binary op),
        // need edge_creation_id to differentiate naming.
        std::string name = producer->name() + "_" + consumer->name() + "_" + ops::Op(op_type).as_string() +
                           std::to_string(rank) + "_" + std::to_string(edge.edge_creation_id);
        graphlib::OpNode *change_rank = dynamic_cast<graphlib::OpNode *>(
            graph->add_node(inherit->clone(name), graph->get_subgraph_id_for_node(producer->id())));
        TT_ASSERT(change_rank);
        change_rank->change_op(op_type, {{"dim", 0}});
        change_rank->set_shape(producer->shape().as_rank(rank));
        change_rank->tag("dont_erase", true);
        auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, edge, change_rank);
        if (try_consteval_op(graph, change_rank))
            return graph->operand_data_edges(consumer)[0];

        // Set dataformat to match producer
        change_rank->set_output_df(producer->output_df());
        return outgoing_edge;
    };

    int orig_producer_size = (int)producer_size;
    while (producer_size < consumer_size)
    {
        producer_size++;
        edge = insert(edge, ops::OpType::Unsqueeze, producer_size);
    }

    while (producer_size > consumer_size)
    {
        producer_size--;
        TT_ASSERT(producer_size > 0);
        edge = insert(edge, ops::OpType::Squeeze, producer_size);
    }

    int diff = (int)producer_size - orig_producer_size;
    for (ops::Op &op : tms)
    {
        if (op.type() == ops::OpType::Broadcast)
        {
            int dim = op.attr_as<int>("dim");
            if (dim >= 0)
            {
                op.set_attr("dim", dim + diff);
            }
        }
    }
    graph->get_edge_attributes(edge)->set_tms(tms);
}

void handle_change_rank(graphlib::Graph *graph, graphlib::Node *node)
{
    for (graphlib::Edge e : graph->operand_data_edges(node)) handle_change_rank(graph, e);
    for (graphlib::Edge e : graph->user_data_edges(node)) handle_change_rank(graph, e);
}

graphlib::Edge clone_input_forking_edge(graphlib::Graph *graph, graphlib::Edge user_edge, bool allow_single_user)
{
    Node *input = graph->node_by_id(user_edge.producer_node_id);
    TT_ASSERT(input->node_type() == NodeType::kInput);
    TT_ASSERT(graph->data_operands(input).empty(), "Cannot clone a loopback input");
    TT_ASSERT(graph->data_users(input).size() > 1 or allow_single_user, "Cannot clone input that doesn't fork");
    Node *clone = graph->add_node(
        input->clone(input->name() + "_fork_clone" + std::to_string(user_edge.consumer_node_id)),
        graph->get_subgraph_id_for_node(input->id()));

    auto edge_attr = graph->get_edge_attributes(user_edge);
    graph->remove_edge(user_edge);
    graphlib::Edge new_edge(
        clone->id(),
        user_edge.producer_output_port_id,
        user_edge.consumer_node_id,
        user_edge.consumer_input_port_id,
        user_edge.edge_type);
    graph->add_edge(new_edge, edge_attr);
    return new_edge;
}

graphlib::Shape default_tm_evaluator(ops::Op const &tm, graphlib::Shape shape, graphlib::IRLevel ir_level)
{
    std::vector<Shape> shapes = {shape};
    std::tuple<Shape, std::vector<DimBroadcast>> shape_data = get_op_shape(tm, shapes);
    shape = std::get<0>(shape_data);
    TT_ASSERT(std::get<1>(shape_data).size() == 0, "TMs should not cause broadcasts");
    return shape;
}

// Calculate node shape from operand shapes, using python callback
void calculate_and_set_node_shape(Graph *graph, Node *node)
{
    log_trace(LogGraphCompiler, "Calculate and set node shape for: {} {}", node->name(), node->get_type());
    // Apply TMs and get post-TM operand shapes
    std::vector<Shape> operand_shapes;

    // Validate / Canonicalize TileDim
    auto op_node = dynamic_cast<graphlib::OpNode *>(node);
    if (op_node)
    {
        validate_tile_dims(graph, op_node);
    }

    for (graphlib::Edge &e : graph->operand_data_edges(node))
    {
        auto operand_shape = graph->node_by_id(e.producer_node_id)->shape();
        std::vector<ops::Op> tms = graph->get_edge_attributes(e)->get_tms();
        for (ops::Op tm : tms)
        {
            std::vector<Shape> shapes = {operand_shape};
            std::tuple<Shape, std::vector<DimBroadcast>> shape_data = get_op_shape(tm, shapes);
            operand_shape = std::get<0>(shape_data);
            TT_ASSERT(std::get<1>(shape_data).size() == 0, "TMs should not cause broadcasts");
            log_trace(LogGraphCompiler, "    TM {} {}", tm.as_string(), operand_shape);
        }
        log_trace(
            LogGraphCompiler,
            "  Operand[{}] {} {}",
            e.consumer_input_port_id,
            operand_shape,
            graph->node_by_id(e.producer_node_id)->name());
        operand_shapes.push_back(operand_shape);
    }

    if ((node->node_type() == graphlib::NodeType::kOutput) || (node->node_type() == graphlib::NodeType::kQueue))
    {
        // Graph shape from first, and only, operand
        TT_ASSERT(operand_shapes.size() == 1, "Node should have exactly one operand");
        node->set_shape(operand_shapes[0]);
        return;
    }

    if (node->node_type() != NodeType::kPyOp)
        return;

    ops::Op op_type = dynamic_cast<graphlib::OpNode *>(node)->op();

    std::tuple<Shape, std::vector<DimBroadcast>> shape_data = get_op_shape(op_type, operand_shapes);

    log_trace(LogGraphCompiler, "  {}", std::get<0>(shape_data));
    node->set_shape(std::get<0>(shape_data));

    // Set broadcast attributes on edges
    for (graphlib::Edge &e : graph->operand_data_edges(node))
    {
        for (DimBroadcast &b : std::get<1>(shape_data))
        {
            log_trace(LogGraphCompiler, "  brcst {} {} {}", std::get<0>(b), std::get<1>(b), std::get<2>(b));

            int operand = std::get<0>(b);
            if (operand == (int)e.consumer_input_port_id)
            {
                int dim = std::get<1>(b);
                int size = std::get<2>(b);
                graph->get_edge_attributes(e)->set_broadcast_dim(dim, size);
            }
        }
    }
}

// Return a vector of pairs of optimizer parameter input nodes and optimizer key names for a given model parameter node
std::vector<std::pair<InputNode *, std::string>> get_optimizer_param_info(
    const Graph *graph, const Node *model_parameter)
{
    // If autograd has run, there will be EdgeType::kAutogradFwdToOptimizer edges. We parse through this
    // list looking for inputs that require its tensors to be populated by the python-side optimizer obj
    std::vector<std::pair<InputNode *, std::string>> ret;
    for (graphlib::Edge edge : graph->user_edges(model_parameter))
    {
        if (edge.edge_type != graphlib::EdgeType::kAutogradFwdToOptimizer)
            continue;
        if (graph->node_by_id(edge.consumer_node_id)->node_type() != NodeType::kInput)
            continue;

        graphlib::InputNode *input = graph->node_by_id(edge.consumer_node_id)->as<graphlib::InputNode>();
        if (not input->is_optimizer_parameter())
        {
            continue;
        }

        // Parse out the optimizer-param suffix string and do a lookup to get the tensor
        std::string optimizer_input_name = input->name();
        std::string::size_type optimizer_param_idx = optimizer_input_name.rfind('.');
        TT_ASSERT(
            optimizer_param_idx != std::string::npos,
            "Expecting optimizer node to have a '.<optimizer-param>' suffix identifier");

        std::string optimizer_param_key = optimizer_input_name.substr(optimizer_param_idx + 1);
        ret.push_back(std::make_pair(input, optimizer_param_key));
    }
    return ret;
}

bool is_constant_input(const Node *node)
{
    graphlib::InputNode const *input = dynamic_cast<graphlib::InputNode const *>(node);
    return input and input->is_constant();
}

bool is_recompute(const Graph *graph, const Node *node)
{
    for (const Edge &edge : graph->operand_edges(node))
    {
        if (edge.edge_type == graphlib::EdgeType::kAutogradFwdToRecompute)
        {
            return true;
        }
    }
    return false;
}

Node *get_fwd_from_recompute(const Graph *graph, const Node *node)
{
    for (const Edge &edge : graph->operand_edges(node))
    {
        if (edge.edge_type == graphlib::EdgeType::kAutogradFwdToRecompute)
        {
            return graph->node_by_id(edge.producer_node_id);
        }
    }
    return nullptr;
}

ConstEvalGraph::ConstEvalGraph(
    std::string const &name, Node *runtime_input, bool promote_input, unsigned int subgraph_id, int unique_id) :
    consteval_graph(IRLevel::IR_CONSTEVAL, name, unique_id == -1 ? Graph::generate_unique_graph_id() : unique_id),
    runtime_input(runtime_input),
    subgraph_id_(subgraph_id)
{
    TT_ASSERT(runtime_input->node_type() == NodeType::kInput);
    if (promote_input)
        promote_node(nullptr, runtime_input, runtime_input->clone());
}

std::unique_ptr<Node> ConstEvalGraph::promote_node(std::unique_ptr<Node> &&consteval_node)
{
    return promote_node(nullptr, nullptr, std::forward<std::unique_ptr<Node>>(consteval_node));
}

std::unique_ptr<Node> ConstEvalGraph::promote_node(Graph *runtime_graph, Node *runtime_node)
{
    return promote_node(runtime_graph, runtime_node, runtime_node->clone());
}

std::unique_ptr<Node> ConstEvalGraph::promote_node(
    Graph *runtime_graph, Node *runtime_node, std::unique_ptr<Node> &&consteval_node_free)
{
    TT_ASSERT(not runtime_graph or runtime_node);
    TT_ASSERT(not runtime_graph or runtime_graph->get_ir_level() == IRLevel::IR_TT_FORGE);

    graph_updated_since_autograd = true;

    Node *consteval_node = consteval_graph.add_node<Node>(std::move(consteval_node_free), subgraph_id_);

    // Promoted consteval nodes are always in the forward epoch for their respective consteval graph
    // ConstEvalGraph will automatically run its own autograd and insert its own, respective BW ops
    consteval_node->set_epoch_type(NodeEpochType::Forward);

    if (consteval_output)
    {
        // Runtime input node needs to always map to the consteval graph output
        auto output_operands = consteval_graph.data_operands(consteval_output);
        TT_ASSERT(output_operands.size() == 1);
        runtime_to_consteval_map[runtime_input->id()] = output_operands[0]->id();
    }

    // Create mapping from runtime node id to consteval
    if (runtime_node)
    {
        runtime_to_consteval_map.insert({runtime_node->id(), consteval_node->id()});
    }

    // Create edges inherited from the runtime_graph
    if (runtime_graph)
    {
        for (Edge const &runtime_edge : runtime_graph->operand_data_edges(runtime_node))
        {
            auto runtime_attr = runtime_graph->get_edge_attributes(runtime_edge);
            int const_producer_id = 0;

            if (runtime_to_consteval_map.find(runtime_edge.producer_node_id) == runtime_to_consteval_map.end())
            {
                InputNode *runtime_operand =
                    dynamic_cast<InputNode *>(runtime_graph->node_by_id(runtime_edge.producer_node_id));
                TT_ASSERT(runtime_operand, "All operands of promoted nodes must be graph inputs");
                Node *consteval_operand = nullptr;

                // Only add the node if it doesn't already exist in the consteval graph
                if (ConstEvalGraph *nested_consteval_graph = runtime_operand->get_consteval_graph())
                    consteval_operand = graft(nested_consteval_graph->get_graph());
                else if (!consteval_graph.has_node_with_name(runtime_operand->name()))
                    consteval_operand = consteval_graph.add_node<Node>(runtime_operand->clone(), subgraph_id_);
                else
                    consteval_operand = consteval_graph.get_node_by_name(runtime_operand->name());

                // Only map the operand if it has 1 user
                if (runtime_graph->user_data_edges(runtime_operand).size() > 1)
                    const_producer_id = consteval_operand->id();
                else if (runtime_graph->user_data_edges(runtime_operand).size() == 1)
                    runtime_to_consteval_map.insert({runtime_operand->id(), consteval_operand->id()});

                runtime_graph->remove_edge(runtime_edge);
                auto users = runtime_graph->user_edges(runtime_operand);
                if (users.empty())
                    runtime_graph->remove_node(runtime_operand);
            }

            Edge consteval_edge = Edge(
                const_producer_id ? const_producer_id : runtime_to_consteval_map.at(runtime_edge.producer_node_id),
                runtime_edge.producer_output_port_id,
                runtime_to_consteval_map.at(runtime_edge.consumer_node_id),
                runtime_edge.consumer_input_port_id,
                runtime_edge.edge_type);

            consteval_graph.add_edge(consteval_edge);
            consteval_graph.get_edge_attributes(consteval_edge)->copy_from(*runtime_attr);
            runtime_attr->get_tms().clear();  // remove all operand runtime tms, they are consumed by consteval
        }
    }
    else if (dynamic_cast<graphlib::OpNode *>(consteval_node))
    {
        TT_ASSERT(consteval_output);
        // If there is no runtime graph then new consteval nodes are simply appended as the new output node
        Edge output_edge = consteval_graph.operand_data_edges(consteval_output).at(0);
        Edge new_edge(
            output_edge.producer_node_id,
            output_edge.producer_output_port_id,
            consteval_node->id(),
            0,
            EdgeType::kData);
        consteval_graph.add_edge(new_edge);
    }

    // Connect to the graph output
    if (consteval_output)
    {
        consteval_graph.remove_edge(consteval_graph.operand_data_edges(consteval_output).at(0));
    }
    else
    {
        consteval_output = consteval_graph.add_node<Node>(
            std::make_unique<OutputNode>(consteval_graph.name() + ".output"), subgraph_id_);
    }

    Edge consteval_edge(consteval_node->id(), 0, consteval_output->id(), 0, EdgeType::kData);
    consteval_graph.add_edge(consteval_edge);

    runtime_input->set_shape(consteval_node->shape());
    runtime_input->set_output_df(consteval_node->output_df());
    consteval_output->set_shape(consteval_node->shape());
    consteval_output->set_output_df(consteval_node->output_df());

    if (runtime_graph)
    {
        if (runtime_graph->operand_data_edges(runtime_node).size() == 1)
        {
            return graphlib::bypass_node(runtime_graph, runtime_node, true /*remove_node*/);
        }
    }
    return nullptr;
}

Node *ConstEvalGraph::graft(Graph *other)
{
    NodeId other_output_op_id = -1;
    std::unordered_map<NodeId, NodeId> node_id_map;
    std::vector<Node *> nodes = other->nodes();
    std::vector<Edge> edges = other->edges(EdgeType::kData);

    // Copy all nodes except for the output node
    for (Node *node : nodes)
    {
        if (node->node_type() == NodeType::kOutput)
        {
            TT_ASSERT(other_output_op_id == -1, "Only one output is supported for consteval graphs");
            other_output_op_id = other->data_operands(node)[0]->id();
            continue;
        }

        // If the graph being graft is from a common ancenstor nodes can overlap
        if (consteval_graph.has_node_with_name(node->name()))
        {
            node_id_map.insert({node->id(), consteval_graph.get_node_by_name(node->name())->id()});
            continue;
        }

        Node *new_node = consteval_graph.add_node<Node>(node->clone(), subgraph_id_);
        node_id_map.insert({node->id(), new_node->id()});
    }

    // Copy all edges except for the output edge
    for (Edge const &edge : edges)
    {
        if (edge.producer_node_id == other_output_op_id)
            continue;

        Edge new_edge(
            node_id_map.at(edge.producer_node_id),
            edge.producer_output_port_id,
            node_id_map.at(edge.consumer_node_id),
            edge.consumer_input_port_id,
            edge.edge_type);
        consteval_graph.add_edge(new_edge);
        consteval_graph.copy_edge_attributes(edge, new_edge, other);
    }

    TT_ASSERT(other_output_op_id != -1);
    TT_ASSERT(node_id_map.find(other_output_op_id) != node_id_map.end());
    Node *output = consteval_graph.node_by_id(node_id_map.at(other_output_op_id));
    return output;
}

std::unique_ptr<ConstEvalGraph> ConstEvalGraph::clone(Node *new_runtime_input, const std::string &new_input_node_name)
{
    TT_ASSERT(new_runtime_input);
    int unique_id = Graph::generate_unique_graph_id();
    std::unique_ptr<ConstEvalGraph> cloned = std::make_unique<ConstEvalGraph>(
        consteval_graph.name() + "." + std::to_string(unique_id), new_runtime_input, false, subgraph_id_, unique_id);

    consteval_graph.clone(&cloned->consteval_graph);
    cloned->needs_autograd = needs_autograd;
    cloned->ran_autograd = ran_autograd;
    cloned->graph_updated_since_autograd = graph_updated_since_autograd;

    if (consteval_output)
        cloned->consteval_output = cloned->consteval_graph.get_node_by_name(consteval_output->name());
    // Map the old ids to cloned ones
    for (auto [runtime_node_id, consteval_node_id] : runtime_to_consteval_map)
    {
        Node *consteval_node = consteval_graph.node_by_id(consteval_node_id);
        std::string node_name = consteval_node->name();

        if (consteval_node->node_type() == NodeType::kInput and new_input_node_name != "")
        {
            std::string const &old_node_name = consteval_node->name();
            cloned->consteval_graph.update_node_name(
                cloned->consteval_graph.get_node_by_name(old_node_name), new_input_node_name);
            node_name = new_input_node_name;
        }
        cloned->runtime_to_consteval_map[runtime_node_id] = cloned->consteval_graph.get_node_by_name(node_name)->id();
    }
    return cloned;
}

void ConstEvalGraph::autograd()
{
    if (not needs_autograd)
        return;

    if (ran_autograd)
    {
        // Remove BW graph and build it again from scratch
        auto bw_nodes = consteval_graph.nodes([](Node *n) { return n->get_epoch_type() == NodeEpochType::Backward; });
        for (Node *bw_node : bw_nodes)
        {
            consteval_graph.remove_node(bw_node);
        }
    }

    autograd::autograd_engine consteval_autograd_engine(&consteval_graph, autograd::autograd_config{});
    consteval_autograd_engine.run();

    ran_autograd = true;
    graph_updated_since_autograd = false;
}

bool is_consteval_capable_input_type(Node *node)
{
    graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);
    return input && (input->is_parameter() || input->is_constant()) &&
           !node->as<graphlib::TaggedNode>()->has_tag("dont_consteval");
}

bool is_consteval_capable_op(Graph *graph, Node *node, bool allow_forks)
{
    graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
    if (not op)
        return false;

    std::vector<graphlib::Node *> operands = graph->data_operands(op);

    if (not std::all_of(operands.begin(), operands.end(), is_consteval_capable_input_type))
        return false;

    bool disable_forks = not allow_forks;

    auto requires_grad = [graph](graphlib::Node *n)
    { return graph->training() and n->as<graphlib::InputNode>()->requires_grad(); };

    auto fork = [graph, disable_forks, requires_grad](graphlib::Node *n)
    { return (requires_grad(n) or disable_forks) and (graph->data_users(n).size() > 1); };

    auto bcast = [graph, requires_grad](graphlib::Node *n)
    {
        bool any_bcast = false;
        for (auto e : graph->user_data_edges(n))
        {
            auto edge_attr = graph->get_edge_attributes(e);
            any_bcast |= edge_attr->has_broadcast_dims();
        }
        return requires_grad(n) and any_bcast;
    };

    if (std::any_of(operands.begin(), operands.end(), fork))
        return false;

    if (std::any_of(operands.begin(), operands.end(), bcast))
        return false;

    if (std::none_of(operands.begin(), operands.end(), requires_grad))
        return true;

    // requires_grad = true
    //   - if grad is required then we limit consteval to tm ops only
    return op->is_tm();
}

bool is_consteval_capable_input_no_operand_forks(Graph *graph, InputNode *input)
{
    if (not is_consteval_capable_input_type(input))
        return false;

    std::vector<Node *> users = graph->data_users(input);
    std::vector<Edge> user_edges = graph->user_data_edges(input);

    if (input->requires_grad() && graph->training())
    {
        return false;
    }

    // If there is only one user then check if that op is consteval capable
    if (users.size() == 1)
        return is_consteval_capable_op(graph, users[0]) and graph->data_operands(users[0]).size() == 1;

    // If there are multiple users....
    // 1. All of the users must have one operand (unary ops)
    // 2. No user edge can have any tms
    // 3. All of the users must have the same op type
    // 4. All of the users must have the exact same op attrs

    if (not std::all_of(users.begin(), users.end(), [graph](Node *n) { return graph->data_operands(n).size() == 1; }))
        return false;

    if (not std::all_of(
            user_edges.begin(),
            user_edges.end(),
            [graph](Edge e) { return graph->get_edge_attributes(e)->get_tms().size() == 0; }))
        return false;

    std::vector<OpNode *> user_ops;
    for (Node *user : users)
        if (auto *op = dynamic_cast<OpNode *>(user))
            user_ops.push_back(op);
        else
            return false;

    ops::OpType op_type = user_ops[0]->op_type();
    if (not std::all_of(user_ops.begin(), user_ops.end(), [op_type](OpNode *n) { return n->op_type() == op_type; }))
        return false;

    auto attrs = user_ops[0]->op_attrs();
    for (OpNode *op : user_ops)
        if (attrs != op->op_attrs())
            return false;

    return true;
}

std::unique_ptr<Node> try_consteval_op(Graph *graph, Node *node, bool dump_graph)
{
    if (not is_consteval_capable_op(graph, node))
        return nullptr;

    std::vector<graphlib::Node *> operands = graph->data_operands(node);
    graphlib::InputNode *input = operands[0]->as<graphlib::InputNode>();
    auto consteval_graph = input->get_consteval_graph(graph, true, true);
    auto ret_node = consteval_graph->promote_node(graph, node);

    if (dump_graph)
        reportify::dump_consteval_graph(graph->name(), input->name(), consteval_graph->get_graph());

    return ret_node;
}

bool try_consteval_input_no_operand_forks(Graph *graph, InputNode *input, bool dump_graph)
{
    if (not is_consteval_capable_input_no_operand_forks(graph, input))
        return false;

    auto consteval_graph = input->get_consteval_graph(graph, true, true);

    auto users = graph->data_users(input);

    // Thanks to is_consteval_capable_input(), we know that each user is identical (same op, same attrs, no edge tms)
    consteval_graph->promote_node(graph, users[0]);

    for (uint32_t i = 1; i < users.size(); i++) bypass_node(graph, users[i], true);

    if (dump_graph)
        reportify::dump_consteval_graph(graph->name(), input->name(), consteval_graph->get_graph());

    return true;
}

Edge retrieve_between_edge(Graph *graph, Node *producer, Node *consumer)
{
    auto producer_user_edges = graph->user_data_edges(producer);
    Edge *edge = nullptr;
    for (auto &e : producer_user_edges)
    {
        if (e.consumer_node_id == consumer->id())
        {
            edge = &e;
            break;
        }
    }
    TT_ASSERT(edge);
    return *edge;
}

bool are_bcasts_between_ops(Graph *graph, Node *producer, Node *consumer)
{
    auto edge = retrieve_between_edge(graph, producer, consumer);
    auto edge_attr = graph->get_edge_attributes(edge);
    return edge_attr->has_broadcast_dims();
}

bool are_different_ranked_shapes_equivalent(Shape a, Shape b)
{
    auto a_vec = a.as_vector();
    auto b_vec = b.as_vector();

    // Remove all pre 1s
    std::vector<int> new_a;
    for (int i = 0; i < (int)a_vec.size(); i++)
    {
        if (a_vec[i] == 1)
        {
            a_vec.erase(a_vec.begin() + i);
            i--;
        }
        else if (a_vec[i] > 1)
            break;
    }
    for (int i = 0; i < (int)b_vec.size(); i++)
    {
        if (b_vec[i] == 1)
        {
            b_vec.erase(b_vec.begin() + i);
            i--;
        }
        else if (b_vec[i] > 1)
            break;
    }

    // Remove all post 1s
    for (int i = (int)a_vec.size() - 1; i >= 0; i--)
    {
        if (a_vec[i] == 1)
        {
            a_vec.erase(a_vec.begin() + i);
        }
        else if (a_vec[i] > 1)
            break;
    }
    for (int i = (int)b_vec.size() - 1; i >= 0; i--)
    {
        if (b_vec[i] == 1)
        {
            b_vec.erase(b_vec.begin() + i);
        }
        else if (b_vec[i] > 1)
            break;
    }

    if (a_vec.size() != b_vec.size())
        return false;

    for (int i = 0; i < (int)a_vec.size(); i++)
    {
        if (a_vec[i] != b_vec[i])
            return false;
    }
    return true;
}

// Check if this is a linked queue.
// Linked queues are output queues which have users nodes connected via partial data copy edges.
//
bool is_linked_queue(const graphlib::Graph *graph, const graphlib::Node *node)
{
    bool output_link_queue = node->node_type() == graphlib::NodeType::kOutput and
                             not graph
                                     ->user_edges(
                                         node,
                                         [](graphlib::Edge e) {  // clang-format off
                                             return e.edge_type == graphlib::EdgeType::kPartialDataCopy or
                                                    e.edge_type == graphlib::EdgeType::kSubgraphLink;
                                         })  // clang-format on
                                     .empty();
    bool input_link_queue = node->node_type() == graphlib::NodeType::kInput and
                            not graph
                                    ->operand_edges(
                                        node,
                                        [](graphlib::Edge e) {  // clang-format off
                                            return e.edge_type == graphlib::EdgeType::kPartialDataCopy or
                                                   e.edge_type == graphlib::EdgeType::kSubgraphLink;
                                        })  // clang-format on
                                    .empty();
    return output_link_queue or input_link_queue;
}

// Check whether queue is input queue on host, meaning it's data resides on host and is accessed via PCIe.
//
bool is_input_host_queue(bool input_queues_on_host, const Graph *graph, const Node *node)
{
    bool input_on_host =
        input_queues_on_host && node->as<graphlib::QueueNode>()->is_input() &&
        (node->as<graphlib::InputNode>()->is_activation() or node->as<graphlib::InputNode>()->is_loss()) &&
        not is_linked_queue(graph, node);

    return input_on_host;
}

// Check whether queue is output queue on host, meaning it's data resides on host and is transferred via PCIe.
//
bool is_output_host_queue(bool output_queues_on_host, const Graph *graph, const Node *node)
{
    bool output_on_host = output_queues_on_host && (node->node_type() == graphlib::NodeType::kOutput) &&
                          node->as<graphlib::OutputNode>()->untilize() && not is_linked_queue(graph, node);
    return output_on_host;
}

NodeGraphContainer::~NodeGraphContainer()
{
    if (remove_from_graph)
    {
        graph->remove_node(node);
    }
}

}  // namespace graphlib

}  // namespace tt
