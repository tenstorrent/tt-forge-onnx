// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "reportify/reportify.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "nlohmann/json.hpp"
#include "reportify/to_json.hpp"
#include "utils/logger.hpp"

// MLIR headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#include "mlir/IR/BuiltinOps.h"
#pragma clang diagnostic pop

using json = nlohmann::json;

namespace tt
{

namespace reportify
{
std::string canonical_dirname(std::string s)
{
    static std::string const chars = "/: ";
    for (char& c : s)
    {
        if (chars.find(c) != std::string::npos)
            c = '_';
    }
    // Linux fs limit is 255 bytes per component. Truncate + hash to stay safe.
    // Leave 16 chars for suffix ("_" + 8-hex hash) and the "_graphs/" suffix (8 chars).
    static const size_t kMaxLen = 220;
    if (s.size() > kMaxLen) {
        // Simple djb2 hash of full string
        uint32_t h = 5381u;
        for (unsigned char c : s) h = ((h << 5) + h) ^ c;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "_%08x", h);
        s = s.substr(0, kMaxLen) + buf;
    }
    return s;
}

template <class T>
std::string stream_operator_to_string(T obj)
{
    std::ostringstream oss;
    oss << obj;
    return oss.str();
}

using JsonNamePair = std::pair<json, std::string>;
using JsonNamePairs = std::vector<JsonNamePair>;

std::vector<std::string> tt_nodes_to_name_strings(const std::vector<graphlib::Node*>& nodes)
{
    std::vector<std::string> ret_vector;
    ret_vector.reserve(nodes.size());

    for (const graphlib::Node* node : nodes)
    {
        ret_vector.push_back(node->name());
    }

    return ret_vector;
}

json node_to_json(const graphlib::Node* node, const graphlib::Graph* graph)
{
    json ret_json;
    ret_json["forge"] = 1;  // marker to reportify to use new colouring scheme
    ret_json["name"] = node->name();
    ret_json["unique_id"] = node->id();
    std::vector<std::string> input_nodes;
    std::vector<std::string> output_nodes;
    std::vector<std::string> port_id_to_name_incoming;
    std::vector<std::string> port_id_to_name_outgoing;
    std::unordered_map<std::string, std::string> input_node_name_to_edge_type;

    for (auto incoming_edge : graph->operand_edges(node))
    {
        graphlib::NodeId incoming_node_id = incoming_edge.producer_node_id;
        graphlib::Node* incoming_node = graph->node_by_id(incoming_node_id);
        std::string edge_type_string = graphlib::edge_type_to_string(incoming_edge.edge_type);

        std::string port_key_string = "port_" + std::to_string(incoming_edge.consumer_input_port_id);
        std::string incoming_port_info = edge_type_string + ": " + incoming_node->name() + " (" + port_key_string + ")";

        port_id_to_name_incoming.push_back(incoming_port_info);

        if (incoming_edge.edge_type != graphlib::EdgeType::kData and
            incoming_edge.edge_type != graphlib::EdgeType::kDataLoopback and
            incoming_edge.edge_type != graphlib::EdgeType::kControlLoop and
            incoming_edge.edge_type != graphlib::EdgeType::kControl and
            incoming_edge.edge_type != graphlib::EdgeType::kPartialDataCopy and
            incoming_edge.edge_type != graphlib::EdgeType::kSubgraphLink)
        {
            continue;  // don't display others for now
        }

        input_nodes.push_back(incoming_node->name());
        input_node_name_to_edge_type.insert({incoming_node->name(), edge_type_string});
    }

    ret_json["input_nodes"] = input_nodes;
    ret_json["incoming_edge_port_info"] = port_id_to_name_incoming;
    ret_json["input_node_to_edge_type"] = input_node_name_to_edge_type;

    for (auto outgoing_edge : graph->user_edges(node))
    {
        graphlib::NodeId outgoing_node_id = outgoing_edge.consumer_node_id;
        graphlib::Node* outgoing_node = graph->node_by_id(outgoing_node_id);
        output_nodes.push_back(outgoing_node->name());

        std::string port_key_string = "port_" + std::to_string(outgoing_edge.producer_output_port_id);
        std::string edge_type_string = graphlib::edge_type_to_string(outgoing_edge.edge_type);
        std::string outgoing_port_info = edge_type_string + ": " + outgoing_node->name() + " (" + port_key_string + ")";

        port_id_to_name_outgoing.push_back(outgoing_port_info);
    }

    ret_json["opcode"] = stream_operator_to_string(node->node_type());
    ret_json["cache"]["shape"] = node->shape().as_vector();

    ret_json["epoch"] = 0;

    ret_json["epoch_type"] = graphlib::node_epoch_type_to_string(node->get_epoch_type());
    ret_json["output_nodes"] = output_nodes;
    ret_json["outgoing_edge_port_info"] = port_id_to_name_outgoing;

    ret_json["type"] = stream_operator_to_string(node->node_type());
    if (node->node_type() == graphlib::NodeType::kInput)
    {
        // Keep constants and accumulators inside the epoch to better visualize what's happening
        if (node->as<graphlib::InputNode>()->is_constant())
        {
            ret_json["class"] = node->as<graphlib::InputNode>()->input_type_string();
            ret_json["type"] = "Constant";

            const graphlib::ConstantInputNode* cnode = node->as<graphlib::ConstantInputNode>();
            if (cnode->is_single_value())
            {
                ret_json["constant_value"] = std::to_string(cnode->constant_value());
                ret_json["constant_dims"] = cnode->constant_dims();
            }
            else if (cnode->is_tensor())
            {
                ret_json["constant_dims"] = cnode->tensor_shape().as_vector();
            }
        }
        else if (node->as<graphlib::InputNode>()->is_accumulator())
        {
            ret_json["class"] = "accumulator";
            ret_json["type"] = "Accumulator";
        }
        else
        {
            ret_json["class"] = "Input::";
            ret_json["type"] = "Input::" + node->as<graphlib::InputNode>()->input_type_string();
        }
        ret_json["queue_type"] = node->as<graphlib::QueueNode>()->queue_type_string();
        ret_json["is_cross_epoch_type"] = node->as<graphlib::QueueNode>()->is_epoch_to_epoch() and
                                          node->as<graphlib::EpochToEpochQueueNode>()->is_cross_epoch_type();
        ret_json["memory_access"] = node->as<graphlib::QueueNode>()->memory_access_type_string();
        ret_json["tile_broadcast"] = node->as<graphlib::InputNode>()->get_tile_broadcast_dims();
        ret_json["requires_grad"] = node->as<graphlib::InputNode>()->requires_grad();
    }
    else if (node->node_type() == graphlib::NodeType::kOutput)
    {
        ret_json["class"] = "Output";
        ret_json["queue_type"] = node->as<graphlib::QueueNode>()->queue_type_string();
        ret_json["is_cross_epoch_type"] = node->as<graphlib::QueueNode>()->is_epoch_to_epoch() and
                                          node->as<graphlib::EpochToEpochQueueNode>()->is_cross_epoch_type();
        ret_json["memory_access"] = node->as<graphlib::QueueNode>()->memory_access_type_string();
        ret_json["is_intermediate"] = node->as<graphlib::OutputNode>()->is_intermediate();
    }
    else if (node->node_type() == graphlib::NodeType::kPyOp)
    {
        const graphlib::PyOpNode* opnode = node->as<graphlib::PyOpNode>();
        ret_json["ir"] = "forge";
        std::stringstream ss;
        ss << opnode->op();
        ret_json["class"] = ss.str();
        ret_json["type"] = opnode->op().as_string();
        to_json(ret_json, opnode->op());
        ret_json["gradient_op"] = opnode->is_gradient_op();
    }
    else if (node->node_type() == graphlib::NodeType::kQueue)
    {
        ret_json["class"] = "ForgeDramQueue::";
        ret_json["queue_type"] = node->as<graphlib::QueueNode>()->queue_type_string();
        ret_json["is_cross_epoch_type"] = node->as<graphlib::QueueNode>()->is_epoch_to_epoch() and
                                          node->as<graphlib::EpochToEpochQueueNode>()->is_cross_epoch_type();
        ret_json["memory_access"] = node->as<graphlib::QueueNode>()->memory_access_type_string();
    }
    std::stringstream ss;
    ss << node->output_df();
    ret_json["output_df"] = ss.str();

    if (auto tagged_node = dynamic_cast<const graphlib::TaggedNode*>(node); tagged_node != nullptr)
    {
        ret_json["tags"] = tagged_node->get_tags();
    }

    // Record input TMs, if any, on the input edges
    for (graphlib::Edge e : graph->operand_data_edges(node))
    {
        std::vector<ops::Op> tms = graph->get_edge_attributes(e)->get_tms();
        ret_json["input_tms"][e.consumer_input_port_id] = json::array();
        if (tms.size() > 0)
        {
            for (const auto& tm : tms)
            {
                json j;
                to_json(j, tm);
                ret_json["input_tms"][e.consumer_input_port_id].push_back(j);
            }
        }
    }

    return ret_json;
}
using JsonNamePair = std::pair<json, std::string>;
using JsonNamePairs = std::vector<JsonNamePair>;

void write_json_to_file(const std::string& path, json json_file, int width = 4)
{
    std::ofstream o(path);
    if (width > 0)
        o << std::setw(4);
    o << json_file;
}

JsonNamePairs create_jsons_for_graph(
    const std::string& graph_prefix,
    const graphlib::Graph* graph,
    std::function<bool(graphlib::Node*)> node_filter = [](graphlib::Node*) { return true; });

void dump_graph(
    const std::string& path,
    const std::string& test_name,
    const std::string& graph_prefix,
    const graphlib::Graph* graph,
    const std::string& report_path)
{
    if (env_as<bool>("FORGE_DISABLE_REPORTIFY_DUMP"))
        return;

    JsonNamePairs json_pairs = create_jsons_for_graph(graph_prefix, graph);

    initalize_reportify_directory(path, test_name);

    std::string sage_report_path = build_report_path(path, test_name, report_path);
    std::string subgraph_path = sage_report_path + graph_prefix + "_graphs/";

    log_debug(tt::LogReportify, "Writing graph to {}", subgraph_path);

    std::filesystem::create_directories(subgraph_path);

    json root_json = json_pairs.back().first;
    std::string root_json_name = json_pairs.back().second;

    std::string root_json_path = sage_report_path + /*graph_prefix +*/ root_json_name;
    write_json_to_file(root_json_path, root_json);
}

void dump_consteval_graph(const std::string& test_name, const std::string& graph_prefix, const graphlib::Graph* graph)
{
    return dump_graph(test_name, canonical_dirname(graph_prefix), graph, "/forge_reports/Consteval/");
}

void dump_epoch_type_graphs(
    const std::string& test_name,
    const std::string& graph_prefix,
    const graphlib::Graph* graph,
    const std::string& directory_path)
{
    if (env_as<bool>("FORGE_DISABLE_REPORTIFY_DUMP"))
        return;

    std::function<bool(graphlib::Node*, graphlib::NodeEpochType epoch_type, const graphlib::Graph* graph)>
        epoch_type_filter = [](graphlib::Node* node, graphlib::NodeEpochType epoch_type, const graphlib::Graph* graph)
    {
        if (node->node_type() == graphlib::NodeType::kInput or node->node_type() == graphlib::NodeType::kQueue)
        {
            // we want to write out input/queue nodes if they happen to be produced/consumed
            // by a node belonging to the queried epoch_type
            for (graphlib::Node* user : graph->data_users(node))
            {
                if (user->get_epoch_type() == epoch_type)
                {
                    return true;
                }
            }
            for (graphlib::Node* operand : graph->data_operands(node))
            {
                if (operand->get_epoch_type() == epoch_type)
                {
                    return true;
                }
            }
            return false;
        }

        return node->get_epoch_type() == epoch_type;
    };

    initalize_reportify_directory(directory_path, test_name);

    std::string report_path = get_epoch_type_report_relative_directory();
    std::string sage_report_path = build_report_path(directory_path, test_name, report_path);

    std::string subgraph_path = sage_report_path + graph_prefix + "_graphs/";

    log_debug(tt::LogReportify, "Writing graph to {}", subgraph_path);

    std::filesystem::create_directories(subgraph_path);

    for (graphlib::NodeEpochType epoch_type :
         {graphlib::NodeEpochType::Forward, graphlib::NodeEpochType::Backward, graphlib::NodeEpochType::Optimizer})
    {
        if ((epoch_type == graphlib::NodeEpochType::Backward and not graph->contains_bwd_nodes()) or
            (epoch_type == graphlib::NodeEpochType::Optimizer and not graph->contains_opt_nodes()))
        {
            continue;
        }

        auto node_epoch_type_filter = std::bind(epoch_type_filter, std::placeholders::_1, epoch_type, graph);
        JsonNamePairs new_json_pairs = create_jsons_for_graph(
            graph_prefix + graphlib::node_epoch_type_to_string(epoch_type), graph, node_epoch_type_filter);

        for (const auto& [json, json_name] : new_json_pairs)
        {
            std::string root_json_path = sage_report_path + graph_prefix + json_name;
            write_json_to_file(root_json_path, json);
        }
    }
}

void dump_epoch_id_graphs(
    const std::string& test_name,
    const std::string& graph_prefix,
    const graphlib::Graph* graph,
    const std::string& directory_path)
{
    return;
}

json create_json_for_graph(const graphlib::Graph* graph, std::function<bool(graphlib::Node*)> node_filter)
{
    json this_json;
    this_json["topological_sorted_nodes"] = {};
    for (graphlib::Node* node : graphlib::topological_sort(*graph))
    {
        if (node_filter(node))
        {
            this_json["nodes"][node->name()] = node_to_json(node, graph);
            this_json["graph"] = std::unordered_map<std::string, std::string>();
            this_json["topological_sorted_nodes"].push_back(node->name());
        }
    }
    return this_json;
}

JsonNamePairs create_jsons_for_graph(
    const std::string& graph_prefix, const graphlib::Graph* graph, std::function<bool(graphlib::Node*)> node_filter)
{
    JsonNamePairs this_json_name_pairs;

    json this_json = create_json_for_graph(graph, node_filter);
    std::string this_name = graph_prefix + ".forge";
    JsonNamePair this_json_name_pair = std::make_pair(this_json, this_name);
    this_json_name_pairs.push_back(this_json_name_pair);

    return this_json_name_pairs;
}

void dump_graph(
    const std::string& test_name,
    const std::string& graph_prefix,
    const graphlib::Graph* graph,
    const std::string& report_path)
{
    std::string default_dir = get_default_reportify_path("");
    dump_graph(default_dir, test_name, graph_prefix, graph, report_path);
}

/**
 * @brief Dumps the MLIR's Operation to a file.
 *
 * This function generates a representation of the given MLIR operation and writes it to a file.
 * The file path is following: `$REPORTIFY_PATH$/$OPERATION_NAME$/mlir_reports/$FILE_NAME$.mlir`.
 * If the environment variable "FORGE_DISABLE_REPORTIFY_DUMP" is set to true, the function returns without performing
 * any action.
 *
 * @param file_name The name of the file to be saved (currently in use are 'ttir' and 'ttnn').
 * @param module_name The name of the module to be saved.
 * @param operation A pointer to the MLIR operation to be dumped.
 */
void dump_mlir(const std::string& file_name, const std::string& module_name, mlir::Operation* operation)
{
    if (env_as<bool>("FORGE_DISABLE_REPORTIFY_DUMP"))
        return;

    std::string path = get_default_reportify_path("");
    std::string report_path = get_mlir_reports_relative_directory();
    std::string full_report_path = build_report_path(path, module_name, report_path);

    std::string outputString;
    llvm::raw_string_ostream outStream(outputString);

    // Print MLIR module
    mlir::OpPrintingFlags printFlags;
    printFlags.enableDebugInfo();

    operation->print(outStream, printFlags);
    outStream.flush();

    std::string root_file_name = file_name + ".mlir";
    std::transform(root_file_name.begin(), root_file_name.end(), root_file_name.begin(), ::tolower);
    std::string root_file_path = full_report_path + root_file_name;

    std::filesystem::create_directories(std::filesystem::path(full_report_path));

    std::ofstream o(root_file_path);
    o << outputString;
    o.flush();
}

}  // namespace reportify
}  // namespace tt
