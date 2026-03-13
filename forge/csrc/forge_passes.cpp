// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "forge_passes.hpp"

#include <algorithm>
#include <map>

#include "backend_api/device_config.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/query.hpp"
#include "graph_lib/utils.hpp"
#include "passes/constant_folding.hpp"
#include "passes/dataformat.hpp"
#include "passes/decompose_nd_reshape_split.hpp"
#include "passes/decomposing_context.hpp"
#include "passes/erase_consecutive_reshape.hpp"
#include "passes/erase_inverse_ops.hpp"
#include "passes/explicate_unsqueeze.hpp"
#include "passes/fuse_conv2d_bias.hpp"
#include "passes/fuse_pad_conv2d.hpp"
#include "passes/fuse_per_channel_ops.hpp"
#include "passes/generate_initial_flops_estimate.hpp"
#include "passes/insert_inverse_on_io.hpp"
#include "passes/mlir_compiler.hpp"
#include "passes/passes_utils.hpp"
#include "passes/pre_lowering_passes.hpp"
#include "passes/print_graph.hpp"
#include "passes/remove_nops.hpp"
#include "passes/replace_incommutable_patterns.hpp"
#include "passes/squeeze_to_reshape.hpp"
#include "python_bindings_common.hpp"
#include "reportify/reportify.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"

namespace tt
{

using NodeType = graphlib::NodeType;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;
using NodeId = graphlib::NodeId;
using PortId = graphlib::PortId;

// *****************************************************************
//  ************************** Main APIs **************************
// *****************************************************************

// ********** Run post initial graph passes **********
std::tuple<std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>>, passes::FractureChipIdAssignments>
run_post_initial_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object, passes::FractureGroups const &fracture_groups)
{
    std::shared_ptr<void> compiler_cfg = make_shared_py_object(compiler_cfg_object);

    passes::print_graph(graph, "INITIAL");
    passes::generate_initial_flops_estimate(graph);
    passes::decompose_nd_reshape_split(graph);
    passes::fuse_pad_conv2d(graph);
    passes::explicate_unsqueeze(graph);
    passes::fuse_conv2d_bias(graph);

    auto inserted_node_id_mapping = decompose_tt_forge_graph<DecomposeEpoch::Initial>(graph, compiler_cfg);
    auto chip_id_assignments = passes::fracture(graph, fracture_groups);

    passes::bypass_nop_tms(graph);
    passes::apply_user_data_format_override(graph, compiler_cfg_object);

    return std::make_tuple(inserted_node_id_mapping, chip_id_assignments);
}

void run_optimization_graph_passes(graphlib::Graph *graph)
{
    passes::print_graph(graph, "PRE OPTIMIZE");

    // Erase all inverse ops possible.
    // Then, if no inverse ops are erased, then attempt to insert inverse ops on the output.
    // Then, if no inverse ops can be inserted on the output, then attempt to insert inverse ops on the input.
    // Then, if no inverse ops can be inserted on the input, then attempt to bypass nop reshapes.
    //        NOTE: The reason we try this last is because nop reshapes may end up being inverses of other ops
    //              and we want to erase them that way first if possible
    // Commuting to input may have introduced clones, so attempt to erase inverse ops again
    // ...

    bool attempt_update = true;
    while (attempt_update)
    {
        passes::hoist_unsqueeze_squeeze_to_reshape(graph);

        bool skip_erase_redundant = false;
        attempt_update = passes::erase_inverse_ops(graph);
        if (not attempt_update)
        {
            attempt_update = passes::insert_inverse_on_outputs(graph);
            if (attempt_update)
                skip_erase_redundant = true;
        }
        if (not attempt_update)
            attempt_update = passes::insert_inverse_on_inputs(graph);
        if (not attempt_update)
        {
            attempt_update = passes::insert_inverse_on_downstream_tms(graph);
            if (attempt_update)
                skip_erase_redundant = true;
        }
        if (not attempt_update)
            attempt_update = passes::replace_incommutable_patterns(graph);

        // These passes erase tms for non-inverse reasons. Usually we are fine with this.
        // However, we might insert tms on top or under of other tms for the purpose of erasing other inverse ops.
        // Skip in that case
        if (not skip_erase_redundant)
        {
            if (not attempt_update)
                attempt_update = passes::erase_consecutive_reshape(graph, true);

            // TODO: Figure out if this is needed. (Issue #152)
            // if (not attempt_update)
            //     attempt_update = passes::fuse_tm_sequences(graph);

            passes::bypass_nop_tms(graph);
        }
    }

    recalculate_shapes(graph);

    passes::erase_consecutive_reshape(graph, true);

    passes::fuse_per_channel_ops(graph);
    if (not env_as<bool>("FORGE_DISABLE_CONSTANT_FOLDING"))
        passes::constant_folding(graph);

    // Issue #152
    // passes::fuse_tm_sequences(graph);

    reportify::dump_graph(graph->name(), "post_erase_inverse_ops", graph);
}

std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_optimize_decompose_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object)
{
    std::shared_ptr<void> compiler_cfg = make_shared_py_object(compiler_cfg_object);

    passes::print_graph(graph, "POST_OPTIMIZE");
    auto inserted_node_id_mapping = decompose_tt_forge_graph<DecomposeEpoch::PostOptimize>(graph, compiler_cfg);

    return inserted_node_id_mapping;
}

// ********** Run post autograd graph passes **********
std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_autograd_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object)
{
    std::shared_ptr<void> compiler_cfg = make_shared_py_object(compiler_cfg_object);

    passes::print_graph(graph, "POST_AUTOGRAD");
    auto inserted_node_id_mapping = decompose_tt_forge_graph<DecomposeEpoch::PostAutograd>(graph, compiler_cfg);
    passes::apply_user_data_format_override(graph, compiler_cfg_object);

    return inserted_node_id_mapping;
}

// ********** Run pre-lowering passes **********
graphlib::Graph *run_pre_lowering_passes(graphlib::Graph *graph, const std::optional<DataFormat> default_df_override)
{
    passes::print_graph(graph, "PRE_MLIR");
    // Recalculate shapes, and figure out implicit broadcasts that are missing
    recalculate_shapes(graph);

    // Manually convert broadcast ops to tms, so insert tile broadcast ops can work generically
    // Note this is not lowering, these are still forge tms
    convert_broadcast_ops_to_tms(graph);

    passes::remove_nops(graph);

    // Recalculate shapes before lowering to MLIR
    recalculate_shapes(graph);

    return graph;
}
}  // namespace tt
