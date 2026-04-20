// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "ops/op_common.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace reduce_avg
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceAvg, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "reduce_avg should have single input tensor.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_avg should have 2 attrs (dim, keep_dim).");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    bool keep_dim = op.attr_as<bool>("keep_dim");

    std::vector<int64_t> dims64(dims.begin(), dims.end());
    return torch::mean(tensors[0], at::IntArrayRef(dims64), keep_dim);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceAvg, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "reduce_avg should have single input shape.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_avg should have 2 attrs (dim, keep_dim).");

    return op_common::reduce_ops_shape(op, in_shapes);
}

tt::graphlib::NodeContext backward(

    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceAvg, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_avg should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    bool keep_dim = op.attr_as<bool>("keep_dim");
    int ndim = inputs[0].shape.size();

    // Normalize negative dims and sort ascending so unsqueezes insert at the
    // correct positions when processed in order.
    for (auto &d : dims)
        if (d < 0)
            d += ndim;
    std::vector<int> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());

    // Number of elements contributing to each averaged output (= product of reduced
    // dim sizes). Each input position contributes dy * (1 / total_elements).
    std::uint32_t total_elements = 1;
    for (int d : sorted_dims) total_elements *= inputs[0].shape[d];

    NodeContext current = gradient;

    // If keep_dim was false, re-insert size-1 dims so the gradient matches input rank.
    if (!keep_dim)
    {
        for (int d : sorted_dims) current = ac.autograd->create_op(ac, Op(OpType::Unsqueeze, {{"dim", d}}), {current});
    }

    // Broadcast the gradient along every reduced dim via Repeat.
    // Repeat (not Broadcast) is used because Broadcast is converted to an edge TM in
    // pre_lowering_passes and is therefore never lowered to MLIR for the backward graph.
    std::vector<int> repeats(ndim, 1);
    for (int d : sorted_dims) repeats[d] = static_cast<int>(inputs[0].shape[d]);

    current = ac.autograd->create_op(ac, Op(OpType::Repeat, {{"repeats", repeats}}), {current});

    // Scale by 1 / total_elements to complete the mean backward.
    NodeContext scale = ac.autograd->create_constant(ac, 1.0 / static_cast<double>(total_elements));
    return ac.autograd->create_op(ac, Op(OpType::Multiply), {current, scale});
}

void decompose_initial(

    const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceAvg, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_avg should have single input.");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int ndim = inputs[0].shape.size();

    for (auto &d : dims)
        if (d < 0)
            d += ndim;

    // Only fuse away the reduce when every reduced dim already has size 1 — then
    // reducing is either a no-op (keep_dim=true) or just a rank adjustment
    // (keep_dim=false, handled via Squeeze).
    bool all_size_one = std::all_of(dims.begin(), dims.end(), [&](int d) { return inputs[0].shape[d] == 1; });

    if (!all_size_one)
        return;

    if (op.attr_as<bool>("keep_dim"))
    {
        NodeContext result = dc.op(Op(OpType::Nop), {inputs[0]});
        dc.fuse(result);
        return;
    }

    // Squeeze each reduced dim; process in descending order to keep indices valid
    // as earlier squeezes shift the positions of later ones.
    std::vector<int> sorted_desc = dims;
    std::sort(sorted_desc.begin(), sorted_desc.end(), std::greater<int>());

    NodeContext current = inputs[0];
    for (int d : sorted_desc) current = dc.op(Op(OpType::Squeeze, {{"dim", d}}), {current});

    dc.fuse(current);
}

}  // namespace reduce_avg
}  // namespace ops
}  // namespace tt
