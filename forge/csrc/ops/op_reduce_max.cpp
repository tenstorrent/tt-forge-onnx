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
namespace reduce_max
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceMax, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "reduce_max should have single input tensor.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_max should have 2 attrs (dim_arg, keep_dim).");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    bool keep_dim = op.attr_as<bool>("keep_dim");

    // torch::amax supports a list of dims; use it for both single- and multi-dim cases.
    std::vector<int64_t> dims64(dims.begin(), dims.end());
    return torch::amax(tensors[0], at::IntArrayRef(dims64), keep_dim);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceMax, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "reduce_max should have single input shape.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_max should have 2 attrs (dim_arg, keep_dim).");

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
    TT_DBG_ASSERT(op.type() == OpType::ReduceMax, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_max should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    bool keep_dim = op.attr_as<bool>("keep_dim");
    int ndim = static_cast<int>(inputs[0].shape.size());

    // Normalize negative dims and sort ascending so unsqueeze inserts at correct positions.
    for (auto &d : dims)
        if (d < 0)
            d += ndim;
    std::vector<int> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());

    // Repeat factors along every reduced dim; identity along the rest.
    std::vector<int> repeats(ndim, 1);
    for (int d : sorted_dims) repeats[d] = static_cast<int>(inputs[0].shape[d]);

    // Step 1: restore the max output to input rank when keep_dim=false by unsqueezing
    // every reduced dim (ascending order keeps indices valid).
    NodeContext max_expanded = output;
    if (!keep_dim)
    {
        for (int d : sorted_dims)
            max_expanded = ac.autograd->create_op(ac, Op(OpType::Unsqueeze, {{"dim", d}}), {max_expanded});
    }

    // Step 2: broadcast max to the full input shape via Repeat. We use Repeat rather
    // than Broadcast because Broadcast is lowered to an edge TM in pre_lowering_passes
    // and never materializes in the MLIR backward graph.
    NodeContext max_repeated = ac.autograd->create_op(ac, Op(OpType::Repeat, {{"repeats", repeats}}), {max_expanded});

    // Step 3: build a mask that is 1.0 at argmax positions and 0.0 elsewhere.
    //
    // A direct equality test (input == max) is unreliable on Wormhole hardware:
    // `ttnn.max` (used in the forward) does not always return a value that exactly
    // matches any input element, so `input - max` is often slightly negative at the
    // true argmax instead of exactly zero. This corrupts a comparison-based mask.
    //
    // Instead, compute a soft mask via a sharp softmax over the reduced dims:
    //   soft_mask = exp(scale * (input - max)) / reduce_sum(exp(scale * (input - max)))
    // and threshold it at 0.5 to obtain a hard one-hot mask. The softmax concentrates
    // the mass at the argmax position regardless of the small hardware precision
    // error in the max value itself, and it relies only on Exp (element-wise, exact)
    // and ReduceSum (precise on TT hardware) — it does not rely on the precise result
    // of ReduceMax.
    //
    // A scale of 2000 keeps exp in numerically safe range for inputs in [0, 1] while
    // making the softmax sharp enough that the argmax position exceeds the 0.5
    // threshold even for reductions of up to ~128 elements.
    NodeContext diff = ac.autograd->create_op(ac, Op(OpType::Subtract), {inputs[0], max_repeated});
    NodeContext scale = ac.autograd->create_constant(ac, 2000.0);
    NodeContext scaled_diff = ac.autograd->create_op(ac, Op(OpType::Multiply), {diff, scale});
    NodeContext exp_diff = ac.autograd->create_op(ac, Op(OpType::Exp), {scaled_diff});
    NodeContext sum_exp =
        ac.autograd->create_op(ac, Op(OpType::ReduceSum, {{"dim_arg", sorted_dims}, {"keep_dim", true}}), {exp_diff});
    NodeContext sum_exp_repeated = ac.autograd->create_op(ac, Op(OpType::Repeat, {{"repeats", repeats}}), {sum_exp});
    NodeContext soft_mask = ac.autograd->create_op(ac, Op(OpType::Divide), {exp_diff, sum_exp_repeated});
    NodeContext half = ac.autograd->create_constant(ac, 0.5);
    NodeContext mask = ac.autograd->create_op(ac, Op(OpType::GreaterEqual), {soft_mask, half});

    // Step 4: lift the incoming gradient to input rank (mirror of Step 1).
    NodeContext grad_expanded = gradient;
    if (!keep_dim)
    {
        for (int d : sorted_dims)
            grad_expanded = ac.autograd->create_op(ac, Op(OpType::Unsqueeze, {{"dim", d}}), {grad_expanded});
    }

    // Step 5: broadcast the gradient to the full input shape via Repeat.
    NodeContext grad_broadcast =
        ac.autograd->create_op(ac, Op(OpType::Repeat, {{"repeats", repeats}}), {grad_expanded});

    // Step 6: route the gradient to argmax positions only.
    return ac.autograd->create_op(ac, Op(OpType::Multiply), {grad_broadcast, mask});
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceMax, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_max should have single input.");

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

}  // namespace reduce_max
}  // namespace ops
}  // namespace tt
