// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace grid_sample
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::GridSample, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "GridSample expects 2 input tensors (input, grid)");

    std::string mode = op.attr_as<std::string>("mode");
    std::string padding_mode = op.attr_as<std::string>("padding_mode");
    bool align_corners = op.attr_as<bool>("align_corners");

    const at::Tensor &input = tensors[0];
    // Grid arrives in TVM relay format (N, 2, H_out, W_out).
    // torch::grid_sample expects (N, H_out, W_out, 2).
    at::Tensor grid = tensors[1].permute({0, 2, 3, 1}).contiguous();

    using GridSampleFuncOptions = torch::nn::functional::GridSampleFuncOptions;
    GridSampleFuncOptions options;
    options.align_corners(align_corners);
    options.padding_mode(torch::kZeros);

    if (mode == "bilinear")
        options.mode(torch::kBilinear);
    else if (mode == "nearest")
        options.mode(torch::kNearest);
    else
        TT_THROW("OpType::GridSample does not support {} interpolation mode", mode);

    return torch::nn::functional::grid_sample(input, grid, options);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::GridSample, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "GridSample expects 2 input shapes");

    const auto &input_shape = in_shapes[0];  // (N, C, H_in, W_in)
    const auto &grid_shape = in_shapes[1];   // (N, 2, H_out, W_out) - TVM relay format

    TT_ASSERT(input_shape.size() == 4, "GridSample input must be 4D");
    TT_ASSERT(grid_shape.size() == 4, "GridSample grid must be 4D");

    std::vector<uint32_t> output_shape = {
        input_shape[0],  // N
        input_shape[1],  // C
        grid_shape[2],   // H_out (dim 2 in TVM format)
        grid_shape[3],   // W_out (dim 3 in TVM format)
    };

    return std::make_tuple(Shape::create(output_shape), std::vector<DimBroadcast>{});
}

NodeContext backward(
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::GridSample, "Wrong op type.");
    TT_THROW("OpType::GridSample does not have backward.");
    unreachable();
}

}  // namespace grid_sample
}  // namespace ops
}  // namespace tt
