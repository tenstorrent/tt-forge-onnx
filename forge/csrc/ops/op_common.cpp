// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "op_common.hpp"

#include <algorithm>
#include <numeric>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "graph_lib/utils.hpp"
#include "lower_to_forge/common.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace op_common
{
std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> eltwise_nary_shape(
    const std::vector<std::vector<uint32_t>> &in_shapes)
{
    std::vector<graphlib::DimBroadcast> broadcast;
    size_t max_dims = 0;
    for (const auto &shape : in_shapes) max_dims = std::max(max_dims, shape.size());

    std::vector<std::vector<uint32_t>> padded_shapes = in_shapes;
    for (auto &shape : padded_shapes)
        while (shape.size() < max_dims) shape.insert(shape.begin(), 1);

    std::vector<uint32_t> output_shape(max_dims);
    for (size_t dim = 0; dim < max_dims; ++dim)
    {
        uint32_t max_size = 1;
        for (const auto &shape : padded_shapes) max_size = std::max(max_size, shape[dim]);

        output_shape[dim] = max_size;

        for (size_t op_idx = 0; op_idx < padded_shapes.size(); ++op_idx)
        {
            if (padded_shapes[op_idx][dim] == max_size)
                continue;

            TT_ASSERT(
                padded_shapes[op_idx][dim] == 1,
                "Eltwise ops must have same shape or operand must be 1 wide to broadcast");

            broadcast.push_back(
                {static_cast<int>(op_idx), static_cast<int>(dim) - static_cast<int>(max_dims), max_size});
        }
    }

    return {graphlib::Shape::create(output_shape), broadcast};
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> compute_elementwise_binary_shape(
    const std::vector<std::vector<uint32_t>> &in_shapes)
{
    TT_ASSERT(in_shapes.size() == 2, "Elementwise binary ops should have exactly two input shapes.");

    std::vector<graphlib::DimBroadcast> broadcast;
    std::vector<uint32_t> output_shape;

    std::vector<uint32_t> shape0 = in_shapes[0];
    std::vector<uint32_t> shape1 = in_shapes[1];

    // Add leading 1s to the shorter shape
    while (shape0.size() < shape1.size())
    {
        shape0.insert(shape0.begin(), 1);
    }

    while (shape1.size() < shape0.size())
    {
        shape1.insert(shape1.begin(), 1);
    }

    output_shape.resize(shape0.size());

    for (size_t dim = 0; dim < shape0.size(); dim++)
    {
        if (shape0[dim] == shape1[dim])
        {
            output_shape[dim] = shape0[dim];
            continue;
        }

        if (shape1[dim] == 1)
        {
            // Broadcast shape1 to shape0
            int neg_dim = static_cast<int>(dim) - static_cast<int>(shape1.size());
            broadcast.push_back(graphlib::DimBroadcast(1, neg_dim, shape0[dim]));
            output_shape[dim] = shape0[dim];
        }
        else
        {
            TT_ASSERT(
                shape0[dim] == 1,
                "Eltwise binary ops must have the same shape in both inputs, or one operand must be 1 wide to "
                "broadcast");
            // Broadcast shape0 to shape1
            int neg_dim = static_cast<int>(dim) - static_cast<int>(shape0.size());
            broadcast.push_back(graphlib::DimBroadcast(0, neg_dim, shape1[dim]));
            output_shape[dim] = shape1[dim];
        }
    }

    return std::make_tuple(graphlib::Shape::create(output_shape), broadcast);
}

tt::graphlib::NodeContext reduce_broadcast_dimensions(
    tt::autograd::autograd_context &ac,
    const tt::graphlib::NodeContext &gradient,
    const tt::graphlib::Shape &input_shape,
    const tt::graphlib::Shape &grad_shape)
{
    // If shapes match, no reduction needed
    if (input_shape == grad_shape)
    {
        return gradient;
    }

    // Shapes don't match, we need to reduce along broadcast dimensions
    tt::graphlib::NodeContext result_grad = gradient;
    auto input_dims = input_shape.as_vector();
    auto grad_dims = grad_shape.as_vector();

    // Pad shapes with 1s at the beginning to match max rank
    size_t max_dims = std::max(input_dims.size(), grad_dims.size());

    std::vector<int> padded_input_dims(max_dims, 1);
    for (size_t i = 0; i < input_dims.size(); i++)
    {
        padded_input_dims[max_dims - input_dims.size() + i] = input_dims[i];
    }

    std::vector<int> padded_grad_dims(max_dims, 1);
    for (size_t i = 0; i < grad_dims.size(); i++)
    {
        padded_grad_dims[max_dims - grad_dims.size() + i] = grad_dims[i];
    }

    // For each dimension, if input_dim < grad_dim, we need to reduce_sum
    for (size_t i = 0; i < max_dims; i++)
    {
        if (padded_input_dims[i] >= padded_grad_dims[i])
            continue;

        int dim = static_cast<int>(i);
        result_grad =
            ac.autograd->create_op(ac, Op(OpType::ReduceSum, {{"keep_dim", true}, {"dim_arg", dim}}), {result_grad});
    }

    return result_grad;
}

long initial_flops_estimate_output_dim(std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape_tuple)
{
    graphlib::Shape out_shape = std::get<0>(shape_tuple);
    return std::accumulate(out_shape.begin(), out_shape.end(), 1L, std::multiplies<int64_t>());
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> reduce_ops_shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    bool keep_dim = op.attr_as<bool>("keep_dim");
    int ndim = static_cast<int>(in_shapes[0].size());

    // Normalize negative dims to positive and validate each against the input rank.
    for (auto &d : dims)
    {
        if (d < 0)
            d += ndim;
        TT_ASSERT(d >= 0 && d < ndim, "Reduce ops should have valid dim.");
    }

    std::vector<std::uint32_t> ret = in_shapes[0];

    if (keep_dim)
    {
        // Set each reduced dim to 1; non-reduced dims are unchanged.
        for (int d : dims) ret[d] = 1;
    }
    else
    {
        // Erase reduced dims in descending order so earlier erasures do not shift
        // the positions of later ones.
        std::vector<int> sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int>());
        for (int d : sorted_dims) ret.erase(ret.begin() + d);
    }

    return {graphlib::Shape::create(ret), {}};
}

std::vector<at::Tensor> promote_floating_dtypes(const std::vector<at::Tensor> &tensors)
{
    std::vector<at::Tensor> result;
    result.reserve(tensors.size());

    at::ScalarType promote_t = torch::kU8;
    for (const auto &t : tensors)
        if (t.is_floating_point())
            promote_t = at::promote_types(promote_t, t.scalar_type());

    for (const auto &t : tensors)
        if (t.is_floating_point() && t.scalar_type() != promote_t)
            result.emplace_back(t.to(promote_t));
        else
            result.emplace_back(t.clone());

    return result;
}

/**
 * \brief Decompose nearest-neighbor interpolation into repeat + conv ops.
 *
 * This function decomposes nearest-neighbor interpolation (1D/2D) into:
 *   - **Upsampling** using `repeat_interleave`, where elements are expanded.
 *   - **Downsampling** using depthwise `conv2d` with a "pick-first" kernel,
 *     which selects the top-left (or first) element in each stride window.
 *
 * ---------------------------
 * How up/down factors are computed:
 * ---------------------------
 * Given:
 *   - input size:  in_size
 *   - target size: out_size
 *   - gcd = gcd(in_size, out_size)
 *
 *   up   = out_size / gcd      # repeat factor (how many times each element is copied)
 *   down = in_size / gcd       # stride factor (how many input elements get collapsed)
 *
 * Meaning:
 *   - If up > 1   → do `repeat_interleave` `up` times.
 *   - If down != 1 → do depthwise `conv2d` with stride=down (pick-first rule).
 *
 * ---------------------------
 * Examples
 * ---------------------------
 *
 * (A) 1D floating scale factor:
 *   Input:   (N, C, 7)
 *   Target:  size=(9)
 *   gcd(7,9)=1
 *   → up=9, down=7
 *   - Step 1: repeat each element 9 times → shape (N, C, 63)
 *   - Step 2: conv with stride=7 (pick-first) → shape (N, C, 9)
 *   - Intuition: expand then shrink to approximate ratio 7→9.
 *
 * (B) 1D downsampling:
 *   Input:   (N, C, 50)
 *   Target:  size=(10)
 *   gcd(50,10)=10
 *   → up=1, down=5
 *   - No repeat (up=1).
 *   - Depthwise conv with stride=5 picks every 5th element → (N, C, 10).
 *
 * (C) 2D floating scale factor:
 *   Input:   (1, 3, 7, 11)   # (N, C, H, W)
 *   Target:  size=(9, 15)
 *
 *   For H: gcd(7,9)=1 → up_h=9, down_h=7
 *   For W: gcd(11,15)=1 → up_w=15, down_w=11
 *
 *   - Repeat each row 9 times (up_h=9).
 *   - Repeat each column 15 times (up_w=15).
 *   - Conv stride=7 along H (down_h=7).
 *   - Conv stride=11 along W (down_w=11).
 *   Result shape: (1, 3, 9, 15).
 *
 *   # Example counts:
 *   - Each pixel expanded to a (9×15) block before conv.
 *   - Conv collapses every group of (7×11).
 *
 * (D) 2D mixed sampling (up in one dim, down in other):
 *   Input:   (1, 3, 5, 7)
 *   Target:  size=(10, 5)
 *
 *   For H: gcd(5,10)=5 → up_h=2, down_h=1
 *       → repeat rows 2× (upsample height).
 *   For W: gcd(7,5)=1 → up_w=5, down_w=7
 *       → repeat cols 5×, then conv stride=7 (downsample width).
 *
 *   Result shape: (1, 3, 10, 5).
 *
 * (E) 2D downsampling only:
 *   Input:   (1, 3, 5, 7)
 *   Target:  size=(3, 5)
 *
 *   For H: gcd(5,3)=1 → up_h=3, down_h=5
 *       → conv stride=5 (no repeat).
 *   For W: gcd(7,5)=1 → up_w=5, down_w=7
 *       → conv stride=7 (no repeat).
 *
 *   Result shape: (1, 3, 3, 5).
 *
 * ---------------------------
 * Summary:
 *   - up_h/up_w   → number of times each pixel is repeated (expand factor).
 *   - down_h/down_w → stride used in conv2d (collapse factor).
 */
void decompose_nearest_interpolation(
    tt::DecomposingContext &dc, const tt::graphlib::NodeContext &activation, std::vector<int> sizes, bool channel_last)
{
    tt::graphlib::NodeContext result = activation;
    graphlib::Shape act_shape = activation.shape;
    const size_t rank = act_shape.size();

    // Helper: repeat_interleave
    // Expands each element along a given dim "repeats" times.
    // Used for upsampling when up > 1.
    auto repeat_if_needed = [&](const tt::graphlib::NodeContext &in, int repeats, int dim) -> tt::graphlib::NodeContext
    {
        if (repeats > 1)
        {
            return dc.op(ops::Op(OpType::RepeatInterleave, {{"repeats", repeats}, {"dim", dim}}), {in});
        }
        return in;
    };

    // Helper: conv2d_with_weight
    // Applies depthwise conv2d with a kernel that has a single "1"
    // at the top-left position, all other values 0.
    // Effect: for stride = down, it keeps the 1st element of each
    // block of length "down", discarding the rest.
    auto conv2d_with_weight = [&](const tt::graphlib::NodeContext &in,
                                  const std::vector<int64_t> &weight_shape,
                                  const std::vector<int> &stride) -> tt::graphlib::NodeContext
    {
        at::ScalarType datatype = tt::graphlib::data_format_to_scalar_type(in.output_df);

        at::Tensor weight_tensor = torch::zeros(weight_shape, torch::TensorOptions().dtype(datatype));
        weight_tensor.index_put_({torch::indexing::Slice(), 0, 0, 0}, 1.0);

        tt::graphlib::NodeContext weight = dc.tensor(weight_tensor);

        std::vector<int> padding = {0, 0, 0, 0};         // no padding
        std::vector<int> dilation = {1, 1};              // no dilation
        int groups = static_cast<int>(weight_shape[0]);  // depthwise = 1 kernel per channel

        return dc.op(
            ops::Op(
                OpType::Conv2d,
                {{"stride", stride},
                 {"groups", groups},
                 {"padding", padding},
                 {"dilation", dilation},
                 {"channel_last", channel_last}}),
            {in, weight});
    };

    // ======================
    // Handle 1D interpolation case (rank=3)
    // Tensor shape: (N, C, W) or (N, W, C)
    // ======================
    if (rank == 3)
    {
        size_t c_dim = channel_last ? rank - 1 : rank - 2;
        size_t w_dim = channel_last ? rank - 2 : rank - 1;

        int input_c = static_cast<int>(act_shape[c_dim]);
        int input_w = static_cast<int>(act_shape[w_dim]);
        int size_w = sizes[0];

        // Compute gcd-based decomposition
        int g_w = std::gcd(size_w, input_w);
        int up_w = size_w / g_w;     // repeat factor (upsample)
        int down_w = input_w / g_w;  // stride factor (downsample)

        // --- Step 1: Upsample if needed
        result = repeat_if_needed(result, up_w, static_cast<int>(w_dim));

        // --- Step 2: Downsample if needed (down_w != 1)
        if (down_w != 1)
        {
            // conv2d expects 4D input, so unsqueeze
            result = dc.op(ops::Op(OpType::Unsqueeze, {{"dim", static_cast<int>(w_dim)}}), {result});

            // Build depthwise kernel of shape (C,1,1,down_w)
            // Stride (1, down_w) collapses every block of width down_w → picks first element
            std::vector<int64_t> weight_shape = {input_c, 1, 1, down_w};
            std::vector<int> stride = {1, down_w};

            result = conv2d_with_weight(result, weight_shape, stride);

            // Remove the temporary dimension back to original rank
            result = dc.op(ops::Op(OpType::Squeeze, {{"dim", static_cast<int>(w_dim)}}), {result});
        }

        // Fuse result into graph
        dc.fuse(result);
    }
    // ======================
    // Handle 2D interpolation case (rank=4)
    // Tensor shape: (N, C, H, W) or (N, H, W, C)
    // ======================
    else if (rank == 4)
    {
        size_t c_dim = channel_last ? rank - 1 : rank - 3;
        size_t h_dim = channel_last ? rank - 3 : rank - 2;
        size_t w_dim = channel_last ? rank - 2 : rank - 1;

        int input_c = static_cast<int>(act_shape[c_dim]);
        int input_h = static_cast<int>(act_shape[h_dim]);
        int input_w = static_cast<int>(act_shape[w_dim]);
        int size_h = sizes[0];
        int size_w = sizes[1];

        // Compute gcd decomposition for both H and W
        int g_h = std::gcd(size_h, input_h);
        int g_w = std::gcd(size_w, input_w);
        int up_h = size_h / g_h;     // repeat factor along height
        int up_w = size_w / g_w;     // repeat factor along width
        int down_h = input_h / g_h;  // stride factor along height
        int down_w = input_w / g_w;  // stride factor along width

        // --- Step 1: Upsample (expand each element if up_h/up_w > 1)
        result = repeat_if_needed(result, up_h, static_cast<int>(h_dim));
        result = repeat_if_needed(result, up_w, static_cast<int>(w_dim));

        // --- Step 2: Downsample width (if down_w != 1)
        if (down_w != 1)
        {
            // Kernel: (C,1,1,down_w), stride (1,down_w)
            // Collapses groups of width down_w into 1 → picks first col in each block
            std::vector<int64_t> weight_shape = {input_c, 1, 1, down_w};
            std::vector<int> stride = {1, down_w};
            result = conv2d_with_weight(result, weight_shape, stride);
        }

        // --- Step 3: Downsample height (if down_h != 1)
        if (down_h != 1)
        {
            // Kernel: (C,1,down_h,1), stride (down_h,1)
            // Collapses groups of height down_h into 1 → picks first row in each block
            std::vector<int64_t> weight_shape = {input_c, 1, down_h, 1};
            std::vector<int> stride = {down_h, 1};
            result = conv2d_with_weight(result, weight_shape, stride);
        }

        dc.fuse(result);
    }
    else
    {
        TT_THROW("Nearest interpolation is supported only for 3d and 4d inputs");
        unreachable();
    }
}

}  // namespace op_common
}  // namespace ops
}  // namespace tt
