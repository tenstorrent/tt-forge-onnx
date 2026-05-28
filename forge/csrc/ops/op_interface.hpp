// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

namespace at
{
class Tensor;  // Forward declaration of torch tensor.
}

namespace tt
{
namespace graphlib
{
struct OpType;
class Shape;
struct NodeContext;
using DimBroadcastTrampoline = std::tuple<int, int, int>;
}  // namespace graphlib

namespace autograd
{
struct autograd_context;
}

class DecomposingContext;

namespace ops
{
class Op;

/**
 * Declaration for ops interface in a separate namespace ns.
 */
#define DECLARE_OP_INTERFACE(ns)                                                                      \
    namespace ns                                                                                      \
    {                                                                                                 \
    at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors);                            \
                                                                                                      \
    std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcastTrampoline>> shape(                 \
        const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs);                         \
                                                                                                      \
    tt::graphlib::NodeContext backward(                                                               \
        const Op &op,                                                                                 \
        tt::autograd::autograd_context &context,                                                      \
        int operand,                                                                                  \
        const std::vector<tt::graphlib::NodeContext> &inputs,                                         \
        const tt::graphlib::NodeContext &output,                                                      \
        const tt::graphlib::NodeContext &gradient);                                                   \
                                                                                                      \
    void decompose_initial(                                                                           \
        const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs);  \
                                                                                                      \
    void decompose_post_optimize(                                                                     \
        const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs);  \
                                                                                                      \
    void decompose_post_autograd(                                                                     \
        const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs);  \
                                                                                                      \
    long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs); \
    }

/**
 * Ops declaration.
 */
DECLARE_OP_INTERFACE(abs);
DECLARE_OP_INTERFACE(adaptive_max_pool_2d);
DECLARE_OP_INTERFACE(add);
DECLARE_OP_INTERFACE(adv_index);
DECLARE_OP_INTERFACE(argmax);
DECLARE_OP_INTERFACE(atan);
DECLARE_OP_INTERFACE(avg_pool_1d);
DECLARE_OP_INTERFACE(avg_pool_2d);
DECLARE_OP_INTERFACE(batchnorm);
DECLARE_OP_INTERFACE(broadcast);
DECLARE_OP_INTERFACE(cast);
DECLARE_OP_INTERFACE(clip);
DECLARE_OP_INTERFACE(concatenate);
DECLARE_OP_INTERFACE(constant);
DECLARE_OP_INTERFACE(constant_pad);
DECLARE_OP_INTERFACE(conv_2d);
DECLARE_OP_INTERFACE(conv_2d_transpose);
DECLARE_OP_INTERFACE(cosine);
DECLARE_OP_INTERFACE(cumulative_sum);
DECLARE_OP_INTERFACE(divide);
DECLARE_OP_INTERFACE(downsample_2d);
DECLARE_OP_INTERFACE(dropout);
DECLARE_OP_INTERFACE(embedding);
DECLARE_OP_INTERFACE(embedding_bw);
DECLARE_OP_INTERFACE(equal);
DECLARE_OP_INTERFACE(erf);
DECLARE_OP_INTERFACE(exp);
DECLARE_OP_INTERFACE(fill_cache);
DECLARE_OP_INTERFACE(gelu);
DECLARE_OP_INTERFACE(gelu_derivative);
DECLARE_OP_INTERFACE(grid_sample);
DECLARE_OP_INTERFACE(greater);
DECLARE_OP_INTERFACE(greater_equal);
DECLARE_OP_INTERFACE(heaviside);
DECLARE_OP_INTERFACE(index);
DECLARE_OP_INTERFACE(index_copy);
DECLARE_OP_INTERFACE(layernorm);
DECLARE_OP_INTERFACE(layernorm_bw);
DECLARE_OP_INTERFACE(leaky_relu);
DECLARE_OP_INTERFACE(less);
DECLARE_OP_INTERFACE(less_equal);
DECLARE_OP_INTERFACE(log);
DECLARE_OP_INTERFACE(log_softmax);
DECLARE_OP_INTERFACE(logical_and);
DECLARE_OP_INTERFACE(bitwise_and);
DECLARE_OP_INTERFACE(logical_not);
DECLARE_OP_INTERFACE(mask);
DECLARE_OP_INTERFACE(matmul);
DECLARE_OP_INTERFACE(max_pool_1d);
DECLARE_OP_INTERFACE(max_pool_2d);
DECLARE_OP_INTERFACE(maximum);
DECLARE_OP_INTERFACE(minimum);
DECLARE_OP_INTERFACE(multiply);
DECLARE_OP_INTERFACE(nop);
DECLARE_OP_INTERFACE(not_equal);
DECLARE_OP_INTERFACE(pad);
DECLARE_OP_INTERFACE(pixel_shuffle);
DECLARE_OP_INTERFACE(pow);
DECLARE_OP_INTERFACE(power);
DECLARE_OP_INTERFACE(reciprocal);
DECLARE_OP_INTERFACE(reduce_avg);
DECLARE_OP_INTERFACE(reduce_max);
DECLARE_OP_INTERFACE(reduce_sum);
DECLARE_OP_INTERFACE(relu);
DECLARE_OP_INTERFACE(remainder);
DECLARE_OP_INTERFACE(repeat);
DECLARE_OP_INTERFACE(repeat_interleave);
DECLARE_OP_INTERFACE(reshape);
DECLARE_OP_INTERFACE(resize_1d);
DECLARE_OP_INTERFACE(resize_2d);
DECLARE_OP_INTERFACE(select);
DECLARE_OP_INTERFACE(sigmoid);
DECLARE_OP_INTERFACE(sine);
DECLARE_OP_INTERFACE(softmax);
DECLARE_OP_INTERFACE(softmax_bw);
DECLARE_OP_INTERFACE(sqrt);
DECLARE_OP_INTERFACE(stack);
DECLARE_OP_INTERFACE(subtract);
DECLARE_OP_INTERFACE(squeeze);
DECLARE_OP_INTERFACE(tanh);
DECLARE_OP_INTERFACE(tile_broadcast);
DECLARE_OP_INTERFACE(transpose);
DECLARE_OP_INTERFACE(unsqueeze);
DECLARE_OP_INTERFACE(update_cache);
DECLARE_OP_INTERFACE(upsample_2d);
DECLARE_OP_INTERFACE(where);

#undef DECLARE_OP_INTERFACE

}  // namespace ops
}  // namespace tt
