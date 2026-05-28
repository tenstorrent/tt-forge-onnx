// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "utils/assert.hpp"

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
enum class DecomposeEpoch : uint8_t;

namespace ops
{

enum class OpType : uint32_t
{
    Abs,
    AdaptiveMaxPool2d,
    Add,
    AdvIndex,
    Argmax,
    Atan,
    AvgPool1d,
    AvgPool2d,
    Batchnorm,
    Broadcast,
    Cast,
    Clip,
    Concatenate,
    Constant,
    ConstantPad,
    Conv2d,
    Conv2dPrestrideAct,
    Conv2dPrestrideWeights,
    Conv2dTranspose,
    Cosine,
    CumulativeSum,
    Divide,
    Downsample2d,
    Dropout,
    Embedding,
    EmbeddingBw,
    Equal,
    Erf,
    Exp,
    FillCache,
    Gelu,
    Greater,
    GreaterEqual,
    Heaviside,
    Index,
    IndexCopy,
    Layernorm,
    LayernormBw,
    LeakyRelu,
    Less,
    LessEqual,
    Log,
    LogSoftmax,
    LogicalAnd,
    LogicalNot,
    BitwiseAnd,
    Mask,
    Matmul,
    MaxPool1d,
    MaxPool2d,
    Maximum,
    Minimum,
    Multiply,
    Nop,
    NotEqual,
    Pad,
    PixelShuffle,
    Pow,
    Power,
    Reciprocal,
    ReduceAvg,
    ReduceMax,
    ReduceSum,
    Relu,
    Remainder,
    Repeat,
    RepeatInterleave,
    Reshape,
    Resize1d,
    Resize2d,
    Select,
    Sigmoid,
    Sine,
    Softmax,
    SoftmaxBw,
    SparseMatmul,
    Sqrt,
    Stack,
    Subtract,
    Squeeze,
    Tanh,
    Transpose,
    Unsqueeze,
    UpdateCache,
    Upsample2d,
    GridSample,
    Where,
};

/**
 * Op attribute.
 */
using Attr = ::std::variant<
    std::string,
    bool,
    int,
    float,
    std::vector<int>,
    std::vector<std::tuple<int, int, int>>,
    std::vector<std::tuple<int, int, int, int>>>;

/**
 * Op attributes.
 * TODO: Conver this into vector. There is no sence to use map/unordered map for few items. Also, once migrated to new
 * ops, we should create attribute key enum, and replace mapping strings.
 */
using Attrs = ::std::map<std::string, Attr>;

class Op
{
   public:
    Op(OpType type, Attrs attrs) : type_(type), attrs_(std::move(attrs)) {}
    Op(OpType type) : type_(type) {}
    Op(const std::string &op_name, Attrs attrs = {});

    bool operator==(const Op &other) const { return type_ == other.type_ && attrs_ == other.attrs_; }
    bool operator!=(const Op &other) const { return !(*this == other); }

    OpType type() const { return type_; }

    const Attrs &attrs() const { return attrs_; }

    template <typename T>
    const T &attr_as(std::string const &name) const
    {
        return std::get<T>(attr(name));
    }

    template <typename T>
    T &attr_as(std::string const &name)
    {
        return std::get<T>(attr(name));
    }

    bool has_attr(const std::string &attr_name) const { return attrs_.find(attr_name) != attrs_.end(); }
    void set_attrs(Attrs attrs) { attrs_ = std::move(attrs); }
    void set_attr(std::string const &name, Attr attr) { attrs_[name] = std::move(attr); }
    bool remove_attr(const std::string &attr_name) { return attrs_.erase(attr_name) > 0; }
    void clear_attrs() { attrs_.clear(); }
    const Attr &get_attr(const std::string &name) const { return attr(name); }

    const std::string &as_string() const;

   private:
    /**
     * Returns attribute based on provided string. Since map::at throws if element does not exist but without any useful
     * info, we will assert to get detailed error message.
     */
    const Attr &attr(const std::string &name) const
    {
        TT_ASSERT(has_attr(name), "Non existing attribute: {}", name);
        return attrs_.at(name);
    }

    Attr &attr(const std::string &name)
    {
        TT_ASSERT(has_attr(name), "Non existing attribute: {}", name);
        return attrs_.at(name);
    }

   public:
    /* ----------------------------------------------------*
     * Calculations segment. All ops must implement these. *
     * ----------------------------------------------------*/

    at::Tensor eval(const std::vector<at::Tensor> &tensors) const;

    std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcastTrampoline>> shape(
        const std::vector<std::vector<std::uint32_t>> &inputs) const;

    tt::graphlib::NodeContext backward(
        tt::autograd::autograd_context &context,
        int operand,
        const std::vector<tt::graphlib::NodeContext> &inputs,
        const tt::graphlib::NodeContext &output,
        const tt::graphlib::NodeContext &gradient) const;

    bool is_tm() const;
    bool is_eltwise() const;
    bool is_eltwise_unary() const;
    bool is_eltwise_binary() const;
    bool is_eltwise_nary() const;

    /* --------------------------*
     * Optional implementations. *
     * --------------------------*/

    /**
     * Note: We will most likely get rid of distinct implementations for decompose, once we investigate why they even
     * exist. They are needed for now in order to unblock ops migration from python to cpp.
     */
    template <DecomposeEpoch epoch>
    void decompose(DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs) const;

    void decompose_initial(DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs) const;
    void decompose_post_optimize(DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs) const;
    void decompose_post_autograd(DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs) const;

    long initial_flops_estimate(const std::vector<std::vector<std::uint32_t>> &inputs) const;

   private:
    OpType type_;
    Attrs attrs_;
};

std::ostream &operator<<(std::ostream &out, const Attr &attr);
std::ostream &operator<<(std::ostream &out, const Op &op);

}  // namespace ops
}  // namespace tt
