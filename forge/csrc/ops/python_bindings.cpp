// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ops/python_bindings.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <utils/raw_ptr.hpp>

#include "ops/op.hpp"

PYBIND11_DECLARE_HOLDER_TYPE(T, tt::raw_ptr<T>);

namespace tt
{

void OpsModule(py::module &m_ops)
{
    py::class_<tt::ops::Op, tt::raw_ptr<tt::ops::Op>>(m_ops, "Op")
        .def(
            py::init([](std::string const &op_name, tt::ops::Attrs const &attrs) { return ops::Op(op_name, attrs); }),
            py::arg("op_name"),
            py::arg("attrs") = tt::ops::Attrs{})
        .def(
            py::init([](ops::OpType type, tt::ops::Attrs const &attrs) { return ops::Op(type, attrs); }),
            py::arg("type"),
            py::arg("attrs") = tt::ops::Attrs{})
        .def("eval", &tt::ops::Op::eval)
        .def("shape", &tt::ops::Op::shape)
        .def("__getattr__", [](tt::ops::Op const &op, std::string const &name) { return op.attrs().at(name); })
        .def(
            "__setattr__",
            [](tt::ops::Op &op_type, std::string const &name, tt::ops::Attr value)
            { return op_type.set_attr(name, value); })
        .def("__repr__", [](tt::ops::Op const &op_type) { return op_type.as_string(); })
        .def("type", &tt::ops::Op::type)
        .def("attrs", &tt::ops::Op::attrs);

    py::enum_<ops::OpType>(m_ops, "OpType")
        .value("Abs", ops::OpType::Abs)
        .value("AdaptiveMaxPool2d", ops::OpType::AdaptiveMaxPool2d)
        .value("Add", ops::OpType::Add)
        .value("AdvIndex", ops::OpType::AdvIndex)
        .value("Argmax", ops::OpType::Argmax)
        .value("Atan", ops::OpType::Atan)
        .value("AvgPool1d", ops::OpType::AvgPool1d)
        .value("AvgPool2d", ops::OpType::AvgPool2d)
        .value("Batchnorm", ops::OpType::Batchnorm)
        .value("Broadcast", ops::OpType::Broadcast)
        .value("Cast", ops::OpType::Cast)
        .value("Clip", ops::OpType::Clip)
        .value("Concatenate", ops::OpType::Concatenate)
        .value("Constant", ops::OpType::Constant)
        .value("Conv2d", ops::OpType::Conv2d)
        .value("Conv2dPrestrideWeights", ops::OpType::Conv2dPrestrideWeights)
        .value("Conv2dTranspose", ops::OpType::Conv2dTranspose)
        .value("Cosine", ops::OpType::Cosine)
        .value("CumulativeSum", ops::OpType::CumulativeSum)
        .value("Divide", ops::OpType::Divide)
        .value("Downsample2d", ops::OpType::Downsample2d)
        .value("Dropout", ops::OpType::Dropout)
        .value("Embedding", ops::OpType::Embedding)
        .value("EmbeddingBw", ops::OpType::EmbeddingBw)
        .value("Equal", ops::OpType::Equal)
        .value("Erf", ops::OpType::Erf)
        .value("Exp", ops::OpType::Exp)
        .value("FillCache", ops::OpType::FillCache)
        .value("Gelu", ops::OpType::Gelu)
        .value("Greater", ops::OpType::Greater)
        .value("GreaterEqual", ops::OpType::GreaterEqual)
        .value("Heaviside", ops::OpType::Heaviside)
        .value("Index", ops::OpType::Index)
        .value("IndexCopy", ops::OpType::IndexCopy)
        .value("Layernorm", ops::OpType::Layernorm)
        .value("LayernormBw", ops::OpType::LayernormBw)
        .value("LeakyRelu", ops::OpType::LeakyRelu)
        .value("Less", ops::OpType::Less)
        .value("LessEqual", ops::OpType::LessEqual)
        .value("Log", ops::OpType::Log)
        .value("LogSoftmax", ops::OpType::LogSoftmax)
        .value("LogicalAnd", ops::OpType::LogicalAnd)
        .value("LogicalNot", ops::OpType::LogicalNot)
        .value("BitwiseAnd", ops::OpType::BitwiseAnd)
        .value("Mask", ops::OpType::Mask)
        .value("Matmul", ops::OpType::Matmul)
        .value("MaxPool1d", ops::OpType::MaxPool1d)
        .value("MaxPool2d", ops::OpType::MaxPool2d)
        .value("Maximum", ops::OpType::Maximum)
        .value("Minimum", ops::OpType::Minimum)
        .value("Multiply", ops::OpType::Multiply)
        .value("Nop", ops::OpType::Nop)
        .value("NotEqual", ops::OpType::NotEqual)
        .value("Pad", ops::OpType::Pad)
        .value("PixelShuffle", ops::OpType::PixelShuffle)
        .value("Pow", ops::OpType::Pow)
        .value("Power", ops::OpType::Power)
        .value("Reciprocal", ops::OpType::Reciprocal)
        .value("ReduceAvg", ops::OpType::ReduceAvg)
        .value("ReduceMax", ops::OpType::ReduceMax)
        .value("ReduceSum", ops::OpType::ReduceSum)
        .value("Relu", ops::OpType::Relu)
        .value("Remainder", ops::OpType::Remainder)
        .value("Repeat", ops::OpType::Repeat)
        .value("RepeatInterleave", ops::OpType::RepeatInterleave)
        .value("Reshape", ops::OpType::Reshape)
        .value("Resize1d", ops::OpType::Resize1d)
        .value("Resize2d", ops::OpType::Resize2d)
        .value("Select", ops::OpType::Select)
        .value("Sigmoid", ops::OpType::Sigmoid)
        .value("Sine", ops::OpType::Sine)
        .value("Softmax", ops::OpType::Softmax)
        .value("SoftmaxBw", ops::OpType::SoftmaxBw)
        .value("Sqrt", ops::OpType::Sqrt)
        .value("Stack", ops::OpType::Stack)
        .value("Subtract", ops::OpType::Subtract)
        .value("Squeeze", ops::OpType::Squeeze)
        .value("Tanh", ops::OpType::Tanh)
        .value("Transpose", ops::OpType::Transpose)
        .value("Unsqueeze", ops::OpType::Unsqueeze)
        .value("UpdateCache", ops::OpType::UpdateCache)
        .value("Upsample2d", ops::OpType::Upsample2d)
        .value("GridSample", ops::OpType::GridSample)
        .value("Where", ops::OpType::Where)
        .export_values();
}

}  // namespace tt
