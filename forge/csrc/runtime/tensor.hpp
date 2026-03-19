// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstring>
#include <optional>
#include <variant>

#include "runtime/tt_device.hpp"
#include "torch/torch.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "utils/assert.hpp"

namespace tt
{

target::DataType torch_scalar_type_to_dt(torch::ScalarType st);
torch::ScalarType dt_to_torch_scalar_type(target::DataType df);

template <typename T>
std::vector<int64_t> as_vec_int64(std::vector<T> const& vec)
{
    std::vector<int64_t> result;
    result.reserve(vec.size());
    for (auto const& v : vec)
    {
        result.push_back(v);
    }
    return result;
}

// Class which wraps the host storage of a tensor.
//
// NOTE: currently this doesn't make much sense, since we only have one type of host storage (torch.Tensor).
// However, i expect that we will soon introduce at least one additional type of storage, so i've decided to introduce
// this class. Remove this if i turn out to be an optimist.
class TensorHostStorage
{
   public:
    explicit TensorHostStorage(torch::Tensor& tensor) : storage(tensor) {}

    void* data_ptr() const
    {
        return std::visit(
            [](auto&& arg) -> void*
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, torch::Tensor>)
                {
                    return arg.data_ptr();
                }
            },
            storage);
    }

    template <typename T>
    static TensorHostStorage from_desc(runtime::TensorDesc desc)
    {
        static_assert(std::is_same_v<T, torch::Tensor>, "Currently only torch::Tensor is supported");

        if constexpr (std::is_same_v<T, torch::Tensor>)
        {
            auto shape = as_vec_int64(desc.shape);
            auto stride = as_vec_int64(desc.stride);
            auto dtype = dt_to_torch_scalar_type(desc.dataType);

            auto tensor = at::empty_strided(shape, stride, dtype);
            return TensorHostStorage(tensor);
        }
    }

    std::variant<torch::Tensor> storage;
};

// Core implementation behind the `Tensor` class.
//
// It holds the tensor description and the actual data.
//
// When created from the host tensor, the host tensor is kept alive during the whole lifetime of the `Tensor` object.
// This way the original host tensor and our `Tensor` are linked, so that it is possible to update the original host
// tensor with new data that was computed on the device (and vice versa). This might turn out to be a bad idea,
// but time will tell.
class TensorImpl : public std::enable_shared_from_this<TensorImpl>
{
   public:
    TensorImpl(torch::Tensor& tensor) : host_storage(tensor), rt_tensor(std::nullopt)
    {
        auto shape = std::vector<uint32_t>(tensor.sizes().begin(), tensor.sizes().end());
        auto stride = std::vector<uint32_t>(tensor.strides().begin(), tensor.strides().end());

        desc.shape = shape;
        desc.stride = stride;
        desc.itemsize = tensor.element_size();
        desc.dataType = torch_scalar_type_to_dt(tensor.scalar_type());
    }

    TensorImpl(runtime::Tensor& tensor, runtime::TensorDesc tensor_desc) :
        host_storage(std::nullopt), desc(tensor_desc), rt_tensor(tensor)
    {
    }

    runtime::Tensor& get_runtime_tensor()
    {
        TT_ASSERT(rt_tensor.has_value());
        return *rt_tensor;
    }

    torch::Tensor to_torch()
    {
        if (!host_storage.has_value())
        {
            host_storage = TensorHostStorage::from_desc<torch::Tensor>(desc);
        }

        if (rt_tensor.has_value())
        {
            update_host_data();
        }

        return std::get<torch::Tensor>(host_storage->storage);
    }

    // Converts the tensor to the given layout via tt::runtime::toLayout (LayoutConverter).
    // Handles all transitions: host->device, device->device layout change, device->host.
    // No-op if the tensor already matches the target layout.
    void to_layout(const size_t device_id, runtime::Layout& layout)
    {
        if (!rt_tensor.has_value())
        {
            TT_ASSERT(host_storage.has_value(), "Tensor has neither runtime tensor nor host storage");
            rt_tensor = runtime::createBorrowedHostTensor(
                host_storage->data_ptr(), desc.shape, desc.stride, desc.itemsize, desc.dataType);
        }

        auto device = TTSystem::get_system().devices[device_id];
        rt_tensor = tt::runtime::toLayout(rt_tensor.value(), *device->rt_device, layout);
    }

    // Updates the host buffer with data from the runtime tensor.
    // Used when the original `tt::Tensor` was created from an existing torch (host) tensor, and later moved and
    // modified on the device.
    //
    // Example use case: optimizer step in training scenario (when executed on the device).
    // The original tensor (weight) was created from a torch tensor and then updated during execution of the optimizer
    // step on device.
    void update_host_data()
    {
        TT_ASSERT(
            rt_tensor.has_value() && host_storage.has_value(),
            "Both runtime tensor and host storage must exist to sync data back to host");

        constexpr bool untilize_tensor = true;
        auto sharded_tensor = tt::runtime::toHost(rt_tensor.value(), untilize_tensor);
        TT_ASSERT(sharded_tensor.size() == 1, "We don't expect sharded tensors, i.e. we expect only one shard");

        auto host = sharded_tensor[0];

        tt::runtime::memcpy(host_storage->data_ptr(), host);
    }

    void detach_from_device() { rt_tensor.reset(); }

    runtime::TensorDesc tensor_desc() const { return desc; }

   private:
    std::optional<TensorHostStorage> host_storage;
    runtime::TensorDesc desc;
    std::optional<runtime::Tensor> rt_tensor;
};

// Main `Tensor` class that wraps a ref-counted tensor and provides an interface to interact with it.
//
// The tensor can be created from a host tensor or from a device runtime handle (runtime::Tensor).
// For more details look at the `TensorImpl` class.
class Tensor
{
   public:
    explicit Tensor(torch::Tensor& tensor) : impl(new TensorImpl(tensor)) {}
    explicit Tensor(runtime::Tensor& tensor, runtime::TensorDesc tensor_desc) :
        impl(new TensorImpl(tensor, tensor_desc))
    {
    }

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;
    Tensor& operator=(const Tensor& other) = default;

    runtime::Tensor& get_runtime_tensor() { return impl->get_runtime_tensor(); }

    // Returns a torch tensor representing this tensor's host storage.
    // If the tensor is on device, it will first be copied to the host.
    torch::Tensor to_torch() const { return impl->to_torch(); }

    void to_layout(const size_t device_id, runtime::Layout& layout) { impl->to_layout(device_id, layout); }

    void update_host_data() { impl->update_host_data(); }

    runtime::TensorDesc tensor_desc() const { return impl->tensor_desc(); }

    void detach_from_device() { impl->detach_from_device(); }

   private:
    std::shared_ptr<TensorImpl> impl;
};

// Container for named tensors.
class TensorPool
{
   public:
    TensorPool() = default;
    TensorPool(TensorPool&&) = default;
    TensorPool(const TensorPool&) = delete;

    void insert(const std::string& name, torch::Tensor& tensor)
    {
        auto t = Tensor(tensor);
        insert(name, std::move(t));
    }

    Tensor get_tensor(const std::string& name) const
    {
        TT_ASSERT(tensor_name_to_value.find(name) != tensor_name_to_value.end(), "Tensor {} not found", name);
        return tensor_name_to_value.at(name);
    }

    bool exists(const std::string& name) const { return tensor_name_to_value.find(name) != tensor_name_to_value.end(); }

    void update_tensor(const std::string& name, tt::Tensor& tensor)
    {
        TT_ASSERT(tensor_name_to_value.find(name) != tensor_name_to_value.end(), "Tensor {} not found", name);
        tensor_name_to_value.at(name).get_runtime_tensor() = tensor.get_runtime_tensor();
    }

   private:
    std::unordered_map<std::string, Tensor> tensor_name_to_value;

    void insert(std::string name, Tensor& tensor)
    {
        if (tensor_name_to_value.find(name) != tensor_name_to_value.end())
        {
            return;
        }

        tensor_name_to_value.emplace(name, tensor);
    }

    void insert(std::string name, Tensor&& tensor)
    {
        if (tensor_name_to_value.find(name) != tensor_name_to_value.end())
        {
            return;
        }

        tensor_name_to_value.emplace(name, tensor);
    }
};

inline target::DataType torch_scalar_type_to_dt(torch::ScalarType st)
{
    switch (st)
    {
        case torch::ScalarType::Byte: return target::DataType::UInt8;
        case torch::ScalarType::Char: return target::DataType::UInt8;
        case torch::ScalarType::Short: return target::DataType::UInt16;
        case torch::ScalarType::Int: return target::DataType::Int32;
        case torch::ScalarType::Long: return target::DataType::UInt32;
        case torch::ScalarType::Half: return target::DataType::Float16;
        case torch::ScalarType::Float: return target::DataType::Float32;
        case torch::ScalarType::BFloat16: return target::DataType::BFloat16;
        case torch::ScalarType::Double:
        case torch::ScalarType::ComplexHalf:
        case torch::ScalarType::ComplexFloat:
        case torch::ScalarType::ComplexDouble:
        case torch::ScalarType::Bool:
        default: TT_THROW(false, "Unhandled scalar type {}", st);
    }

    __builtin_unreachable();
}

inline torch::ScalarType dt_to_torch_scalar_type(target::DataType df)
{
    switch (df)
    {
        case target::DataType::UInt8: return torch::ScalarType::Byte;
        case target::DataType::UInt16: return torch::ScalarType::Short;
        case target::DataType::UInt32: return torch::ScalarType::Int;
        case target::DataType::Int32: return torch::ScalarType::Int;
        case target::DataType::Float16: return torch::ScalarType::Half;
        case target::DataType::Float32: return torch::ScalarType::Float;
        case target::DataType::BFloat16: return torch::ScalarType::BFloat16;
        default: TT_THROW(false, "Unhandled dtype {}", target::EnumNameDataType(df));
    }

    __builtin_unreachable();
}

}  // namespace tt
