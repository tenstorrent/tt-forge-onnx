// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Standalone runner for resnet50_forward.so.
// Loads activation and persistent inputs from the TTNN binary format,
// runs the compiled forward pass on device, collects outputs, and compares
// them against golden outputs produced by compile_resnet50_emitc.py.
//
// Usage (from repo root):
//   ./build/.../run_resnet50_forward [so] [act.bin] [pers.bin] [golden.bin]

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "runtime/tensor.hpp"
#include "runtime/testutils/testutils.hpp"
#include "runtime/tt_device.hpp"
#include "tt/runtime/runtime.h"
#include "tt/runtime/test/ttnn/dylib.h"
#include "ttmlir/Target/Common/types_generated.h"

static constexpr const char* SO_PATH         = "emitc/so_files/resnet50_forward.so";
static constexpr const char* ACT_INPUTS_PATH = "emitc/inputs/activation_inputs.bin";
static constexpr const char* PERS_INPUTS_PATH = "emitc/inputs/persistent_inputs.bin";
static constexpr const char* GOLDEN_PATH     = "emitc/inputs/golden_outputs.bin";
static constexpr const char* FUNC_NAME       = "forward";

// ---------------------------------------------------------------------------
// TTNN binary format (little-endian):
//   magic(4B) + version(u32) + count(u32)
//   per tensor: ndim(u32) + shape(i64*ndim) + dtype(u32) + size(u64) + data
// ---------------------------------------------------------------------------
static std::vector<torch::Tensor> load_tensor_list(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open: " + path);

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "TTNN")
        throw std::runtime_error("Bad magic in: " + path);

    uint32_t version = 0, num_tensors = 0;
    f.read(reinterpret_cast<char*>(&version),     4);
    f.read(reinterpret_cast<char*>(&num_tensors), 4);

    static const c10::ScalarType kDtypes[] = {
        c10::kFloat,    // 0 float32
        c10::kHalf,     // 1 float16
        c10::kBFloat16, // 2 bfloat16
        c10::kInt,      // 3 int32
        c10::kLong,     // 4 int64
        c10::kChar,     // 5 int8
        c10::kByte,     // 6 uint8
    };

    std::vector<torch::Tensor> result;
    result.reserve(num_tensors);

    for (uint32_t i = 0; i < num_tensors; ++i)
    {
        uint32_t ndim = 0;
        f.read(reinterpret_cast<char*>(&ndim), 4);

        std::vector<int64_t> shape(ndim);
        if (ndim > 0)
            f.read(reinterpret_cast<char*>(shape.data()), static_cast<std::streamsize>(ndim) * 8);

        uint32_t dtype_code = 0;
        uint64_t data_size  = 0;
        f.read(reinterpret_cast<char*>(&dtype_code), 4);
        f.read(reinterpret_cast<char*>(&data_size),  8);

        auto tensor = torch::empty(shape, kDtypes[dtype_code < 7u ? dtype_code : 0u]);
        f.read(reinterpret_cast<char*>(tensor.data_ptr()), static_cast<std::streamsize>(data_size));
        result.push_back(std::move(tensor));
    }

    if (result.empty())
        throw std::runtime_error("No tensors found in: " + path);

    return result;
}

static tt::runtime::Tensor make_runtime_tensor(const torch::Tensor& t)
{
    torch::Tensor tc = t.contiguous().cpu();
    std::vector<uint32_t> shape(tc.sizes().begin(),   tc.sizes().end());
    std::vector<uint32_t> stride(tc.strides().begin(), tc.strides().end());
    return tt::runtime::createOwnedHostTensor(
        tc.data_ptr(),
        shape,
        stride,
        static_cast<uint32_t>(tc.element_size()),
        tt::torch_scalar_type_to_dt(tc.scalar_type()));
}

static c10::ScalarType dt_to_scalar_type(tt::target::DataType dt)
{
    switch (dt) {
        case tt::target::DataType::Float32:  return c10::kFloat;
        case tt::target::DataType::Float16:  return c10::kHalf;
        case tt::target::DataType::BFloat16: return c10::kBFloat16;
        case tt::target::DataType::Int32:    return c10::kInt;
        case tt::target::DataType::Int8:     return c10::kChar;
        case tt::target::DataType::UInt8:    return c10::kByte;
        case tt::target::DataType::UInt32:   return c10::kInt;
        default:                             return c10::kFloat;
    }
}

static torch::Tensor runtime_tensor_to_torch(const tt::runtime::Tensor& rt)
{
    auto shape_u32 = tt::runtime::getTensorShape(rt);
    std::vector<int64_t> shape(shape_u32.begin(), shape_u32.end());
    auto sc_type = dt_to_scalar_type(tt::runtime::getTensorDataType(rt));
    auto buf = tt::runtime::getTensorDataBuffer(rt);

    auto out = torch::empty(shape, sc_type);
    std::memcpy(out.data_ptr(), buf.data(), buf.size());
    return out;
}

static bool approx_compare(
    const std::vector<tt::runtime::Tensor>& emitc_host,
    const std::vector<torch::Tensor>&       golden,
    float rtol = 1e-2f,
    float atol = 1e-2f)
{
    if (emitc_host.size() != golden.size()) {
        std::cerr << "  Output count mismatch: emitc=" << emitc_host.size()
                  << " golden=" << golden.size() << std::endl;
        return false;
    }
    bool all_pass = true;
    for (std::size_t i = 0; i < emitc_host.size(); ++i) {
        auto emitc_t  = runtime_tensor_to_torch(emitc_host[i]).to(c10::kFloat);
        auto golden_t = golden[i].to(c10::kFloat);

        bool match = torch::allclose(emitc_t, golden_t, rtol, atol);
        auto diff   = (emitc_t - golden_t).abs();
        std::cout << "  Output[" << i << "]: " << (match ? "PASS" : "FAIL")
                  << "  max_diff=" << diff.max().item<float>()
                  << "  mean_diff=" << diff.mean().item<float>() << std::endl;
        all_pass &= match;
    }
    return all_pass;
}

int main(int argc, char* argv[])
{
    const std::string so_path     = (argc > 1) ? argv[1] : SO_PATH;
    const std::string act_path    = (argc > 2) ? argv[2] : ACT_INPUTS_PATH;
    const std::string pers_path   = (argc > 3) ? argv[3] : PERS_INPUTS_PATH;
    const std::string golden_path = (argc > 4) ? argv[4] : GOLDEN_PATH;

    std::cout << "ResNet-50 forward runner\n"
              << "  .so     : " << so_path     << "\n"
              << "  act     : " << act_path    << "\n"
              << "  pers    : " << pers_path   << "\n"
              << "  golden  : " << golden_path << "\n"
              << "  func    : " << FUNC_NAME   << std::endl;

    std::cout << "\nLoading inputs ..." << std::endl;
    auto act_torch    = load_tensor_list(act_path);
    auto pers_torch   = load_tensor_list(pers_path);
    auto golden_torch = load_tensor_list(golden_path);
    std::cout << "  activation: " << act_torch.size()    << " tensor(s)\n"
              << "  persistent: " << pers_torch.size()   << " tensor(s)\n"
              << "  golden    : " << golden_torch.size() << " tensor(s)" << std::endl;

    std::cout << "\nOpening device ..." << std::endl;
    auto& system = tt::TTSystem::get_system();
    for (auto& dev : system.devices)
        if (!dev->is_open())
            dev->open_device();

    if (system.devices.empty() || !system.devices[0]->is_open())
        throw std::runtime_error("No TT device available");

    tt::runtime::Device device = *system.devices[0]->rt_device;

    std::cout << "Opening shared object ..." << std::endl;
    void* so_handle = tt::runtime::testutils::open_so(so_path);

    std::cout << "Querying input layouts from .so ..." << std::endl;
    auto layout_templates =
        tt::runtime::test::ttnn::createInputs(so_handle, FUNC_NAME, device, so_path);

    std::size_t total_inputs = act_torch.size() + pers_torch.size();
    if (layout_templates.size() != total_inputs)
        throw std::runtime_error(
            ".so expects " + std::to_string(layout_templates.size()) +
            " inputs, got " + std::to_string(total_inputs));

    std::cout << "Moving tensors to device ..." << std::endl;
    std::vector<tt::runtime::Tensor> device_inputs;
    device_inputs.reserve(total_inputs);

    std::size_t idx = 0;
    for (const auto& t : act_torch)
    {
        auto layout = tt::runtime::getTensorLayout(layout_templates[idx++]);
        device_inputs.push_back(tt::runtime::toLayout(make_runtime_tensor(t), device, layout));
    }
    for (const auto& t : pers_torch)
    {
        auto layout = tt::runtime::getTensorLayout(layout_templates[idx++]);
        device_inputs.push_back(tt::runtime::toLayout(make_runtime_tensor(t), device, layout));
    }

    std::cout << "Running forward ..." << std::endl;
    std::vector<tt::runtime::Tensor> outputs =
        tt::runtime::test::ttnn::runSoProgram(so_handle, FUNC_NAME, device_inputs, device);

    std::cout << "\nCollecting outputs to host ..." << std::endl;
    std::vector<tt::runtime::Tensor> host_outputs;
    for (auto& out : outputs)
    {
        auto shards = tt::runtime::toHost(out, /*untilize=*/true);
        assert(!shards.empty());
        host_outputs.insert(host_outputs.end(), shards.begin(), shards.end());
    }
    std::cout << "  " << host_outputs.size() << " output shard(s) on host." << std::endl;

    std::cout << "\nComparing against golden ..." << std::endl;
    bool match = approx_compare(host_outputs, golden_torch);
    std::cout << "  Overall: " << (match ? "PASS" : "FAIL") << std::endl;

    tt::runtime::testutils::close_so(so_handle);

    std::cout << "\nDone." << std::endl;
    return match ? 0 : 1;
}
