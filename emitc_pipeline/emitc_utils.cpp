// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "emitc_utils.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

// =============================================================================
// ANSI colors
// =============================================================================

namespace color {

static bool enabled = false;

void        init(bool force_off) { enabled = !force_off && (isatty(STDOUT_FILENO) != 0); }
const char* reset()    { return enabled ? "\033[0m"  : ""; }
const char* bold()     { return enabled ? "\033[1m"  : ""; }
const char* green()    { return enabled ? "\033[32m" : ""; }
const char* cyan()     { return enabled ? "\033[36m" : ""; }
const char* pass_tag() { return enabled ? "\033[1;32mPASS\033[0m" : "PASS"; }
const char* fail_tag() { return enabled ? "\033[1;31mFAIL\033[0m" : "FAIL"; }

} // namespace color

// =============================================================================
// Terminal layout
// =============================================================================

void print_separator(char c)
{
    std::cout << std::string(kLineWidth, c) << "\n";
}

void print_phase(int step, int total, const char* title)
{
    std::cout << "\n" << std::string(kLineWidth, '-') << "\n";
    std::printf("  %s[%d/%d]%s  %s\n", color::cyan(), step, total, color::reset(), title);
    std::cout << std::string(kLineWidth, '-') << "\n";
}

// =============================================================================
// TTNN binary loader
//
// File layout (little-endian):
//   magic(4B) + version(u32) + count(u32)
//   per tensor: ndim(u32) + shape(i64 * ndim) + dtype(u32) + byte_size(u64) + data
// =============================================================================

static const c10::ScalarType kDtypeTable[] = {
    c10::kFloat,    // 0  float32
    c10::kHalf,     // 1  float16
    c10::kBFloat16, // 2  bfloat16
    c10::kInt,      // 3  int32
    c10::kLong,     // 4  int64
    c10::kChar,     // 5  int8
    c10::kByte,     // 6  uint8
};

const char* dtype_name(c10::ScalarType st)
{
    switch (st) {
        case c10::kFloat:    return "float32";
        case c10::kHalf:     return "float16";
        case c10::kBFloat16: return "bfloat16";
        case c10::kInt:      return "int32";
        case c10::kLong:     return "int64";
        case c10::kChar:     return "int8";
        case c10::kByte:     return "uint8";
        default:             return "unknown";
    }
}

std::string format_shape(const torch::Tensor& t)
{
    std::ostringstream ss;
    ss << "[";
    for (int64_t i = 0; i < t.dim(); ++i) {
        if (i > 0) ss << " x ";
        ss << t.size(i);
    }
    ss << "]";
    return ss.str();
}

std::vector<torch::Tensor> load_tensor_list(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "TTNN")
        throw std::runtime_error("bad magic in: " + path);

    uint32_t version = 0, num_tensors = 0;
    f.read(reinterpret_cast<char*>(&version),     4);
    f.read(reinterpret_cast<char*>(&num_tensors), 4);

    std::vector<torch::Tensor> result;
    result.reserve(num_tensors);

    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint32_t ndim = 0;
        f.read(reinterpret_cast<char*>(&ndim), 4);

        std::vector<int64_t> shape(ndim);
        if (ndim > 0)
            f.read(reinterpret_cast<char*>(shape.data()),
                   static_cast<std::streamsize>(ndim) * 8);

        uint32_t dtype_code = 0;
        uint64_t data_size  = 0;
        f.read(reinterpret_cast<char*>(&dtype_code), 4);
        f.read(reinterpret_cast<char*>(&data_size),  8);

        c10::ScalarType st = kDtypeTable[dtype_code < 7u ? dtype_code : 0u];
        auto tensor = torch::empty(shape, st);
        f.read(reinterpret_cast<char*>(tensor.data_ptr()),
               static_cast<std::streamsize>(data_size));
        result.push_back(std::move(tensor));
    }

    if (result.empty())
        throw std::runtime_error("no tensors found in: " + path);

    return result;
}

// =============================================================================
// Tensor conversion
// =============================================================================

tt::runtime::Tensor make_runtime_tensor(const torch::Tensor& t)
{
    torch::Tensor tc = t.contiguous().cpu();
    std::vector<uint32_t> shape(tc.sizes().begin(),   tc.sizes().end());
    std::vector<int64_t>  stride(tc.strides().begin(), tc.strides().end());
    return tt::runtime::createOwnedHostTensor(
        tc.data_ptr(), shape, stride,
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

torch::Tensor runtime_tensor_to_torch(const tt::runtime::Tensor& rt)
{
    auto shape_u32 = tt::runtime::getTensorShape(rt);
    std::vector<int64_t> shape(shape_u32.begin(), shape_u32.end());
    auto buf = tt::runtime::getTensorDataBuffer(rt);

    auto out = torch::empty(shape, dt_to_scalar_type(tt::runtime::getTensorDataType(rt)));
    std::memcpy(out.data_ptr(), buf.data(),
                std::min(buf.size(), static_cast<std::size_t>(out.nbytes())));
    return out;
}

// =============================================================================
// Validation
// =============================================================================

static float compute_pcc(const torch::Tensor& a, const torch::Tensor& b)
{
    auto fa = a.to(c10::kFloat).flatten();
    auto fb = b.to(c10::kFloat).flatten();
    if (fa.numel() <= 1) return 1.0f;

    auto da = fa - fa.mean();
    auto db = fb - fb.mean();
    float sa = da.pow(2).mean().sqrt().item<float>();
    float sb = db.pow(2).mean().sqrt().item<float>();
    if (sa < 1e-8f || sb < 1e-8f) return 1.0f;

    return ((da * db).mean() / (da.pow(2).mean().sqrt() * db.pow(2).mean().sqrt()))
               .item<float>();
}

bool validate_outputs(
    const std::vector<tt::runtime::Tensor>& device_host,
    const std::vector<torch::Tensor>&       golden,
    float                                   threshold)
{
    if (device_host.size() != golden.size()) {
        std::fprintf(stderr, "  output count mismatch: device=%zu  golden=%zu\n",
                     device_host.size(), golden.size());
        return false;
    }

    bool all_pass = true;
    for (std::size_t i = 0; i < device_host.size(); ++i) {
        auto dev_t    = runtime_tensor_to_torch(device_host[i]).to(c10::kFloat);
        auto golden_t = golden[i].to(c10::kFloat);

        if (dev_t.sizes() != golden_t.sizes()) {
            std::printf("  Output[%zu]  shape mismatch  device=%s  golden=%s\n",
                        i, format_shape(dev_t).c_str(), format_shape(golden_t).c_str());
            all_pass = false;
            continue;
        }

        auto  diff      = (dev_t - golden_t).abs();
        float max_diff  = diff.max().item<float>();
        float mean_diff = diff.mean().item<float>();

        bool pass;
        char metric[72];
        if (dev_t.numel() > 1) {
            float pcc = compute_pcc(dev_t, golden_t);
            pass = (pcc >= threshold);
            std::snprintf(metric, sizeof(metric), "PCC %.6f  (thr %.3f)", pcc, threshold);
        } else {
            pass = (max_diff <= 0.1f);
            std::snprintf(metric, sizeof(metric), "atol %.6f  (thr 0.100)", max_diff);
        }

        std::printf("  Output[%zu]  %s  |  %s  |  max_diff %.4f  mean_diff %.4f\n",
                    i, pass ? color::pass_tag() : color::fail_tag(),
                    metric, max_diff, mean_diff);
        all_pass &= pass;
    }
    return all_pass;
}

// =============================================================================
// Device
// =============================================================================

tt::runtime::Device open_device(bool enable_program_cache)
{
    auto& system = tt::TTSystem::get_system();
    for (auto& dev : system.devices) {
        if (!dev->is_open()) {
            if (enable_program_cache) {
                tt::DeviceSettings ds;
                ds.enable_program_cache = true;
                dev->open_device(ds);
            } else {
                dev->open_device();
            }
        }
    }
    if (system.devices.empty() || !system.devices[0]->is_open())
        throw std::runtime_error("no TT device available");

    return *system.devices[0]->rt_device;
}
