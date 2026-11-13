// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "emitc_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

// =============================================================================
// Terminal layout
// =============================================================================

void print_separator(char c)
{
    std::cout << std::string(kLineWidth, c) << "\n";
}

void print_phase(int step, int total, const char* title)
{
    std::printf("\n[%d/%d]  %s\n", step, total, title);
    print_separator('-');
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
    // Borrow the torch buffer instead of copying it; the caller keeps the tensors alive.
    if (!t.is_contiguous() || !t.device().is_cpu()) {
        torch::Tensor tc = t.contiguous().cpu();
        std::vector<uint32_t> shape(tc.sizes().begin(),   tc.sizes().end());
        std::vector<int64_t>  stride(tc.strides().begin(), tc.strides().end());
        return tt::runtime::createOwnedHostTensor(
            tc.data_ptr(), shape, stride,
            static_cast<uint32_t>(tc.element_size()),
            tt::torch_scalar_type_to_dt(tc.scalar_type()));
    }

    std::vector<uint32_t> shape(t.sizes().begin(),   t.sizes().end());
    std::vector<int64_t>  stride(t.strides().begin(), t.strides().end());
    return tt::runtime::createBorrowedHostTensor(
        const_cast<void*>(t.data_ptr()), shape, stride,
        static_cast<uint32_t>(t.element_size()),
        tt::torch_scalar_type_to_dt(t.scalar_type()));
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

const PccOverrides& default_pcc_overrides()
{
    static const PccOverrides overrides = {
        {{"seq_89cf_1708956532201476864", 0}, 0.98f},
        {{"seq_89cf_1708956532268134400", 0}, 0.98f},
    };
    return overrides;
}

float required_pcc(const std::string& seq_id, std::size_t output_idx, float default_threshold)
{
    if (seq_id.empty()) return default_threshold;
    const auto& o = default_pcc_overrides();
    auto it = o.find({seq_id, output_idx});
    return it == o.end() ? default_threshold : it->second;
}

void print_pcc_overrides()
{
    for (const auto& [key, thr] : default_pcc_overrides())
        std::printf("    [%s] output %zu = %.3f\n", key.first.c_str(), key.second, thr);
}

bool validate_outputs(
    const std::vector<tt::runtime::Tensor>& device_host,
    const std::vector<torch::Tensor>&       golden,
    float                                   default_threshold,
    const std::string&                      seq_id,
    const std::vector<std::string>&         names)
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
        char metric[80];
        if (dev_t.numel() > 1) {
            float pcc = compute_pcc(dev_t, golden_t);
            float threshold = required_pcc(seq_id, i, default_threshold);
            pass = (pcc >= threshold);
            std::snprintf(metric, sizeof(metric), "PCC %.6f  (>= %.3f%s)", pcc, threshold,
                          threshold != default_threshold ? ", override" : "");
        } else {
            pass = (max_diff <= 0.1f);
            std::snprintf(metric, sizeof(metric), "atol %.6f  (<= 0.100)", max_diff);
        }

        std::printf("    [%zu]  %s  %-34s  max %7.4f  mean %7.4f  %s\n",
                    i, pass ? "PASS" : "FAIL", metric, max_diff, mean_diff,
                    i < names.size() ? names[i].c_str() : "");
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

void close_devices()
{
    // Closing the device flushes the device profiler.
    auto& system = tt::TTSystem::get_system();
    for (auto& dev : system.devices) {
        if (dev->is_open()) {
            dev->close_device();
        }
    }
}

// =============================================================================
// Benchmark helpers
// =============================================================================

std::vector<std::string> discover_sequences(const std::string& seq_dir)
{
    namespace fs = std::filesystem;
    std::vector<std::string> seqs;
    for (auto& entry : fs::directory_iterator(seq_dir)) {
        if (!entry.is_directory()) continue;
        if (fs::exists(entry.path() / kActFilename) &&
            fs::exists(entry.path() / kGoldenFilename))
            seqs.push_back(entry.path().filename().string());
    }
    std::sort(seqs.begin(), seqs.end());
    return seqs;
}

BenchStats compute_stats(const std::vector<double>& ms, int batch_size)
{
    auto sorted = ms;
    std::sort(sorted.begin(), sorted.end());

    auto percentile = [&](double p) {
        double      idx = p * static_cast<double>(sorted.size() - 1);
        std::size_t lo  = static_cast<std::size_t>(idx);
        std::size_t hi  = std::min(lo + 1, sorted.size() - 1);
        return sorted[lo] + (idx - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
    };

    double m = std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(ms.size());
    double var = 0.0;
    for (double x : ms) var += (x - m) * (x - m);

    return { m,
             ms.size() > 1 ? std::sqrt(var / static_cast<double>(ms.size() - 1)) : 0.0,
             sorted.front(),
             percentile(0.50),
             percentile(0.95),
             sorted.back(),
             batch_size * 1000.0 / m };
}

double mean_of(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

void print_progress(const char* label, int done, int total, std::size_t seq_idx, std::size_t n_seqs,
                    double last_ms, double avg_ms)
{
    std::printf("  %-7s %*d/%-*d  seq %zu/%zu  %7.2f ms  (avg %7.2f ms)\n",
                label, 2, done, 2, total, seq_idx + 1, n_seqs, last_ms, avg_ms);
}

void print_results_table(int n_iters, int n_warmup, int batch_size, const BenchStats& s,
                         std::size_t n_seqs)
{
    std::printf("  Benchmark Results    %d iters  ·  %d warmup  ·  batch %d  ·  %zu sequences\n\n",
                n_iters, n_warmup, batch_size, n_seqs);
    std::printf("  %-32s  %8.2f ± %5.2f ms\n", "Inference (H2D + kernel + D2H)", s.mean_ms, s.stddev_ms);
    std::printf("  %-32s  %8.2f ms\n", "    Min",          s.min_ms);
    std::printf("  %-32s  %8.2f ms\n", "    Median (P50)", s.median_ms);
    std::printf("  %-32s  %8.2f ms\n", "    P95",          s.p95_ms);
    std::printf("  %-32s  %8.2f ms\n", "    Max",          s.max_ms);
    print_separator('-');
    std::printf("  %-32s  %8.2f samples/sec\n", "Throughput", s.fps);
}

void print_phase_breakdown(const std::vector<double>& h2d, const std::vector<double>& run,
                           const std::vector<double>& d2h, double total_ms)
{
    const double mh = mean_of(h2d), mr = mean_of(run), md = mean_of(d2h);
    const auto pct = [total_ms](double x) { return total_ms > 0.0 ? 100.0 * x / total_ms : 0.0; };

    std::printf("\n  Per-frame breakdown\n");
    std::printf("  %-32s  %8.2f ms  (%4.1f%%)\n", "  H2D  input staging",   mh, pct(mh));
    std::printf("  %-32s  %8.2f ms  (%4.1f%%)\n", "  RUN  model call",      mr, pct(mr));
    std::printf("  %-32s  %8.2f ms  (%4.1f%%)\n", "  D2H  output readback", md, pct(md));
    print_separator('=');
}

void print_validation_summary(const std::vector<std::string>& seq_ids,
                              const std::vector<bool>& results, float pcc_threshold)
{
    int n_pass = 0;
    for (bool r : results) if (r) ++n_pass;

    std::printf("\n  Validation Summary    %zu sequence(s)  ·  PCC >= %.3f\n\n",
                seq_ids.size(), pcc_threshold);
    constexpr int kSeqCol = 40;
    std::printf("  %-*s  %s\n", kSeqCol, "Sequence", "Result");
    for (std::size_t i = 0; i < seq_ids.size(); ++i)
        std::printf("  %-*s  %s\n", kSeqCol, seq_ids[i].c_str(), results[i] ? "PASS" : "FAIL");

    std::printf("\n  %d/%zu sequences PASS\n", n_pass, seq_ids.size());
    std::printf("  per-output PCC overrides (default %.3f):\n", pcc_threshold);
    print_pcc_overrides();
}
