// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Shared utilities for EmitC model runners.

#pragma once

#include <chrono>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "runtime/tensor.hpp"
#include "runtime/tt_device.hpp"
#include "tt/runtime/runtime.h"
#include "ttmlir/Target/Common/types_generated.h"

// Terminal layout
static constexpr int kLineWidth = 62;
void print_separator(char c = '=');
void print_phase(int step, int total, const char* title);

// TTNN binary loader (.bin files produced by the EmitC compilation scripts)
std::vector<torch::Tensor> load_tensor_list(const std::string& path);
const char*                dtype_name(c10::ScalarType st);
std::string                format_shape(const torch::Tensor& t);

// Tensor conversion between torch and TT runtime representations
tt::runtime::Tensor make_runtime_tensor(const torch::Tensor& t);
torch::Tensor       runtime_tensor_to_torch(const tt::runtime::Tensor& rt);

// PCC-based output validation
//
// Per-(sequence, output index) PCC threshold overrides, mirroring PCC_OVERRIDES in bev_model_benchmark.py. Output 0 = height map, 1 = semantic logits, 2 = visibility logits.
using PccOverrides = std::map<std::pair<std::string, std::size_t>, float>;
const PccOverrides& default_pcc_overrides();

// Threshold that applies to one (sequence, output) pair.
float required_pcc(const std::string& seq_id, std::size_t output_idx, float default_threshold);

// Prints the active overrides, one per line.
void print_pcc_overrides();

// seq_id selects the overrides above; leave it empty to apply `threshold` to every output.
// names, when given, label the outputs in the report (by output index).
bool validate_outputs(
    const std::vector<tt::runtime::Tensor>& device_host,
    const std::vector<torch::Tensor>&       golden,
    float                                   threshold,
    const std::string&                      seq_id = "",
    const std::vector<std::string>&         names  = {});

// Open device 0; the program cache must be enabled for a .so that uses trace capture.
tt::runtime::Device open_device(bool enable_program_cache = false);

// Close every open device; needed before exit when profiling.
void close_devices();

// =============================================================================
// Benchmark helpers shared by the EmitC and the profiling benchmark runners
// =============================================================================

// Directory conventions (produced by compile_bev_emitc.py --flatbuffer)
static constexpr const char* kActFilename    = "bev_activation_inputs.bin";
static constexpr const char* kGoldenFilename = "bev_device_outputs.bin";
static constexpr const char* kPersDefault    = "bev_persistent_inputs.bin";

// Subdirectories of seq_dir that hold both an activation and a golden file, sorted.
std::vector<std::string> discover_sequences(const std::string& seq_dir);

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

inline TimePoint now() { return Clock::now(); }
inline double elapsed_ms(TimePoint t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct BenchStats {
    double mean_ms, stddev_ms, min_ms, median_ms, p95_ms, max_ms, fps;
};
BenchStats compute_stats(const std::vector<double>& ms, int batch_size);
double     mean_of(const std::vector<double>& v);

// One inference pass: the outputs on the host plus the phase split.
struct RunResult {
    std::vector<tt::runtime::Tensor> host_outputs;
    double infer_ms = 0.0;
    double h2d_ms   = 0.0;   // input staging
    double run_ms   = 0.0;   // model call, device drained
    double d2h_ms   = 0.0;   // output readback
};

void print_progress(const char* label, int done, int total, std::size_t seq_idx, std::size_t n_seqs,
                    double last_ms, double avg_ms);
void print_results_table(int n_iters, int n_warmup, int batch_size, const BenchStats& s,
                         std::size_t n_seqs);
void print_phase_breakdown(const std::vector<double>& h2d, const std::vector<double>& run,
                           const std::vector<double>& d2h, double total_ms);
void print_validation_summary(const std::vector<std::string>& seq_ids,
                              const std::vector<bool>& results, float pcc_threshold);
