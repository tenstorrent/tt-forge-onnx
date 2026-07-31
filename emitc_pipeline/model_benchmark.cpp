// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Generic EmitC benchmark runner.
//
// Loads activation inputs, persistent weights, and golden outputs from TTNN
// binary files.  Runs N warmup iterations (the first performs trace capture),
// then N timed iterations measuring full H2D + kernel + D2H wall-clock time.
// Prints a statistics table and validates the last output against golden.
//
// Usage:
//   model_benchmark <so> <act.bin> <pers.bin> <golden.bin> [options]
//
// Required:
//   <so>               compiled EmitC shared object
//   <act.bin>          activation inputs   (TTNN binary format)
//   <pers.bin>         persistent weights  (TTNN binary format)
//   <golden.bin>       golden outputs      (TTNN binary format)
//
// Options:
//   --func <name>          entry-point function name  (default: forward)
//   --warmup <N>           warmup iterations          (default: 3, min: 1)
//   --iters <N>            timed iterations           (default: 10)
//   --batch-size <N>       batch size for FPS calc    (default: 1)
//   --pcc-threshold <f>    PCC pass threshold         (default: 0.99)
//   --no-color             disable ANSI color output
//   --help                 show this message and exit

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "emitc_utils.hpp"
#include "runtime/testutils/testutils.hpp"
#include "tt/runtime/test/ttnn/dylib.h"

// =============================================================================
// CLI
// =============================================================================

struct Args {
    std::string so_path;
    std::string act_path;
    std::string pers_path;
    std::string golden_path;
    std::string func_name     = "forward";
    int         n_warmup      = 3;
    int         n_iters       = 10;
    int         batch_size    = 1;
    float       pcc_threshold = 0.99f;
    bool        no_color      = false;
};

static void print_usage(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s <so> <act.bin> <pers.bin> <golden.bin> [options]\n"
        "\n"
        "Required:\n"
        "  <so>               compiled EmitC shared object (.so)\n"
        "  <act.bin>          activation inputs   (TTNN binary format)\n"
        "  <pers.bin>         persistent weights  (TTNN binary format)\n"
        "  <golden.bin>       golden outputs      (TTNN binary format)\n"
        "\n"
        "Options:\n"
        "  --func <name>          entry-point function name  (default: forward)\n"
        "  --warmup <N>           warmup iterations          (default: 3)\n"
        "  --iters <N>            timed iterations           (default: 10)\n"
        "  --batch-size <N>       batch size for FPS calc    (default: 1)\n"
        "  --pcc-threshold <f>    PCC pass threshold         (default: 0.99)\n"
        "  --no-color             disable ANSI color output\n"
        "  --help                 show this message and exit\n",
        prog);
}

static Args parse_args(int argc, char* argv[])
{
    Args a;
    int  positional = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]); std::exit(0);
        } else if (arg == "--func"          && i + 1 < argc) { a.func_name     = argv[++i];
        } else if (arg == "--warmup"        && i + 1 < argc) { a.n_warmup      = std::stoi(argv[++i]);
        } else if (arg == "--iters"         && i + 1 < argc) { a.n_iters       = std::stoi(argv[++i]);
        } else if (arg == "--batch-size"    && i + 1 < argc) { a.batch_size    = std::stoi(argv[++i]);
        } else if (arg == "--pcc-threshold" && i + 1 < argc) { a.pcc_threshold = std::stof(argv[++i]);
        } else if (arg == "--no-color") { a.no_color = true;
        } else if (arg.rfind("--", 0) != 0) {
            switch (positional++) {
                case 0: a.so_path     = arg; break;
                case 1: a.act_path    = arg; break;
                case 2: a.pers_path   = arg; break;
                case 3: a.golden_path = arg; break;
                default:
                    std::fprintf(stderr, "error: unexpected argument: %s\n\n", arg.c_str());
                    print_usage(argv[0]); std::exit(1);
            }
        } else {
            std::fprintf(stderr, "error: unknown option: %s\n\n", arg.c_str());
            print_usage(argv[0]); std::exit(1);
        }
    }

    bool missing = false;
    if (a.so_path.empty())     { std::fprintf(stderr, "error: missing required argument <so>\n");         missing = true; }
    if (a.act_path.empty())    { std::fprintf(stderr, "error: missing required argument <act.bin>\n");    missing = true; }
    if (a.pers_path.empty())   { std::fprintf(stderr, "error: missing required argument <pers.bin>\n");   missing = true; }
    if (a.golden_path.empty()) { std::fprintf(stderr, "error: missing required argument <golden.bin>\n"); missing = true; }
    if (missing) { std::fprintf(stderr, "\n"); print_usage(argv[0]); std::exit(1); }

    // First warmup = trace capture; at least one is required.
    if (a.n_warmup < 1)   { std::fprintf(stderr, "warning: --warmup clamped to 1 (trace capture requires it)\n"); a.n_warmup   = 1; }
    if (a.n_iters   < 1)  { std::fprintf(stderr, "warning: --iters clamped to 1\n");                              a.n_iters    = 1; }
    if (a.batch_size < 1) { std::fprintf(stderr, "warning: --batch-size clamped to 1\n");                         a.batch_size = 1; }

    return a;
}

// =============================================================================
// Timing and statistics
// =============================================================================

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

static inline TimePoint now() { return Clock::now(); }
static inline double elapsed_ms(TimePoint t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct BenchStats {
    double mean_ms, stddev_ms, min_ms, median_ms, p95_ms, max_ms, fps;
};

static BenchStats compute_stats(const std::vector<double>& ms, int batch_size)
{
    auto sorted = ms;
    std::sort(sorted.begin(), sorted.end());

    auto percentile = [&](double p) {
        double idx = p * static_cast<double>(sorted.size() - 1);
        std::size_t lo = static_cast<std::size_t>(idx);
        std::size_t hi = std::min(lo + 1, sorted.size() - 1);
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

// =============================================================================
// Terminal UI — benchmark-specific
// =============================================================================

static void print_progress(const char* label, int done, int total,
                           double last_ms = -1.0, double avg_ms = -1.0)
{
    constexpr int BAR = 28;
    int filled = (total > 0) ? (done * BAR / total) : BAR;

    std::printf("\r  %-8s  [%s%s%s%s]  %d/%d",
                label,
                color::green(),
                std::string(filled, '#').c_str(),
                color::reset(),
                std::string(BAR - filled, '-').c_str(),
                done, total);

    if (last_ms >= 0.0) std::printf("  %6.2f ms", last_ms);
    if (avg_ms  >= 0.0) std::printf("  (avg %6.2f ms)", avg_ms);
    if (done == total)  std::printf("\n");
    std::fflush(stdout);
}

static void print_results_table(const Args& a, const BenchStats& s)
{
    print_separator('=');
    std::printf("  %sBenchmark Results%s"
                "     %d iters  ·  %d warmup  ·  batch %d\n",
                color::bold(), color::reset(),
                a.n_iters, a.n_warmup, a.batch_size);
    print_separator('=');
    std::printf("  %-32s  %8.2f ± %5.2f ms\n",
                "Inference (H2D + kernel + D2H)", s.mean_ms, s.stddev_ms);
    std::printf("  %-32s  %8.2f ms\n", "    Min",          s.min_ms);
    std::printf("  %-32s  %8.2f ms\n", "    Median (P50)", s.median_ms);
    std::printf("  %-32s  %8.2f ms\n", "    P95",          s.p95_ms);
    std::printf("  %-32s  %8.2f ms\n", "    Max",          s.max_ms);
    print_separator('-');
    std::printf("  %-32s  %8.2f samples/sec\n", "Throughput", s.fps);
    print_separator('=');
}

// =============================================================================
// Single inference pass — timed as one wall-clock lap (H2D + kernel + D2H).
// Reporting the phases separately would be misleading on a pipelined device
// where their boundaries overlap.
// =============================================================================

struct RunResult {
    std::vector<tt::runtime::Tensor> host_outputs;
    double infer_ms;
};

static RunResult run_one(
    void*                                   so_handle,
    const std::string&                      func_name,
    const std::vector<torch::Tensor>&       act_torch,
    const std::vector<tt::runtime::Tensor>& pers_device,
    const std::vector<tt::runtime::Layout>& act_layouts,
    tt::runtime::Device&                    device)
{
    RunResult r{};
    auto t0 = now();

    std::vector<tt::runtime::Tensor> all_inputs;
    all_inputs.reserve(act_torch.size() + pers_device.size());
    for (std::size_t i = 0; i < act_torch.size(); ++i)
        all_inputs.push_back(
            tt::runtime::toLayout(make_runtime_tensor(act_torch[i]), device, act_layouts[i]));
    for (const auto& p : pers_device)
        all_inputs.push_back(p);

    auto outputs =
        tt::runtime::test::ttnn::runSoProgram(so_handle, func_name, all_inputs, device);

    // Enqueue all D2H reads non-blocking, then wait once to amortize the sync
    // cost across all output shards.
    for (auto& out : outputs) {
        auto shards = tt::runtime::toHost(out, /*untilize=*/true, /*blocking=*/false);
        assert(!shards.empty());
        r.host_outputs.insert(r.host_outputs.end(), shards.begin(), shards.end());
    }
    tt::runtime::wait(r.host_outputs);

    r.infer_ms = elapsed_ms(t0);
    return r;
}

// =============================================================================
// Device input preparation
// =============================================================================

// Persistent weights are uploaded once before the timed loop — they are
// constant across all inference calls so per-iteration transfer is wasteful.
static std::vector<tt::runtime::Tensor> upload_persistent_weights(
    const std::vector<torch::Tensor>&       pers_torch,
    const std::vector<tt::runtime::Tensor>& layout_templates,
    std::size_t                             pers_offset,
    tt::runtime::Device&                    device)
{
    std::vector<tt::runtime::Tensor> pers_device;
    pers_device.reserve(pers_torch.size());
    for (std::size_t i = 0; i < pers_torch.size(); ++i) {
        auto layout = tt::runtime::getTensorLayout(layout_templates[pers_offset + i]);
        pers_device.push_back(
            tt::runtime::toLayout(make_runtime_tensor(pers_torch[i]), device, layout));
    }
    return pers_device;
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[])
{
    Args a = parse_args(argc, argv);
    color::init(a.no_color);

    print_separator('=');
    std::printf("  %sEmitC Model Benchmark%s\n", color::bold(), color::reset());
    print_separator('=');
    std::printf("  %-16s  %s\n",   "shared object",  a.so_path.c_str());
    std::printf("  %-16s  %s\n",   "activation",     a.act_path.c_str());
    std::printf("  %-16s  %s\n",   "persistent",     a.pers_path.c_str());
    std::printf("  %-16s  %s\n",   "golden",         a.golden_path.c_str());
    std::printf("  %-16s  %s\n",   "function",       a.func_name.c_str());
    std::printf("  %-16s  %d\n",   "warmup iters",   a.n_warmup);
    std::printf("  %-16s  %d\n",   "timed iters",    a.n_iters);
    std::printf("  %-16s  %d\n",   "batch size",     a.batch_size);
    std::printf("  %-16s  %.3f\n", "pcc threshold",  a.pcc_threshold);
    print_separator('=');

    constexpr int kPhases = 5;

    // -------------------------------------------------------------------------
    print_phase(1, kPhases, "Loading inputs");

    auto act_torch    = load_tensor_list(a.act_path);
    auto pers_torch   = load_tensor_list(a.pers_path);
    auto golden_torch = load_tensor_list(a.golden_path);

    std::printf("  activation  %3zu tensor(s)  %s  %s\n", act_torch.size(),
                act_torch.empty() ? "" : format_shape(act_torch[0]).c_str(),
                act_torch.empty() ? "" : dtype_name(act_torch[0].scalar_type()));
    std::printf("  persistent  %3zu tensor(s)  %s  %s\n", pers_torch.size(),
                pers_torch.empty() ? "" : format_shape(pers_torch[0]).c_str(),
                pers_torch.empty() ? "" : dtype_name(pers_torch[0].scalar_type()));
    std::printf("  golden      %3zu tensor(s)  %s  %s\n", golden_torch.size(),
                golden_torch.empty() ? "" : format_shape(golden_torch[0]).c_str(),
                golden_torch.empty() ? "" : dtype_name(golden_torch[0].scalar_type()));

    // -------------------------------------------------------------------------
    print_phase(2, kPhases, "Opening device & shared object");

    tt::runtime::Device device = open_device(/*enable_program_cache=*/true);
    std::printf("  device opened  (program cache %senabled%s)\n",
                color::green(), color::reset());

    void* so_handle = tt::runtime::testutils::open_so(a.so_path);
    std::printf("  shared object loaded\n");

    // -------------------------------------------------------------------------
    print_phase(3, kPhases, "Preparing device inputs");

    auto layout_templates =
        tt::runtime::test::ttnn::createInputs(so_handle, a.func_name, device, a.so_path);

    std::size_t total_expected = act_torch.size() + pers_torch.size();
    if (layout_templates.size() != total_expected)
        throw std::runtime_error(
            "input count mismatch: .so expects " +
            std::to_string(layout_templates.size()) +
            ", got " + std::to_string(total_expected));

    std::printf("  %zu layout(s) queried  (%zu activation + %zu persistent)\n",
                layout_templates.size(), act_torch.size(), pers_torch.size());

    std::vector<tt::runtime::Layout> act_layouts;
    act_layouts.reserve(act_torch.size());
    for (std::size_t i = 0; i < act_torch.size(); ++i)
        act_layouts.push_back(tt::runtime::getTensorLayout(layout_templates[i]));

    auto pers_device =
        upload_persistent_weights(pers_torch, layout_templates, act_torch.size(), device);
    std::printf("  %zu persistent tensor(s) on device\n", pers_device.size());

    // -------------------------------------------------------------------------
    print_phase(4, kPhases, "Running benchmark");

    std::printf("  %sWarmup%s  (%d iteration(s), iter 1 = trace capture) ...\n",
                color::cyan(), color::reset(), a.n_warmup);
    for (int i = 0; i < a.n_warmup; ++i) {
        run_one(so_handle, a.func_name, act_torch, pers_device, act_layouts, device);
        print_progress("warmup", i + 1, a.n_warmup);
    }

    std::printf("\n  %sTimed run%s  (%d iteration(s)) ...\n",
                color::cyan(), color::reset(), a.n_iters);

    std::vector<double> infer_ms(a.n_iters);
    std::vector<tt::runtime::Tensor> last_outputs;
    double running_sum = 0.0;

    for (int i = 0; i < a.n_iters; ++i) {
        auto r      = run_one(so_handle, a.func_name, act_torch, pers_device, act_layouts, device);
        infer_ms[i] = r.infer_ms;
        running_sum += r.infer_ms;
        last_outputs = std::move(r.host_outputs);
        print_progress("timed", i + 1, a.n_iters,
                       r.infer_ms, running_sum / static_cast<double>(i + 1));
    }

    // -------------------------------------------------------------------------
    print_phase(5, kPhases, "Results & validation");

    print_results_table(a, compute_stats(infer_ms, a.batch_size));

    std::printf("\n  PCC validation  (threshold >= %.3f)\n\n", a.pcc_threshold);
    bool pass = validate_outputs(last_outputs, golden_torch, a.pcc_threshold);
    std::printf("\n  Overall:  %s\n", pass ? color::pass_tag() : color::fail_tag());

    std::cout.flush();
    std::fflush(stdout);
    tt::runtime::testutils::close_so(so_handle);
    std::printf("\nDone.\n");
    std::fflush(stdout);

    // _Exit skips atexit / static-dtor cleanup to avoid a SIGSEGV in _dl_fini.
    // _ttnncpp.so static dtors run after libtt_metal.so frees its
    // thread_local GraphTracker TLS, making any Tensor dtor a use-after-free.
    std::_Exit(pass ? 0 : 1);
}
