// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Generic EmitC benchmark runner.
//
// Runs a compiled EmitC shared object on the device: loads the activation inputs,
// persistent weights and golden outputs from TTNN binary files, validates one frame
// against the golden, then runs the warmup iterations (on a traced model the first
// call is the trace capture) and the timed iterations, and prints the statistics.
// For Tracy profiling or flatbuffer binaries use model_benchmark_profile.
//
// Usage:
//   model_benchmark <so> <act.bin> <pers.bin> [<golden.bin>] [options]
//
// Inputs:
//   <so>                   compiled EmitC shared object
//   <act.bin>              activation inputs   (TTNN binary format)
//   <pers.bin>             persistent weights  (TTNN binary format)
//   <golden.bin>           golden outputs      (TTNN binary format; not needed with
//                          --skip-validation)
//
// Options:
//   --func <name>          entry-point function name  (default: forward)
//   --warmup <N>           warmup iterations          (default: 3, min: 1)
//   --iters <N>            timed iterations           (default: 10)
//   --batch-size <N>       batch size for FPS calc    (default: 1)
//   --pcc-threshold <f>    PCC pass threshold         (default: 0.99)
//   --skip-validation      skip the PCC validation pass
//   --blocking-d2h         synchronise on every D2H read (default: enqueue all
//                          reads, wait once per frame)
//   --no-device-sync       do not drain the device after the model call; RUN then
//                          measures host dispatch only and device time lands in D2H
//   --help                 show this message and exit
//
// Phases of one timed frame:
//   H2D  host tensors -> runtime tensors in the layout the program expects
//   RUN  the model call, followed by a device barrier (see emitc_device_sync.cpp)
//        so RUN covers device execution, not just enqueue
//   D2H  outputs back to host, untilized

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "emitc_utils.hpp"
#include "runtime/testutils/testutils.hpp"
#include "tt/runtime/runtime.h"
#include "tt/runtime/test/ttnn/dylib.h"

namespace
{

// =============================================================================
// CLI
// =============================================================================

struct Args
{
    std::string so_path;
    std::string act_bin;
    std::string pers_bin;
    std::string golden_bin;
    std::string func_name = "forward";
    int n_warmup = 3;
    int n_iters = 10;
    int batch_size = 1;
    float pcc_threshold = 0.99f;
    bool skip_validation = false;
    bool blocking_d2h = false;
    bool device_sync = true;
};

void print_usage(const char* prog)
{
    std::fprintf(
        stderr,
        "Usage: %s <so> <act.bin> <pers.bin> [<golden.bin>] [options]\n"
        "\n"
        "Inputs:\n"
        "  <so>                   compiled EmitC shared object (.so)\n"
        "  <act.bin>              activation inputs   (TTNN binary format)\n"
        "  <pers.bin>             persistent weights  (TTNN binary format)\n"
        "  <golden.bin>           golden outputs      (TTNN binary format; not needed\n"
        "                         with --skip-validation)\n"
        "\n"
        "Options:\n"
        "  --func <name>          entry-point function name  (default: forward)\n"
        "  --warmup <N>           warmup iterations          (default: 3, min 1)\n"
        "  --iters <N>            timed iterations           (default: 10)\n"
        "  --batch-size <N>       batch size for FPS calc    (default: 1)\n"
        "  --pcc-threshold <f>    PCC pass threshold         (default: 0.99)\n"
        "  --skip-validation      skip the PCC validation pass\n"
        "  --blocking-d2h         synchronise on every D2H read  (default: enqueue all\n"
        "                         reads and wait once per frame)\n"
        "  --no-device-sync       skip the device barrier after the model call; RUN\n"
        "                         then measures host dispatch only\n"
        "  --help                 show this message and exit\n",
        prog);
}

Args parse_args(int argc, char* argv[])
{
    Args a;
    int positional = 0;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return argv[++i]; };
        bool has_value = i + 1 < argc;

        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--func" && has_value)
            a.func_name = next();
        else if (arg == "--warmup" && has_value)
            a.n_warmup = std::stoi(next());
        else if (arg == "--iters" && has_value)
            a.n_iters = std::stoi(next());
        else if (arg == "--batch-size" && has_value)
            a.batch_size = std::stoi(next());
        else if (arg == "--pcc-threshold" && has_value)
            a.pcc_threshold = std::stof(next());
        else if (arg == "--skip-validation")
            a.skip_validation = true;
        else if (arg == "--blocking-d2h")
            a.blocking_d2h = true;
        else if (arg == "--no-device-sync")
            a.device_sync = false;
        else if (arg.rfind("--", 0) != 0 && positional < 4)
        {
            std::string* slots[] = {&a.so_path, &a.act_bin, &a.pers_bin, &a.golden_bin};
            *slots[positional++] = arg;
        }
        else
        {
            std::fprintf(stderr, "error: unknown option or unexpected argument: %s\n\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    bool bad = false;
    if (a.so_path.empty() || a.act_bin.empty() || a.pers_bin.empty())
    {
        std::fprintf(stderr, "error: need <so> <act.bin> <pers.bin>\n");
        bad = true;
    }
    if (!a.skip_validation && a.golden_bin.empty())
    {
        std::fprintf(stderr, "error: validation needs <golden.bin>; or pass --skip-validation\n");
        bad = true;
    }
    if (bad)
    {
        std::fprintf(stderr, "\n");
        print_usage(argv[0]);
        std::exit(1);
    }

    if (a.n_warmup < 1)
    {
        std::fprintf(stderr, "warning: --warmup clamped to 1 (iteration 1 is the trace capture)\n");
        a.n_warmup = 1;
    }
    if (a.n_iters < 1)
    {
        std::fprintf(stderr, "warning: --iters clamped to 1\n");
        a.n_iters = 1;
    }
    if (a.batch_size < 1)
    {
        std::fprintf(stderr, "warning: --batch-size clamped to 1\n");
        a.batch_size = 1;
    }
    return a;
}

// =============================================================================
// Timing and statistics
// =============================================================================

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

TimePoint now() { return Clock::now(); }

double elapsed_ms(TimePoint t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

double mean_of(const std::vector<double>& v)
{
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

struct BenchStats
{
    double mean_ms, stddev_ms, min_ms, median_ms, p95_ms, max_ms, fps;
};

BenchStats compute_stats(const std::vector<double>& ms, int batch_size)
{
    auto sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](double p)
    {
        double idx = p * static_cast<double>(sorted.size() - 1);
        std::size_t lo = static_cast<std::size_t>(idx);
        std::size_t hi = std::min(lo + 1, sorted.size() - 1);
        return sorted[lo] + (idx - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
    };

    double m = mean_of(ms);
    double var = 0.0;
    for (double x : ms) var += (x - m) * (x - m);

    return {
        m,
        ms.size() > 1 ? std::sqrt(var / static_cast<double>(ms.size() - 1)) : 0.0,
        sorted.front(),
        percentile(0.50),
        percentile(0.95),
        sorted.back(),
        batch_size * 1000.0 / m};
}

// =============================================================================
// One inference pass (H2D + RUN + D2H)
// =============================================================================

struct RunResult
{
    std::vector<tt::runtime::Tensor> host_outputs;
    double infer_ms = 0.0;
    double h2d_ms = 0.0;
    double run_ms = 0.0;
    double d2h_ms = 0.0;
};

struct DeviceInputs
{
    std::vector<tt::runtime::Layout> act_layouts;  // layouts the program expects for the activations
    std::vector<tt::runtime::Tensor> pers_device;  // weights, on device for the whole run
};

RunResult run_one(
    void* so_handle,
    const std::vector<torch::Tensor>& act_torch,
    const DeviceInputs& in,
    tt::runtime::Device& device,
    const Args& a)
{
    RunResult r{};
    auto t0 = now();

    auto t_h2d = now();
    std::vector<tt::runtime::Tensor> all_inputs;
    all_inputs.reserve(act_torch.size() + in.pers_device.size());
    for (std::size_t i = 0; i < act_torch.size(); ++i)
    {
        auto rt = make_runtime_tensor(act_torch[i]);
        all_inputs.push_back(
            tt::runtime::hasLayout(rt, in.act_layouts[i]) ? rt : tt::runtime::toLayout(rt, device, in.act_layouts[i]));
    }
    for (const auto& p : in.pers_device) all_inputs.push_back(p);
    r.h2d_ms = elapsed_ms(t_h2d);

    // runSoProgram() only enqueues the work. Without the barrier the wait for the
    // device lands in the D2H readback, which then reports most of the frame.
    auto t_run = now();
    std::vector<tt::runtime::Tensor> outputs =
        tt::runtime::test::ttnn::runSoProgram(so_handle, a.func_name, all_inputs, device);
    if (a.device_sync)
        device_synchronize(device);
    r.run_ms = elapsed_ms(t_run);

    auto t_d2h = now();
    for (auto& out : outputs)
    {
        auto shards = tt::runtime::toHost(out, /*untilize=*/true, /*blocking=*/a.blocking_d2h);
        assert(!shards.empty());
        r.host_outputs.insert(r.host_outputs.end(), shards.begin(), shards.end());
    }
    if (!a.blocking_d2h)
        tt::runtime::wait(r.host_outputs);
    r.d2h_ms = elapsed_ms(t_d2h);

    r.infer_ms = elapsed_ms(t0);
    return r;
}

// =============================================================================
// Reporting
// =============================================================================

void print_progress(const char* label, int done, int total, double last_ms, double avg_ms)
{
    std::printf("  %-7s %3d/%-3d  %8.2f ms  (avg %8.2f ms)\n", label, done, total, last_ms, avg_ms);
}

void print_config(const Args& a)
{
    print_separator('=');
    std::printf("  EmitC Model Benchmark\n");
    print_separator('=');
    std::printf("  %-18s  %s\n", "shared object", a.so_path.c_str());
    std::printf("  %-18s  %s\n", "activation inputs", a.act_bin.c_str());
    std::printf("  %-18s  %s\n", "persistent inputs", a.pers_bin.c_str());
    std::printf("  %-18s  %s\n", "golden", a.skip_validation ? "(validation skipped)" : a.golden_bin.c_str());
    std::printf("  %-18s  %s\n", "function", a.func_name.c_str());
    std::printf("  %-18s  %d\n", "warmup iters", a.n_warmup);
    std::printf("  %-18s  %d\n", "timed iters", a.n_iters);
    std::printf("  %-18s  %d\n", "batch size", a.batch_size);
    std::printf("  %-18s  %.3f\n", "pcc threshold", a.pcc_threshold);
    std::printf("  %-18s  %s\n", "D2H", a.blocking_d2h ? "blocking per output" : "enqueue all, wait once");
    std::printf("  %-18s  %s\n", "device sync", a.device_sync ? "after model call" : "off");
    print_separator('=');
}

void print_results(
    const Args& a,
    const BenchStats& s,
    const std::vector<double>& h2d,
    const std::vector<double>& run,
    const std::vector<double>& d2h)
{
    std::printf("  Benchmark Results    %d iters  ·  %d warmup  ·  batch %d\n\n", a.n_iters, a.n_warmup, a.batch_size);
    std::printf("  %-32s  %8.2f ± %5.2f ms\n", "Inference (H2D + RUN + D2H)", s.mean_ms, s.stddev_ms);
    std::printf("  %-32s  %8.2f ms\n", "    Min", s.min_ms);
    std::printf("  %-32s  %8.2f ms\n", "    Median (P50)", s.median_ms);
    std::printf("  %-32s  %8.2f ms\n", "    P95", s.p95_ms);
    std::printf("  %-32s  %8.2f ms\n", "    Max", s.max_ms);
    print_separator('-');
    std::printf("  %-32s  %8.2f samples/sec\n", "Throughput", s.fps);

    const double mh = mean_of(h2d), mr = mean_of(run), md = mean_of(d2h);
    auto pct = [&](double x) { return s.mean_ms > 0.0 ? 100.0 * x / s.mean_ms : 0.0; };
    std::printf("\n  Per-frame breakdown\n");
    std::printf("  %-32s  %8.2f ms  (%4.1f%%)\n", "  H2D  input staging", mh, pct(mh));
    std::printf(
        "  %-32s  %8.2f ms  (%4.1f%%)  %s\n",
        "  RUN  model call",
        mr,
        pct(mr),
        a.device_sync ? "[device drained]" : "[host dispatch only]");
    std::printf("  %-32s  %8.2f ms  (%4.1f%%)\n", "  D2H  output readback", md, pct(md));
    print_separator('=');
}

// =============================================================================
// Phases
// =============================================================================

// Phase 2: query the program's input layouts and move the weights to the device.
// Program input order is activations first, then persistent tensors.
DeviceInputs prepare_inputs(
    void* so_handle,
    const Args& a,
    tt::runtime::Device& device,
    std::size_t n_act,
    const std::vector<torch::Tensor>& pers_torch)
{
    const std::size_t n_pers = pers_torch.size();
    auto templates = tt::runtime::test::ttnn::createInputs(so_handle, a.func_name, device, a.so_path);
    if (templates.size() != n_act + n_pers)
        throw std::runtime_error(
            "input count mismatch: " + a.so_path + " expects " + std::to_string(templates.size()) +
            " inputs, the .bin files hold " + std::to_string(n_act + n_pers) +
            " (were they produced by the same compiler config?)");
    std::printf("  %zu layout(s) queried  (%zu activation + %zu persistent)\n", templates.size(), n_act, n_pers);

    DeviceInputs in;
    for (std::size_t i = 0; i < n_act; ++i) in.act_layouts.push_back(tt::runtime::getTensorLayout(templates[i]));

    // Persistent weights are uploaded once: they are constant across iterations.
    in.pers_device.reserve(n_pers);
    for (std::size_t i = 0; i < n_pers; ++i)
        in.pers_device.push_back(tt::runtime::toLayout(
            make_runtime_tensor(pers_torch[i]), device, tt::runtime::getTensorLayout(templates[n_act + i])));
    std::printf("  %zu persistent tensor(s) on device\n", in.pers_device.size());
    return in;
}

// Phase 3: one frame against the golden. On a traced model this first call is the
// trace capture and the warmup iterations that follow are replays.
template <typename RunFrame>
bool run_validation(const Args& a, RunFrame&& run_frame)
{
    if (a.skip_validation)
    {
        std::printf("  Validation skipped (--skip-validation).\n");
        return true;
    }
    auto golden = load_tensor_list(a.golden_bin);
    std::printf("  Running the frame once against %zu golden output(s) ...\n", golden.size());
    auto r = run_frame();
    const bool pass = validate_outputs(r.host_outputs, golden, a.pcc_threshold);
    std::printf("\n  Validation: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

struct TimedRuns
{
    std::vector<double> infer_ms, h2d_ms, run_ms, d2h_ms;
};

// Phase 4: warmup, then the timed frames.
template <typename RunFrame>
TimedRuns run_benchmark(const Args& a, RunFrame&& run_frame)
{
    std::printf("  Warmup  (%d iteration(s)%s)\n", a.n_warmup, a.skip_validation ? ", iter 1 = trace capture" : "");
    double warm_sum = 0.0;
    for (int i = 0; i < a.n_warmup; ++i)
    {
        auto r = run_frame();
        warm_sum += r.infer_ms;
        print_progress("warmup", i + 1, a.n_warmup, r.infer_ms, warm_sum / (i + 1));
    }

    std::printf("\n  Timed run  (%d iteration(s))\n", a.n_iters);
    TimedRuns t;
    double running = 0.0;
    for (int i = 0; i < a.n_iters; ++i)
    {
        auto r = run_frame();
        t.infer_ms.push_back(r.infer_ms);
        t.h2d_ms.push_back(r.h2d_ms);
        t.run_ms.push_back(r.run_ms);
        t.d2h_ms.push_back(r.d2h_ms);
        running += r.infer_ms;
        print_progress("timed", i + 1, a.n_iters, r.infer_ms, running / (i + 1));
    }
    return t;
}

}  // namespace

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[])
{
    const Args a = parse_args(argc, argv);
    print_config(a);
    constexpr int kPhases = 5;

    print_phase(1, kPhases, "Loading inputs & opening device");
    auto act_torch = load_tensor_list(a.act_bin);
    auto pers_torch = load_tensor_list(a.pers_bin);
    std::printf("  activation  %zu tensor(s)  [%s]\n", act_torch.size(), a.act_bin.c_str());
    std::printf("  persistent  %zu tensor(s)  [%s]\n", pers_torch.size(), a.pers_bin.c_str());
    tt::runtime::Device device = open_device(/*enable_program_cache=*/true);
    std::printf("  device opened  (program cache enabled)\n");
    void* so_handle = tt::runtime::testutils::open_so(a.so_path);
    std::printf("  shared object loaded\n");

    print_phase(2, kPhases, "Preparing device inputs");
    const DeviceInputs in = prepare_inputs(so_handle, a, device, act_torch.size(), pers_torch);

    auto run_frame = [&]() { return run_one(so_handle, act_torch, in, device, a); };

    print_phase(3, kPhases, "Validation");
    const bool val_pass = run_validation(a, run_frame);

    print_phase(4, kPhases, "Benchmark");
    const TimedRuns t = run_benchmark(a, run_frame);

    print_phase(5, kPhases, "Results");
    print_results(a, compute_stats(t.infer_ms, a.batch_size), t.h2d_ms, t.run_ms, t.d2h_ms);

    std::fflush(stdout);
    tt::runtime::testutils::close_so(so_handle);
    std::printf("\nDone.\n");
    std::fflush(stdout);
    // _Exit skips atexit / static-destructor cleanup to avoid a SIGSEGV in _dl_fini:
    // _ttnncpp.so's static destructors run after libtt_metal.so frees its
    // thread_local GraphTracker TLS, making any Tensor destructor a use-after-free.
    std::_Exit(val_pass ? 0 : 1);
}
