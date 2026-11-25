// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark runner for the EVO50 EmitC and flatbuffer pipelines, with Tracy support.
//
// Ported from the BEV runner (Forge4 emitc_pipeline/model_benchmark_profile.cpp) to
// EVO50, which has a single input frame rather than BEV's per-sequence directories.
//
// It runs either the EmitC shared object or the flatbuffer binary through the same
// timing path, so the two pipelines can be compared directly. It also emits Tracy
// signposts around each timed iteration and around its input / model / readback
// phases, so an ops-perf report can be sliced to exactly the timed region.
//
// Inputs (all produced by evo50_model/compile_evo50_emitc.py --flatbuffer):
//   evo50_activation_inputs.bin      activation inputs   (TTNN binary format)
//   evo50_persistent_inputs.bin      persistent weights  (TTNN binary format)
//   evo50_golden_outputs_device.bin  device golden       (TTNN binary format)
// Pass --inputs-dir to use those three names, or name each file explicitly.
//
// Usage:
//   model_benchmark_profile <so>                --inputs-dir <dir> [options]
//   model_benchmark_profile --flatbuffer <ttnn> --inputs-dir <dir> [options]
//
// Options:
//   --inputs-dir <dir>     directory holding the three .bin files above
//   --act-bin <path>       activation inputs   (overrides --inputs-dir)
//   --pers-bin <path>      persistent weights  (overrides --inputs-dir)
//   --golden-bin <path>    device golden       (overrides --inputs-dir)
//   --flatbuffer <path>    benchmark a flatbuffer (.ttnn) instead of an EmitC .so
//   --func <name>          entry-point function name  (default: forward)
//   --warmup <N>           warmup iterations          (default: 3, min: 1)
//   --iters <N>            timed iterations           (default: 10)
//   --batch-size <N>       batch size for FPS calc    (default: 1)
//   --pcc-threshold <f>    PCC pass threshold         (default: 0.99)
//   --skip-validation      skip the PCC validation pass
//   --tracy                profiling preset: 1 warmup, 1 timed iteration, no
//                          validation, device-profiler read after every frame.
//                          An explicit --warmup / --iters wins.
//   --blocking-d2h         synchronise on every D2H read (default: enqueue all
//                          reads, wait once per frame)
//   --no-device-sync       do not drain the device after the model call; RUN then
//                          measures host dispatch only and device time lands in D2H
//   --help                 show this message and exit
//
// Phases of one timed frame (each bracketed by a Tracy signpost):
//   H2D  host tensors -> runtime tensors in the layout the program expects
//   RUN  the model call, followed by a device barrier (see emitc_device_sync.cpp)
//        so RUN covers device execution, not just enqueue
//   D2H  outputs back to host, untilized

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "emitc_utils.hpp"
#include "runtime/testutils/testutils.hpp"
#include "tracy/Tracy.hpp"
#include "tt/runtime/runtime.h"
#include "tt/runtime/test/ttnn/dylib.h"

namespace fs = std::filesystem;

namespace
{

// File names written by compile_evo50_emitc.py --flatbuffer
constexpr const char* kActFilename = "evo50_activation_inputs.bin";
constexpr const char* kPersFilename = "evo50_persistent_inputs.bin";
constexpr const char* kGoldenFilename = "evo50_golden_outputs_device.bin";

// =============================================================================
// CLI
// =============================================================================

struct Args
{
    std::string so_path;
    std::string binary_path;
    std::string inputs_dir;
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
    bool tracy_mode = false;
    bool explicit_warmup = false;
    bool explicit_iters = false;
};

void print_usage(const char* prog)
{
    std::fprintf(
        stderr,
        "Usage: %s <so>                --inputs-dir <dir> [options]\n"
        "       %s --flatbuffer <ttnn> --inputs-dir <dir> [options]\n"
        "\n"
        "Inputs:\n"
        "  <so>                   compiled EmitC shared object (.so)\n"
        "  --flatbuffer <path>    benchmark this flatbuffer (.ttnn) instead of an EmitC\n"
        "                         .so, through the identical timing path\n"
        "  --inputs-dir <dir>     directory holding %s,\n"
        "                         %s and %s\n"
        "  --act-bin <path>       activation inputs   (overrides --inputs-dir)\n"
        "  --pers-bin <path>      persistent weights  (overrides --inputs-dir)\n"
        "  --golden-bin <path>    device golden       (overrides --inputs-dir)\n"
        "\n"
        "Options:\n"
        "  --func <name>          entry-point function name  (default: forward)\n"
        "  --warmup <N>           warmup iterations          (default: 3, min 1)\n"
        "  --iters <N>            timed iterations           (default: 10)\n"
        "  --batch-size <N>       batch size for FPS calc    (default: 1)\n"
        "  --pcc-threshold <f>    PCC pass threshold         (default: 0.99)\n"
        "  --skip-validation      skip the PCC validation pass. Use when profiling:\n"
        "                         the validation inference lands in the profile too\n"
        "  --tracy                profiling preset: 1 warmup, 1 timed iteration, no\n"
        "                         validation, device-profiler read after every frame\n"
        "                         (an explicit --warmup / --iters wins)\n"
        "  --blocking-d2h         synchronise on every D2H read  (default: enqueue all\n"
        "                         reads and wait once per frame)\n"
        "  --no-device-sync       skip the device barrier after the model call; RUN\n"
        "                         then measures host dispatch only\n"
        "  --help                 show this message and exit\n",
        prog,
        prog,
        kActFilename,
        kPersFilename,
        kGoldenFilename);
}

Args parse_args(int argc, char* argv[])
{
    Args a;
    bool got_so = false;

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
        else if (arg == "--inputs-dir" && has_value)
            a.inputs_dir = next();
        else if (arg == "--act-bin" && has_value)
            a.act_bin = next();
        else if (arg == "--pers-bin" && has_value)
            a.pers_bin = next();
        else if (arg == "--golden-bin" && has_value)
            a.golden_bin = next();
        else if (arg == "--flatbuffer" && has_value)
            a.binary_path = next();
        else if (arg == "--func" && has_value)
            a.func_name = next();
        else if (arg == "--warmup" && has_value)
        {
            a.n_warmup = std::stoi(next());
            a.explicit_warmup = true;
        }
        else if (arg == "--iters" && has_value)
        {
            a.n_iters = std::stoi(next());
            a.explicit_iters = true;
        }
        else if (arg == "--batch-size" && has_value)
            a.batch_size = std::stoi(next());
        else if (arg == "--pcc-threshold" && has_value)
            a.pcc_threshold = std::stof(next());
        else if (arg == "--skip-validation")
            a.skip_validation = true;
        else if (arg == "--tracy")
            a.tracy_mode = true;
        else if (arg == "--blocking-d2h")
            a.blocking_d2h = true;
        else if (arg == "--no-device-sync")
            a.device_sync = false;
        else if (arg.rfind("--", 0) != 0 && !got_so)
        {
            a.so_path = arg;
            got_so = true;
        }
        else
        {
            std::fprintf(stderr, "error: unknown option or unexpected argument: %s\n\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    auto from_dir = [&](std::string& path, const char* name)
    {
        if (path.empty() && !a.inputs_dir.empty())
            path = (fs::path(a.inputs_dir) / name).string();
    };
    from_dir(a.act_bin, kActFilename);
    from_dir(a.pers_bin, kPersFilename);
    from_dir(a.golden_bin, kGoldenFilename);

    if (a.tracy_mode)
    {
        if (!a.explicit_warmup)
            a.n_warmup = 1;
        if (!a.explicit_iters)
            a.n_iters = 1;
        a.skip_validation = true;
    }

    bool bad = false;
    if (a.so_path.empty() == a.binary_path.empty())
    {
        std::fprintf(stderr, "error: pass exactly one of <so> or --flatbuffer <ttnn>\n");
        bad = true;
    }
    if (a.act_bin.empty() || a.pers_bin.empty())
    {
        std::fprintf(stderr, "error: need --inputs-dir, or both --act-bin and --pers-bin\n");
        bad = true;
    }
    if (!a.skip_validation && a.golden_bin.empty())
    {
        std::fprintf(stderr, "error: validation needs --golden-bin (or --inputs-dir); or pass --skip-validation\n");
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
// Tracy
// =============================================================================

// Same wire format as tools/tracy signpost(): the ops-perf post-processor turns a
// message "`TT_SIGNPOST: <name>`" into a signpost row, sorted among the op rows by
// timestamp, so the report can be sliced between two of them.
void signpost(const char* name)
{
    char msg[128];
    int len = std::snprintf(msg, sizeof(msg), "`TT_SIGNPOST: %s`", name);
    len = std::min(len, static_cast<int>(sizeof(msg)) - 1);
    if (len > 0)
        TracyMessageC(msg, static_cast<std::size_t>(len), 0xf0f8ff);
    (void)msg;
    (void)len;  // both unused when TracyMessageC compiles to nothing (no TRACY_ENABLE)
}

// True when the process runs under the tt-metal device profiler, i.e. was started
// by `python -m tracy` (it sets TT_METAL_DEVICE_PROFILER=1 in the child's env).
bool device_profiler_enabled()
{
    const char* v = std::getenv("TT_METAL_DEVICE_PROFILER");
    return v != nullptr && std::strcmp(v, "0") != 0 && v[0] != '\0';
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

struct Model
{
    void* so_handle = nullptr;
    std::optional<tt::runtime::Binary> binary;
    std::string func_name;
};

RunResult run_one(
    Model& model,
    const std::vector<torch::Tensor>& act_torch,
    const std::vector<tt::runtime::Tensor>& pers_device,
    const std::vector<tt::runtime::Layout>& act_layouts,
    tt::runtime::Device& device,
    const Args& a)
{
    RunResult r{};
    auto t0 = now();

    signpost("emitc_frame_h2d-start");
    auto t_h2d = now();
    std::vector<tt::runtime::Tensor> all_inputs;
    all_inputs.reserve(act_torch.size() + pers_device.size());
    for (std::size_t i = 0; i < act_torch.size(); ++i)
    {
        auto rt = make_runtime_tensor(act_torch[i]);
        all_inputs.push_back(
            tt::runtime::hasLayout(rt, act_layouts[i]) ? rt : tt::runtime::toLayout(rt, device, act_layouts[i]));
    }
    for (const auto& p : pers_device) all_inputs.push_back(p);
    signpost("emitc_frame_h2d-end");
    r.h2d_ms = elapsed_ms(t_h2d);

    signpost("emitc_frame_run-start");
    auto t_run = now();
    std::vector<tt::runtime::Tensor> outputs =
        model.binary.has_value()
            ? tt::runtime::submit(device, *model.binary, /*programIndex=*/0, all_inputs)
            : tt::runtime::test::ttnn::runSoProgram(model.so_handle, model.func_name, all_inputs, device);
    if (a.device_sync)
        device_synchronize(device);
    signpost("emitc_frame_run-end");
    r.run_ms = elapsed_ms(t_run);

    signpost("emitc_frame_d2h-start");
    auto t_d2h = now();
    for (auto& out : outputs)
    {
        auto shards = tt::runtime::toHost(out, /*untilize=*/true, /*blocking=*/a.blocking_d2h);
        assert(!shards.empty());
        r.host_outputs.insert(r.host_outputs.end(), shards.begin(), shards.end());
    }
    if (!a.blocking_d2h)
        tt::runtime::wait(r.host_outputs);
    signpost("emitc_frame_d2h-end");
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

void print_results(
    const Args& a,
    const BenchStats& s,
    const std::vector<double>& h2d,
    const std::vector<double>& run,
    const std::vector<double>& d2h)
{
    std::printf(
        "  Benchmark Results    %d iters  ·  %d warmup  ·  batch %d  ·  %s\n\n",
        a.n_iters,
        a.n_warmup,
        a.batch_size,
        a.binary_path.empty() ? "EmitC .so" : "flatbuffer");
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

// Under the device profiler, read (and so drain) the profiler buffers after every
// frame. Nothing else does it on this path: the flatbuffer runtime reads per
// program, but the EmitC .so never does, so without this the whole run must fit
// in the DRAM profiler buffer and anything past it is dropped. The read sits
// outside the timed windows.
struct Profiling
{
    bool device_profiler = false;
    bool read_after_frame = false;
};

void print_config(const Args& a, const Profiling& prof)
{
    const bool is_so = a.binary_path.empty();
    print_separator('=');
    std::printf("  EVO50 EmitC / Flatbuffer Benchmark (profile)\n");
    print_separator('=');
    std::printf(
        "  %-18s  %s\n", is_so ? "shared object" : "flatbuffer", is_so ? a.so_path.c_str() : a.binary_path.c_str());
    std::printf("  %-18s  %s\n", "activation inputs", a.act_bin.c_str());
    std::printf("  %-18s  %s\n", "persistent inputs", a.pers_bin.c_str());
    std::printf("  %-18s  %s\n", "device golden", a.skip_validation ? "(validation skipped)" : a.golden_bin.c_str());
    std::printf("  %-18s  %s\n", "function", a.func_name.c_str());
    std::printf("  %-18s  %d\n", "warmup iters", a.n_warmup);
    std::printf("  %-18s  %d\n", "timed iters", a.n_iters);
    std::printf("  %-18s  %d\n", "batch size", a.batch_size);
    std::printf("  %-18s  %.3f\n", "pcc threshold", a.pcc_threshold);
    std::printf("  %-18s  %s\n", "D2H", a.blocking_d2h ? "blocking per output" : "enqueue all, wait once");
    std::printf("  %-18s  %s\n", "device sync", a.device_sync ? "after model call" : "off");
    std::printf(
        "  %-18s  %s%s\n",
        "device profiler",
        prof.device_profiler ? "ON" : "off",
        prof.read_after_frame ? " (read after every frame)" : "");
    std::printf("  %-18s  %s\n", "tracy client", TracyIsConnected ? "connected" : "not connected");
    print_separator('=');
}

// Phase 1: the .so or flatbuffer binary.
Model load_model(const Args& a)
{
    Model model;
    model.func_name = a.func_name;
    if (a.binary_path.empty())
    {
        model.so_handle = tt::runtime::testutils::open_so(a.so_path);
        std::printf("  shared object loaded\n");
    }
    else
    {
        model.binary = tt::runtime::Binary::loadFromPath(a.binary_path.c_str());
        tt::runtime::setCompatibleDeviceRuntime(*model.binary);
        std::printf("  flatbuffer binary loaded\n");
    }
    return model;
}

struct DeviceInputs
{
    std::vector<tt::runtime::Layout> act_layouts;  // layouts the program expects for the activations
    std::vector<tt::runtime::Tensor> pers_device;  // weights, on device for the whole run
};

// Phase 2: query the program's input layouts and move the weights to the device.
// Program input order is activations first, then persistent tensors.
DeviceInputs prepare_inputs(
    const Model& model,
    const Args& a,
    tt::runtime::Device& device,
    std::size_t n_act,
    const std::vector<torch::Tensor>& pers_torch)
{
    const std::size_t n_pers = pers_torch.size();
    std::vector<tt::runtime::Layout> layouts;
    if (model.binary.has_value())
    {
        for (std::size_t i = 0; i < n_act + n_pers; ++i)
            layouts.push_back(tt::runtime::getLayout(*model.binary, /*programIndex=*/0, static_cast<std::uint32_t>(i)));
    }
    else
    {
        auto templates = tt::runtime::test::ttnn::createInputs(model.so_handle, a.func_name, device, a.so_path);
        if (templates.size() != n_act + n_pers)
            throw std::runtime_error(
                "input count mismatch: " + a.so_path + " expects " + std::to_string(templates.size()) +
                " inputs, the .bin files hold " + std::to_string(n_act + n_pers) +
                " (were they produced by the same compiler config?)");
        for (const auto& t : templates) layouts.push_back(tt::runtime::getTensorLayout(t));
    }
    std::printf("  %zu layout(s) queried  (%zu activation + %zu persistent)\n", layouts.size(), n_act, n_pers);

    DeviceInputs in;
    in.act_layouts.assign(layouts.begin(), layouts.begin() + static_cast<std::ptrdiff_t>(n_act));

    // retain=true: a flatbuffer program frees its inputs as it runs, and the weights
    // must outlive every iteration.
    in.pers_device.reserve(n_pers);
    for (std::size_t i = 0; i < n_pers; ++i)
        in.pers_device.push_back(
            tt::runtime::toLayout(make_runtime_tensor(pers_torch[i]), device, layouts[n_act + i], /*retain=*/true));
    std::printf("  %zu persistent tensor(s) on device\n", in.pers_device.size());
    return in;
}

// Phase 3: one frame against the device golden. On a traced model this first call
// is the trace capture and the warmup iterations that follow are replays.
template <typename RunFrame, typename Drain>
bool run_validation(const Args& a, RunFrame&& run_frame, Drain&& drain_profiler)
{
    if (a.skip_validation)
    {
        std::printf("  Validation skipped%s.\n", a.tracy_mode ? " (--tracy)" : " (--skip-validation)");
        return true;
    }
    auto golden = load_tensor_list(a.golden_bin);
    std::printf("  Running the frame once against %zu golden output(s) ...\n", golden.size());
    auto r = run_frame();
    drain_profiler();
    const bool pass = validate_outputs(r.host_outputs, golden, a.pcc_threshold);
    std::printf("\n  Validation: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

struct TimedRuns
{
    std::vector<double> infer_ms, h2d_ms, run_ms, d2h_ms;
};

// Phase 4: warmup, then the timed frames, each bracketed by Tracy signposts. The
// profiler read follows the end signpost, so it stays out of the sliced window.
template <typename RunFrame, typename Drain>
TimedRuns run_benchmark(const Args& a, RunFrame&& run_frame, Drain&& drain_profiler)
{
    std::printf("  Warmup  (%d iteration(s)%s)\n", a.n_warmup, a.skip_validation ? ", iter 1 = trace capture" : "");
    double warm_sum = 0.0;
    for (int i = 0; i < a.n_warmup; ++i)
    {
        auto r = run_frame();
        drain_profiler();
        warm_sum += r.infer_ms;
        print_progress("warmup", i + 1, a.n_warmup, r.infer_ms, warm_sum / (i + 1));
    }

    std::printf("\n  Timed run  (%d iteration(s))\n", a.n_iters);
    TimedRuns t;
    double running = 0.0;
    for (int i = 0; i < a.n_iters; ++i)
    {
        signpost("emitc_benchmark_iter-start");
        auto r = run_frame();
        signpost("emitc_benchmark_iter-end");
        drain_profiler();
        t.infer_ms.push_back(r.infer_ms);
        t.h2d_ms.push_back(r.h2d_ms);
        t.run_ms.push_back(r.run_ms);
        t.d2h_ms.push_back(r.d2h_ms);
        running += r.infer_ms;
        print_progress("timed", i + 1, a.n_iters, r.infer_ms, running / (i + 1));
    }
    return t;
}

// Closing the device is what writes the device profiler's final data; do it
// whenever profiling, since main's _Exit skips TTSystem's destructor.
void shutdown(Model& model, const Profiling& prof)
{
    std::fflush(stdout);
    if (model.so_handle != nullptr)
        tt::runtime::testutils::close_so(model.so_handle);

    if (prof.read_after_frame || TracyIsConnected)
    {
        std::printf("Profiling: closing device to flush the device profiler ...\n");
        std::fflush(stdout);
        close_devices();
        // Give Tracy's worker thread time to ship its queue before _Exit.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

}  // namespace

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[])
{
    const Args a = parse_args(argc, argv);
    Profiling prof;
    prof.device_profiler = device_profiler_enabled();
    prof.read_after_frame = prof.device_profiler || a.tracy_mode;

    print_config(a, prof);
    constexpr int kPhases = 5;

    print_phase(1, kPhases, "Loading inputs & opening device");
    auto act_torch = load_tensor_list(a.act_bin);
    auto pers_torch = load_tensor_list(a.pers_bin);
    std::printf("  activation  %zu tensor(s)  [%s]\n", act_torch.size(), a.act_bin.c_str());
    std::printf("  persistent  %zu tensor(s)  [%s]\n", pers_torch.size(), a.pers_bin.c_str());
    tt::runtime::Device device = open_device(/*enable_program_cache=*/true);
    std::printf("  device opened  (program cache enabled)\n");
    Model model = load_model(a);

    print_phase(2, kPhases, "Preparing device inputs");
    const DeviceInputs in = prepare_inputs(model, a, device, act_torch.size(), pers_torch);

    // One frame (H2D + RUN + D2H), and the per-frame profiler read when profiling.
    auto run_frame = [&]() { return run_one(model, act_torch, in.pers_device, in.act_layouts, device, a); };
    auto drain_profiler = [&]()
    {
        if (prof.read_after_frame)
            tt::runtime::readDeviceProfilerResults(device);
    };

    print_phase(3, kPhases, "Validation");
    const bool val_pass = run_validation(a, run_frame, drain_profiler);

    print_phase(4, kPhases, "Benchmark");
    const TimedRuns t = run_benchmark(a, run_frame, drain_profiler);

    print_phase(5, kPhases, "Results");
    print_results(a, compute_stats(t.infer_ms, a.batch_size), t.h2d_ms, t.run_ms, t.d2h_ms);

    shutdown(model, prof);
    std::printf("\nDone.\n");
    std::fflush(stdout);
    // _Exit skips atexit / static-destructor cleanup (SIGSEGV in _dl_fini otherwise).
    // A skipped validation still exits 0 so tracy post-processes the profile.
    std::_Exit(val_pass ? 0 : 1);
}
