// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark runner for both pipelines, with Tracy profiling support.
//
// Runs either an EmitC shared object or a flatbuffer binary through the same
// timing path, so the two can be compared directly, and records Tracy signposts
// around the iterations and the input / model / readback phases so a profile
// can be scoped to the timed region. For an EmitC-only benchmark without any
// Tracy dependency use model_benchmark.
//
// Discovers every subdirectory of --seq-dir that contains:
//   bev_activation_inputs.bin   activation inputs  (TTNN binary format)
//   bev_device_outputs.bin      golden outputs     (TTNN binary format)
//
// Persistent weights (shared across sequences) are loaded from --pers-bin,
// which defaults to <seq-dir>/bev_persistent_inputs.bin.
//
// Usage:
//   model_benchmark_profile <so> --seq-dir <dir> [--pers-bin <path>] [options]
//   model_benchmark_profile --flatbuffer <ttnn> --seq-dir <dir> [options]
//
// Options:
//   --pers-bin <path>      persistent weights  (default: <seq-dir>/bev_persistent_inputs.bin)
//   --func <name>          entry-point function name  (default: forward)
//   --warmup <N>           warmup iterations          (default: 3, min: 1)
//   --iters <N>            timed iterations           (default: 10)
//   --batch-size <N>       batch size for FPS calc    (default: 1)
//   --pcc-threshold <f>    PCC pass threshold         (default: 0.99)
//   --flatbuffer <path>    benchmark a flatbuffer binary instead of an EmitC .so
//   --skip-validation      skip the per-sequence PCC validation pass
//   --tracy                profiling preset: 1 warmup, 1 timed iteration and no
//                          validation, so the capture holds a single signposted
//                          frame. An explicit --warmup / --iters wins.
//   --blocking-d2h         synchronise on every D2H read (default: enqueue all, wait once per frame)
//   --help                 show this message and exit

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>


#include "tracy/Tracy.hpp"

#include "emitc_utils.hpp"
#include "runtime/testutils/testutils.hpp"
#include "tt/runtime/runtime.h"
#include "tt/runtime/test/ttnn/dylib.h"

namespace fs = std::filesystem;

// =============================================================================
// CLI
// =============================================================================

struct Args {
    std::string so_path;
    std::string binary_path;
    std::string seq_dir;
    std::string pers_bin;
    std::string func_name     = "forward";
    int         n_warmup      = 3;
    int         n_iters       = 10;
    int         batch_size    = 1;
    float       pcc_threshold = 0.99f;
    bool        skip_validation = false;
    bool        blocking_d2h  = false;
    bool        tracy_mode    = false;
    bool        explicit_warmup = false;
    bool        explicit_iters  = false;
};

static void print_usage(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s <so> --seq-dir <dir> [--pers-bin <path>] [options]\n"
        "\n"
        "Required:\n"
        "  <so>               compiled EmitC shared object (.so)\n"
        "  --seq-dir <dir>    directory whose subdirectories each contain:\n"
        "                       %-36s activation inputs\n"
        "                       %-36s golden outputs\n"
        "\n"
        "Options:\n"
        "  --pers-bin <path>      persistent weights\n"
        "                         (default: <seq-dir>/%s)\n"
        "  --func <name>          entry-point function name  (default: forward)\n"
        "  --warmup <N>           warmup iterations          (default: 3)\n"
        "  --iters <N>            timed iterations           (default: 10)\n"
        "  --batch-size <N>       batch size for FPS calc    (default: 1)\n"
        "  --pcc-threshold <f>    PCC pass threshold         (default: 0.99)\n"
        "  --flatbuffer <path>    benchmark this flatbuffer (.ttnn) binary instead\n"
        "                         of an EmitC .so, through the identical timing\n"
        "                         path. Makes the two pipelines comparable.\n"
        "  --skip-validation      skip the per-sequence PCC validation pass.\n"
        "                         Use when profiling: validation runs the model once\n"
        "                         per sequence and those extra inferences land in the\n"
        "                         profile alongside the timed ones.\n"
        "  --tracy                profiling preset: 1 warmup, 1 timed iteration, no\n"
        "                         validation (an explicit --warmup / --iters wins)\n"
        "  --blocking-d2h         synchronise on every D2H read  (default: enqueue all\n"
        "                         reads and wait once per frame)\n"
        "  --help                 show this message and exit\n",
        prog, kActFilename, kGoldenFilename, kPersDefault);
}

static Args parse_args(int argc, char* argv[])
{
    Args a;
    bool got_so = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]); std::exit(0);
        } else if (arg == "--seq-dir"       && i + 1 < argc) { a.seq_dir       = argv[++i];
        } else if (arg == "--pers-bin"      && i + 1 < argc) { a.pers_bin      = argv[++i];
        } else if (arg == "--func"          && i + 1 < argc) { a.func_name     = argv[++i];
        } else if (arg == "--warmup"        && i + 1 < argc) { { a.n_warmup = std::stoi(argv[++i]); a.explicit_warmup = true; }
        } else if (arg == "--iters"         && i + 1 < argc) { { a.n_iters = std::stoi(argv[++i]); a.explicit_iters = true; }
        } else if (arg == "--batch-size"    && i + 1 < argc) { a.batch_size    = std::stoi(argv[++i]);
        } else if (arg == "--pcc-threshold" && i + 1 < argc) { a.pcc_threshold = std::stof(argv[++i]);
        } else if (arg == "--flatbuffer"    && i + 1 < argc) { a.binary_path   = argv[++i];
        } else if (arg == "--skip-validation") { a.skip_validation = true;
        } else if (arg == "--tracy") { a.tracy_mode = true;
        } else if (arg == "--blocking-d2h") { a.blocking_d2h = true;
        } else if (arg.rfind("--", 0) != 0 && !got_so) {
            a.so_path = arg; got_so = true;
        } else {
            std::fprintf(stderr, "error: unknown option or unexpected argument: %s\n\n", arg.c_str());
            print_usage(argv[0]); std::exit(1);
        }
    }

    bool missing = false;
    if (a.so_path.empty() && a.binary_path.empty()) {
        std::fprintf(stderr, "error: pass either <so> or --flatbuffer <binary>\n");
        missing = true;
    }
    if (!a.so_path.empty() && !a.binary_path.empty()) {
        std::fprintf(stderr, "error: <so> and --flatbuffer are mutually exclusive\n");
        missing = true;
    }
    if (a.seq_dir.empty())  { std::fprintf(stderr, "error: missing required option --seq-dir\n"); missing = true; }
    if (missing) { std::fprintf(stderr, "\n"); print_usage(argv[0]); std::exit(1); }

    if (a.tracy_mode) {
        if (!a.explicit_warmup) a.n_warmup = 1;
        if (!a.explicit_iters)  a.n_iters  = 1;
        a.skip_validation = true;
    }

    if (a.pers_bin.empty())
        a.pers_bin = (fs::path(a.seq_dir) / kPersDefault).string();

    if (a.n_warmup < 1)   { std::fprintf(stderr, "warning: --warmup clamped to 1\n");     a.n_warmup   = 1; }
    if (a.n_iters   < 1)  { std::fprintf(stderr, "warning: --iters clamped to 1\n");      a.n_iters    = 1; }
    if (a.batch_size < 1) { std::fprintf(stderr, "warning: --batch-size clamped to 1\n"); a.batch_size = 1; }

    return a;
}

// =============================================================================
// Single timed inference pass (H2D + kernel + D2H)
// =============================================================================

static void signpost(const std::string& header);

static RunResult run_one(
    void*                                   so_handle,
    const std::optional<tt::runtime::Binary>& binary,
    const std::string&                      func_name,
    const std::vector<torch::Tensor>&       act_torch,
    const std::vector<tt::runtime::Tensor>& pers_device,
    const std::vector<tt::runtime::Layout>& act_layouts,
    tt::runtime::Device&                    device,
    bool                                    blocking_d2h)
{
    RunResult r{};
    auto t0 = now();

    signpost("emitc_frame_h2d-start");
    auto t_h2d = now();
    std::vector<tt::runtime::Tensor> all_inputs;
    all_inputs.reserve(act_torch.size() + pers_device.size());
    for (std::size_t i = 0; i < act_torch.size(); ++i) {
        auto rt = make_runtime_tensor(act_torch[i]);
        all_inputs.push_back(tt::runtime::hasLayout(rt, act_layouts[i])
                                 ? rt
                                 : tt::runtime::toLayout(rt, device, act_layouts[i]));
    }
    for (const auto& p : pers_device)
        all_inputs.push_back(p);
    signpost("emitc_frame_h2d-end");
    r.h2d_ms = elapsed_ms(t_h2d);

    signpost("emitc_frame_run-start");
    auto t_run = now();
    auto outputs = binary.has_value()
        ? tt::runtime::submit(device, *binary, /*programIndex=*/0, all_inputs)
        : tt::runtime::test::ttnn::runSoProgram(so_handle, func_name, all_inputs, device);
    // drain the device so RUN covers device execution
    tt::runtime::deviceSynchronize(device);
    signpost("emitc_frame_run-end");
    r.run_ms = elapsed_ms(t_run);

    signpost("emitc_frame_d2h-start");
    auto t_d2h = now();
    for (auto& out : outputs) {
        auto shards = tt::runtime::toHost(out, /*untilize=*/true, blocking_d2h);
        assert(!shards.empty());
        r.host_outputs.insert(r.host_outputs.end(), shards.begin(), shards.end());
    }
    if (!blocking_d2h)
        tt::runtime::wait(r.host_outputs);
    signpost("emitc_frame_d2h-end");
    r.d2h_ms = elapsed_ms(t_d2h);

    r.infer_ms = elapsed_ms(t0);
    return r;
}

// =============================================================================
// main
// =============================================================================

// Tracy signpost in the form tools/tracy emits, so the ops report can be scoped to it.
static void signpost(const std::string& header)
{
    const std::string msg = "`TT_SIGNPOST: " + header + "`";
    TracyMessageC(msg.c_str(), msg.size(), 0xf0f8ff);
}

int main(int argc, char* argv[])
{
    Args a = parse_args(argc, argv);

    // ------------------------------------------------------------------
    print_separator('=');
    std::printf("  EmitC / Flatbuffer Model Benchmark (profile)\n");
    print_separator('=');
    std::printf("  %-18s  %s\n",   "shared object",  a.so_path.c_str());
    std::printf("  %-18s  %s\n",   "seq-dir",        a.seq_dir.c_str());
    std::printf("  %-18s  %s\n",   "pers-bin",       a.pers_bin.c_str());
    std::printf("  %-18s  %s\n",   "function",       a.func_name.c_str());
    std::printf("  %-18s  %d\n",   "warmup iters",   a.n_warmup);
    std::printf("  %-18s  %d\n",   "timed iters",    a.n_iters);
    std::printf("  %-18s  %d\n",   "batch size",     a.batch_size);
    std::printf("  %-18s  %.3f\n", "pcc threshold",  a.pcc_threshold);
    print_separator('=');

    constexpr int kPhases = 6;

    // ------------------------------------------------------------------
    print_phase(1, kPhases, "Discovering sequences");

    auto seq_ids = discover_sequences(a.seq_dir);
    if (seq_ids.empty()) {
        std::fprintf(stderr,
            "error: no sequences found under '%s'\n"
            "       (expected subdirectories containing %s and %s)\n",
            a.seq_dir.c_str(), kActFilename, kGoldenFilename);
        std::_Exit(1);
    }
    std::printf("  %zu sequence(s):\n", seq_ids.size());
    for (auto& id : seq_ids)
        std::printf("    %s\n", id.c_str());

    // ------------------------------------------------------------------
    print_phase(2, kPhases, "Loading inputs & opening device");

    auto pers_torch = load_tensor_list(a.pers_bin);
    std::printf("  persistent  %zu tensor(s)  [%s]\n",
                pers_torch.size(), a.pers_bin.c_str());

    std::vector<std::vector<torch::Tensor>> all_act(seq_ids.size());
    for (std::size_t si = 0; si < seq_ids.size(); ++si) {
        all_act[si] = load_tensor_list(
            (fs::path(a.seq_dir) / seq_ids[si] / kActFilename).string());
        std::printf("  [%s]  %zu activation tensor(s)\n",
                    seq_ids[si].c_str(), all_act[si].size());
    }

    tt::runtime::Device device = open_device(/*enable_program_cache=*/true);
    std::printf("  device opened  (program cache enabled)\n");

    void*                              so_handle = nullptr;
    std::optional<tt::runtime::Binary> binary;
    if (a.binary_path.empty()) {
        so_handle = tt::runtime::testutils::open_so(a.so_path);
        std::printf("  shared object loaded\n");
    } else {
        binary = tt::runtime::Binary::loadFromPath(a.binary_path.c_str());
        tt::runtime::setCompatibleDeviceRuntime(*binary);
        std::printf("  flatbuffer binary loaded\n");
    }

    // ------------------------------------------------------------------
    print_phase(3, kPhases, "Preparing device inputs");

    std::vector<tt::runtime::Tensor> layout_templates;
    std::vector<tt::runtime::Layout> binary_layouts;
    std::size_t                      n_inputs = 0;
    if (binary.has_value()) {
        std::size_t n_act = all_act[0].size(), n_pers = pers_torch.size();
        n_inputs = n_act + n_pers;
        binary_layouts.reserve(n_inputs);
        for (std::size_t i = 0; i < n_inputs; ++i)
            binary_layouts.push_back(tt::runtime::getLayout(*binary, /*programIndex=*/0, i));
    } else {
        layout_templates =
            tt::runtime::test::ttnn::createInputs(so_handle, a.func_name, device, a.so_path);
        n_inputs = layout_templates.size();
    }
    auto layout_at = [&](std::size_t i) {
        return binary.has_value() ? binary_layouts[i]
                                  : tt::runtime::getTensorLayout(layout_templates[i]);
    };

    std::size_t total_expected = all_act[0].size() + pers_torch.size();
    if (n_inputs != total_expected)
        throw std::runtime_error(
            "input count mismatch: executable expects " + std::to_string(n_inputs) +
            ", got " + std::to_string(total_expected));

    std::printf("  %zu layout(s) queried  (%zu activation + %zu persistent)\n",
                n_inputs, all_act[0].size(), pers_torch.size());

    std::vector<tt::runtime::Layout> act_layouts;
    act_layouts.reserve(all_act[0].size());
    for (std::size_t i = 0; i < all_act[0].size(); ++i)
        act_layouts.push_back(layout_at(i));

    // retain=true: a flatbuffer program frees its inputs as it runs, the weights must outlive it.
    std::vector<tt::runtime::Tensor> pers_device;
    pers_device.reserve(pers_torch.size());
    for (std::size_t i = 0; i < pers_torch.size(); ++i) {
        auto layout = layout_at(all_act[0].size() + i);
        pers_device.push_back(tt::runtime::toLayout(
            make_runtime_tensor(pers_torch[i]), device, layout, /*retain=*/true));
    }
    std::printf("  %zu persistent tensor(s) on device\n", pers_device.size());

    std::size_t n_seqs = seq_ids.size();

    // ------------------------------------------------------------------
    print_phase(4, kPhases, "Validation");

    auto run_seq = [&](std::size_t si) {
        return run_one(so_handle, binary, a.func_name, all_act[si], pers_device, act_layouts, device, a.blocking_d2h);
    };

    std::printf("  Running each sequence once for PCC validation ...\n");

    std::vector<bool> val_results;
    val_results.reserve(n_seqs);

    for (std::size_t si = 0; a.skip_validation ? false : si < n_seqs; ++si) {
        const auto& seq_id = seq_ids[si];
        auto golden_torch  = load_tensor_list((fs::path(a.seq_dir) / seq_id / kGoldenFilename).string());

        std::printf("\n  [%zu/%zu]  %s\n", si + 1, n_seqs, seq_id.c_str());
        auto r = run_seq(si);
        val_results.push_back(validate_outputs(r.host_outputs, golden_torch, a.pcc_threshold, seq_id));
    }

    if (a.skip_validation)
        std::printf("\n  Validation skipped (--skip-validation).\n");
    else
        print_validation_summary(seq_ids, val_results, a.pcc_threshold);

    // ------------------------------------------------------------------
    print_phase(5, kPhases, "Benchmark  (cycling through sequences)");

    std::printf("  Warmup  (%d iteration(s), iter 1 = trace capture)\n", a.n_warmup);
    double warm_sum = 0.0;
    for (int i = 0; i < a.n_warmup; ++i) {
        std::size_t si = static_cast<std::size_t>(i) % n_seqs;
        auto r         = run_seq(si);
        warm_sum      += r.infer_ms;
        print_progress("warmup", i + 1, a.n_warmup, si, n_seqs, r.infer_ms, warm_sum / (i + 1));
    }

    std::printf("\n  Timed run  (%d iteration(s))\n", a.n_iters);

    std::vector<double> infer_ms(a.n_iters);
    std::vector<double> h2d_ms(a.n_iters), run_ms(a.n_iters), d2h_ms(a.n_iters);
    double running_sum = 0.0;

    for (int i = 0; i < a.n_iters; ++i) {
        std::size_t si  = static_cast<std::size_t>(i) % n_seqs;
        signpost("emitc_benchmark_iter-start");
        auto r = run_seq(si);
        signpost("emitc_benchmark_iter-end");
        infer_ms[i]     = r.infer_ms;
        h2d_ms[i]       = r.h2d_ms;
        run_ms[i]       = r.run_ms;
        d2h_ms[i]       = r.d2h_ms;
        running_sum    += r.infer_ms;
        print_progress("timed", i + 1, a.n_iters, si, n_seqs, r.infer_ms, running_sum / (i + 1));
    }

    // ------------------------------------------------------------------
    print_phase(6, kPhases, "Results");

    auto _stats = compute_stats(infer_ms, a.batch_size);
    print_results_table(a.n_iters, a.n_warmup, a.batch_size, _stats, n_seqs);
    print_phase_breakdown(h2d_ms, run_ms, d2h_ms, _stats.mean_ms);

    int n_pass = 0;
    for (bool r : val_results) if (r) ++n_pass;
    // --skip-validation must still exit 0 so tracy post-processes the profile.
    bool all_pass = a.skip_validation || (n_pass == static_cast<int>(n_seqs));

    std::cout.flush();
    std::fflush(stdout);
    if (so_handle != nullptr) {
        tt::runtime::testutils::close_so(so_handle);
    }
    std::printf("\nDone.\n");
    std::fflush(stdout);

#if defined(TRACY_ENABLE)
    if (TracyIsConnected) {
        // close the device so the device profiler is flushed before _Exit.
        std::printf("Profiling: closing device to flush the device profiler ...\n");
        std::fflush(stdout);
        close_devices();

        // give Tracy's worker thread time to ship its queue before _Exit.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
#endif

    // _Exit skips atexit / static-dtor cleanup to avoid a SIGSEGV in _dl_fini.
    std::_Exit(all_pass ? 0 : 1);
}
