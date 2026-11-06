// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// EmitC benchmark runner — cycles through all sequences under a directory.
//
// Discovers every subdirectory of --seq-dir that contains:
//   bev_activation_inputs.bin   activation inputs  (TTNN binary format)
//   bev_device_outputs.bin      golden outputs     (TTNN binary format)
//
// Persistent weights (shared across sequences) are loaded from --pers-bin,
// which defaults to <seq-dir>/bev_persistent_inputs.bin.
//
// Each sequence is validated against its golden outputs before benchmarking.
// Warmup and timed iterations then cycle across all discovered sequences.
//
// Usage:
//   model_benchmark <so> --seq-dir <dir> [--pers-bin <path>] [options]
//
// Required:
//   <so>               compiled EmitC shared object
//   --seq-dir <dir>    directory containing per-sequence subdirectories
//
// Options:
//   --pers-bin <path>      persistent weights  (default: <seq-dir>/bev_persistent_inputs.bin)
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
// Directory conventions (produced by compile_bev_emitc.py --flatbuffer)
// =============================================================================

static constexpr const char* kActFilename    = "bev_activation_inputs.bin";
static constexpr const char* kGoldenFilename = "bev_device_outputs.bin";
static constexpr const char* kPersDefault    = "bev_persistent_inputs.bin";

// =============================================================================
// CLI
// =============================================================================

struct Args {
    std::string so_path;
    // Benchmark a flatbuffer binary instead of an EmitC .so, through this same
    // runner. Without it the two pipelines can only be compared across
    // different harnesses -- this runner versus the pytest benchmark -- which
    // time different envelopes, so any difference is unattributable.
    std::string binary_path;
    std::string seq_dir;
    std::string pers_bin;
    std::string func_name     = "forward";
    int         n_warmup      = 3;
    int         n_iters       = 10;
    int         batch_size    = 1;
    float       pcc_threshold = 0.99f;
    bool        skip_validation = false;
    // D2H readback mode. Blocking (the default, and toHost's own) synchronises on
    // every output; non-blocking enqueues all reads and waits once per frame. Kept
    // as a runtime flag so both can be measured with the identical binary.
    bool        blocking_d2h  = true;
    bool        no_color      = false;
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
        "  --nonblocking-d2h      enqueue all D2H reads non-blocking and wait once\n"
        "                         per frame, instead of synchronising on each output\n"
        "  --no-color             disable ANSI color output\n"
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
        } else if (arg == "--warmup"        && i + 1 < argc) { a.n_warmup      = std::stoi(argv[++i]);
        } else if (arg == "--iters"         && i + 1 < argc) { a.n_iters       = std::stoi(argv[++i]);
        } else if (arg == "--batch-size"    && i + 1 < argc) { a.batch_size    = std::stoi(argv[++i]);
        } else if (arg == "--pcc-threshold" && i + 1 < argc) { a.pcc_threshold = std::stof(argv[++i]);
        } else if (arg == "--flatbuffer"    && i + 1 < argc) { a.binary_path   = argv[++i];
        } else if (arg == "--skip-validation") { a.skip_validation = true;
        } else if (arg == "--nonblocking-d2h") { a.blocking_d2h = false;
        } else if (arg == "--no-color") { a.no_color = true;
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

    if (a.pers_bin.empty())
        a.pers_bin = (fs::path(a.seq_dir) / kPersDefault).string();

    if (a.n_warmup < 1)   { std::fprintf(stderr, "warning: --warmup clamped to 1\n");     a.n_warmup   = 1; }
    if (a.n_iters   < 1)  { std::fprintf(stderr, "warning: --iters clamped to 1\n");      a.n_iters    = 1; }
    if (a.batch_size < 1) { std::fprintf(stderr, "warning: --batch-size clamped to 1\n"); a.batch_size = 1; }

    return a;
}

// =============================================================================
// Sequence discovery
// =============================================================================

static std::vector<std::string> discover_sequences(const std::string& seq_dir)
{
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

// =============================================================================
// Terminal UI
// =============================================================================

static void print_progress(const char* label, int done, int total,
                           const std::string& seq_id,
                           double last_ms = -1.0, double avg_ms = -1.0)
{
    constexpr int BAR = 24;
    int filled = (total > 0) ? (done * BAR / total) : BAR;

    std::printf("\r  %-8s  [%s%s%s%s]  %d/%d  seq=%.20s",
                label,
                color::green(),
                std::string(filled, '#').c_str(),
                color::reset(),
                std::string(BAR - filled, '-').c_str(),
                done, total, seq_id.c_str());

    if (last_ms >= 0.0) std::printf("  %6.2f ms", last_ms);
    if (avg_ms  >= 0.0) std::printf("  (avg %6.2f ms)", avg_ms);
    if (done == total)  std::printf("\n");
    std::fflush(stdout);
}

static void print_results_table(const Args& a, const BenchStats& s,
                                 std::size_t n_seqs)
{
    print_separator('=');
    std::printf("  %sBenchmark Results%s"
                "     %d iters  ·  %d warmup  ·  batch %d  ·  %zu sequences\n",
                color::bold(), color::reset(),
                a.n_iters, a.n_warmup, a.batch_size, n_seqs);
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
// Single timed inference pass (H2D + kernel + D2H)
// =============================================================================

struct RunResult {
    std::vector<tt::runtime::Tensor> host_outputs;
    double infer_ms;
    // Phase split, so a frame can be attributed to input staging vs the model
    // call vs output readback. Needed to compare against harnesses that time a
    // different envelope -- the pytest benchmark, for instance, times only the
    // model call and reports input assembly and output realisation separately.
    double h2d_ms = 0.0;
    double run_ms = 0.0;
    double d2h_ms = 0.0;
};

// Defined below, next to the Tracy include commentary.
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

    // Phase signposts, so a profile can attribute the frame to input staging vs
    // the model call vs the readback. Worth having: on BEV block A the frame
    // looked device-bound until these showed a 17 ms host prologue before the
    // traced graph was even enqueued.
    signpost("emitc_frame_h2d-start");
    auto t_h2d = now();
    std::vector<tt::runtime::Tensor> all_inputs;
    all_inputs.reserve(act_torch.size() + pers_device.size());
    for (std::size_t i = 0; i < act_torch.size(); ++i) {
        // Skip the layout conversion when the tensor already matches, as forge's
        // Tensor::to_layout does. Without the check every activation is
        // converted on every frame even when nothing changed.
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
    // Identical everything else -- input staging above, readback below, and the
    // timed region -- so the only difference between the two pipelines is which
    // executable runs the graph.
    auto outputs = binary.has_value()
        ? tt::runtime::submit(device, *binary, /*programIndex=*/0, all_inputs)
        : tt::runtime::test::ttnn::runSoProgram(so_handle, func_name, all_inputs, device);
    // Drain the device before stopping the RUN timer. submit()/runSoProgram()
    // only enqueue work, so without this barrier the wait for device execution
    // lands in whichever call blocks next -- that is the D2H readback, which
    // then reports ~60 ms for 918 KB and hides where the time actually goes.
    // With the barrier, RUN covers dispatch plus device execution and D2H
    // covers only the untilize and the transfer.
    tt::runtime::deviceSynchronize(device);
    signpost("emitc_frame_run-end");
    r.run_ms = elapsed_ms(t_run);

    // Blocking reads synchronise on each output; non-blocking enqueues them all and
    // waits once. Either way the timed region below covers the full H2D + run + D2H,
    // because the non-blocking path still waits before leaving this scope.
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
// Validation summary
// =============================================================================

static void print_validation_summary(
    const std::vector<std::string>& seq_ids,
    const std::vector<bool>&        results,
    float                           pcc_threshold)
{
    int n_pass = 0;
    for (bool r : results) if (r) ++n_pass;

    print_separator('=');
    std::printf("  %sValidation Summary%s"
                "  (%zu sequence(s)  ·  PCC threshold %.3f)\n",
                color::bold(), color::reset(),
                seq_ids.size(), pcc_threshold);
    print_separator('=');

    constexpr int kSeqCol = 46;
    std::printf("  %-*s  %s\n", kSeqCol, "Sequence", "Result");
    print_separator('-');
    for (std::size_t i = 0; i < seq_ids.size(); ++i)
        std::printf("  %-*s  %s\n", kSeqCol, seq_ids[i].c_str(),
                    results[i] ? color::pass_tag() : color::fail_tag());
    print_separator('-');
    std::printf("  %d/%zu sequences PASS\n", n_pass, seq_ids.size());
    print_separator('=');
}

// =============================================================================
// main
// =============================================================================

// Tracy signpost in the exact form tools/tracy emits, so that
// process_ops_logs.py recognises it (it keys on "TT_SIGNPOST" and splits on
// ": ") and the ops report can be scoped to the timed region with
// --start-signpost / --end-signpost. The backtick wrapping matters: it is the
// quotechar tracy-csvexport uses for the message column.
//
// TracyMessageC is what op_profiler::tracy_message expands to, so this records
// the same MessageSourceType::User / MessageSeverity::Info message the
// Python-side signposts do. Without TRACY_ENABLE the macro is a no-op, which
// is the right behaviour for a non-profiling build.
static void signpost(const std::string& header)
{
    const std::string msg = "`TT_SIGNPOST: " + header + "`";
    TracyMessageC(msg.c_str(), msg.size(), 0xf0f8ff);
}

static double mean_of(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// Where the frame actually went.
//
// The timed region here spans input staging, the model call and output
// readback. Other harnesses time a narrower envelope -- the pytest benchmark
// times only the model call, reporting input assembly and output realisation as
// separate lines -- so two correct numbers can differ by milliseconds while
// measuring the same binary. Printing the split makes that visible instead of
// leaving it to be rediscovered.
static void print_phase_breakdown(const std::vector<double>& h2d,
                                  const std::vector<double>& run,
                                  const std::vector<double>& d2h,
                                  double total_ms)
{
    const double mh = mean_of(h2d), mr = mean_of(run), md = mean_of(d2h);
    const auto pct = [total_ms](double x) { return total_ms > 0.0 ? 100.0 * x / total_ms : 0.0; };
    print_separator('-');
    std::printf("  %sPer-frame breakdown%s\n", color::bold(), color::reset());
    std::printf("  %-32s  %8.2f ms  (%4.1f%%)\n", "  H2D  input staging",    mh, pct(mh));
    std::printf("  %-32s  %8.2f ms  (%4.1f%%)\n", "  RUN  model call",       mr, pct(mr));
    std::printf("  %-32s  %8.2f ms  (%4.1f%%)\n", "  D2H  output readback",  md, pct(md));
    print_separator('=');
}

int main(int argc, char* argv[])
{
    Args a = parse_args(argc, argv);
    color::init(a.no_color);

    // ------------------------------------------------------------------
    print_separator('=');
    std::printf("  %sEmitC Model Benchmark%s\n", color::bold(), color::reset());
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
    std::printf("  device opened  (program cache %senabled%s)\n",
                color::green(), color::reset());

    void*                              so_handle = nullptr;
    std::optional<tt::runtime::Binary> binary;
    if (a.binary_path.empty()) {
        so_handle = tt::runtime::testutils::open_so(a.so_path);
        std::printf("  shared object loaded\n");
    } else {
        binary = tt::runtime::Binary::loadFromPath(a.binary_path.c_str());
        // Pick the runtime the binary was built for before any device call.
        tt::runtime::setCompatibleDeviceRuntime(*binary);
        std::printf("  flatbuffer binary loaded\n");
    }

    // ------------------------------------------------------------------
    print_phase(3, kPhases, "Preparing device inputs");

    // Both pipelines describe their expected input layouts, just through
    // different queries: the .so hands back template tensors, the flatbuffer
    // exposes a Layout per input index.
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

    // Upload persistent weights once — shared across all sequences.
    //
    // retain=true is required, not cosmetic. A flatbuffer program deallocates
    // its own inputs as it runs (runtime deallocate ops honour the tensor's
    // retain flag), so with the default retain=false the first submit() frees
    // these weights and the second aborts with "Input Tensor A is not
    // allocated". They are uploaded once and reused by every iteration, so they
    // must outlive the program that consumes them.
    //
    // The EmitC path does not hit this only because TTNNAdjustDeallocs strips
    // the deallocs for const/parameter operands from the generated C++ -- the
    // same asymmetry that shows up as 600 vs 1198 ttnn.deallocate in `forward`.
    // Setting it for both backends keeps the two paths honest and costs nothing.
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

    std::printf("  Running each sequence once for PCC validation ...\n");

    std::vector<bool> val_results;
    val_results.reserve(n_seqs);

    // Validation runs a full inference per sequence. Under a profiler those extra
    // inferences are indistinguishable from the timed ones in the ops report, so
    // allow skipping them to profile exactly the requested warmup + iters.
    for (std::size_t si = 0; a.skip_validation ? false : si < n_seqs; ++si) {
        const auto& seq_id = seq_ids[si];
        std::printf("\n  [%zu/%zu]  %s%s%s\n",
                    si + 1, n_seqs, color::bold(), seq_id.c_str(), color::reset());

        auto golden_torch = load_tensor_list(
            (fs::path(a.seq_dir) / seq_id / kGoldenFilename).string());

        auto r = run_one(so_handle, binary, a.func_name, all_act[si], pers_device, act_layouts, device, a.blocking_d2h);

        std::printf("  PCC validation  (threshold >= %.3f)\n\n", a.pcc_threshold);
        bool pass = validate_outputs(r.host_outputs, golden_torch, a.pcc_threshold);
        std::printf("\n  Result:  %s\n", pass ? color::pass_tag() : color::fail_tag());
        val_results.push_back(pass);
    }

    if (a.skip_validation)
        std::printf("\n  Validation skipped (--skip-validation).\n");
    else
        print_validation_summary(seq_ids, val_results, a.pcc_threshold);

    // ------------------------------------------------------------------
    print_phase(5, kPhases, "Benchmark  (cycling through sequences)");

    std::printf("  %sWarmup%s  (%d iteration(s), iter 1 = trace capture) ...\n",
                color::cyan(), color::reset(), a.n_warmup);
    for (int i = 0; i < a.n_warmup; ++i) {
        std::size_t si = static_cast<std::size_t>(i) % n_seqs;
        run_one(so_handle, binary, a.func_name, all_act[si], pers_device, act_layouts, device, a.blocking_d2h);
        print_progress("warmup", i + 1, a.n_warmup, seq_ids[si]);
    }

    std::printf("\n  %sTimed run%s  (%d iteration(s)) ...\n",
                color::cyan(), color::reset(), a.n_iters);

    std::vector<double> infer_ms(a.n_iters);
    std::vector<double> h2d_ms(a.n_iters), run_ms(a.n_iters), d2h_ms(a.n_iters);
    double running_sum = 0.0;

    for (int i = 0; i < a.n_iters; ++i) {
        std::size_t si  = static_cast<std::size_t>(i) % n_seqs;
        signpost("emitc_benchmark_iter-start");
        auto r          = run_one(so_handle, binary, a.func_name, all_act[si], pers_device, act_layouts, device, a.blocking_d2h);
        signpost("emitc_benchmark_iter-end");
        infer_ms[i]     = r.infer_ms;
        h2d_ms[i]       = r.h2d_ms;
        run_ms[i]       = r.run_ms;
        d2h_ms[i]       = r.d2h_ms;
        running_sum    += r.infer_ms;
        print_progress("timed", i + 1, a.n_iters, seq_ids[si],
                       r.infer_ms, running_sum / static_cast<double>(i + 1));
    }

    // ------------------------------------------------------------------
    print_phase(6, kPhases, "Results");

    auto _stats = compute_stats(infer_ms, a.batch_size);
    print_results_table(a, _stats, n_seqs);
    print_phase_breakdown(h2d_ms, run_ms, d2h_ms, _stats.mean_ms);

    int n_pass = 0;
    for (bool r : val_results) if (r) ++n_pass;
    // Nothing to validate is not a failure: --skip-validation must still exit 0,
    // or `python3 -m tracy` (which runs the command with check=True) aborts
    // before it post-processes the profile.
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
        // The device-side profiler buffers its kernel timestamps on-chip and is
        // only written to profile_log_device.csv by the mesh device's closing
        // read (ProfilerReadState::LAST_FD_READ). _Exit below skips that
        // teardown, so without closing the device here the device profile comes
        // out empty and process_ops_logs.py aborts on the missing device rows.
        std::printf("Profiling: closing device to flush the device profiler ...\n");
        std::fflush(stdout);
        close_devices();

        // _Exit also skips atexit, where Tracy's client would drain its own
        // queue, so the tail of the profile (the closing signpost included)
        // would be dropped. Tracy exposes no explicit flush, but its worker
        // thread ships the queue every few ms, so give it a moment.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
#endif

    // _Exit skips atexit / static-dtor cleanup to avoid a SIGSEGV in _dl_fini.
    std::_Exit(all_pass ? 0 : 1);
}
