// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// EmitC inference runner — validates all sequences under a directory.
//
// Discovers every subdirectory of --seq-dir that contains:
//   bev_activation_inputs.bin   activation inputs  (TTNN binary format)
//   bev_device_outputs.bin      golden outputs     (TTNN binary format)
//
// Persistent weights (shared across sequences) are loaded from --pers-bin,
// which defaults to <seq-dir>/bev_persistent_inputs.bin.
//
// Usage:
//   model_inference <so> --seq-dir <dir> [--pers-bin <path>] [options]
//
// Required:
//   <so>               compiled EmitC shared object
//   --seq-dir <dir>    directory containing per-sequence subdirectories
//
// Options:
//   --pers-bin <path>      persistent weights  (default: <seq-dir>/bev_persistent_inputs.bin)
//   --func <name>          entry-point function name  (default: forward)
//   --pcc-threshold <f>    PCC pass threshold         (default: 0.99)
//   --no-color             disable ANSI color output
//   --help                 show this message and exit

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "emitc_utils.hpp"
#include "runtime/testutils/testutils.hpp"
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
    std::string seq_dir;
    std::string pers_bin;
    std::string func_name     = "forward";
    float       pcc_threshold = 0.99f;
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
        "  --pcc-threshold <f>    PCC pass threshold         (default: 0.99)\n"
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
        } else if (arg == "--pcc-threshold" && i + 1 < argc) { a.pcc_threshold = std::stof(argv[++i]);
        } else if (arg == "--no-color") { a.no_color = true;
        } else if (arg.rfind("--", 0) != 0 && !got_so) {
            a.so_path = arg; got_so = true;
        } else {
            std::fprintf(stderr, "error: unknown option or unexpected argument: %s\n\n", arg.c_str());
            print_usage(argv[0]); std::exit(1);
        }
    }

    bool missing = false;
    if (a.so_path.empty())  { std::fprintf(stderr, "error: missing required argument <so>\n");        missing = true; }
    if (a.seq_dir.empty())  { std::fprintf(stderr, "error: missing required option --seq-dir\n");     missing = true; }
    if (missing) { std::fprintf(stderr, "\n"); print_usage(argv[0]); std::exit(1); }

    if (a.pers_bin.empty())
        a.pers_bin = (fs::path(a.seq_dir) / kPersDefault).string();

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
// Per-sequence inference pass
// =============================================================================

static std::vector<tt::runtime::Tensor> run_inference(
    void*                                   so_handle,
    const std::string&                      func_name,
    const std::vector<torch::Tensor>&       act_torch,
    const std::vector<torch::Tensor>&       pers_torch,
    const std::vector<tt::runtime::Tensor>& layout_templates,
    tt::runtime::Device&                    device)
{
    std::vector<tt::runtime::Tensor> device_inputs;
    device_inputs.reserve(act_torch.size() + pers_torch.size());

    std::size_t idx = 0;
    for (const auto& t : act_torch) {
        auto layout = tt::runtime::getTensorLayout(layout_templates[idx++]);
        device_inputs.push_back(tt::runtime::toLayout(make_runtime_tensor(t), device, layout));
    }
    for (const auto& t : pers_torch) {
        auto layout = tt::runtime::getTensorLayout(layout_templates[idx++]);
        device_inputs.push_back(tt::runtime::toLayout(make_runtime_tensor(t), device, layout));
    }

    auto outputs =
        tt::runtime::test::ttnn::runSoProgram(so_handle, func_name, device_inputs, device);

    std::vector<tt::runtime::Tensor> host_outputs;
    for (auto& out : outputs) {
        auto shards = tt::runtime::toHost(out, /*untilize=*/true);
        assert(!shards.empty());
        host_outputs.insert(host_outputs.end(), shards.begin(), shards.end());
    }
    return host_outputs;
}

// =============================================================================
// Validation summary
// =============================================================================

static void print_summary(
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

int main(int argc, char* argv[])
{
    Args a = parse_args(argc, argv);
    color::init(a.no_color);

    // ------------------------------------------------------------------
    print_separator('=');
    std::printf("  %sEmitC Model Inference%s\n", color::bold(), color::reset());
    print_separator('=');
    std::printf("  %-18s  %s\n",   "shared object",  a.so_path.c_str());
    std::printf("  %-18s  %s\n",   "seq-dir",        a.seq_dir.c_str());
    std::printf("  %-18s  %s\n",   "pers-bin",       a.pers_bin.c_str());
    std::printf("  %-18s  %s\n",   "function",       a.func_name.c_str());
    std::printf("  %-18s  %.3f\n", "pcc threshold",  a.pcc_threshold);
    print_separator('=');

    // ------------------------------------------------------------------
    constexpr int kPhases = 4;

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
    print_phase(2, kPhases, "Loading persistent weights & opening device");

    auto pers_torch = load_tensor_list(a.pers_bin);
    std::printf("  persistent  %zu tensor(s)  [%s]\n",
                pers_torch.size(), a.pers_bin.c_str());

    tt::runtime::Device device = open_device(/*enable_program_cache=*/true);
    std::printf("  device opened  (program cache %senabled%s)\n",
                color::green(), color::reset());

    void* so_handle = tt::runtime::testutils::open_so(a.so_path);
    std::printf("  shared object loaded\n");

    // Query layout templates using the first sequence's activation inputs
    std::string first_act_path =
        (fs::path(a.seq_dir) / seq_ids[0] / kActFilename).string();
    auto first_act = load_tensor_list(first_act_path);

    auto layout_templates =
        tt::runtime::test::ttnn::createInputs(so_handle, a.func_name, device, a.so_path);

    std::size_t total_expected = first_act.size() + pers_torch.size();
    if (layout_templates.size() != total_expected)
        throw std::runtime_error(
            "input count mismatch: .so expects " +
            std::to_string(layout_templates.size()) +
            ", got " + std::to_string(total_expected));

    std::printf("  %zu layout(s) queried  (%zu activation + %zu persistent)\n",
                layout_templates.size(), first_act.size(), pers_torch.size());

    // ------------------------------------------------------------------
    print_phase(3, kPhases, "Running inference & validation on each sequence");

    std::vector<bool> results;
    results.reserve(seq_ids.size());

    for (std::size_t si = 0; si < seq_ids.size(); ++si) {
        const auto& seq_id = seq_ids[si];
        std::printf("\n  [%zu/%zu]  %s%s%s\n",
                    si + 1, seq_ids.size(),
                    color::bold(), seq_id.c_str(), color::reset());

        auto act_torch    = load_tensor_list(
            (fs::path(a.seq_dir) / seq_id / kActFilename).string());
        auto golden_torch = load_tensor_list(
            (fs::path(a.seq_dir) / seq_id / kGoldenFilename).string());

        std::printf("  activation  %3zu tensor(s)   golden  %3zu tensor(s)\n",
                    act_torch.size(), golden_torch.size());
        std::printf("  calling %s%s%s ...\n",
                    color::cyan(), a.func_name.c_str(), color::reset());

        auto host_outputs = run_inference(
            so_handle, a.func_name, act_torch, pers_torch, layout_templates, device);

        std::printf("  PCC validation  (threshold >= %.3f)\n\n", a.pcc_threshold);
        bool pass = validate_outputs(host_outputs, golden_torch, a.pcc_threshold);
        std::printf("\n  Result:  %s\n", pass ? color::pass_tag() : color::fail_tag());

        results.push_back(pass);
    }

    // ------------------------------------------------------------------
    print_phase(4, kPhases, "Summary");

    print_summary(seq_ids, results, a.pcc_threshold);

    int n_pass = 0;
    for (bool r : results) if (r) ++n_pass;
    bool all_pass = (n_pass == static_cast<int>(seq_ids.size()));

    std::cout.flush();
    std::fflush(stdout);
    tt::runtime::testutils::close_so(so_handle);
    std::printf("\nDone.\n");
    std::fflush(stdout);

    // _Exit skips atexit / static-dtor cleanup to avoid a SIGSEGV in _dl_fini.
    std::_Exit(all_pass ? 0 : 1);
}
