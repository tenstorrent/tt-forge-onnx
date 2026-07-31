// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Generic EmitC inference runner.
//
// Loads activation inputs, persistent weights, and golden outputs from TTNN
// binary files, runs a single forward pass on device, and validates the output
// against the golden using PCC.
//
// Usage:
//   model_inference <so> <act.bin> <pers.bin> <golden.bin> [options]
//
// Required:
//   <so>               compiled EmitC shared object
//   <act.bin>          activation inputs   (TTNN binary format)
//   <pers.bin>         persistent weights  (TTNN binary format)
//   <golden.bin>       golden outputs      (TTNN binary format)
//
// Options:
//   --func <name>          entry-point function name  (default: forward)
//   --pcc-threshold <f>    PCC pass threshold         (default: 0.99)
//   --no-color             disable ANSI color output
//   --help                 show this message and exit

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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

    return a;
}

// =============================================================================
// Input preparation
// =============================================================================

static std::vector<tt::runtime::Tensor> transfer_inputs_to_device(
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
    return device_inputs;
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[])
{
    Args a = parse_args(argc, argv);
    color::init(a.no_color);

    print_separator('=');
    std::printf("  %sEmitC Model Inference%s\n", color::bold(), color::reset());
    print_separator('=');
    std::printf("  %-16s  %s\n",   "shared object",  a.so_path.c_str());
    std::printf("  %-16s  %s\n",   "activation",     a.act_path.c_str());
    std::printf("  %-16s  %s\n",   "persistent",     a.pers_path.c_str());
    std::printf("  %-16s  %s\n",   "golden",         a.golden_path.c_str());
    std::printf("  %-16s  %s\n",   "function",       a.func_name.c_str());
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

    // The compiled .so uses trace capture internally and requires the program
    // cache even for a single inference call.
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

    auto device_inputs =
        transfer_inputs_to_device(act_torch, pers_torch, layout_templates, device);
    std::printf("  %zu tensor(s) on device\n", device_inputs.size());

    // -------------------------------------------------------------------------
    print_phase(4, kPhases, "Running inference");

    std::printf("  calling %s%s%s ...\n", color::cyan(), a.func_name.c_str(), color::reset());

    auto outputs =
        tt::runtime::test::ttnn::runSoProgram(so_handle, a.func_name, device_inputs, device);

    std::vector<tt::runtime::Tensor> host_outputs;
    for (auto& out : outputs) {
        auto shards = tt::runtime::toHost(out, /*untilize=*/true);
        assert(!shards.empty());
        host_outputs.insert(host_outputs.end(), shards.begin(), shards.end());
    }
    std::printf("  %zu output shard(s) on host\n", host_outputs.size());

    // -------------------------------------------------------------------------
    print_phase(5, kPhases, "Validation");

    std::printf("  PCC validation  (threshold >= %.3f)\n\n", a.pcc_threshold);
    bool pass = validate_outputs(host_outputs, golden_torch, a.pcc_threshold);
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
