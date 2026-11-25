// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Generic EmitC inference runner.
//
// Runs a compiled EmitC shared object once on the device: loads the activation
// inputs, persistent weights and golden outputs from TTNN binary files, runs a
// single forward pass and validates every output against the golden (PCC).
// For timing use model_benchmark; for Tracy profiling or flatbuffer binaries use
// model_benchmark_profile.
//
// Usage:
//   model_inference <so> <act.bin> <pers.bin> [<golden.bin>] [options]
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
//   --pcc-threshold <f>    PCC pass threshold         (default: 0.99)
//   --skip-validation      run the forward pass without validating it
//   --help                 show this message and exit

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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
    float pcc_threshold = 0.99f;
    bool skip_validation = false;
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
        "  --pcc-threshold <f>    PCC pass threshold         (default: 0.99)\n"
        "  --skip-validation      run the forward pass without validating it\n"
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
        else if (arg == "--pcc-threshold" && has_value)
            a.pcc_threshold = std::stof(next());
        else if (arg == "--skip-validation")
            a.skip_validation = true;
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
    return a;
}

void print_config(const Args& a)
{
    print_separator('=');
    std::printf("  EmitC Model Inference\n");
    print_separator('=');
    std::printf("  %-18s  %s\n", "shared object", a.so_path.c_str());
    std::printf("  %-18s  %s\n", "activation inputs", a.act_bin.c_str());
    std::printf("  %-18s  %s\n", "persistent inputs", a.pers_bin.c_str());
    std::printf("  %-18s  %s\n", "golden", a.skip_validation ? "(validation skipped)" : a.golden_bin.c_str());
    std::printf("  %-18s  %s\n", "function", a.func_name.c_str());
    std::printf("  %-18s  %.3f\n", "pcc threshold", a.pcc_threshold);
    print_separator('=');
}

// =============================================================================
// Phases
// =============================================================================

struct DeviceInputs
{
    std::vector<tt::runtime::Layout> act_layouts;  // layouts the program expects for the activations
    std::vector<tt::runtime::Tensor> pers_device;  // weights, on device
};

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

    in.pers_device.reserve(n_pers);
    for (std::size_t i = 0; i < n_pers; ++i)
        in.pers_device.push_back(tt::runtime::toLayout(
            make_runtime_tensor(pers_torch[i]), device, tt::runtime::getTensorLayout(templates[n_act + i])));
    std::printf("  %zu persistent tensor(s) on device\n", in.pers_device.size());
    return in;
}

// Phase 3: stage the activations, run the forward pass and read every output back
// to host (all reads enqueued, then one wait).
std::vector<tt::runtime::Tensor> run_inference(
    void* so_handle,
    const Args& a,
    const std::vector<torch::Tensor>& act_torch,
    const DeviceInputs& in,
    tt::runtime::Device& device)
{
    std::vector<tt::runtime::Tensor> all_inputs;
    all_inputs.reserve(act_torch.size() + in.pers_device.size());
    for (std::size_t i = 0; i < act_torch.size(); ++i)
    {
        auto rt = make_runtime_tensor(act_torch[i]);
        all_inputs.push_back(
            tt::runtime::hasLayout(rt, in.act_layouts[i]) ? rt : tt::runtime::toLayout(rt, device, in.act_layouts[i]));
    }
    for (const auto& p : in.pers_device) all_inputs.push_back(p);

    std::printf("  calling %s ...\n", a.func_name.c_str());
    std::vector<tt::runtime::Tensor> outputs =
        tt::runtime::test::ttnn::runSoProgram(so_handle, a.func_name, all_inputs, device);

    std::vector<tt::runtime::Tensor> host_outputs;
    for (auto& out : outputs)
    {
        auto shards = tt::runtime::toHost(out, /*untilize=*/true, /*blocking=*/false);
        assert(!shards.empty());
        host_outputs.insert(host_outputs.end(), shards.begin(), shards.end());
    }
    tt::runtime::wait(host_outputs);
    std::printf("  %zu output(s) on host\n", host_outputs.size());
    return host_outputs;
}

// Phase 4: every output against the golden.
bool run_validation(const Args& a, const std::vector<tt::runtime::Tensor>& host_outputs)
{
    if (a.skip_validation)
    {
        std::printf("  Validation skipped (--skip-validation).\n");
        return true;
    }
    auto golden = load_tensor_list(a.golden_bin);
    std::printf(
        "  PCC validation against %zu golden output(s)  (threshold >= %.3f)\n\n", golden.size(), a.pcc_threshold);
    const bool pass = validate_outputs(host_outputs, golden, a.pcc_threshold);
    std::printf("\n  Validation: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[])
{
    const Args a = parse_args(argc, argv);
    print_config(a);
    constexpr int kPhases = 4;

    print_phase(1, kPhases, "Loading inputs & opening device");
    auto act_torch = load_tensor_list(a.act_bin);
    auto pers_torch = load_tensor_list(a.pers_bin);
    std::printf("  activation  %zu tensor(s)  [%s]\n", act_torch.size(), a.act_bin.c_str());
    std::printf("  persistent  %zu tensor(s)  [%s]\n", pers_torch.size(), a.pers_bin.c_str());
    // A traced .so needs the program cache even for a single call.
    tt::runtime::Device device = open_device(/*enable_program_cache=*/true);
    std::printf("  device opened  (program cache enabled)\n");
    void* so_handle = tt::runtime::testutils::open_so(a.so_path);
    std::printf("  shared object loaded\n");

    print_phase(2, kPhases, "Preparing device inputs");
    const DeviceInputs in = prepare_inputs(so_handle, a, device, act_torch.size(), pers_torch);

    print_phase(3, kPhases, "Inference");
    const auto host_outputs = run_inference(so_handle, a, act_torch, in, device);

    print_phase(4, kPhases, "Validation");
    const bool pass = run_validation(a, host_outputs);

    std::fflush(stdout);
    tt::runtime::testutils::close_so(so_handle);
    std::printf("\nDone.\n");
    std::fflush(stdout);
    // _Exit skips atexit / static-destructor cleanup to avoid a SIGSEGV in _dl_fini:
    // _ttnncpp.so's static destructors run after libtt_metal.so frees its
    // thread_local GraphTracker TLS, making any Tensor destructor a use-after-free.
    std::_Exit(pass ? 0 : 1);
}
