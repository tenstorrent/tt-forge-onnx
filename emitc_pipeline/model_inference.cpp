// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// EmitC inference runner — runs every sequence under a directory once and
// checks the result three ways:
//
//   1. run          the call completes and returns the expected outputs: same
//                   count and shapes as the golden outputs, every value finite.
//   2. golden       PCC against bev_device_outputs.bin, the outputs the
//                   flatbuffer program produced on the device.
//   3. ground truth PCC against the reference outputs of the model, one raw
//                   float32 file per output: <gt-dir>/<sequence>/<output>.bin
//                   (the same files the Python benchmark validates against).
//
// Discovers every subdirectory of --seq-dir that contains:
//   bev_activation_inputs.bin   activation inputs  (TTNN binary format)
//   bev_device_outputs.bin      golden outputs     (TTNN binary format)
//
// Persistent weights (shared across sequences) are loaded from --pers-bin,
// which defaults to <seq-dir>/bev_persistent_inputs.bin. The exit code is
// non-zero if any check of any sequence fails.
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
//   --pcc-threshold <f>    golden PCC pass threshold  (default: 0.99)
//   --gt-dir <dir>         ground-truth directory     (default: $BEV_ASSETS_DIR/output_samples,
//                                                      else BEV_MODEL/model_artifacts/output_samples)
//   --gt-pcc-threshold <f> ground-truth PCC threshold (default: 0.99)
//   --output-names <list>  comma-separated output names, in output order
//                          (default: occupancy_mid_range_height_map,
//                                    aux_semantic_logits_mid,aux_visibility_logits_mid)
//   --no-gt                skip the ground-truth check
//   --blocking-d2h         synchronise on every D2H read (default: enqueue all, wait once)
//   --help                 show this message and exit

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "emitc_utils.hpp"
#include "runtime/testutils/testutils.hpp"
#include "tt/runtime/runtime.h"
#include "tt/runtime/test/ttnn/dylib.h"

namespace fs = std::filesystem;

static const char* kDefaultOutputNames =
    "occupancy_mid_range_height_map,aux_semantic_logits_mid,aux_visibility_logits_mid";

// =============================================================================
// CLI
// =============================================================================

struct Args {
    std::string so_path;
    std::string seq_dir;
    std::string pers_bin;
    std::string func_name        = "forward";
    float       pcc_threshold    = 0.99f;
    std::string gt_dir;
    float       gt_pcc_threshold = 0.99f;
    std::string output_names     = kDefaultOutputNames;
    bool        no_gt            = false;
    bool        blocking_d2h     = false;
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
        "  --pcc-threshold <f>    golden PCC pass threshold  (default: 0.99)\n"
        "  --gt-dir <dir>         ground-truth directory, <dir>/<sequence>/<output>.bin\n"
        "                         (default: $BEV_ASSETS_DIR/output_samples, else\n"
        "                          BEV_MODEL/model_artifacts/output_samples)\n"
        "  --gt-pcc-threshold <f> ground-truth PCC threshold (default: 0.99)\n"
        "  --output-names <list>  comma-separated output names in output order\n"
        "                         (default: %s)\n"
        "  --no-gt                skip the ground-truth check\n"
        "  --blocking-d2h         synchronise on every D2H read  (default: enqueue all\n"
        "                         reads and wait once)\n"
        "  --help                 show this message and exit\n",
        prog, kActFilename, kGoldenFilename, kPersDefault, kDefaultOutputNames);
}

static Args parse_args(int argc, char* argv[])
{
    Args a;
    bool got_so = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]); std::exit(0);
        } else if (arg == "--seq-dir"          && i + 1 < argc) { a.seq_dir          = argv[++i];
        } else if (arg == "--pers-bin"         && i + 1 < argc) { a.pers_bin         = argv[++i];
        } else if (arg == "--func"             && i + 1 < argc) { a.func_name        = argv[++i];
        } else if (arg == "--pcc-threshold"    && i + 1 < argc) { a.pcc_threshold    = std::stof(argv[++i]);
        } else if (arg == "--gt-dir"           && i + 1 < argc) { a.gt_dir           = argv[++i];
        } else if (arg == "--gt-pcc-threshold" && i + 1 < argc) { a.gt_pcc_threshold = std::stof(argv[++i]);
        } else if (arg == "--output-names"     && i + 1 < argc) { a.output_names     = argv[++i];
        } else if (arg == "--no-gt") { a.no_gt = true;
        } else if (arg == "--blocking-d2h") { a.blocking_d2h = true;
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

    if (a.gt_dir.empty()) {
        const char* assets = std::getenv("BEV_ASSETS_DIR");
        a.gt_dir = assets ? (fs::path(assets) / "output_samples").string()
                          : "BEV_MODEL/model_artifacts/output_samples";
    }

    return a;
}

static std::vector<std::string> split_names(const std::string& list)
{
    std::vector<std::string> out;
    std::stringstream ss(list);
    for (std::string item; std::getline(ss, item, ',');)
        if (!item.empty()) out.push_back(item);
    return out;
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
    tt::runtime::Device&                    device,
    bool                                    blocking_d2h)
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
        auto shards = tt::runtime::toHost(out, /*untilize=*/true, blocking_d2h);
        assert(!shards.empty());
        host_outputs.insert(host_outputs.end(), shards.begin(), shards.end());
    }
    if (!blocking_d2h)
        tt::runtime::wait(host_outputs);
    return host_outputs;
}

// =============================================================================
// Checks
// =============================================================================

// Check 1: output count and shapes match the golden tensors and all values are finite.
static bool check_run(const std::vector<tt::runtime::Tensor>& host_outputs,
                      const std::vector<torch::Tensor>&       golden,
                      const std::vector<std::string>&         names)
{
    bool ok = true;
    if (host_outputs.size() != golden.size()) {
        std::printf("    output count %zu, expected %zu\n", host_outputs.size(), golden.size());
        return false;
    }
    for (std::size_t i = 0; i < host_outputs.size(); ++i) {
        auto t = runtime_tensor_to_torch(host_outputs[i]).to(c10::kFloat);
        const char* name = i < names.size() ? names[i].c_str() : "";
        bool shape_ok  = t.sizes() == golden[i].sizes();
        bool finite_ok = torch::isfinite(t).all().item<bool>();
        std::printf("    [%zu]  %s  shape %-22s  %s  finite  %s  %s\n",
                    i, shape_ok && finite_ok ? "PASS" : "FAIL", format_shape(t).c_str(),
                    shape_ok ? "ok  " : "FAIL", finite_ok ? "ok  " : "FAIL", name);
        ok &= shape_ok && finite_ok;
    }
    return ok;
}

// Check 3 input: reference outputs of one sequence, shaped like the golden tensors.
static bool load_ground_truth(const std::string&               gt_dir,
                              const std::string&               seq_id,
                              const std::vector<std::string>&  names,
                              const std::vector<torch::Tensor>& golden,
                              std::vector<torch::Tensor>&      out)
{
    out.clear();
    if (names.size() != golden.size()) {
        std::printf("  ground truth  %zu output name(s) for %zu output(s); pass --output-names\n",
                    names.size(), golden.size());
        return false;
    }
    for (std::size_t i = 0; i < golden.size(); ++i) {
        fs::path p = fs::path(gt_dir) / seq_id / (names[i] + ".bin");
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) { std::printf("  ground truth  missing %s\n", p.string().c_str()); return false; }
        std::streamsize bytes = f.tellg();
        std::int64_t expect = golden[i].numel() * static_cast<std::int64_t>(sizeof(float));
        if (bytes != expect) {
            std::printf("  ground truth  %s has %lld bytes, expected %lld for shape %s\n",
                        p.string().c_str(), static_cast<long long>(bytes),
                        static_cast<long long>(expect), format_shape(golden[i]).c_str());
            return false;
        }
        f.seekg(0);
        auto t = torch::empty(golden[i].sizes(), torch::kFloat);
        f.read(reinterpret_cast<char*>(t.data_ptr<float>()), bytes);
        out.push_back(t);
    }
    return true;
}

struct SeqResult { bool run, golden, gt; };

static void print_summary(const std::vector<std::string>& seq_ids,
                          const std::vector<SeqResult>&   results,
                          const Args&                     a)
{
    int n_pass = 0;
    for (const auto& r : results) if (r.run && r.golden && (a.no_gt || r.gt)) ++n_pass;

    std::printf("\n  %zu sequence(s)  ·  golden PCC >= %.3f", seq_ids.size(), a.pcc_threshold);
    if (!a.no_gt) std::printf("  ·  ground-truth PCC >= %.3f", a.gt_pcc_threshold);
    std::printf("\n\n");

    constexpr int kSeqCol = 40;
    auto tag = [](bool b) { return b ? "PASS" : "FAIL"; };
    std::printf("  %-*s  %-6s  %-6s  %-6s\n", kSeqCol, "Sequence", "run", "golden", "gt");
    for (std::size_t i = 0; i < seq_ids.size(); ++i)
        std::printf("  %-*s  %-6s  %-6s  %-6s\n", kSeqCol, seq_ids[i].c_str(), tag(results[i].run),
                    tag(results[i].golden), a.no_gt ? "skip" : tag(results[i].gt));

    std::printf("\n  %d/%zu sequences PASS\n", n_pass, seq_ids.size());
    std::printf("  per-output PCC overrides (default %.3f):\n", a.pcc_threshold);
    print_pcc_overrides();
    print_separator('=');
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[])
{
    Args a = parse_args(argc, argv);
    const auto names = split_names(a.output_names);

    // ------------------------------------------------------------------
    print_separator('=');
    std::printf("  EmitC Model Inference\n");
    print_separator('=');
    std::printf("  %-18s  %s\n",   "shared object",  a.so_path.c_str());
    std::printf("  %-18s  %s\n",   "seq-dir",        a.seq_dir.c_str());
    std::printf("  %-18s  %s\n",   "pers-bin",       a.pers_bin.c_str());
    std::printf("  %-18s  %s\n",   "function",       a.func_name.c_str());
    std::printf("  %-18s  %.3f\n", "golden pcc",     a.pcc_threshold);
    if (a.no_gt) {
        std::printf("  %-18s  %s\n", "ground truth", "skipped (--no-gt)");
    } else {
        std::printf("  %-18s  %s\n",   "ground-truth dir", a.gt_dir.c_str());
        std::printf("  %-18s  %.3f\n", "ground-truth pcc", a.gt_pcc_threshold);
    }
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
    std::printf("  device opened  (program cache enabled)\n");

    void* so_handle = tt::runtime::testutils::open_so(a.so_path);
    std::printf("  shared object loaded\n");

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
    print_phase(3, kPhases, "Running inference & checking each sequence");

    std::vector<SeqResult> results;
    results.reserve(seq_ids.size());

    for (std::size_t si = 0; si < seq_ids.size(); ++si) {
        const auto& seq_id = seq_ids[si];
        std::printf("\n  [%zu/%zu]  %s\n", si + 1, seq_ids.size(), seq_id.c_str());

        auto act_torch    = load_tensor_list(
            (fs::path(a.seq_dir) / seq_id / kActFilename).string());
        auto golden_torch = load_tensor_list(
            (fs::path(a.seq_dir) / seq_id / kGoldenFilename).string());

        std::printf("  %zu activation tensor(s), %zu golden tensor(s); calling %s\n",
                    act_torch.size(), golden_torch.size(), a.func_name.c_str());

        SeqResult r{false, false, a.no_gt};
        std::vector<tt::runtime::Tensor> host_outputs;
        try {
            host_outputs = run_inference(
                so_handle, a.func_name, act_torch, pers_torch, layout_templates, device, a.blocking_d2h);
        } catch (const std::exception& e) {
            std::printf("  the call failed: %s\n", e.what());
            results.push_back(r);
            continue;
        }

        std::printf("\n  run\n");
        r.run = check_run(host_outputs, golden_torch, names);

        std::printf("\n  golden  (PCC vs %s, >= %.3f)\n", kGoldenFilename, a.pcc_threshold);
        r.golden = validate_outputs(host_outputs, golden_torch, a.pcc_threshold, seq_id, names);

        if (!a.no_gt) {
            std::printf("\n  ground truth  (PCC vs %s/%s, >= %.3f)\n",
                        a.gt_dir.c_str(), seq_id.c_str(), a.gt_pcc_threshold);
            std::vector<torch::Tensor> gt;
            r.gt = load_ground_truth(a.gt_dir, seq_id, names, golden_torch, gt) &&
                   validate_outputs(host_outputs, gt, a.gt_pcc_threshold, seq_id, names);
        }

        results.push_back(r);
    }

    // ------------------------------------------------------------------
    print_phase(4, kPhases, "Summary");

    print_summary(seq_ids, results, a);

    bool all_pass = true;
    for (const auto& r : results) all_pass &= r.run && r.golden && r.gt;

    std::cout.flush();
    std::fflush(stdout);
    tt::runtime::testutils::close_so(so_handle);
    std::printf("\nDone.\n");
    std::fflush(stdout);

    // _Exit skips atexit / static-dtor cleanup to avoid a SIGSEGV in _dl_fini.
    std::_Exit(all_pass ? 0 : 1);
}
