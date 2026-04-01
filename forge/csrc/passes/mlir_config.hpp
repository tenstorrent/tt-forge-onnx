// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "lower_to_forge/common.hpp"
#include "nlohmann/json_fwd.hpp"

namespace tt::passes
{

// ============================================================================
// MemoryLayoutAnalysisPolicy
//
// Maps to the `memory-layout-analysis-policy` option of
// TTIRToTTNNDevicePipelineOptions (TTNNPipelines.h).
//
// Selects the search strategy used by the sharding analysis sub-pass of the
// TTNNOptimizer when it searches for the best tensor memory layout across the
// full compute graph.  The option only takes effect when both
//   enable_optimizer = true
//   enable_memory_layout_analysis = true
// are set (or optimization_level >= 2).
//
// Pipeline option: memory-layout-analysis-policy   default: DFSharding
// ============================================================================
enum class MemoryLayoutAnalysisPolicy
{
    /// Depth-first data-flow driven sharding (pipeline default).
    /// Propagates sharding decisions top-to-bottom through the dataflow graph.
    DFSharding,

    /// Greedy search that first promotes tensors to L1-interleaved layouts
    /// before attempting block/height/width sharding.
    GreedyL1Interleaved,

    /// Breadth-first interleaved strategy.  Analyses all ops at a given depth
    /// before moving deeper, favouring interleaved layouts throughout.
    BFInterleaved,
};

/// Returns the pipeline option string for the given MemoryLayoutAnalysisPolicy.
/// Throws std::invalid_argument for unknown values.
inline std::string to_pipeline_string(MemoryLayoutAnalysisPolicy p)
{
    switch (p)
    {
        case MemoryLayoutAnalysisPolicy::DFSharding: return "DFSharding";
        case MemoryLayoutAnalysisPolicy::GreedyL1Interleaved: return "GreedyL1Interleaved";
        case MemoryLayoutAnalysisPolicy::BFInterleaved: return "BFInterleaved";
    }
    throw std::invalid_argument("Unknown MemoryLayoutAnalysisPolicy value");
}

/// Parses a pipeline option string back into a MemoryLayoutAnalysisPolicy.
/// Throws std::invalid_argument for unknown strings.
inline MemoryLayoutAnalysisPolicy memory_layout_policy_from_string(const std::string& s)
{
    if (s == "DFSharding")
        return MemoryLayoutAnalysisPolicy::DFSharding;
    if (s == "GreedyL1Interleaved")
        return MemoryLayoutAnalysisPolicy::GreedyL1Interleaved;
    if (s == "BFInterleaved")
        return MemoryLayoutAnalysisPolicy::BFInterleaved;
    throw std::invalid_argument("Unknown MemoryLayoutAnalysisPolicy string: " + s);
}

void to_json(nlohmann::json& j, MemoryLayoutAnalysisPolicy p);
void from_json(const nlohmann::json& j, MemoryLayoutAnalysisPolicy& p);

/// Converts tt::MathFidelity to the MLIR pipeline option string.
/// MathFidelity::Invalid maps to "undefined" (backend-selected fidelity).
inline std::string to_pipeline_string(tt::MathFidelity f)
{
    switch (f)
    {
        case tt::MathFidelity::LoFi: return "lofi";
        case tt::MathFidelity::HiFi2: return "hifi2";
        case tt::MathFidelity::HiFi3: return "hifi3";
        case tt::MathFidelity::HiFi4: return "hifi4";
        case tt::MathFidelity::Invalid: return "undefined";
    }
    throw std::invalid_argument("Unknown MathFidelity value");
}

/// Parses a math-fidelity pipeline string back into tt::MathFidelity.
/// "undefined" maps to MathFidelity::Invalid.
inline tt::MathFidelity math_fidelity_from_string(const std::string& s)
{
    if (s == "lofi")
        return tt::MathFidelity::LoFi;
    if (s == "hifi2")
        return tt::MathFidelity::HiFi2;
    if (s == "hifi3")
        return tt::MathFidelity::HiFi3;
    if (s == "hifi4")
        return tt::MathFidelity::HiFi4;
    if (s == "undefined")
        return tt::MathFidelity::Invalid;
    throw std::invalid_argument("Unknown MathFidelity pipeline string: " + s);
}

/// Converts a weight-dtype DataFormat value to the MLIR pipeline option string.
/// Only DataFormat::Bfp8_b and DataFormat::Bfp4_b are accepted; all other
/// values throw std::invalid_argument.
inline std::string weight_dtype_to_pipeline_string(tt::DataFormat d)
{
    switch (d)
    {
        case tt::DataFormat::Bfp8_b: return "bfp_bf8";
        case tt::DataFormat::Bfp4_b: return "bfp_bf4";
        default:
            throw std::invalid_argument(
                "experimental_weight_dtype only accepts DataFormat::Bfp8_b or "
                "DataFormat::Bfp4_b");
    }
}

/// Parses a weight-dtype pipeline string back into a tt::DataFormat value.
inline tt::DataFormat weight_dtype_from_string(const std::string& s)
{
    if (s == "bfp_bf8")
        return tt::DataFormat::Bfp8_b;
    if (s == "bfp_bf4")
        return tt::DataFormat::Bfp4_b;
    throw std::invalid_argument("Unknown weight dtype pipeline string: " + s);
}

// ============================================================================
// MLIRConfig
//
// Typed configuration wrapper for TTIRToTTNNDevicePipelineOptions
// (TTNNPipelines.h).  Used to pass compile-time options from Python / C++
// callers through to the MLIR pipeline without requiring knowledge of the
// internal MLIR option-string format.
//
// Each field is std::optional.  When left as std::nullopt, the option is
// omitted from the pipeline string and the pipeline uses its built-in default
// ============================================================================
struct MLIRConfig
{
    // -------------------------------------------------------------------------
    // Optimization level shorthand
    // Pipeline option : optimization-level   default: 0
    //
    //  0 — All optimizer passes disabled. Fastest compile, baseline runtime.
    //      Equivalent to: enable_optimizer=false, memory_layout_analysis=false.
    //  1 — Optimizer on; Conv2d-multiply fusing on; sharding (MLA) off.
    //      Equivalent to: enable_optimizer=true,
    //                     enable_fusing_conv2d_with_multiply_pattern=true.
    //  2 — All optimizer passes including sharding.  Longest compile, best
    //      runtime.  Equivalent to level 1 + enable_memory_layout_analysis=true.
    //
    // -------------------------------------------------------------------------
    std::optional<int> optimization_level = std::nullopt;

    // -------------------------------------------------------------------------
    // Optimizer control
    // -------------------------------------------------------------------------

    /// Pipeline option: enable-const-eval   default: true
    /// Folds constant sub-graphs at compile time, eliminating redundant
    /// on-device computation for weights and static inputs.
    std::optional<bool> enable_consteval = std::nullopt;

    /// Pipeline option: enable-optimizer   default: false
    /// Master switch for the TTNNOptimizer pass.  When enabled, the compiler
    /// determines the optimal Tensix core grid, sharding strategy, and memory
    /// placement (L1 vs DRAM) for every operation.
    /// Note: requires TTMLIR_ENABLE_OPMODEL=ON at build time.
    /// Automatically enabled when optimization_level >= 1.
    std::optional<bool> enable_optimizer = std::nullopt;

    /// Pipeline option: memory-layout-analysis-enabled   default: false
    /// Enables the sharding analysis sub-pass of the optimizer.  Searches the
    /// full compute graph for the best L1-sharding layout for each tensor.
    /// Requires enable_optimizer = true.
    /// Automatically enabled when optimization_level >= 2.
    std::optional<bool> enable_memory_layout_analysis = std::nullopt;

    // -------------------------------------------------------------------------
    // Memory layout options
    // -------------------------------------------------------------------------

    /// Pipeline option: memory-layout-analysis-policy   default: DFSharding
    /// Selects the sharding search strategy used by the memory layout analysis
    /// pass.
    std::optional<MemoryLayoutAnalysisPolicy> memory_layout_analysis_policy = std::nullopt;

    /// Pipeline option: l1-interleaved-fallback-analysis-enabled   default: false
    /// Lightweight pass that promotes DRAM-interleaved tensors to L1-interleaved
    /// when sufficient on-chip SRAM is available.  Works independently of the
    /// full sharding analysis (enable_memory_layout_analysis not required).
    std::optional<bool> enable_l1_interleaved_fallback_analysis = std::nullopt;

    /// Pipeline option: memreconfig-enabled   default: true
    /// Inserts ToLayout ops between operations whose output/input memory
    /// layouts are incompatible.  Disable only for debugging — may produce
    /// incorrect results when off.
    std::optional<bool> enable_memreconfig = std::nullopt;

    /// Pipeline option: max-legal-layouts   default: 8
    /// Maximum number of sharded layout candidates the optimizer generates and
    /// evaluates per operation during legal layout analysis.
    std::optional<int64_t> max_legal_layouts = std::nullopt;

    /// Pipeline option: row-major-enabled   default: false
    /// Includes row-major layout as a candidate during legal layout analysis.
    /// Can improve ops that run faster in row-major; increases the search space.
    std::optional<bool> enable_row_major = std::nullopt;

    // -------------------------------------------------------------------------
    // Compute kernel configuration
    // -------------------------------------------------------------------------

    /// Pipeline option: compute-cfg-math-fidelity   default: hifi4
    /// Arithmetic fidelity for all Tensix compute kernels that expose a
    /// ComputeKernelConfig.  Uses the canonical tt::MathFidelity enum from
    /// lower_to_forge/common.hpp.
    ///
    ///   MathFidelity::LoFi    — lowest precision, fastest throughput
    ///   MathFidelity::HiFi2   — good accuracy/speed trade-off
    ///   MathFidelity::HiFi3   — high accuracy
    ///   MathFidelity::HiFi4   — highest accuracy (pipeline default)
    ///   MathFidelity::Invalid — let the backend choose ("undefined")
    std::optional<tt::MathFidelity> compute_cfg_math_fidelity = std::nullopt;

    /// Pipeline option: compute-cfg-fp32-dest-acc-en   default: true
    /// When true, intermediate sums are accumulated in FP32 before being
    /// written back to the output tensor.  Setting false uses the native
    /// accumulator width (typically the same as the input dtype), which is
    /// faster at the cost of slightly lower precision on FP16/BF16 workloads.
    std::optional<bool> compute_cfg_fp32_dest_acc_en = std::nullopt;

    // -------------------------------------------------------------------------
    // Data type / quantization options
    // -------------------------------------------------------------------------

    /// Pipeline option: experimental-weight-dtype   default: none (nullopt)
    /// Experimental: converts weight tensors in matmul / linear operations to a
    /// reduced-precision block-floating-point format, reducing DRAM bandwidth.
    /// Uses tt::DataFormat from lower_to_forge/common.hpp.
    ///
    ///   DataFormat::Bfp8_b — BFP BFloat8 (8-bit mantissa)
    ///   DataFormat::Bfp4_b — BFP BFloat4 (4-bit mantissa)
    ///   std::nullopt        — no conversion (pipeline default "none")
    ///
    /// Only DataFormat::Bfp8_b and DataFormat::Bfp4_b are accepted; all other
    /// DataFormat values are rejected by set_experimental_weight_dtype().
    std::optional<tt::DataFormat> experimental_weight_dtype = std::nullopt;

    // -------------------------------------------------------------------------
    // Graph transformation passes
    // -------------------------------------------------------------------------

    /// Pipeline option: enable-erase-inverse-ops-pass   default: true
    /// Cancels pairs of inverse operations, reducing unnecessary data movement.
    std::optional<bool> enable_erase_inverse_ops = std::nullopt;

    /// Pipeline option: enable-implicit-broadcast-folding-pass   default: true
    /// Folds implicit broadcast operations into downstream consumers,
    /// eliminating standalone broadcast ops where possible.
    std::optional<bool> enable_implicit_broadcast_folding = std::nullopt;

    /// Pipeline option: enable-permute-matmul-fusion   default: false
    /// Fuses permute / transpose operations into adjacent matmul operations
    /// so that no separate transpose kernel is dispatched.
    std::optional<bool> enable_permute_matmul_fusion = std::nullopt;

    /// Pipeline option: enable-dram-space-saving-optimization-pass   default: false
    /// Reduces peak DRAM memory usage by safely reusing buffers across ops
    /// whose live ranges do not overlap.
    std::optional<bool> enable_dram_space_saving_optimization = std::nullopt;

    /// Pipeline option: enable-remove-dead-values   default: false
    /// Eliminates dead (unused) values and their producers from the compute
    /// graph, reducing both compile time and runtime overhead.
    /// Note: this pass can significantly increase peak memory consumption
    /// during compilation; disabled by default for this reason.
    std::optional<bool> enable_remove_dead_values = std::nullopt;

    // -------------------------------------------------------------------------
    // Const-eval (constant folding) options
    // -------------------------------------------------------------------------

    /// Pipeline option: enable-cpu-hoisted-const-eval   default: true
    /// Hoists constant sub-graphs to CPU execution, folding them at compile
    /// time rather than dispatching them to device at every inference call.
    std::optional<bool> enable_cpu_hoisted_consteval = std::nullopt;

    /// Pipeline option: enable-const-eval-inputs-to-system-memory   default: true
    /// Stores the output of const-eval sub-graphs in system (host) memory
    /// instead of device DRAM.  Disable if device DRAM capacity is not a
    /// bottleneck and lower host-to-device latency is preferred.
    std::optional<bool> enable_consteval_inputs_to_system_memory = std::nullopt;

    // -------------------------------------------------------------------------
    // Fusing passes
    // -------------------------------------------------------------------------

    /// Pipeline option: enable-fusing-pass   default: true
    /// Master switch for the general op-fusion pass.
    std::optional<bool> enable_fusing = std::nullopt;

    /// Pipeline option: enable-fusing-conv2d-with-multiply-pattern   default: false
    /// Fuses Conv2d with a subsequent elementwise multiply (e.g. channel
    /// scaling or batch-norm weight fold).  Automatically enabled at
    /// optimization_level >= 1.
    std::optional<bool> enable_fusing_conv2d_with_multiply_pattern = std::nullopt;

    /// Pipeline option: enable-d2m-fusing-pass   default: false
    /// Fuses device-to-memory (D2M) data movement operations into adjacent
    /// compute kernels where possible.
    std::optional<bool> enable_d2m_fusing = std::nullopt;

    // -------------------------------------------------------------------------
    // Execution options
    // -------------------------------------------------------------------------

    /// Pipeline option: enable-trace   default: false
    /// Enables TTNN program tracing.  The first inference call captures the
    /// compute graph; subsequent calls replay the captured trace, reducing
    /// host-side dispatch overhead for repeated inference.
    std::optional<bool> enable_trace = std::nullopt;

    // -------------------------------------------------------------------------
    // Workaround passes
    // -------------------------------------------------------------------------

    /// Pipeline option: disable-workarounds   default: false
    /// Disables all hardware compatibility workaround passes in one flag.
    /// Use only for debugging; likely to produce incorrect results on real
    /// Tenstorrent hardware.
    std::optional<bool> disable_workarounds = std::nullopt;

    /// Pipeline option: enable-layout-workaround-pass   default: true
    /// Inserts layout conversion ops required for operations whose expected
    /// input layout differs from their producer's output layout.
    /// Automatically overridden to false when enable_optimizer = true.
    std::optional<bool> enable_layout_workaround = std::nullopt;

    /// Pipeline option: enable-decomposition-workaround-pass   default: true
    /// Decomposes compound ops that are not natively supported on the target
    /// hardware into sequences of supported primitives.
    std::optional<bool> enable_decomposition_workaround = std::nullopt;

    // -------------------------------------------------------------------------
    // Performance metrics and diagnostics
    // -------------------------------------------------------------------------

    /// Pipeline option: ttnn-perf-metrics-enabled   default: false
    /// Instruments the compiled program to collect per-op TTNN performance
    /// counters (cycles, bandwidth, compute utilisation).
    std::optional<bool> enable_ttnn_perf_metrics = std::nullopt;

    /// Pipeline option: ttnn-perf-metrics-output-file   default: ""
    /// Path to the file where the TTNN performance report is written.
    /// When empty, the pipeline generates a filename based on the module or
    /// function name under the "perf_metrics" directory.
    /// Only meaningful when enable_ttnn_perf_metrics = true.
    std::optional<std::string> ttnn_perf_metrics_output_file = std::nullopt;

    /// Pipeline option: ttnn-perf-metrics-verbose-output-enabled   default: true
    /// Enables verbose mode in the TTNN performance metrics report, including
    /// per-operation details.
    std::optional<bool> enable_ttnn_perf_metrics_verbose = std::nullopt;

    // -------------------------------------------------------------------------
    // Custom configuration passthrough
    //
    // Any pipeline option not covered by the typed fields above can be supplied
    // as a raw, space-separated option string.  This string is appended verbatim
    // to the generated pipeline option string and takes precedence over any
    // conflicting structured field.
    // -------------------------------------------------------------------------
    std::string custom_config = "";

    MLIRConfig& set_optimization_level(int level)
    {
        if (level < 0 || level > 2)
            throw std::out_of_range("optimization_level must be 0, 1, or 2");
        optimization_level = level;
        return *this;
    }

    // Optimizer control
    MLIRConfig& set_enable_consteval(bool v)
    {
        enable_consteval = v;
        return *this;
    }
    MLIRConfig& set_enable_optimizer(bool v)
    {
        enable_optimizer = v;
        return *this;
    }
    MLIRConfig& set_enable_memory_layout_analysis(bool v)
    {
        enable_memory_layout_analysis = v;
        return *this;
    }

    // Memory layout options
    MLIRConfig& set_memory_layout_analysis_policy(MemoryLayoutAnalysisPolicy p)
    {
        memory_layout_analysis_policy = p;
        return *this;
    }
    MLIRConfig& set_enable_l1_interleaved_fallback_analysis(bool v)
    {
        enable_l1_interleaved_fallback_analysis = v;
        return *this;
    }
    MLIRConfig& set_enable_memreconfig(bool v)
    {
        enable_memreconfig = v;
        return *this;
    }

    MLIRConfig& set_max_legal_layouts(int64_t max)
    {
        if (max <= 0)
            throw std::out_of_range("max_legal_layouts must be greater than 0");
        max_legal_layouts = max;
        return *this;
    }
    MLIRConfig& set_enable_row_major(bool v)
    {
        enable_row_major = v;
        return *this;
    }

    MLIRConfig& set_compute_cfg_math_fidelity(tt::MathFidelity f)
    {
        compute_cfg_math_fidelity = f;
        return *this;
    }
    MLIRConfig& set_compute_cfg_fp32_dest_acc_en(bool v)
    {
        compute_cfg_fp32_dest_acc_en = v;
        return *this;
    }

    MLIRConfig& set_experimental_weight_dtype(tt::DataFormat d)
    {
        if (d != tt::DataFormat::Bfp8_b && d != tt::DataFormat::Bfp4_b)
            throw std::invalid_argument(
                "experimental_weight_dtype only accepts "
                "DataFormat::Bfp8_b or DataFormat::Bfp4_b");
        experimental_weight_dtype = d;
        return *this;
    }

    MLIRConfig& set_enable_erase_inverse_ops(bool v)
    {
        enable_erase_inverse_ops = v;
        return *this;
    }
    MLIRConfig& set_enable_implicit_broadcast_folding(bool v)
    {
        enable_implicit_broadcast_folding = v;
        return *this;
    }
    MLIRConfig& set_enable_permute_matmul_fusion(bool v)
    {
        enable_permute_matmul_fusion = v;
        return *this;
    }
    MLIRConfig& set_enable_dram_space_saving_optimization(bool v)
    {
        enable_dram_space_saving_optimization = v;
        return *this;
    }
    MLIRConfig& set_enable_remove_dead_values(bool v)
    {
        enable_remove_dead_values = v;
        return *this;
    }

    MLIRConfig& set_enable_cpu_hoisted_consteval(bool v)
    {
        enable_cpu_hoisted_consteval = v;
        return *this;
    }
    MLIRConfig& set_enable_consteval_inputs_to_system_memory(bool v)
    {
        enable_consteval_inputs_to_system_memory = v;
        return *this;
    }

    MLIRConfig& set_enable_fusing(bool v)
    {
        enable_fusing = v;
        return *this;
    }
    MLIRConfig& set_enable_d2m_fusing(bool v)
    {
        enable_d2m_fusing = v;
        return *this;
    }
    MLIRConfig& set_enable_fusing_conv2d_with_multiply_pattern(bool v)
    {
        enable_fusing_conv2d_with_multiply_pattern = v;
        return *this;
    }

    MLIRConfig& set_enable_trace(bool v)
    {
        enable_trace = v;
        return *this;
    }

    MLIRConfig& set_disable_workarounds(bool v)
    {
        disable_workarounds = v;
        return *this;
    }
    MLIRConfig& set_enable_layout_workaround(bool v)
    {
        enable_layout_workaround = v;
        return *this;
    }
    MLIRConfig& set_enable_decomposition_workaround(bool v)
    {
        enable_decomposition_workaround = v;
        return *this;
    }

    MLIRConfig& set_enable_ttnn_perf_metrics(bool v)
    {
        enable_ttnn_perf_metrics = v;
        return *this;
    }
    MLIRConfig& set_ttnn_perf_metrics_output_file(const std::string& path)
    {
        ttnn_perf_metrics_output_file = path;
        return *this;
    }
    MLIRConfig& set_enable_ttnn_perf_metrics_verbose(bool v)
    {
        enable_ttnn_perf_metrics_verbose = v;
        return *this;
    }

    MLIRConfig& set_custom_config(const std::string& config)
    {
        custom_config = config;
        return *this;
    }
};

void to_json(nlohmann::json& j, const MLIRConfig& p);
void from_json(const nlohmann::json& j, MLIRConfig& p);

std::string config_to_pipeline_options(const std::optional<MLIRConfig>& mlir_config);

}  // namespace tt::passes
