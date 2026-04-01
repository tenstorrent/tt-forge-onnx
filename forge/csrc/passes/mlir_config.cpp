// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_config.hpp"

#include <sstream>

#include "nlohmann/json.hpp"
#include "shared_utils/json_extension.hpp"

namespace tt::passes
{

void to_json(nlohmann::json& j, MemoryLayoutAnalysisPolicy p) { j = to_pipeline_string(p); }

void from_json(const nlohmann::json& j, MemoryLayoutAnalysisPolicy& p)
{
    p = memory_layout_policy_from_string(j.get<std::string>());
}

void to_json(::nlohmann::json& j, const MLIRConfig& p)
{
    const auto fidelity_val = [&]() -> nlohmann::json
    {
        if (!p.compute_cfg_math_fidelity.has_value())
            return nullptr;
        return to_pipeline_string(*p.compute_cfg_math_fidelity);
    }();

    const auto weight_dtype_val = [&]() -> nlohmann::json
    {
        if (!p.experimental_weight_dtype.has_value())
            return nullptr;
        return weight_dtype_to_pipeline_string(*p.experimental_weight_dtype);
    }();

    j = nlohmann::json{// Optimization level shorthand
                       {"optimization_level", p.optimization_level},
                       // Optimizer control
                       {"enable_consteval", p.enable_consteval},
                       {"enable_optimizer", p.enable_optimizer},
                       {"enable_memory_layout_analysis", p.enable_memory_layout_analysis},
                       // Memory layout options
                       {"memory_layout_analysis_policy", p.memory_layout_analysis_policy},
                       {"enable_l1_interleaved_fallback_analysis", p.enable_l1_interleaved_fallback_analysis},
                       {"enable_memreconfig", p.enable_memreconfig},
                       {"max_legal_layouts", p.max_legal_layouts},
                       {"enable_row_major", p.enable_row_major},
                       // Compute kernel configuration
                       {"compute_cfg_math_fidelity", fidelity_val},
                       {"compute_cfg_fp32_dest_acc_en", p.compute_cfg_fp32_dest_acc_en},
                       // Data type / quantization options
                       {"experimental_weight_dtype", weight_dtype_val},
                       // Graph transformation passes
                       {"enable_erase_inverse_ops", p.enable_erase_inverse_ops},
                       {"enable_implicit_broadcast_folding", p.enable_implicit_broadcast_folding},
                       {"enable_permute_matmul_fusion", p.enable_permute_matmul_fusion},
                       {"enable_dram_space_saving_optimization", p.enable_dram_space_saving_optimization},
                       {"enable_remove_dead_values", p.enable_remove_dead_values},
                       // Const-eval options
                       {"enable_cpu_hoisted_consteval", p.enable_cpu_hoisted_consteval},
                       {"enable_consteval_inputs_to_system_memory", p.enable_consteval_inputs_to_system_memory},
                       // Fusing passes
                       {"enable_fusing", p.enable_fusing},
                       {"enable_d2m_fusing", p.enable_d2m_fusing},
                       {"enable_fusing_conv2d_with_multiply_pattern", p.enable_fusing_conv2d_with_multiply_pattern},
                       // Execution options
                       {"enable_trace", p.enable_trace},
                       // Workaround passes
                       {"disable_workarounds", p.disable_workarounds},
                       {"enable_layout_workaround", p.enable_layout_workaround},
                       {"enable_decomposition_workaround", p.enable_decomposition_workaround},
                       // Performance metrics and diagnostics
                       {"enable_ttnn_perf_metrics", p.enable_ttnn_perf_metrics},
                       {"ttnn_perf_metrics_output_file", p.ttnn_perf_metrics_output_file},
                       {"enable_ttnn_perf_metrics_verbose", p.enable_ttnn_perf_metrics_verbose},
                       // Custom configuration passthrough
                       {"custom_config", p.custom_config}};
}

void from_json(const ::nlohmann::json& j, MLIRConfig& p)
{
    // Optimization level shorthand
    j.at("optimization_level").get_to(p.optimization_level);
    // Optimizer control
    j.at("enable_consteval").get_to(p.enable_consteval);
    j.at("enable_optimizer").get_to(p.enable_optimizer);
    j.at("enable_memory_layout_analysis").get_to(p.enable_memory_layout_analysis);
    // Memory layout options
    j.at("memory_layout_analysis_policy").get_to(p.memory_layout_analysis_policy);
    j.at("enable_l1_interleaved_fallback_analysis").get_to(p.enable_l1_interleaved_fallback_analysis);
    j.at("enable_memreconfig").get_to(p.enable_memreconfig);
    j.at("max_legal_layouts").get_to(p.max_legal_layouts);
    j.at("enable_row_major").get_to(p.enable_row_major);
    // Compute kernel configuration
    {
        const auto& jf = j.at("compute_cfg_math_fidelity");
        p.compute_cfg_math_fidelity =
            jf.is_null() ? std::nullopt : std::make_optional(math_fidelity_from_string(jf.get<std::string>()));
    }
    j.at("compute_cfg_fp32_dest_acc_en").get_to(p.compute_cfg_fp32_dest_acc_en);
    {
        const auto& jd = j.at("experimental_weight_dtype");
        p.experimental_weight_dtype =
            jd.is_null() ? std::nullopt : std::make_optional(weight_dtype_from_string(jd.get<std::string>()));
    }
    // Graph transformation passes
    j.at("enable_erase_inverse_ops").get_to(p.enable_erase_inverse_ops);
    j.at("enable_implicit_broadcast_folding").get_to(p.enable_implicit_broadcast_folding);
    j.at("enable_permute_matmul_fusion").get_to(p.enable_permute_matmul_fusion);
    j.at("enable_dram_space_saving_optimization").get_to(p.enable_dram_space_saving_optimization);
    j.at("enable_remove_dead_values").get_to(p.enable_remove_dead_values);
    // Const-eval options
    j.at("enable_cpu_hoisted_consteval").get_to(p.enable_cpu_hoisted_consteval);
    j.at("enable_consteval_inputs_to_system_memory").get_to(p.enable_consteval_inputs_to_system_memory);
    // Fusing passes
    j.at("enable_fusing").get_to(p.enable_fusing);
    j.at("enable_d2m_fusing").get_to(p.enable_d2m_fusing);
    j.at("enable_fusing_conv2d_with_multiply_pattern").get_to(p.enable_fusing_conv2d_with_multiply_pattern);
    // Execution options
    j.at("enable_trace").get_to(p.enable_trace);
    // Workaround passes
    j.at("disable_workarounds").get_to(p.disable_workarounds);
    j.at("enable_layout_workaround").get_to(p.enable_layout_workaround);
    j.at("enable_decomposition_workaround").get_to(p.enable_decomposition_workaround);
    // Performance metrics and diagnostics
    j.at("enable_ttnn_perf_metrics").get_to(p.enable_ttnn_perf_metrics);
    j.at("ttnn_perf_metrics_output_file").get_to(p.ttnn_perf_metrics_output_file);
    j.at("enable_ttnn_perf_metrics_verbose").get_to(p.enable_ttnn_perf_metrics_verbose);
    // Custom configuration passthrough
    j.at("custom_config").get_to(p.custom_config);
}

std::string config_to_pipeline_options(const std::optional<MLIRConfig>& mlir_config)
{
    std::stringstream options;

    if (!mlir_config.has_value())
        return options.str();

    // -----------------------------------------------------------------------
    // Optimization level shorthand
    // -----------------------------------------------------------------------
    if (mlir_config->optimization_level.has_value())
        options << " optimization-level=" << *mlir_config->optimization_level;

    // -----------------------------------------------------------------------
    // Optimizer control
    // -----------------------------------------------------------------------
    if (mlir_config->enable_consteval.has_value())
        options << " enable-const-eval=" << *mlir_config->enable_consteval;
    if (mlir_config->enable_optimizer.has_value())
        options << " enable-optimizer=" << *mlir_config->enable_optimizer;
    if (mlir_config->enable_memory_layout_analysis.has_value())
        options << " memory-layout-analysis-enabled=" << *mlir_config->enable_memory_layout_analysis;

    // -----------------------------------------------------------------------
    // Memory layout options
    // -----------------------------------------------------------------------
    if (mlir_config->memory_layout_analysis_policy.has_value())
        options << " memory-layout-analysis-policy=" << to_pipeline_string(*mlir_config->memory_layout_analysis_policy);
    if (mlir_config->enable_l1_interleaved_fallback_analysis.has_value())
        options << " l1-interleaved-fallback-analysis-enabled="
                << *mlir_config->enable_l1_interleaved_fallback_analysis;
    if (mlir_config->enable_memreconfig.has_value())
        options << " memreconfig-enabled=" << *mlir_config->enable_memreconfig;
    if (mlir_config->max_legal_layouts.has_value())
        options << " max-legal-layouts=" << *mlir_config->max_legal_layouts;
    if (mlir_config->enable_row_major.has_value())
        options << " row-major-enabled=" << *mlir_config->enable_row_major;

    // -----------------------------------------------------------------------
    // Compute kernel configuration
    // -----------------------------------------------------------------------
    if (mlir_config->compute_cfg_math_fidelity.has_value())
        options << " compute-cfg-math-fidelity=" << to_pipeline_string(*mlir_config->compute_cfg_math_fidelity);
    if (mlir_config->compute_cfg_fp32_dest_acc_en.has_value())
        options << " compute-cfg-fp32-dest-acc-en=" << *mlir_config->compute_cfg_fp32_dest_acc_en;

    // -----------------------------------------------------------------------
    // Data type / quantization options
    // -----------------------------------------------------------------------
    if (mlir_config->experimental_weight_dtype.has_value())
        options << " experimental-weight-dtype="
                << weight_dtype_to_pipeline_string(*mlir_config->experimental_weight_dtype);

    // -----------------------------------------------------------------------
    // Graph transformation passes
    // -----------------------------------------------------------------------
    if (mlir_config->enable_erase_inverse_ops.has_value())
        options << " enable-erase-inverse-ops-pass=" << *mlir_config->enable_erase_inverse_ops;
    if (mlir_config->enable_implicit_broadcast_folding.has_value())
        options << " enable-implicit-broadcast-folding-pass=" << *mlir_config->enable_implicit_broadcast_folding;
    if (mlir_config->enable_permute_matmul_fusion.has_value())
        options << " enable-permute-matmul-fusion=" << *mlir_config->enable_permute_matmul_fusion;
    if (mlir_config->enable_dram_space_saving_optimization.has_value())
        options << " enable-dram-space-saving-optimization-pass="
                << *mlir_config->enable_dram_space_saving_optimization;
    if (mlir_config->enable_remove_dead_values.has_value())
        options << " enable-remove-dead-values=" << *mlir_config->enable_remove_dead_values;

    // -----------------------------------------------------------------------
    // Const-eval options
    // -----------------------------------------------------------------------
    if (mlir_config->enable_cpu_hoisted_consteval.has_value())
        options << " enable-cpu-hoisted-const-eval=" << *mlir_config->enable_cpu_hoisted_consteval;
    if (mlir_config->enable_consteval_inputs_to_system_memory.has_value())
        options << " enable-const-eval-inputs-to-system-memory="
                << *mlir_config->enable_consteval_inputs_to_system_memory;

    // -----------------------------------------------------------------------
    // Fusing passes
    // -----------------------------------------------------------------------
    if (mlir_config->enable_fusing.has_value())
        options << " enable-fusing-pass=" << *mlir_config->enable_fusing;
    if (mlir_config->enable_d2m_fusing.has_value())
        options << " enable-d2m-fusing-pass=" << *mlir_config->enable_d2m_fusing;
    if (mlir_config->enable_fusing_conv2d_with_multiply_pattern.has_value())
        options << " enable-fusing-conv2d-with-multiply-pattern="
                << *mlir_config->enable_fusing_conv2d_with_multiply_pattern;

    // -----------------------------------------------------------------------
    // Execution options
    // -----------------------------------------------------------------------
    if (mlir_config->enable_trace.has_value())
        options << " enable-trace=" << *mlir_config->enable_trace;

    // -----------------------------------------------------------------------
    // Workaround passes
    // -----------------------------------------------------------------------
    if (mlir_config->disable_workarounds.has_value())
        options << " disable-workarounds=" << *mlir_config->disable_workarounds;
    if (mlir_config->enable_layout_workaround.has_value())
        options << " enable-layout-workaround-pass=" << *mlir_config->enable_layout_workaround;
    if (mlir_config->enable_decomposition_workaround.has_value())
        options << " enable-decomposition-workaround-pass=" << *mlir_config->enable_decomposition_workaround;

    // -----------------------------------------------------------------------
    // Performance metrics and diagnostics
    // -----------------------------------------------------------------------
    if (mlir_config->enable_ttnn_perf_metrics.has_value())
        options << " ttnn-perf-metrics-enabled=" << *mlir_config->enable_ttnn_perf_metrics;
    if (mlir_config->ttnn_perf_metrics_output_file.has_value() && !mlir_config->ttnn_perf_metrics_output_file->empty())
        options << " ttnn-perf-metrics-output-file=" << *mlir_config->ttnn_perf_metrics_output_file;
    if (mlir_config->enable_ttnn_perf_metrics_verbose.has_value())
        options << " ttnn-perf-metrics-verbose-output-enabled=" << *mlir_config->enable_ttnn_perf_metrics_verbose;

    // -----------------------------------------------------------------------
    // Custom configuration passthrough
    // -----------------------------------------------------------------------
    if (!mlir_config->custom_config.empty())
        options << " " << mlir_config->custom_config;

    return options.str();
}

}  // namespace tt::passes
