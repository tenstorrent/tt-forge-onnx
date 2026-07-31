// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Shared utilities for EmitC model runners.

#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "runtime/tensor.hpp"
#include "runtime/tt_device.hpp"
#include "tt/runtime/runtime.h"
#include "ttmlir/Target/Common/types_generated.h"

// ANSI colors — automatically disabled when stdout is not a TTY.
namespace color {
    void        init(bool force_off);
    const char* reset();
    const char* bold();
    const char* green();
    const char* cyan();
    const char* pass_tag();
    const char* fail_tag();
}

// Terminal layout
static constexpr int kLineWidth = 62;
void print_separator(char c = '=');
void print_phase(int step, int total, const char* title);

// TTNN binary loader (.bin files produced by the EmitC compilation scripts)
std::vector<torch::Tensor> load_tensor_list(const std::string& path);
const char*                dtype_name(c10::ScalarType st);
std::string                format_shape(const torch::Tensor& t);

// Tensor conversion between torch and TT runtime representations
tt::runtime::Tensor make_runtime_tensor(const torch::Tensor& t);
torch::Tensor       runtime_tensor_to_torch(const tt::runtime::Tensor& rt);

// PCC-based output validation
bool validate_outputs(
    const std::vector<tt::runtime::Tensor>& device_host,
    const std::vector<torch::Tensor>&       golden,
    float                                   threshold);

// Open device 0; enable_program_cache=true is required whenever the .so uses
// trace capture (which all EmitC-generated .so files currently do).
tt::runtime::Device open_device(bool enable_program_cache = false);
