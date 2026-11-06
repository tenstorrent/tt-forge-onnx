// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/python.h>
#include <torch/torch.h>

#include "tensor.hpp"
#include "tt/runtime/types.h"

namespace tt
{

// Helper function to load the binary from the file and run a program - might be useful for testing/debugging.
runtime::Binary load_binary_from_file(std::string const& filename);

std::vector<tt::Tensor> run_program_from_file(
    std::string const& filename, int program_idx, std::vector<torch::Tensor> const& inputs);

// Entry point for invoking tt-mlir runtime and running the specific program from the binary on the device.
std::vector<tt::Tensor> run_program(runtime::Binary& binary, int program_idx, std::vector<tt::Tensor>& inputs);

}  // namespace tt
