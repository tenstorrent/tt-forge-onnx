// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "runtime/state.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <utils/logger.hpp>

#include "tt/runtime/runtime.h"

namespace tt
{

std::ostream& operator<<(std::ostream& os, ProgramType program_type)
{
    switch (program_type)
    {
        case ProgramType::Forward: os << "Forward"; break;
        case ProgramType::Backward: os << "Backward"; break;
        case ProgramType::Optimizer: os << "Optimizer"; break;
        default: os << "Unknown"; break;
    }
    return os;
}

ProgramState create_program_state(
    ProgramType program_type, const TensorPool& tensor_pool, std::vector<std::string> persistent_input_names)
{
    std::vector<tt::Tensor> persistent_inputs;
    persistent_inputs.reserve(persistent_input_names.size());

    for (auto& name : persistent_input_names)
    {
        auto tensor = tensor_pool.get_tensor(name);
        persistent_inputs.emplace_back(tensor);
    }
    return ProgramState{program_type, persistent_inputs, {}};
}

void ModelState::run_program(ProgramType program_type, std::vector<tt::Tensor> act_inputs)
{
    // ISSUE(#1346): So far, the device_id is hardcoded to 0 - make it user-configurable.
    constexpr size_t device_id = 0;
    size_t pg_id = program_idx(program_type);
    std::optional<ProgramState>& opt_program_state = program_states[pg_id];

    TT_ASSERT(opt_program_state.has_value(), "Program state for {} not initialized", program_type);

    if (!TTSystem::get_system().devices[device_id]->is_open())
    {
        TTSystem::get_system().devices[device_id]->open_device();
    }

    auto& program_state = opt_program_state.value();

    // Clear the outputs from the previous run.
    program_state.outputs.clear();

    std::vector<tt::Tensor> inputs;
    inputs.reserve(act_inputs.size() + program_state.persistent_inputs.size());

    // Convert each input to the layout expected by the compiled program.
    // This handles all necessary transitions: host->device, device->host, and layout changes.
    //
    // NOTE: there is an ordering requirement for the activation inputs and the persistent inputs, i.e.
    // in the input list (`inputs` vector here), the activation inputs come first. Unfortunately, there isn't any
    // mechanism which enforces this, yet. It's an informal contract between the compiler and the runtime.
    static const bool host_timing = std::getenv("FORGE_HOST_TIMING") != nullptr;
    using clk = std::chrono::steady_clock;
    const auto ht0 = clk::now();
    size_t input_idx = 0;
    for (auto tensor : act_inputs)
    {
        auto layout = tt::runtime::getLayout(binary, pg_id, input_idx);
        tensor.to_layout(device_id, layout);
        inputs.emplace_back(tensor);
        input_idx++;
    }
    const auto ht1 = clk::now();

    // Convert each persistent to the expected layout.
    for (auto& persistent_input : program_state.persistent_inputs)
    {
        auto layout = tt::runtime::getLayout(binary, pg_id, input_idx);
        persistent_input.to_layout(device_id, layout);
        inputs.emplace_back(persistent_input);
        input_idx++;
    }
    const auto ht2 = clk::now();

    program_state.outputs = ::tt::run_program(binary, pg_id, inputs);
    if (host_timing)
    {
        const auto ht3 = clk::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::fprintf(
            stderr,
            "[HOST_TIMING] forge run_program: act_to_layout=%.2f ms (n=%zu) persistent_to_layout=%.2f ms (n=%zu) submit=%.2f ms\n",
            ms(ht0, ht1), act_inputs.size(), ms(ht1, ht2), program_state.persistent_inputs.size(), ms(ht2, ht3));
    }
}

};  // namespace tt
