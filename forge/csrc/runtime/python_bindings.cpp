// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime/python_bindings.hpp"

#include <stdexcept>

#include "runtime/runtime.hpp"
#include "runtime/state.hpp"
#include "runtime/testutils/testutils.hpp"
#include "runtime/tt_device.hpp"
#include "tt/runtime/types.h"

namespace tt
{

void RuntimeModule(py::module &m_runtime)
{
    // Main runtime APIs
    py::class_<runtime::Binary>(m_runtime, "Binary")
        .def("get_program_inputs", &runtime::Binary::getProgramInputs)
        .def("get_program_outputs", &runtime::Binary::getProgramOutputs)
        .def("store", &runtime::Binary::store)
        .def("as_json", &runtime::Binary::asJson);
    m_runtime.def("run_program", &tt::run_program);

    py::class_<Tensor>(m_runtime, "Tensor")
        .def(py::init<torch::Tensor &>())
        .def("to_torch", &Tensor::to_torch)
        .def("update_host_data", &Tensor::update_host_data)
        .def("detach_from_device", &Tensor::detach_from_device);
    py::class_<TensorPool>(m_runtime, "TensorPool")
        .def(py::init<>())
        .def("get_tensor", &TensorPool::get_tensor)
        .def(
            "insert",
            [](TensorPool &self, const std::string &name, torch::Tensor &tensor) { self.insert(name, tensor); })
        .def("update_tensor", &TensorPool::update_tensor);

    py::enum_<ProgramType>(m_runtime, "ProgramType")
        .value("Forward", ProgramType::Forward)
        .value("Backward", ProgramType::Backward)
        .value("Optimizer", ProgramType::Optimizer)
        .export_values();

    py::class_<ProgramState>(m_runtime, "ProgramState")
        .def(py::init<ProgramType, std::vector<Tensor>, std::vector<Tensor>>());

    py::class_<ModelState>(m_runtime, "ModelState")
        .def(py::init<runtime::Binary>())
        .def_property_readonly("tensor_pool", &ModelState::get_tensor_pool)
        .def(
            "init_program_state",
            [](ModelState &self, ProgramState &program_state) { self.add_program_state(program_state); })
        .def(
            "run_program",
            [](ModelState &self, ProgramType program_type, std::vector<tt::Tensor> &act_inputs)
            { self.run_program(program_type, act_inputs); })
        .def(
            "get_outputs",
            [](ModelState &self, ProgramType program_type)
            {
                std::optional<ProgramState> &opt_program_state = self.program_states[program_idx(program_type)];
                TT_ASSERT(opt_program_state.has_value(), "Program state for {} not initialized", program_type);
                return opt_program_state.value().outputs;
            });

    m_runtime.def("create_program_state", &tt::create_program_state);

    // Experimental APIs
    py::module m_experimental = m_runtime.def_submodule("experimental");
    py::class_<tt::DeviceSettings>(m_experimental, "DeviceSettings")
        .def(py::init<>())
        .def_readwrite("enable_program_cache", &tt::DeviceSettings::enable_program_cache);

    // TTDevice bindings
    py::class_<tt::TTDevice, std::shared_ptr<tt::TTDevice>>(m_experimental, "TTDevice")
        .def_readonly("arch", &tt::TTDevice::arch)
        .def_readonly("mmio", &tt::TTDevice::mmio)
        .def_readonly("index", &tt::TTDevice::index)
        .def("is_open", &tt::TTDevice::is_open)
        .def("open_device", &tt::TTDevice::open_device, py::arg("settings") = DeviceSettings())
        .def("close_device", &tt::TTDevice::close_device)
        .def("configure_device", &tt::TTDevice::configure_device, py::arg("settings") = DeviceSettings());

    // TTSystem bindings
    py::class_<tt::TTSystem>(m_experimental, "TTSystem")
        .def_readonly("chip_ids", &tt::TTSystem::chip_ids)
        .def_readonly("devices", &tt::TTSystem::devices)
        .def("close_devices", &tt::TTSystem::close_devices)
        .def("configure_devices", &tt::TTSystem::configure_devices, py::arg("settings") = DeviceSettings())
        .def_static(
            "get_system",
            &tt::TTSystem::get_system,
            py::return_value_policy::reference,
            "Get the singleton TTSystem instance")
        .def_static(
            "is_initialized", &tt::TTSystem::is_initialized, "Check if the TTSystem singleton has been initialized");

    m_experimental.def(
        "configure_devices",
        [](DeviceSettings settings) -> void { tt::TTSystem::get_system().configure_devices(settings); },
        py::arg("device_settings") = DeviceSettings(),
        "Configure all devices with the given settings.");

    m_experimental.def(
        "save_system_desc",
        [](const std::string &path) -> void
        {
            if (path.empty())
            {
                throw std::invalid_argument("save_system_desc: path must not be empty");
            }
            // Opens a device if one is not already open, which is the point: the
            // descriptor being captured is the attached device's.
            tt::TTSystem &system = tt::TTSystem::get_system();
            TT_ASSERT(system.system_desc.handle, "System descriptor handle is null");
            system.system_desc.store(path.c_str());
        },
        py::arg("path"),
        "Write the attached device's system descriptor to a .ttsys flatbuffer.\n"
        "\n"
        "The counterpart to MLIRConfig.set_system_desc_path: capture once on the\n"
        "machine that has the device (including a simulated one), then compile\n"
        "against that exact descriptor anywhere. Equivalent to `ttrt query\n"
        "--save-artifacts` without needing ttrt installed.");

    // TestUtils APIs
    py::module m_testutils = m_runtime.def_submodule("testutils");
    m_testutils
        .def(
            "test_so",
            [](std::string so_path,
               std::string func_name,
               std::vector<tt::Tensor> &inputs,
               std::vector<tt::Tensor> &consts_and_params,
               std::vector<tt::Tensor> &outputs)
            { return tt::runtime::testutils::test_so(so_path, func_name, inputs, consts_and_params, outputs); })
        .def(
            "get_persistent_inputs",
            [](ProgramType program_type, ModelState &model_state) -> std::vector<tt::Tensor>
            {
                std::optional<ProgramState> &opt_program_state = model_state.program_states[program_idx(program_type)];
                TT_ASSERT(opt_program_state.has_value(), "Program state for {} not initialized", program_type);
                return opt_program_state.value().persistent_inputs;
            });
}

}  // namespace tt
