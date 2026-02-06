// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_device.hpp"

#include <optional>

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "utils/assert.hpp"
#include "utils/logger.hpp"

namespace tt
{

static bool system_is_initialized = false;

TTSystem detect_available_devices()
{
    auto system_desc = runtime::getCurrentSystemDesc();
    std::vector<int> chip_ids;
    std::vector<std::shared_ptr<TTDevice>> devices;
    int logical_device_index = 0;
    ARCH arch = ARCH::Invalid;
    for (std::uint32_t chip_desc_index : *system_desc->chip_desc_indices())
    {
        chip_ids.push_back(static_cast<int>(chip_desc_index));
        target::ChipDesc const* chip_desc = system_desc->chip_descs()->Get(chip_desc_index);
        target::ChipCapability chip_capabilities = system_desc->chip_capabilities()->Get(logical_device_index);

        bool mmio = bool(chip_capabilities & target::ChipCapability::HostMMIO);
        if (not mmio)
        {
            continue;
        }

        switch (chip_desc->arch())
        {
            case target::Arch::Wormhole_b0: arch = ARCH::WORMHOLE_B0; break;
            case target::Arch::Blackhole: arch = ARCH::BLACKHOLE; break;
            default: log_fatal(LogTTDevice, "Unknown chip type {}", target::EnumNameArch(chip_desc->arch()));
        }

        auto device = std::make_shared<TTDevice>(std::nullopt, system_desc, arch, mmio, logical_device_index);
        devices.push_back(device);
        ++logical_device_index;
    }

    system_is_initialized = true;
    return TTSystem{system_desc, chip_ids, devices};
}

TTSystem& TTSystem::get_system()
{
    static TTSystem system = detect_available_devices();
    return system;
}

bool TTSystem::is_initialized() { return system_is_initialized; }

void TTDevice::open_device(const DeviceSettings& settings)
{
    TT_ASSERT(!is_open());
    static constexpr std::uint32_t num_hw_cqs = 1;
    runtime::MeshDeviceOptions options;
    options.numHWCQs = num_hw_cqs;
    options.enableProgramCache = settings.enable_program_cache;
    options.meshShape = {1, 1};
    rt_device = runtime::openMeshDevice(options);
}

void TTDevice::close_device()
{
    TT_ASSERT(is_open());
    runtime::closeMeshDevice(rt_device.value());
    rt_device.reset();
}

void TTDevice::configure_device(const DeviceSettings& settings)
{
    if (is_open())
    {
        close_device();
    }

    open_device(settings);
}

}  // namespace tt
