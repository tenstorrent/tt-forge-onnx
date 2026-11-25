// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Device barrier for the EmitC runners.
//
// A runner needs to know when the device has *finished* the model call, not just
// when the host returned from it (the EmitC .so only enqueues work). The runtime in
// this checkout has no such entry point:
//   - tt::runtime::wait(tensors) returns immediately for tensors that carry no
//     MeshEvent, which is every tensor the EmitC .so returns
//     (runtime/lib/ttnn/runtime.cpp, wait(): `if (!meshEvent.has_value()) return;`).
//   - tt::runtime::deviceSynchronize exists only in a later tt-mlir.
//
// tt-metal does export the barrier we need:
//   tt::tt_metal::distributed::Synchronize(MeshDevice*, std::optional<uint8_t>,
//                                          std::span<const SubDeviceId>)
// It is declared here instead of including <tt-metalium/distributed.hpp>, which
// would pull the full tt-metal/ttnn include tree (fmt, reflect, umd, ...) into the
// runner build. The two headers below are header-only and dependency-free, and they
// give the exact parameter types, so the mangled name matches the exported symbol
// (verified with `nm -DC libtt_metal.so | grep distributed::Synchronize`).
//
// The runtime Device handle stores a ::ttnn::MeshDevice (an alias of
// tt::tt_metal::distributed::MeshDevice) behind a shared_ptr<void>; Device::as<T>()
// is a static_cast from void*, which is valid for an incomplete T.

#include <cstdint>
#include <optional>
#include <span>
#include <tt-metalium/sub_device_types.hpp>

#include "emitc_utils.hpp"

namespace tt::tt_metal::distributed
{
class MeshDevice;
void Synchronize(
    MeshDevice* device, std::optional<uint8_t> cq_id, std::span<const tt::tt_metal::SubDeviceId> sub_device_ids);
}  // namespace tt::tt_metal::distributed

void device_synchronize(tt::runtime::Device device)
{
    auto& mesh = device.as<tt::tt_metal::distributed::MeshDevice>(tt::runtime::DeviceRuntime::TTNN);
    // cq_id = nullopt: wait for every command queue, as the runtime's own
    // deviceSynchronize does in later tt-mlir.
    tt::tt_metal::distributed::Synchronize(&mesh, std::nullopt, {});
}
