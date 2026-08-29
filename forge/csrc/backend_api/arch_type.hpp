// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace tt
{
enum class ARCH
{
    JAWBRIDGE = 0,
    WORMHOLE = 2,
    WORMHOLE_B0 = 3,
    BLACKHOLE = 4,
    // NOTE: this enum is forge-local and its values intentionally do NOT match
    // UMD's tt::ARCH (WORMHOLE_B0=2, BLACKHOLE=3, QUASAR=4). QUASAR is 5 here
    // only to avoid colliding with forge's BLACKHOLE=4.
    QUASAR = 5,
    Invalid = 0xFF,
};

std::string to_string_arch(ARCH ar);
std::string to_string_arch_lower(ARCH arch);
ARCH to_arch_type(const std::string& arch_string);
}  // namespace tt
