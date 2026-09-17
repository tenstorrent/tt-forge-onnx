// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

// MLIRConfig must be complete here: std::optional<MLIRConfig> is instantiated in
// the signature below, and a forward declaration is not enough for that.
#include "passes/mlir_config.hpp"

namespace tt
{
class ForgeGraphModule;
}

namespace mlir
{
class MLIRContext;
class ModuleOp;
template <typename OpTy>
class OwningOpRef;
}  // namespace mlir

namespace tt::passes
{
// Public API for generating MLIR from a Forge module (set of graphs).
//
// When mlir_config names a target arch or a system descriptor path, the module is
// left without a ttcore.system_desc attribute so the tt-mlir pipeline can build
// the descriptor from those options instead. Otherwise the descriptor is read from
// the attached device, which requires one to be present.
mlir::OwningOpRef<mlir::ModuleOp> lower_to_mlir(
    tt::ForgeGraphModule& module,
    mlir::MLIRContext& context,
    const std::optional<MLIRConfig>& mlir_config = std::nullopt);
}  // namespace tt::passes
