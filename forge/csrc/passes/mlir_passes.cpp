// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_passes.hpp"

// Standard headers
#include <stdexcept>

// Forge headers
#include "mlir_compiler.hpp"
#include "mlir_config.hpp"
#include "utils/logger.hpp"

// MLIR headers
#include "mlir/IR/BuiltinOps.h"

// TTMLIR headers
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace tt::passes
{

void register_mlir_passes()
{
    // Static (only once) initialization of the MLIR passes.
    static bool _ = []()
    {
        // Register required passes
        mlir::tt::ttir::registerPasses();
        mlir::tt::ttnn::registerPasses();

        // Register pass pipelines
        // This will internally register the pipelines in the MLIR pipeline registry. Then,
        // the registry can be used to lookup the pipeline by its name and add it to the pass manager.
        mlir::tt::ttnn::registerTTNNPipelines();

        return true;
    }();
    (void)_;
}

template <MLIROutputKind output>
void run_mlir_passes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, const std::optional<MLIRConfig> &mlir_config)
{
    // Register the MLIR passes.
    register_mlir_passes();

    // Create a pass manager.
    mlir::PassManager pm(mlir_module.get()->getName());

    // Get the pipeline info for the wanted pipeline.
    static_assert(
        output == MLIROutputKind::Flatbuffer || output == MLIROutputKind::Cpp || output == MLIROutputKind::SharedObject,
        "Unsupported MLIROutputKind: only Flatbuffer, Cpp, and SharedObject are handled.");

    const char *pipeline_name;
    if constexpr (output == MLIROutputKind::Flatbuffer)
    {
        pipeline_name = "ttir-to-ttnn-backend-pipeline";
    }
    else if constexpr (output == MLIROutputKind::Cpp)
    {
        pipeline_name = "ttir-to-emitc-pipeline";
    }
    else  // SharedObject
    {
        pipeline_name = "ttir-to-emitc-pipeline";
    }

    const auto pipelineInfo = mlir::PassPipelineInfo::lookup(pipeline_name);

    // Error handler for the pipeline. Will be called if there is an error during parsing of the pipeline options.
    auto err_handler = [](const mlir::Twine &location)
    {
        log_error(LogMLIRCompiler, "Error during parsing pipeline options: {}", location.str());
        return mlir::failure();
    };

    std::string options{config_to_pipeline_options(mlir_config)};

    // Add target-dylib flag for SharedObject output
    if constexpr (output == MLIROutputKind::SharedObject)
    {
        options += " target-dylib=true";
    }

    auto result = pipelineInfo->addToPipeline(pm, options, err_handler);
    if (mlir::failed(result))
    {
        throw std::runtime_error("Failed to add the pipeline to the pass manager!");
    }

    // Run the pass manager.
    if (mlir::failed(pm.run(mlir_module.get())))
    {
        throw std::runtime_error("Failed to run MLIR compiler pass pipeline.");
    }

#ifdef DEBUG
    // Create a string to store the output
    std::string moduleStr;
    llvm::raw_string_ostream rso(moduleStr);

    // Print the MLIR module
    mlir::OpPrintingFlags printFlags;
    printFlags.enableDebugInfo();
    mlir_module.get()->print(rso, printFlags);

    rso.flush();

    log_trace(LogMLIRCompiler, "MLIR module after running passes:\n{}", moduleStr);
#endif
}

// Explicit templates instantiation.
template void run_mlir_passes<MLIROutputKind::Flatbuffer>(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, const std::optional<MLIRConfig> &mlir_config);
template void run_mlir_passes<MLIROutputKind::Cpp>(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, const std::optional<MLIRConfig> &mlir_config);
template void run_mlir_passes<MLIROutputKind::SharedObject>(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, const std::optional<MLIRConfig> &mlir_config);

}  // namespace tt::passes
