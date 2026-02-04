// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_compiler.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "graph_lib/defines.hpp"
#include "llvm/Support/raw_ostream.h"
#include "lower_to_mlir.hpp"
#include "mlir_passes.hpp"
#include "shared_utils/json_extension.hpp"

// Forge headers
#include "graph_lib/graph.hpp"
#include "nlohmann/json.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include "graph_lib/node_types.hpp"
#pragma clang diagnostic pop
#include "utils/logger.hpp"

// MLIR headers
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#include "mlir/InitAllDialects.h"
#pragma clang diagnostic pop

// TTMLIR headers
#include "compile_so.hpp"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "tt/runtime/types.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToCpp.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

// Reportify headers
#include "reportify/reportify.hpp"

namespace fs = std::filesystem;

namespace tt::passes
{

// Template function to run the MLIR compiler pipeline, depending on the desired output.
template <MLIROutputKind output>
auto run_mlir_compiler_generic(tt::ForgeGraphModule& module, const std::optional<MLIRConfig>& mlir_config)
{
    // Register all the required dialects.
    mlir::DialectRegistry registry;

    registry.insert<
        mlir::tt::ttcore::TTCoreDialect,
        mlir::tt::ttir::TTIRDialect,
        mlir::tt::ttnn::TTNNDialect,
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect,
        mlir::ml_program::MLProgramDialect,
        mlir::tensor::TensorDialect,
        mlir::LLVM::LLVMDialect>();

    mlir::func::registerInlinerExtension(registry);

    // Register the LLVM dialect inliner extension
    mlir::LLVM::registerInlinerInterface(registry);

    // Register BufferizableOpInterface for dialects that may be used during bufferization.
    // This is required for bufferization passes that run during CPU-hoisting const-eval.
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

    // Register SubsetInsertionOpInterface for dialects that support subset operations.
    // This is also required for bufferization passes during CPU-hoisting const-eval.
    mlir::linalg::registerSubsetOpInterfaceExternalModels(registry);
    mlir::tensor::registerSubsetOpInterfaceExternalModels(registry);

    // Register LLVM translation interfaces for all dialects.
    // This is required to convert MLIR operations to LLVM IR
    mlir::registerAllToLLVMIRTranslations(registry);

    // Create a context with all registered dialects.
    mlir::MLIRContext context(registry);

#ifdef DEBUG
    // Context setting to have mlir print out stacktrace whenever errors occur
    context.printStackTraceOnDiagnostic(true);
#endif

    // Load all available dialects
    context.loadAllAvailableDialects();

    // Generate MLIR from the Forge graph.
    mlir::OwningOpRef<mlir::ModuleOp> mlir_module = lower_to_mlir(module, context);

    // Run MLIR pipeline.
    run_mlir_passes<output>(mlir_module, mlir_config);

    tt::log_info(LogMLIRCompiler, "MLIR passes run successfully.");

    // Dump TTNN module only at trace log level
    std::string moduleStr;
    llvm::raw_string_ostream rso(moduleStr);
    mlir_module->print(rso);
    rso.flush();
    log_trace(LogMLIRCompiler, "TTNN module after running mlir passes:\n{}", moduleStr);

    if constexpr (output == MLIROutputKind::Flatbuffer)
    {
        // Save generated ttnn module to a file named "{name}.mlir".
        reportify::dump_mlir("ttnn", mlir_module->getName()->str(), mlir_module.get());

        // Generate binary from the MLIR module.
        auto binary = mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());
        if (binary == nullptr)
        {
            throw std::runtime_error("Failed to generate flatbuffer binary.");
        }

        tt::log_info(LogMLIRCompiler, "Flatbuffer binary generated successfully.");
        return binary;
    }
    else if constexpr (output == MLIROutputKind::Cpp)
    {
        std::string cpp_source;
        llvm::raw_string_ostream rso(cpp_source);

        log_info(LogMLIRCompiler, "Generating C++ code from MLIR module.");
        auto res = mlir::emitc::translateToCpp(mlir_module.get(), rso);
        if (mlir::failed(res))
        {
            throw std::runtime_error("Failed to generate C++ code.");
        }

        rso.flush();

        tt::log_info(LogMLIRCompiler, "C++ code generated successfully.");
        return cpp_source;
    }
    else if constexpr (output == MLIROutputKind::SharedObject)
    {
        std::string cpp_source;
        llvm::raw_string_ostream rso(cpp_source);

        log_info(LogMLIRCompiler, "Generating a shared object from MLIR module.");

        mlir::LogicalResult res = mlir::emitc::translateToCpp(mlir_module.get(), rso);
        if (mlir::failed(res))
        {
            throw std::runtime_error("Failed to generate C++ code.");
        }
        rso.flush();

        tt::log_info(LogMLIRCompiler, "C++ code for SharedObject generated successfully.");

        const char* TT_METAL_RUNTIME_ROOT = std::getenv("TT_METAL_RUNTIME_ROOT");
        const char* FORGE_HOME = std::getenv("FORGE_HOME");
        if (TT_METAL_RUNTIME_ROOT == nullptr)
        {
            throw std::runtime_error("TT_METAL_RUNTIME_ROOT environment variable is not set.");
        }
        if (FORGE_HOME == nullptr)
        {
            throw std::runtime_error("FORGE_HOME environment variable is not set.");
        }

        fs::path metal_src_dir;
        fs::path metal_lib_dir;
        fs::path standalone_dir;
        if (std::getenv("FORGE_IN_WHEEL"))
        {
            metal_src_dir = fs::path(std::string(TT_METAL_RUNTIME_ROOT));
            metal_lib_dir = fs::path(std::string(FORGE_HOME)) / "forge/lib";
            standalone_dir = fs::path(std::string(FORGE_HOME)) / "forge/tools/ttnn-standalone";
        }
        else if (std::getenv("FORGE_IN_SOURCE"))
        {
            metal_src_dir = fs::path(std::string(TT_METAL_RUNTIME_ROOT));
            metal_lib_dir = fs::path(std::string(TT_METAL_RUNTIME_ROOT)).parent_path() / "tt-metal/build/lib";
            standalone_dir =
                fs::path(std::string(FORGE_HOME)).parent_path() / "third_party/tt-mlir/tools/ttnn-standalone";
        }
        else
        {
            throw std::runtime_error("Neither FORGE_IN_WHEEL nor FORGE_IN_SOURCE environment variables are set.");
        }

        std::string soPathStr = compileCppToSo(
            cpp_source, "/tmp/", metal_src_dir.string(), metal_lib_dir.string(), standalone_dir.string());

        tt::log_info(LogMLIRCompiler, "SharedObject generated successfully at path: {}.", soPathStr);

        return soPathStr;
    }
}

runtime::Binary run_mlir_compiler(tt::ForgeGraphModule& module, const std::optional<MLIRConfig>& mlir_config)
{
    return run_mlir_compiler_generic<MLIROutputKind::Flatbuffer>(module, mlir_config);
}

std::string run_mlir_compiler_to_cpp(tt::ForgeGraphModule& module, const std::optional<MLIRConfig>& mlir_config)
{
    return run_mlir_compiler_generic<MLIROutputKind::Cpp>(module, mlir_config);
}

std::string run_mlir_compiler_to_shared_object(
    tt::ForgeGraphModule& module, const std::optional<MLIRConfig>& mlir_config)
{
    return run_mlir_compiler_generic<MLIROutputKind::SharedObject>(module, mlir_config);
}

void to_json(::nlohmann::json& j, const MLIRConfig& p)
{
    j = nlohmann::json{
        {"enable_consteval", p.enable_consteval},
        {"enable_optimizer", p.enable_optimizer},
        {"enable_memory_layout_analysis", p.enable_memory_layout_analysis},
        {"enable_fusing", p.enable_fusing},
        {"enable_fusing_conv2d_with_multiply_pattern", p.enable_fusing_conv2d_with_multiply_pattern},
        {"custom_config", p.custom_config}};
}

void from_json(const ::nlohmann::json& j, MLIRConfig& p)
{
    j.at("enable_consteval").get_to(p.enable_consteval);
    j.at("enable_optimizer").get_to(p.enable_optimizer);
    j.at("enable_memory_layout_analysis").get_to(p.enable_memory_layout_analysis);
    j.at("enable_fusing").get_to(p.enable_fusing);
    j.at("enable_fusing_conv2d_with_multiply_pattern").get_to(p.enable_fusing_conv2d_with_multiply_pattern);
    j.at("custom_config").get_to(p.custom_config);
}

}  // namespace tt::passes
