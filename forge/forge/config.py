# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

from enum import Enum
from typing import Tuple, Dict, List, Optional, Set
from dataclasses import dataclass, field
from forge._C import DataFormat, MathFidelity, AMPNodeProperties, MLIRConfig
import forge.query as query
from dataclasses_json import dataclass_json, config

from forge.utils import (
    as_json,
    list_as_json,
    optional_as_json,
)
from loguru import logger


class CompileDepth(Enum):
    INIT_COMPILE = 0
    GENERATE_INITIAL_GRAPH = 1
    POST_INITIAL_GRAPH_PASS = 2
    CONSTEVAL_GRAPH = 3
    POST_PATTERN_MATCHER = 4
    OPTIMIZED_GRAPH = 5
    AUTOGRAD = 6
    POST_AUTOGRAD_PASS = 7
    PRE_LOWERING_PASS = 8
    SPLIT_GRAPH = 9
    RUN_MLIR_COMPILER = 10
    FINISH_COMPILE = 11
    FULL = 12

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def to_json(cls, value):
        return value.name

    @classmethod
    def from_json(cls, value):
        return cls[value.upper()]


@dataclass_json
@dataclass
class CompilerConfig:
    # enable training; run autograd & training passes
    enable_training: bool = False
    # enable training recompute during autograd
    enable_recompute: bool = False
    # invokes pattern_matcher to compact isomorphic subgraphs
    match_subgraph_patterns: Optional[int] = None
    # enable optimization passes (soon to be removed, only until mlir creates proper optimization passes for resnet)
    enable_optimization_passes: bool = False

    # enable promotion of nodes to be constant evaluated where possible
    enable_consteval: bool = True
    # Compile each disjoint graph separately into its own program
    compile_subgraphs: bool = False
    # Enable linking of past key-value pairs in the graph
    enable_pt2_fx_graph_link: bool = False

    # Defines compilation depth. Used to limit scope of some unit tests
    compile_depth: CompileDepth = field(default=CompileDepth.FULL, metadata=as_json(CompileDepth))
    # Create cpu device for unsupported forge ops
    enable_tvm_cpu_fallback: bool = False
    # Types of ops to fall back to CPU for
    cpu_fallback_ops: Set[str] = field(default_factory=lambda: set(["embedding"]))
    # Extend CPU fallback for TM ops
    enable_tm_cpu_fallback: bool = False
    # Max search depth for extended CPU fallback
    tm_cpu_fallback_max_depth: int = 10

    # (Temporary): Remove when forge supports dropout
    enable_tvm_dropout: bool = False
    # Create "unsupported" forge ops in python file, allowing user to modify later
    enable_tvm_unsupported_ops: bool = False
    # Should we constant prop in tvm
    enable_tvm_constant_prop: bool = False
    # Convert framework params to relay params
    convert_framework_params_to_tvm: bool = False
    # Convert JAX model to TF through XLA
    enable_xla_jax_convert: bool = False
    # When model param is larger than 2GB, Protobuf will error out. This flag will enable large model tracing
    enable_tvm_jax_freeze_large_model: bool = True
    # List of output names specified by framework
    framework_model_output_names: List[str] = field(default_factory=lambda: list())
    # Which parameters should be constant propped by tvm
    tvm_constnat_prop_mask: Set[str] = field(default_factory=lambda: set())
    # instead of generating a direct forge graph from TVM, generate a forge python class
    compile_tvm_to_python: bool = True
    # Whether to keep generated python code, or load and delete
    retain_tvm_python_files: bool = False

    # Transpiler Path Configuration (new)
    compile_transpiler_to_python: bool = False  # Generate Python code from TIRGraph
    # Transpiler-specific options
    transpiler_enable_debug: bool = False  # Enable debug mode for transpiler
    # Defines store path of serilized TVM graphs.
    tvm_graph_store_path: str = ""
    # Defines load path of serilized TVM graphs.
    tvm_graph_load_path: str = ""
    # Number of patterns to match for each module
    tvm_module_to_num_patterns: Dict[str, int] = field(default_factory=lambda: dict())

    # If enabled, for given test, it only extracts the unique operation configuration.
    extract_tvm_unique_ops_config: bool = False

    # Export the unique operations configurations information to the excel file
    export_tvm_unique_ops_config_details: bool = False

    # Outputs will be kept on device and looped back to inputs of subsequent runs
    loopback_outputs: Dict[str, int] = field(default_factory=lambda: dict())
    # Default override for all node data formats, None means automatically inferred
    default_df_override: Optional[DataFormat] = field(default=None, metadata=optional_as_json(DataFormat))
    # Accumulation format, for chips that support it
    default_accumulate_df: Optional[DataFormat] = field(default=None, metadata=optional_as_json(DataFormat))
    # if true, large broadcasts will be split into multiple edges with nops between them
    enable_broadcast_splitting: bool = False
    # default math fidelity for all ops
    default_math_fidelity: MathFidelity = field(default=MathFidelity.HiFi3, metadata=as_json(MathFidelity))

    # Configure Automatic Mixed Precision (AMP) level. By default it's set to 'None' (0), which means no AMP is applied. However, there
    # are few levels of AMP that can be applied:
    # 1: Matmuls inputs/outputs are set to BFP8_b;  Fused ops, Softmax, LayerNorm ops are set to FP16_b;
    # 2: Matmuls inputs/outputs are set to BFP8;    Fused ops, Softmax, LayerNorm ops are set to FP16;  GELU is BFP8;
    #
    # Have in mind that in each AMP level, non-mentioned op types are left with default data format (usually set by user; i.e. FP32).
    amp_level: Optional[int] = None

    # see insert_fracture_group
    fracture_groups: List[Tuple[List[Tuple[str, int, int]], List[str], List[int]]] = field(
        default_factory=lambda: list()
    )
    amp_properties: List[AMPNodeProperties] = field(
        default_factory=lambda: list(), metadata=list_as_json(AMPNodeProperties)
    )

    mlir_config: Optional[MLIRConfig] = field(default=None, metadata=optional_as_json(MLIRConfig))

    # TODO: add reportify dir

    def apply_env_config_overrides(self):
        if "FORGE_COMPILE_DEPTH" in os.environ:
            self.compile_depth = {
                "init_compile": CompileDepth.INIT_COMPILE,
                "generate_initial_graph": CompileDepth.GENERATE_INITIAL_GRAPH,
                "post_initial_graph_pass": CompileDepth.POST_INITIAL_GRAPH_PASS,
                "consteval_graph": CompileDepth.CONSTEVAL_GRAPH,
                "post_pattern_matcher": CompileDepth.POST_PATTERN_MATCHER,
                "optimized_graph": CompileDepth.OPTIMIZED_GRAPH,
                "autograd": CompileDepth.AUTOGRAD,
                "post_autograd_pass": CompileDepth.POST_AUTOGRAD_PASS,
                "pre_lowering_pass": CompileDepth.PRE_LOWERING_PASS,
                "split_graph": CompileDepth.SPLIT_GRAPH,
                "run_mlir_compiler": CompileDepth.RUN_MLIR_COMPILER,
                "finish_compile": CompileDepth.FINISH_COMPILE,
                "full": CompileDepth.FULL,
            }[os.environ["FORGE_COMPILE_DEPTH"].lower()]

        if "FORGE_CONVERT_PARAMS_TO_TVM" in os.environ:
            self.convert_framework_params_to_tvm = bool(int(os.environ["FORGE_CONVERT_PARAMS_TO_TVM"]))

        if "FORGE_DEFAULT_DF" in os.environ:
            self.default_df_override = DataFormat.from_json(os.environ["FORGE_DEFAULT_DF"])

        if "FORGE_EXTRACT_TVM_UNIQUE_OPS_CONFIG" in os.environ:
            self.extract_tvm_unique_ops_config = bool(int(os.environ["FORGE_EXTRACT_TVM_UNIQUE_OPS_CONFIG"]))

        if "FORGE_EXPORT_TVM_UNIQUE_OPS_CONFIG_DETAILS" in os.environ:
            self.export_tvm_unique_ops_config_details = bool(
                int(os.environ["FORGE_EXPORT_TVM_UNIQUE_OPS_CONFIG_DETAILS"])
            )

    def __post_init__(self):
        self.apply_env_config_overrides()

    def enable_amp_light(self, level: int = 1):
        if level == 0:
            return

        level_to_config = {
            1: (8, MathFidelity.HiFi2),
            2: (4, MathFidelity.HiFi2),
            3: (4, MathFidelity.LoFi),
        }
        mantissa_bits, math_fidelity = level_to_config[level]
        target_mm_weights, target_mm_bias = True, True
        input_parameter_indices = [
            (operand_index, mantissa_bits)
            for use_lower_precision, operand_index in zip((target_mm_weights, target_mm_bias), range(1, 3))
            if use_lower_precision
        ]
        self.amp_properties.append(
            AMPNodeProperties(
                op_type="matmul",
                math_fidelity=math_fidelity,
                input_parameter_indices_to_optimize=input_parameter_indices,
            )
        )
