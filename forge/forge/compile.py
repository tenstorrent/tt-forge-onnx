# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

import torch
from loguru import logger

import forge
from forge.compiled_graph_state import CompiledGraphState, CompiledModel, CompileResults
from forge.config import (
    CompilerConfig,
    CompileDepth,
)
from forge._C import (
    run_post_initial_graph_passes,
    run_optimization_graph_passes,
    run_post_optimize_decompose_graph_passes,
    run_consteval_graph_pass,
    run_post_autograd_graph_passes,
    run_pre_lowering_passes,
    dump_graph,
    extract_unique_op_configuration,
)
from forge._C import ForgeGraphModule, GraphType
import forge._C.autograd as pyautograd
import forge._C.graph as pygraph
from forge._C.graph import Graph
from forge._C.ops import Op, OpType
from forge._C.runtime import Binary
import forge.ci as ci
from forge.module import Module, ForgeModule, wrap_module, AnyModule
from forge.parameter import Parameter
from forge.forgeglobal import state_changed, clear_state_changed
import forge.query as query
from forge.tensor import Tensor, to_pt_tensors, AnyTensor
from forge.verify import DeprecatedVerifyConfig, do_verify, _generate_random_losses, _run_pytorch_backward
from forge.forge_property_utils import (
    ExecutionStage,
    ExecutionRunMode,
    record_execution,
    record_execution_run_mode,
    record_compiler_config,
    record_flatbuffer_details,
)
from forge.tensor import TensorFromPytorch

LAST_SUCCESSFUL_STAGE = None


def init_log_last_successful_compile_stage():
    global LAST_SUCCESSFUL_STAGE
    LAST_SUCCESSFUL_STAGE = None


def dump_compiler_cfg(backend_output_directory, compiler_cfg, graph_name):
    import yaml

    try:
        int(os.environ["FORGE_DUMP_CONFIG"])
        path = f"{graph_name}_config.yaml"
    except ValueError:
        path = os.environ["FORGE_DUMP_CONFIG"]
    with open(os.path.join(backend_output_directory, path), "w") as fd:
        yaml.dump(compiler_cfg.to_dict(), fd, indent=2)


def load_compiler_cfg(compiler_cfg, clobber=False):
    import yaml
    import json

    path = os.environ["FORGE_LOAD_CONFIG"]
    loader = json.load if os.path.splitext(path)[1] == ".json" else lambda f: yaml.load(f, yaml.SafeLoader)
    with open(path) as fd:
        d = compiler_cfg.to_dict()
        overrides = loader(fd)
        for k, v in overrides.items():
            d[k] = v
        return CompilerConfig.from_dict(d)


def generate_override_config(graph, balancer_solution, placer_solution, nop_instructions, graph_name):
    import yaml

    try:
        int(os.environ["FORGE_GENERATE_OVERRIDE_CONFIG"])
        path = f"{graph_name}_override_config.yaml"
    except ValueError:
        path = os.environ["FORGE_GENERATE_OVERRIDE_CONFIG"]

    overrides = {}
    overrides["balancer_op_overrides"] = {
        k: {
            "grid_shape": [v.grid_shape.r, v.grid_shape.c],
            "t_stream_dir": str(v.t_stream_factor.dir).split(".")[1],
            "t_stream_shape": [v.t_stream_factor.r, v.t_stream_factor.c],
            "fracture_factor": v.fracture_factor,
        }
        for k, v in balancer_solution.op_models.items()
    }

    overrides["buffering_nops_to_insert"] = [NopInsertionInstruction.to_json(n) for n in nop_instructions]

    overrides["insert_queues"] = list(list(v) for v in balancer_solution.cut_edges_as_override(graph))

    with open(path, "w") as fd:
        yaml.dump(overrides, fd, indent=2)


@dataclass
class CompileContext:
    modules: List[Module]
    graph_name: str
    compiler_cfg: CompilerConfig
    verify_cfg: DeprecatedVerifyConfig
    microbatch_size: int
    microbatch_count: int
    inputs: Union[torch.Tensor, List[torch.Tensor]]
    optimizer: Optional[Union[torch.optim.Optimizer, forge.optimizers.Optimizer]] = None
    training: bool = False
    graph: Optional[Graph] = None
    losses: Optional[List[Tensor]] = None
    output_kwargs: Dict[str, Any] = field(default_factory=dict)
    targets: List[Tensor] = field(default_factory=list)
    initial_graph: Optional[Graph] = None
    final_graph: Optional[Graph] = None
    stage: CompileDepth = CompileDepth.INIT_COMPILE
    initial_graph_copy: Optional[Graph] = None
    outputs: Tuple[Tensor, ...] = field(default_factory=tuple)
    intermediate_tensors: Dict[int, Tensor] = field(default_factory=dict)
    parameter_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    input_grads: List[torch.Tensor] = field(default_factory=list)
    netlist_filename: Optional[str] = None
    perf_model_results: Optional[Dict[str, float]] = None
    use_interactive_placer: bool = False
    fracture_chip_id_assignments: Dict[str, int] = field(default_factory=dict)
    forge_targets: List[Tensor] = field(default_factory=list)
    forge_losses: List[Tensor] = field(default_factory=list)
    placer_retry_count: int = 0
    backend_output_directory: str = ""
    in_recompile: bool = False
    recompile_count: int = 0
    target_cycles_offset: int = 0
    forge_module: Optional[ForgeGraphModule] = None
    compiled_binary: Optional[Binary] = None
    attach_to: Optional[CompiledModel] = None

    def optimizer_on_device(self):
        # For now we support only Forge optimizer on device.
        return self.optimizer is not None and isinstance(self.optimizer, forge.optimizers.Optimizer)


def calculate_grads(outputs: Tuple[Tensor, ...], intermediate_golden_tensors: Dict, losses=None):
    """
    Verify graph vs. pytorch golden
    """

    # retain intermediate gradients for verification
    for t in intermediate_golden_tensors.values():
        if t.requires_grad == True:
            t.retain_grad()

    # Calculate pytorch gradients
    run_backward = False
    for o in outputs:
        # Check if we need to run, or if gradients have been calculated already
        if o.value().grad is None and o.requires_grad:
            run_backward = True
            break

    if losses is None:
        losses = _generate_random_losses(outputs)

    if run_backward:
        _run_pytorch_backward(outputs, losses)

    return losses


def compile_main(
    module: AnyModule,
    sample_inputs: List[torch.Tensor],
    module_name: Optional[str] = None,
    optimizer: Optional[Union[torch.optim.Optimizer, forge.optimizers.Optimizer]] = None,
    training: bool = False,
    attach_to: Optional[CompiledModel] = None,
    compiler_cfg: CompilerConfig = CompilerConfig(),
    verify_cfg: DeprecatedVerifyConfig = DeprecatedVerifyConfig(),
) -> CompiledModel:
    """
    Main entry point for compiling modules from different frameworks for Tenstorrent devices.

    Parameters
    ----------
    module: AnyModule
        Torch, TensorFlow, Paddle, ONNX, Jax(Flax) or Forge module to compile

    sample_inputs: List[torch.Tensor]
        List of sample inputs for the module (used to infer shapes)

    module_name: Optional[str]
        Name of the module. If not provided, the class name of the provided module will be used.

    optimizer: Optional[torch.optim.Optimizer]
        Optimizer for training.

    training: bool
        Whether to compile the module for training.
        If true, the compiled module will contain both forward and backward passes.

    attach_to: Optional[CompiledModel]
        If provided, the compiled module will be "attached" to this module. Meaning that the backward pass
        of this module is tied to the backward pass of the attached module.
        NOTE: This part of the API is still in development and will probably change.

    Returns
    -------
    CompiledModel - Callable object that can be used to run the compiled module on device.

    """
    assert isinstance(module, AnyModule), f"Forge only supports: {AnyModule}."

    if module_name is None:
        module_name = module.__class__.__name__

    assert module_name is not None

    logger.info("Compiling module {}", module_name)

    if sample_inputs is None:
        logger.error("No sample inputs provided for module {}", module_name)
        assert False

    assert sample_inputs is not None

    modules = [wrap_module(module, module_name)]
    training = training or optimizer is not None

    record_compiler_config(compiler_cfg)
    # Record information needed to differentiate between inference and training compilation
    record_execution_run_mode(ExecutionRunMode.from_training_param(training))
    record_execution(ExecutionStage.FAILED_TVM_RELAY_IRMODULE_GENERATION)

    compile_context: CompileContext = CompileContext(
        modules=modules,
        graph_name=module_name,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
        microbatch_size=1,
        microbatch_count=1,
        inputs=sample_inputs,
        optimizer=optimizer,
        training=training,
        attach_to=attach_to,
    )

    return forge_compile_from_context(compile_context)


def forge_compile_from_context(context: CompileContext) -> CompiledModel:
    """
    Run front-end compile passes and generate a Forge netlist, with a given compile context.

    Parameters
    ----------
    context: CompileContext
        Contains all needed info to run compile passes.

    Returns
    -------
    CompileResults

    """

    # Map stages to functions which execute them.
    stage_to_func = {
        CompileDepth.INIT_COMPILE: init_compile,
        CompileDepth.GENERATE_INITIAL_GRAPH: generate_initial_graph,
        CompileDepth.POST_INITIAL_GRAPH_PASS: run_post_initial_graph_pass,
        CompileDepth.CONSTEVAL_GRAPH: run_consteval_pass,
        CompileDepth.POST_PATTERN_MATCHER: run_post_pattern_matcher,
        CompileDepth.OPTIMIZED_GRAPH: run_optimization_pass,
        CompileDepth.AUTOGRAD: run_autograd_pass,
        CompileDepth.POST_AUTOGRAD_PASS: run_post_autograd_pass,
        CompileDepth.PRE_LOWERING_PASS: run_pre_lowering_pass,
        CompileDepth.SPLIT_GRAPH: split_graph,
        CompileDepth.RUN_MLIR_COMPILER: run_mlir_compiler,
        CompileDepth.FINISH_COMPILE: finish_compile,
    }

    while context.stage != CompileDepth.FULL:
        logger.info("Running compile stage {}", context.stage.name.lower())

        current_stage = context.stage
        verify_cfg = context.verify_cfg
        compiler_cfg = context.compiler_cfg

        # Execute the current stage.
        next_stage = stage_to_func[current_stage](context)

        # Check if we need to stop compilation or perform verifications in the current stage.
        should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, current_stage)
        should_verify = current_stage in verify_cfg.stages_for_intermediate_verification

        can_verify = (
            current_stage != CompileDepth.INIT_COMPILE
            and current_stage != CompileDepth.PRE_LOWERING_PASS
            and current_stage != CompileDepth.POST_PATTERN_MATCHER
        )
        if (
            verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation) or should_verify
        ) and can_verify:
            do_verify(
                current_stage.name.lower(),
                context.graph,
                context.inputs,
                context.parameter_dict,
                context.input_grads,
                context.outputs,
                context.intermediate_tensors,  # intermediate golden tensors
                verify_cfg,  # DeprecatedVerifyConfig
                losses=context.losses,
                targets=context.targets,
                optimizer=context.optimizer,
            )

        if should_early_stop_compilation:
            logger.info("Early stopping compilation at stage {}", current_stage.name.lower())
            return generate_compile_results(
                context.verify_cfg,
                context.initial_graph_copy,
                context.outputs,
                context.intermediate_tensors,
                context.final_graph,
                pass_specific_output_kwargs=context.output_kwargs,
            )

        context.stage = next_stage

    compile_results = generate_compile_results(
        verify_cfg,
        context.initial_graph_copy,
        context.outputs,
        context.intermediate_tensors,
        final_graph=context.final_graph,
        pass_specific_output_kwargs=context.output_kwargs,
    )

    assert context.forge_module is not None
    fwd_compiled_graph_state = CompiledGraphState.from_compiled_graph(
        context.modules[0], context.forge_module.get_graph(GraphType.Forward)
    )
    bwd_compiled_graph_state = None
    opt_compiled_graph_state = None
    if context.training:
        bwd_compiled_graph_state = CompiledGraphState.from_compiled_graph(
            context.modules[0], context.forge_module.get_graph(GraphType.Backward)
        )

        if context.optimizer_on_device():
            # TODO(#924): This is not good. We are linking the optimizer parameters (such as learning rate, momentum, etc.)
            # to the optimizer parameters in the optimizer graph via their names. Due to the nature of the previous stack,
            # this was maybe a good enough solution, but we should redesign this.
            #
            # E.g. if the optimizer has a learning rate parameter named "lr" and the compiled graph has a parameter named
            # "l1.weight", the optimizer parameter in the graph will be named "input_opt_l1.weight_0.lr". This needs to be
            # matched with ('l1.weight', 'lr') in the parameters that we extract from the optimizer.
            #
            # Additionally, all trainable parameters will have its own copy of the learning rate optimizer parameter, which
            # is not ideal as well.
            module_params = context.modules[0].get_parameters()
            context.optimizer.set_parameters_to_optimize(module_params)

            # Get all parameters of the optimizer (learning rate, momentum, etc.) for the compiled graph.
            assert context.optimizer is not None
            opt_params = context.optimizer.get_optimizer_params()
            graph_opt_params = context.graph.get_optimizer_parameter_nodes()

            # Now convert the names of the optimizer parameters to the names of the parameters in the compiled graph.
            module_param_names = [param.get_name() for param in module_params]
            converted_opt_params = {}
            for opt_param_node in graph_opt_params:
                for opt_param in opt_params:
                    param_name, opt_param_name = opt_param
                    assert param_name in module_param_names, f"Parameter {param_name} not found in module parameters"

                    if param_name in opt_param_node.name and opt_param_name in opt_param_node.name:
                        converted_opt_params[opt_param_node.name] = opt_params[opt_param]

            opt_compiled_graph_state = CompiledGraphState.from_compiled_graph(
                context.modules[0], context.forge_module.get_graph(GraphType.Optimizer), converted_opt_params
            )

    assert context.compiled_binary is not None

    compiled_module = CompiledModel(
        context.forge_module,
        fwd_compiled_graph_state,
        bwd_compiled_graph_state,
        opt_compiled_graph_state,
        context.compiled_binary,
        context.modules[0],
        context.attach_to,
    )

    if context.optimizer_on_device():
        # Link the module to the optimizer, so that the user can call `optimizer.step()` which can in turn
        # execute the optimizer graphs of all linked modules.
        context.optimizer.link_module(compiled_module)

    record_execution(ExecutionStage.FAILED_TTNN_BINARY_EXECUTION)
    logger.info("Compilation completed.")

    return compiled_module


def forge_compile(
    graph_name: str,
    *inputs: Union[Tensor, List[Any], Dict[str, Any]],
    targets: List[Tensor] = [],
    compiler_cfg: CompilerConfig = CompilerConfig(),
    verify_cfg: Optional[DeprecatedVerifyConfig] = None,
    losses: Optional[List[Tensor]] = None,
    microbatch_size: int = 1,
    microbatch_count: int = 1,
    training: bool = False,
) -> CompileResults:
    """
    Run front-end compile passes and generate a Forge netlist for given input tensors. Optionally verify
    against PyTorch model.

    This version has significant amount of verification built-in, and is primarily used for testing. A "deliverable"
    version that does only the compile will be written in the future.

    Parameters
    ----------
    dev: TTDevice
        Device to compile modules for. Modules should already be placed on the device.

    graph_name: str
        Name to be used in the netlist

    *inputs: Tuple[Tensor, ...]
        Input tensors to compile for. Tensors must have set shapes, but values are only needed for
        automatic verification.

    targets: List[Tensor], optional
        Optional list of target tensors, if this device has a loss module

    verify_cfg: Optional[DeprecatedVerifyConfig]
        If set, automatic verification vs. pytorch golden result will be performed, with given parameters
        must contain data.


    Returns
    -------
    CompileResults

    """

    inputs = list(inputs)
    if verify_cfg is None:
        verify_cfg = DeprecatedVerifyConfig.disabled()  # no verification config provided, disable by default

    compiler_cfg.apply_env_config_overrides()

    compile_context: CompileContext = CompileContext(
        graph_name=graph_name,
        inputs=inputs,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
        microbatch_size=microbatch_size,
        microbatch_count=microbatch_count,
        targets=targets,
        losses=losses,
        training=training,
    )

    return forge_compile_from_context(compile_context)


def check_for_compilation_early_stop(desired_stage, current_stage):
    """
    Determines should current compilation process stop or not based on desired
    and current phase of execution.

    Parameters
    ----------
    desired_stage: CompileDepth
        Desired phase for compiler early stopping.

    current_stage: CompileDepth
        Current phase for compiler early stopping.

    Returns
    -------
    Boolean
    """
    # update global compile stage variable
    global LAST_SUCCESSFUL_STAGE
    LAST_SUCCESSFUL_STAGE = str(CompileDepth(current_stage.value).name)

    if not CompileDepth.has_value(desired_stage.value):
        raise Exception("Invalid compilation depth flag: {}".format(desired_stage.name))

    if desired_stage.value == current_stage.value:
        logger.info("Compilation early stopping after {}".format(current_stage.name))

        return True

    return False


def placer_breaks_eval(value):
    if type(value) is query.NodePredicateBuilder:
        return value.eval()
    elif type(value) is list:
        return [placer_breaks_eval(v) for v in value]
    else:
        assert type(value) is str
        return value


def placer_op_overrides_eval(value):
    assert type(value) is tuple
    if type(value[0]) is query.NodePredicateBuilder:
        return (value[0].eval(), value[1])
    else:
        return value


def generate_compile_results(
    verify_cfg=None,
    initial_graph=None,
    outputs=None,
    intermediate_tensors=None,
    final_graph=None,
    *,
    pass_specific_output_kwargs=None,
):
    """
    Wrapper for generating result from the graph compiler. Contains initial and final graphs, output tensors,
    and, optionally golden results for final output and intermediates, if desired.

    Parameters
    ----------
    verify_cfg: DeprecatedVerifyConfig
        Value verification config

    initial_graph: Graph
        Initial graph, immediately after conversion from the input framework

    outputs: Tuple[Tensor, ...]
        Output tensors

    intermediate_tensors: Dict[str, Tensor]
        Intermediated tensors

    final_graph: Graph
        Forge graph

    netlist_filename: str
        Netlist file name

    Returns
    -------
    CompileResults
    """
    ret = CompileResults()

    ret.initial_graph = initial_graph
    ret.outputs = outputs
    if verify_cfg and verify_cfg.intermediates:
        ret.golden_intermediates = {
            initial_graph.get_node_name(node_id): tensor
            for node_id, tensor in intermediate_tensors.items()
            if initial_graph.has_node_with_id(node_id)
        }
    ret.final_graph = final_graph

    if outputs:
        ret.golden_outputs = [out.value() if out.has_value() else None for out in outputs]

    if pass_specific_output_kwargs:
        ret.pass_specific_output_kwargs = pass_specific_output_kwargs

    return ret


def init_compile(context: CompileContext) -> CompileDepth:

    compiler_cfg = context.compiler_cfg
    graph_name = context.graph_name

    force_full = bool(int(os.environ.get("FORGE_FORCE_FULL_COMPILE_DEPTH", "0")))
    if force_full:
        compiler_cfg.compile_depth = CompileDepth.FULL

    # compiler_cfg is fully formed
    if "FORGE_LOAD_CONFIG" in os.environ:
        compiler_cfg = load_compiler_cfg(compiler_cfg)
    elif "FORGE_DUMP_CONFIG" in os.environ:
        dump_compiler_cfg(context.backend_output_directory, compiler_cfg, graph_name)

    init_log_last_successful_compile_stage()

    return CompileDepth.GENERATE_INITIAL_GRAPH


def generate_initial_graph(context: CompileContext) -> CompileDepth:
    """
    Generates initial graph from the input framework.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """

    modules_ = []
    # Check if we should convert to Forge module (either TVM or transpiler path)
    should_convert = (
        context.compiler_cfg.compile_tvm_to_python or context.compiler_cfg.compile_transpiler_to_python
    ) and context.graph is None

    if should_convert:
        logger.info("Converting framework modules to Forge modules")
        module_inputs = context.inputs
        for module in context.modules:
            if not isinstance(module, ForgeModule):
                module, module_inputs = convert_to_forge_module(
                    module,
                    module_inputs,
                    context.compiler_cfg,
                    context.verify_cfg,
                )
                assert isinstance(module, ForgeModule)

                context.inputs = module_inputs

            modules_.append(module)
        logger.info("Framework to Forge module conversion completed")

    record_execution(ExecutionStage.FAILED_FORGE_INITIAL_GRAPH_PASS)

    if context.graph is None:
        logger.info("Generating initial Forge graph from modules")
        context.graph, context.outputs, context.intermediate_tensors, context.inputs, _ = generate_graph(
            context,
            modules_,
            return_intermediate=context.verify_cfg.intermediates,
            target_tensors=context.targets,
        )
        logger.info("Initial Forge graph generation completed")

    context.graph.set_microbatch(context.microbatch_size)
    dump_graph(context.graph, context.graph_name, "initial_graph")
    extract_unique_op_configuration(context.graph, context.stage.name.upper())

    context.initial_graph_copy = context.graph.clone()  # save the original graph for verification and analysis
    context.input_grads = []

    context.parameter_dict = {}
    for module in modules_:
        if isinstance(module, forge.module.Module):
            for p in module.get_parameters():
                context.parameter_dict[p.get_name()] = p.value()
        elif isinstance(module, torch.fx.GraphModule):
            for name, value in module.named_parameters():
                context.parameter_dict[name] = value

    return CompileDepth.POST_INITIAL_GRAPH_PASS


def run_post_initial_graph_pass(context: CompileContext) -> CompileDepth:
    """
    Runs post initial graph passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph_name = context.graph_name
    graph, intermediate_tensors = context.graph, context.intermediate_tensors

    record_execution(ExecutionStage.FAILED_FORGE_POST_INITIAL_GRAPH_PASS)

    inserted_node_id_mapping, context.fracture_chip_id_assignments = run_post_initial_graph_passes(
        graph, compiler_cfg, compiler_cfg.fracture_groups
    )

    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        # If we have multi-level of decomposition, some node id might not in the original
        # intermediate tensor dict.
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    dump_graph(graph, graph_name, "decomposed_graph")
    extract_unique_op_configuration(context.graph, context.stage.name.upper())

    if compiler_cfg.match_subgraph_patterns:
        return CompileDepth.POST_PATTERN_MATCHER
    if compiler_cfg.enable_optimization_passes:
        return CompileDepth.OPTIMIZED_GRAPH
    if context.training:
        return CompileDepth.AUTOGRAD
    return CompileDepth.POST_AUTOGRAD_PASS


def run_consteval_pass(context: CompileContext) -> CompileDepth:
    """
    Runs consteval pass.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    graph = context.graph
    graph_name = context.graph_name
    record_execution(ExecutionStage.FAILED_FORGE_CONSTEVAL)

    run_consteval_graph_pass(graph)
    dump_graph(graph, graph_name, "consteval_graph")
    extract_unique_op_configuration(context.graph, context.stage.name.upper())

    return CompileDepth.PRE_LOWERING_PASS


def run_post_pattern_matcher(context: CompileContext) -> CompileDepth:
    """
    Runs post pattern matcher passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph = context.graph
    graph_name = context.graph_name

    graph, match_result = pypattern_matcher.lower_forge_to_pattern_matcher(graph, compiler_cfg.match_subgraph_patterns)
    context.output_kwargs["match_result"] = match_result

    if match_result.is_subgraph_loopable:
        dump_graph(graph, graph_name, "looped_graph")

    if compiler_cfg.enable_optimization_passes:
        return CompileDepth.OPTIMIZED_GRAPH
    if context.training:
        return CompileDepth.AUTOGRAD

    return CompileDepth.POST_AUTOGRAD_PASS


def run_optimization_pass(context: CompileContext) -> CompileDepth:
    """
    Runs optimization passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph_name = context.graph_name
    graph, intermediate_tensors = context.graph, context.intermediate_tensors
    record_execution(ExecutionStage.FAILED_FORGE_OPTIMIZATION_GRAPH_PASS)

    run_optimization_graph_passes(graph)
    dump_graph(graph, graph_name, "optimized_graph")

    record_execution(ExecutionStage.FAILED_FORGE_POST_OPTIMIZATION_DECOMP)

    inserted_node_id_mapping = run_post_optimize_decompose_graph_passes(graph, compiler_cfg)
    dump_graph(graph, graph_name, "decomposed_optimized_graph")
    extract_unique_op_configuration(context.graph, context.stage.name.upper())
    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    next_stage = CompileDepth.POST_AUTOGRAD_PASS
    if context.training:
        next_stage = CompileDepth.AUTOGRAD

    return next_stage


def append_losses_to_inputs(
    inputs: Union[TensorFromPytorch, List[TensorFromPytorch]], losses: List[torch.Tensor]
) -> List[torch.Tensor]:
    # Ensure inputs is a list
    if isinstance(inputs, TensorFromPytorch):
        inputs = [inputs]
    inputs = [input.value() for input in inputs]

    # Append losses if they exist
    if losses is not None:
        inputs.extend(losses)

    return inputs


def run_autograd_pass(context: CompileContext) -> CompileDepth:
    """
    Runs autograd pass.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph_name = context.graph_name
    graph, intermediate_tensors, outputs = context.graph, context.intermediate_tensors, context.outputs

    record_execution(ExecutionStage.FAILED_FORGE_AUTOGRAD_PASS)

    graph.set_training(True)

    optimizer = None
    if context.optimizer_on_device():
        #  Set optimizer parameters from module, because we need optimizer parameters in intermediate verification.
        module_params = context.modules[0].get_parameters()
        context.optimizer.set_parameters_to_optimize(module_params)
        # If we should run the optimizer on the device, pass it so that the autograd engine can create the optimizer graph.
        optimizer = context.optimizer

    autograd_config = pyautograd.AutogradConfig(recompute=compiler_cfg.enable_recompute, optimizer=optimizer)
    autograd_engine = pyautograd.AutogradEngine(graph, autograd_config)

    graph = autograd_engine.run()
    dump_graph(graph, graph_name, "post_autograd")
    extract_unique_op_configuration(context.graph, context.stage.name.upper())

    # GOLDEN:
    context.losses = calculate_grads(outputs, intermediate_tensors, context.losses)
    # Append losses to inputs, because we need losses for calculating backward gradients.
    # Loss is one of the inputs of backward graph.
    context.inputs = append_losses_to_inputs(context.inputs, context.losses)

    # Gradients of input activations, if they are trainable (require grad)
    context.input_grads = [i.grad for i in context.inputs if i.requires_grad and i.grad is not None]

    return CompileDepth.POST_AUTOGRAD_PASS


def run_post_autograd_pass(context: CompileContext) -> CompileDepth:
    """
    Runs post autograd passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph_name = context.graph_name
    graph, intermediate_tensors, losses, outputs = (
        context.graph,
        context.intermediate_tensors,
        context.losses,
        context.outputs,
    )
    record_execution(ExecutionStage.FAILED_FORGE_POST_AUTOGRAD_DECOMP)

    inserted_node_id_mapping = run_post_autograd_graph_passes(graph, compiler_cfg)
    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    dump_graph(graph, graph_name, "post_autograd_passes")
    extract_unique_op_configuration(context.graph, context.stage.name.upper())

    if compiler_cfg.enable_consteval:
        return CompileDepth.CONSTEVAL_GRAPH

    return CompileDepth.PRE_LOWERING_PASS


def run_pre_lowering_pass(context: CompileContext) -> CompileDepth:
    """
    Runs pre lowering passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph_name = context.graph_name
    graph = context.graph
    record_execution(ExecutionStage.FAILED_FORGE_PRE_LOWERING)

    graph = run_pre_lowering_passes(graph, compiler_cfg.default_df_override)
    dump_graph(graph, graph_name, "pre_lowering")
    extract_unique_op_configuration(context.graph, context.stage.name.upper())

    context.final_graph = graph
    return CompileDepth.SPLIT_GRAPH


def split_graph(context: CompileContext) -> CompileDepth:
    """
    Splits graph into multiple graphs which will lower to different MLIR functions,
    i.e. forward, backward, etc.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    record_execution(ExecutionStage.FAILED_FORGE_GRAPH_SPLIT)

    assert context.graph is not None
    context.forge_module = forge._C.split_graph(context.graph)

    return CompileDepth.RUN_MLIR_COMPILER


def run_mlir_compiler(context: CompileContext) -> CompileDepth:
    forge_module, compiler_cfg = (
        context.forge_module,
        context.compiler_cfg,
    )
    assert forge_module is not None
    assert compiler_cfg is not None

    record_execution(ExecutionStage.FAILED_FORGE_MLIR_COMPILATION)

    context.compiled_binary = forge._C.run_mlir_compiler(forge_module, compiler_cfg.mlir_config)

    record_flatbuffer_details(context.compiled_binary.as_json())

    return CompileDepth.FINISH_COMPILE


def finish_compile(context: CompileContext) -> CompileDepth:
    """
    Doesn't do anything (for now)

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    verify_cfg = context.verify_cfg

    return CompileDepth.FULL


def convert_to_forge_module(
    module: AnyModule,
    module_inputs: Union[AnyTensor, List[AnyTensor]],
    compiler_cfg: CompilerConfig,
    verify_cfg: DeprecatedVerifyConfig,
) -> ForgeModule:
    """
    Converts given module to a Forge module, along with the module_inputs (which will be converted to Forge tensors).

    Supports two paths:
    1. TVM Path: Framework Model → TVM Relay IR  → Forge Module
    2. TIR Path: Framework Model →   TIRGraph    → Forge Module

    Returns
    -------
    ForgeModule, Tuple[Tensor, ...]
    """

    prev_state = state_changed()

    if module_inputs is None:
        logger.error("No inputs provided for module {}", module.name)
        assert False

    # Determine which path to use
    # Default to TVM path, but if transpiler is enabled, use it instead
    use_transpiler_path = compiler_cfg.compile_transpiler_to_python
    use_tvm_path = compiler_cfg.compile_tvm_to_python and not use_transpiler_path

    module_inputs = to_pt_tensors(module_inputs)

    if use_transpiler_path:
        # Transpiler Path: Framework Model → TIRGraph → Forge Module
        from .transpiler.codegen.transpiler_to_forge import generate_forge_module_from_transpiler

        # Generate Forge module from transpiler (all logic is inside this function)
        forge_module, module_inputs = generate_forge_module_from_transpiler(
            framework_mod=module,
            module_inputs=module_inputs,
            compiler_cfg=compiler_cfg,
            graph_name=module.name,
            verify_cfg=verify_cfg,
        )

    elif use_tvm_path:
        # TVM Path: Framework Model → TVM Relay IR → Forge Module (existing)
        from .tvm_to_python import generate_forge_module

        forge_module, dev_types, module_inputs = generate_forge_module(
            module,
            to_pt_tensors(module_inputs),
            compiler_cfg,
            module.name,
            verify_cfg,
        )
    else:
        raise ValueError(
            "Either compile_tvm_to_python or compile_transpiler_to_python must be True. "
            f"Current config: compile_tvm_to_python={compiler_cfg.compile_tvm_to_python}, "
            f"compile_transpiler_to_python={compiler_cfg.compile_transpiler_to_python}"
        )

    assert len(forge_module) == 1, "Attemping to load split model onto single devices"

    if not (prev_state):
        clear_state_changed()

    if isinstance(module_inputs, Tensor):
        module_inputs = (module_inputs,)  # Force a tuple

    return forge_module[0], module_inputs


def generate_graph(
    context: CompileContext,
    modules: List[ForgeModule],
    target_tensors: List[Tensor] = [],
    return_intermediate: bool = False,
    trace_only: bool = False,
) -> Tuple[Graph, Tuple[Tensor, ...], Dict[str, Tensor], Tuple[Tensor, ...], Optional[Tensor]]:
    """
    Generate a forge graph from the passed modules, and return the graph and output tensors.
    If input tensors have a value set, the output tensor will also have the calculated output value
    set.

    Parameters
    ----------
    inputs: Tuple[Tensor, ....]
        Input tensors

    target_tensors: List[Tensor]
        Target inputs. Optional, if trace_only is set. Otherwise, value must be provided.

    return_intermediate: bool
        Optional. If set, a dictionary of node IDs -> tensors will be return with intermediate values, for data mismatch debug.

    trace_only: bool
        If set, the graph is made for a quick trace only and shouldn't have side-effects

    Returns
    -------
    Graph, Tuple[Tensor, ...], Dict[str, Tensor], Tuple[Tensor, ...], Optional[Tensor]
        Forge graph, outputs, optional intermediates, original inputs, target tensor
    """

    """
    TODO: This function was copied over from ttdevice.py with some modifications. Probably needs to be refactored (possibly moved to cpp)
    """

    from .forgeglobal import start_tracing, stop_tracing
    from forge.tvm_utils import flatten_inputs
    from collections import deque
    import inspect

    from forge._C.graph import (
        create_output,
        create_parameter_input,
        create_data_edge,
        create_activation_input,
        create_constant_input,
        create_op_node,
        create_target_input,
    )

    inputs = context.inputs
    graph_name = context.graph_name
    compiler_cfg = context.compiler_cfg

    output_to_module_map: Dict[Tensor, ForgeModule] = {}
    output_to_subgraph_index = {}

    # Create the graph
    graph = Graph(graph_name)
    graph.set_microbatch(1)

    # Trace through the modules
    all_subgraph_outputs = []
    outputs = inputs
    for idx, module in enumerate(modules):

        assert isinstance(module, ForgeModule), "This function only supports ForgeModule instances"

        if compiler_cfg.compile_subgraphs:
            outputs = inputs[idx]

        start_tracing()
        outputs = module.forward(*outputs)
        stop_tracing()
        if isinstance(outputs, Tensor):
            outputs = (outputs,)  # Force a tuple

        for output in outputs:
            output_to_module_map[output] = module
            if compiler_cfg.compile_subgraphs:
                assert (
                    output not in output_to_subgraph_index
                ), "Output tensor {} is produced by multiple modules".format(output)

            output_to_subgraph_index[output] = module.subgraph_idx

        all_subgraph_outputs += outputs

    if trace_only:
        return graph, all_subgraph_outputs, {}, inputs, target_tensors

    visited_tensors = {}
    pending_tensors = deque()
    intermediate = {}
    module_input_tensor_to_node: Dict[str, Tensor] = {}
    module_output_tensor_to_node: Dict[str, Tensor] = {}
    module_target_tensor_to_node: Dict[str, Tensor] = {}
    module_loopback_tensor_to_node: Dict[str, Tensor] = {}
    passthroughs: Set = set()

    input_node_names = []
    input_names_known = True
    if isinstance(inputs[0], Tensor):
        inputs = (inputs,)
    for index, (module, submodule_input) in enumerate(zip(modules, inputs)):
        submodule_input_node_names = list(
            inspect.signature(super(ForgeModule, module).__getattribute__("forward")).parameters.keys()
        )
        if len(modules) > 1:
            submodule_input_node_names = [f"{input_name}_{index}" for input_name in submodule_input_node_names]
        input_node_names += submodule_input_node_names
        if len(submodule_input_node_names) != len(submodule_input):
            input_names_known = False
    inputs, _, _ = flatten_inputs(inputs)

    for out in all_subgraph_outputs:
        module = output_to_module_map[out]
        assert module is not None
        module_name = module.get_name()

        if out.src_op is None:

            # No source op. It could be a pass-through, so let's compare to inputs
            found = False
            for input in inputs:
                if input == out:
                    # Found a passthrough
                    outq = create_output(
                        graph,
                        module_name + f".output_passthrough_{len(passthroughs)}",
                        out.shape.get_pytorch_shape(),
                        out.data_format,
                        module.is_loss,
                        output_to_subgraph_index.get(out, 0),
                    )
                    passthroughs.add(input)
                    found = True
                    break

            if not found:
                raise RuntimeError("Untraced output tensor encountered")

        else:
            outq = create_output(
                graph,
                module_name + ".output_" + out.src_op.name,
                out.shape.get_pytorch_shape(),
                out.data_format,
                module.is_loss,
                output_to_subgraph_index.get(out, 0),
            )
        module_output_tensor_to_node[out] = outq
        pending_tensors.append((out, outq, 0, [], output_to_subgraph_index.get(out, 0)))

    recorded_parameters = {}

    while pending_tensors:

        tensor, output, port_index, operand_broadcast, subgraph_idx = pending_tensors.popleft()

        if tensor in visited_tensors:
            # Already created the note - let's add the edge and move on
            create_data_edge(graph, visited_tensors[tensor], 0, output, port_index, operand_broadcast)
            continue

        if isinstance(tensor, int):
            # integer constant. Don't add to visited tensors.
            assert False  # not supported any more

        if isinstance(tensor, Parameter):
            # parameter tensor
            if tensor.get_name() is not None:
                name = tensor.get_name()
            else:
                name = "parameter_" + graph.get_node_name(output)

            if name in recorded_parameters:
                # Multiple subgraphs might use the same parameter. If it is used in the same subgraph,
                # we should have already found it in the visited_tensors dictionary. Putting an assert here
                # to catch fallouts.
                assert (
                    graph.get_subgraph_id_for_node(recorded_parameters[name]) != subgraph_idx
                ), "Trying to add parameter with name: {} that is used in the same subgraph".format(name)
                create_data_edge(graph, recorded_parameters[name], 0, output, port_index, operand_broadcast)
                continue

            inq = create_parameter_input(
                graph, name, tensor.shape.get_pytorch_shape(), tensor.requires_grad, tensor.data_format, subgraph_idx
            )
            create_data_edge(graph, inq, 0, output, port_index, operand_broadcast)
            visited_tensors[tensor] = inq
            recorded_parameters[name] = inq
            continue

        if tensor.src_op is None:
            input_name = (
                input_node_names[inputs.index(tensor)]
                if input_names_known and tensor in inputs
                else "input_" + str(port_index) + "_" + graph.get_node_name(output)
            )
            if tensor in passthroughs:
                # passthrough input->output, add a nop
                inq = create_activation_input(
                    graph,
                    input_name,
                    tensor.shape.get_pytorch_shape(),
                    tensor.requires_grad,
                    tensor.data_format,
                    subgraph_idx,
                )

                nop = create_op_node(
                    graph,
                    f"_passthrough_nop_{output}",
                    Op(OpType.Nop),
                    tensor.shape.get_pytorch_shape(),
                    tensor.data_format,
                    subgraph_idx,
                    {},
                )

                create_data_edge(graph, inq, 0, nop, 0, operand_broadcast)
                create_data_edge(graph, nop, 0, output, 0, operand_broadcast)
                visited_tensors[tensor] = inq
                module_input_tensor_to_node[tensor] = inq
                continue

            elif tensor in target_tensors:
                # Target input
                inq = create_target_input(
                    graph,
                    input_name,
                    tensor.shape.get_pytorch_shape(),
                    tensor.requires_grad,
                    tensor.data_format,
                    subgraph_idx,
                )
                create_data_edge(graph, inq, 0, output, port_index, operand_broadcast)
                visited_tensors[tensor] = inq
                module_target_tensor_to_node[tensor] = inq
                continue

            elif tensor.is_constant():
                # Target input
                inq = create_constant_input(
                    graph,
                    input_name,
                    tensor.value(),
                    tensor.shape.get_pytorch_shape(),
                    tensor.data_format,
                    subgraph_idx,
                )
                create_data_edge(graph, inq, 0, output, port_index, operand_broadcast)
                visited_tensors[tensor] = inq
                module_target_tensor_to_node[tensor] = inq
                continue

            else:
                # input tensor
                input_creator = (
                    create_activation_input
                    if input_name not in compiler_cfg.loopback_outputs
                    else create_parameter_input
                )

                if input_name in compiler_cfg.loopback_outputs:
                    module.add_parameter(input_name, Parameter(tensor.value(), requires_grad=True, name=input_name))

                inq = input_creator(
                    graph,
                    input_name,
                    tensor.shape.get_pytorch_shape(),
                    tensor.requires_grad,
                    tensor.data_format,
                    subgraph_idx,
                )
                create_data_edge(graph, inq, 0, output, port_index, operand_broadcast)
                visited_tensors[tensor] = inq
                if input_name not in compiler_cfg.loopback_outputs:
                    module_input_tensor_to_node[tensor] = inq
                elif input_name in compiler_cfg.loopback_outputs:
                    module_loopback_tensor_to_node[tensor] = inq
                    recorded_parameters[input_name] = inq
                continue

        elif tensor.src_op.op.type() == OpType.Constant:
            constant_value = tensor.src_op.op.attrs()["c"]
            constant = create_constant_input(
                graph,
                "constant_" + str(port_index) + "_" + graph.get_node_name(output),
                constant_value,
                tensor.data_format,
                subgraph_idx,
            )

            create_data_edge(graph, constant, 0, output, port_index, operand_broadcast)
            visited_tensors[tensor] = constant
            continue

        """
        print("ttdevice.py, create_op_node")
        print(f"graph type: {type(graph)}")
        print(f"src_op name: {tensor.src_op.name}")
        print(f"src_op op_type: {tensor.src_op.op_type}")
        print(f"src_op attrs: {tensor.src_op.attrs}")
        print(f"shape: {tensor.shape.get_pytorch_shape()}")
        print(f"data format: {tensor.data_format}")
        """

        tags = {}
        if tensor.src_layer is not None:
            tags["layer"] = tensor.src_layer
        op = create_op_node(
            graph,
            tensor.src_op.name,
            tensor.src_op.op,
            tensor.shape.get_pytorch_shape(),
            tensor.data_format,
            subgraph_idx,
            tags,
        )

        visited_tensors[tensor] = op
        if return_intermediate and tensor.has_value():
            intermediate[op] = tensor.value()

        create_data_edge(graph, op, 0, output, port_index, operand_broadcast)

        for i, t in enumerate(tensor.src_op.operands):
            pending_tensors.append((t, op, i, tensor.src_op.operand_broadcast, subgraph_idx))

    # Register input/output order of the module to the graph now that the nodes are created
    module_inputs = [
        module_input_tensor_to_node[input_tensor]
        for input_tensor in inputs
        if input_tensor in module_input_tensor_to_node
    ]
    module_outputs = [
        module_output_tensor_to_node[output_tensor]
        for output_tensor in all_subgraph_outputs
        if output_tensor in module_output_tensor_to_node
    ]
    module_targets = [module_target_tensor_to_node[target_tensor] for target_tensor in target_tensors]
    out_requires_grad = [
        output_tensor.requires_grad
        for output_tensor in all_subgraph_outputs
        if output_tensor in module_output_tensor_to_node
    ]

    # Remove unused inputs from list of module inputs
    inputs = [
        input_tensor
        for input_tensor in inputs
        if input_tensor in module_input_tensor_to_node or input_tensor in module_output_tensor_to_node
    ]

    # Remove loopback inputs from list of module inputs
    inputs = [input_tensor for input_tensor in inputs if input_tensor not in module_loopback_tensor_to_node]

    if len(compiler_cfg.loopback_outputs):
        output_to_remove = []
        out_requires_grad_to_remove = []
        for input_name, output_indices in compiler_cfg.loopback_outputs.items():
            if isinstance(output_indices, int):
                output_indices = [output_indices]
            for output_index in output_indices:
                input_id = graph.get_node_id(input_name)
                output_id = module_outputs[output_index]
                add_partial_datacopy_edge(graph, output_id, 0, input_id, 0)
                output_to_remove.append(module_outputs[output_index])
                out_requires_grad_to_remove.append(out_requires_grad[output_index])
        [module_outputs.remove(value) for value in output_to_remove]
        [out_requires_grad.remove(value) for value in out_requires_grad_to_remove]

    graph.register_module_inputs(module_inputs)
    graph.register_module_targets(module_targets)
    graph.register_module_outputs(module_outputs)

    if return_intermediate:
        return graph, outputs, intermediate, inputs, target_tensors

    return graph, outputs, {}, inputs, target_tensors
