# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from loguru import logger
import torch
from typing import Dict, List, Any, Optional, Union


from forge._C import ForgeGraphModule, DataFormat
from forge.tensor import forge_dataformat_to_pytorch_dtype
from forge._C.graph import Graph
import forge._C.graph as pygraph
from forge._C.runtime import (
    Binary,
    TensorPool,
    Tensor as CTensor,
    ModelState,
    create_program_state,
    ProgramType,
)
from forge._C import run_mlir_compiler_to_cpp, run_mlir_compiler_to_shared_object
from forge.tensor import Tensor, get_post_const_eval_tensors, to_pt_tensors, cast_unsupported_torch_dtype, AnyTensor
from forge.module import Module, PyTorchModule, AnyModule


class CompileResults:
    """
    Wrapper for result from the graph compiler. Contains initial and final graphs, output tensors,
    and, optionally golden results for final output and intermediates, if desired.
    """

    outputs: List[Tensor]
    golden_outputs: List[torch.Tensor]
    golden_intermediates: Dict[str, torch.Tensor]
    initial_graph: Graph
    final_graph: Graph
    loss_module: Optional[Module]
    optimizer: Optional[torch.optim.Optimizer]

    pass_specific_output_kwargs: Dict[str, Any] = {}


@dataclass_json
@dataclass()
class CompiledGraphState:
    graph: Graph
    ordered_input_names: List[str]
    ordered_input_gradient_names: List[str]
    ordered_output_names: List[str]
    ordered_external_output_names: List[str]
    ordered_target_names: List[str]
    ordered_constant_node_names: List[str]
    ordered_parameter_node_names: List[str]
    ordered_intermediate_names: List[str]
    aliased_outputs: Dict[str, str]

    # JSON trace for each node in the graph which can be consteval'd
    consteval_trace: Dict[str, Any]
    post_const_eval_constants: Dict[str, torch.Tensor]
    post_const_eval_parameters: Dict[str, torch.Tensor]

    # Hold cpu-evaluated outputs loaded from TTI
    cpueval_outputs: Optional[List[torch.Tensor]] = field(default=None)

    has_cache_buffers: bool = False

    @staticmethod
    def from_compiled_graph(
        module: Module, graph: Graph, optimizer_params: Optional[Dict[str, Tensor]] = None
    ) -> "CompiledGraphState":
        ordered_input_names = graph.get_ordered_input_names()
        ordered_output_names = graph.get_ordered_output_names()
        ordered_external_output_names = graph.get_ordered_external_output_names()
        ordered_target_names = graph.get_ordered_target_names()
        ordered_intermediate_names = graph.get_ordered_intermediate_names()
        ordered_output_nodes = graph.get_ordered_output_nodes()
        aliased_outputs: Dict[str, str] = {}
        for node in ordered_output_nodes:
            assert isinstance(node, pygraph.OutputNode)
            if node.is_aliased:
                aliased_outputs[node.name] = node.alias

        ordered_constant_node_names = [constant_node.name for constant_node in graph.get_constant_nodes()]
        ordered_parameter_node_names = [parameter_node.name for parameter_node in graph.get_parameter_nodes()]
        ordered_optimizer_parameter_node_names = [
            parameter_node.name for parameter_node in graph.get_optimizer_parameter_nodes()
        ]
        if len(ordered_optimizer_parameter_node_names) > 0:
            ordered_parameter_node_names.extend(ordered_optimizer_parameter_node_names)
        ordered_input_gradient_names = graph.get_ordered_input_gradient_names()

        consteval_trace = pygraph.record_consteval_operations(graph)

        has_cache_buffers = False

        constant_to_tensor: Dict[str, torch.Tensor] = {}
        if isinstance(module, Module):
            for p in module.get_parameters():
                value = p.value()
                if value == None:
                    raise ValueError(f"Parameter {p.get_name()} has no value")
                constant_to_tensor[p.get_name()] = p.value()

        if optimizer_params is not None:
            for name, opt_param in optimizer_params.items():
                constant_to_tensor[name] = opt_param.value()

        post_const_eval_constants: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph,
            constant_to_tensor,
            consteval_trace,
            ordered_constant_node_names,
        )

        post_const_eval_parameters: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph,
            constant_to_tensor,
            consteval_trace,
            ordered_parameter_node_names,
        )

        return CompiledGraphState(
            graph=graph,
            ordered_input_names=ordered_input_names,
            ordered_input_gradient_names=ordered_input_gradient_names,
            ordered_output_names=ordered_output_names,
            aliased_outputs=aliased_outputs,
            ordered_external_output_names=ordered_external_output_names,
            ordered_target_names=ordered_target_names,
            ordered_constant_node_names=ordered_constant_node_names,
            ordered_parameter_node_names=ordered_parameter_node_names,
            ordered_intermediate_names=ordered_intermediate_names,
            consteval_trace=consteval_trace,
            post_const_eval_constants=post_const_eval_constants,
            post_const_eval_parameters=post_const_eval_parameters,
            has_cache_buffers=has_cache_buffers,
        )

    def get_tensor(self, name_to_tensor: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        assert name in name_to_tensor
        return name_to_tensor[name]

    def get_constant_tensor(self, name: str) -> torch.Tensor:
        return self.get_tensor(self.post_const_eval_constants, name)

    def get_ordered_constant_tensors(self) -> List[torch.Tensor]:
        return [self.get_constant_tensor(name) for name in self.ordered_constant_node_names]

    def get_parameter_tensor(self, name: str) -> torch.Tensor:
        return self.get_tensor(self.post_const_eval_parameters, name)

    def get_ordered_parameter_tensors(self) -> List[torch.Tensor]:
        return [self.get_parameter_tensor(name) for name in self.ordered_parameter_node_names]


class ProgramId(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    OPTIMIZER = 2


class CompiledModel:
    """
    Callable object for running the compiled model on the device(s).

    If the model is compiled for inference, only forward pass can be executed.
    In case of training - forward, backward, loss and optimizer steps can be executed - depending on which of these
    is compiled for the device, and which are set up to be ran separately on the CPU.
    """

    runtime_model_state: ModelState

    fwd_compiled_graph_state: CompiledGraphState
    tensor_pool: TensorPool
    bwd_compiled_graph_state: Optional[CompiledGraphState]
    opt_compiled_graph_state: Optional[CompiledGraphState]

    # Compiled flatbuffer binary composed of programs which execute compiled graphs (e.g., forward, backward, etc.)
    compiled_binary: Binary

    inputs: List[CTensor]
    outputs: Dict[str, CTensor]
    intermediates: List[CTensor]

    # Original user-defined module.
    framework_module: AnyModule

    # Forge graph module, currently used for exporting the model to a cpp file.
    # Needed by the lower to MLIR logic.
    # Issue(#1350): current state of `CompiledModel` is a bit messy, we should clean it up.
    forge_graph_module: ForgeGraphModule

    # Gradients to be passed into the backward pass.
    # Used when CompiledModel.backward() is part of a chain of backward passes.
    gradient_inputs: List[Optional[CTensor]]
    gradient_outputs: List[CTensor]

    attached_module: Optional["CompiledModel"]

    df_override_dtype: Optional[Union[torch.dtype, DataFormat]] = None

    def __init__(
        self,
        forge_graph_module: ForgeGraphModule,
        fwd_compiled_graph_state: CompiledGraphState,
        bwd_compiled_graph_state: Optional[CompiledGraphState],
        opt_compiled_graph_state: Optional[CompiledGraphState],
        compiled_binary: Binary,
        framework_module: AnyModule,
        attached_module: Optional["CompiledModel"] = None,
        df_override_dtype: Optional[Union[torch.dtype, DataFormat]] = None,
    ):
        self.forge_graph_module = forge_graph_module

        self.runtime_model_state = ModelState(compiled_binary)
        self.tensor_pool = self.runtime_model_state.tensor_pool

        self.fwd_compiled_graph_state = fwd_compiled_graph_state
        self.create_program_state(ProgramType.Forward, self.fwd_compiled_graph_state)

        self.bwd_compiled_graph_state = bwd_compiled_graph_state
        if self.bwd_compiled_graph_state is not None:
            self.create_program_state(ProgramType.Backward, self.bwd_compiled_graph_state)

        self.opt_compiled_graph_state = opt_compiled_graph_state
        if self.opt_compiled_graph_state is not None:
            self.create_program_state(ProgramType.Optimizer, self.opt_compiled_graph_state)

        self.compiled_binary = compiled_binary
        self.inputs = []
        self.framework_module = framework_module
        self.intermediates = []
        if self.bwd_compiled_graph_state is not None:
            self.gradient_inputs = [None] * len(self.bwd_compiled_graph_state.ordered_input_gradient_names)
        self.outputs = {}
        self.attached_module = attached_module
        self.gradient_outputs = []
        self.df_override_dtype = df_override_dtype

    def create_persistent_inputs(self, tensor_pool: TensorPool, compiled_graph_state: CompiledGraphState):
        persistent_inputs = []
        for name, value in zip(
            compiled_graph_state.ordered_constant_node_names, compiled_graph_state.get_ordered_constant_tensors()
        ):
            persistent_inputs.append((name, value))

        for name, value in zip(
            compiled_graph_state.ordered_parameter_node_names, compiled_graph_state.get_ordered_parameter_tensors()
        ):
            persistent_inputs.append((name, value))

        for name, tensor in persistent_inputs:
            tensor_pool.insert(name, tensor)

    def create_program_state(self, program_type: ProgramType, compiled_graph_state: CompiledGraphState):
        self.create_persistent_inputs(self.tensor_pool, compiled_graph_state)

        persistent_tensors = [
            *compiled_graph_state.ordered_constant_node_names,
            *compiled_graph_state.ordered_parameter_node_names,
        ]

        pstate = create_program_state(program_type, self.tensor_pool, persistent_tensors)
        self.runtime_model_state.init_program_state(pstate)

    def tie_grad_fn(self, grad_id: int, grad: torch.Tensor) -> None:
        """
        Hook function to tie the gradients produced by torch as inputs to the backward pass which will be ran on the
        TT device.

        NOTE: Should be used only when loss is computed on CPU (outside of our runtime).
        """
        assert len(self.gradient_inputs) > grad_id, "More gradients than expected."
        self.gradient_inputs[grad_id] = CTensor(grad)

    def __call__(self, *inputs: AnyTensor) -> List[torch.Tensor]:
        """
        Run inference on the compiled model.

        Parameters
        ----------
        inputs: [Tensor, ...]
            Input tensors

        Returns
        -------
        List[Tensor]
            Output tensors
        """
        # If a lower-precision data-format override was requested at compile time (e.g. bfloat16
        # for ONNX models where ONNXRuntime cannot run CPUExecutionProvider in bfloat16), cast
        # all floating-point input tensors to the target dtype before runtime dispatch.
        # Integer / boolean tensors are left unchanged.
        if self.df_override_dtype is not None:
            if isinstance(self.df_override_dtype, DataFormat):
                df_override_dtype = forge_dataformat_to_pytorch_dtype(self.df_override_dtype)
            else:
                df_override_dtype = self.df_override_dtype
            inputs = [
                t.to(df_override_dtype) if isinstance(t, torch.Tensor) and torch.is_floating_point(t) else t
                for t in inputs
            ]

        # Handle mixed inputs: torch tensors and CTensors
        inputs_runtime = []
        for t in inputs:
            if isinstance(t, CTensor):
                inputs_runtime.append(t)
            else:
                torch_tensor = to_pt_tensors([t])[0]
                torch_tensor = cast_unsupported_torch_dtype(torch_tensor)
                inputs_runtime.append(CTensor(torch_tensor))

        if self.training() and isinstance(self.framework_module, PyTorchModule):
            for name, param in self.framework_module.module.named_parameters():
                if param.requires_grad:
                    our_tensor = self.fwd_compiled_graph_state.get_parameter_tensor(name)

                    # NOTE: for parameters that require gradients, we want to share the same tensor with the PyTorch
                    # module. This is because we want to be able to optimize the parameters both on the device
                    # (through our runtime) and via the torch optimizers. So this ensures that whichever side updates
                    # the parameter value, the other side can see the change.
                    #
                    # This could change in the future, but for now ensure that our premise is correct.
                    assert param is our_tensor

            if not self.optimizer_on_device():
                self.remove_weights_from_device()

        logger.info(
            f"Running model {self.framework_module.get_name()} {self.fwd_compiled_graph_state.graph.get_name()} on device..."
        )

        self.inputs = inputs_runtime
        self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)

        all_outputs = self.runtime_model_state.get_outputs(ProgramType.Forward)

        self.intermediates = []

        # The model_outputs will contain outputs that we need to return to the user, i.e. external outputs.
        model_outputs = []
        for idx, output_name in enumerate(self.fwd_compiled_graph_state.ordered_output_names):
            output = all_outputs[idx]
            if output_name in self.fwd_compiled_graph_state.ordered_intermediate_names:
                self.intermediates.append(output)
            if output_name in self.fwd_compiled_graph_state.ordered_external_output_names:
                self.outputs[output_name] = output
                model_outputs.append(output.to_torch())

        if self.training():
            # For executing loss and its backward graph on CPU, we need to tell torch to compute gradients.
            for idx, output in enumerate(model_outputs):
                output.requires_grad = True
                # NOTE: the default idx parameter for the lambda is used to capture the idx by value. Otherwise, the lambda
                # would capture the idx by reference, and all the lambdas would have the same idx value.
                output.register_hook(lambda grad, idx=idx: self.tie_grad_fn(idx, grad))

        return model_outputs

    def forward(self, *inputs: AnyTensor) -> List[torch.Tensor]:
        return self(*inputs)

    def backward(self) -> List[CTensor]:
        assert self.training(), "Model not compiled for training."
        assert self.bwd_compiled_graph_state is not None, "Backward graph should be present for training."

        for grad in self.gradient_inputs:
            assert grad is not None, "Gradients not provided for backward pass."

        # Inputs from forward pass are needed in backward pass only if
        # they are used in the backward pass computation
        # They will be used if there is backward operation that explicitly requires them
        # as in other cases, intermediate tensors can be used if they exists
        inputs = [
            self.inputs[i]
            for i, name in enumerate(self.fwd_compiled_graph_state.ordered_input_names)
            if name in self.bwd_compiled_graph_state.ordered_input_names
        ]

        logger.info(
            f"Running backward pass on model {self.framework_module.get_name()} {self.bwd_compiled_graph_state.graph.get_name()} on device..."
        )

        bwd_inputs = [*self.gradient_inputs, *self.intermediates, *inputs]
        assert all([isinstance(t, CTensor) for t in bwd_inputs]), "All inputs should be CTensors by now."

        self.runtime_model_state.run_program(ProgramType.Backward, bwd_inputs)
        grads = self.runtime_model_state.get_outputs(ProgramType.Backward)

        if self.optimizer_on_device():
            if self.gradient_outputs is None or len(self.gradient_outputs) == 0:
                self.gradient_outputs = grads
            else:
                # TODO (Issue #1530): Handle gradient accumulation
                assert len(self.gradient_outputs) == len(grads), "Number of gradients does not match number of outputs"
                assert False, "Gradient accumulation for grads on device not implemented yet"
        else:
            self.gradient_outputs = grads
            # Accumulate gradients in the PyTorch module
            if isinstance(self.framework_module, PyTorchModule):
                for name, param in self.framework_module.module.named_parameters():
                    for idx, grad_output_name in enumerate(self.bwd_compiled_graph_state.ordered_output_names):
                        if name in grad_output_name:
                            grad_tensor = grads[idx].to_torch()
                            if param.shape != grad_tensor.shape:
                                # Our gradients have additional dimensions which PyTorch not expects (e.g. [1, 1, N, M] -> [N, M])
                                assert (torch.squeeze(grad_tensor)).shape == param.shape
                                grad_tensor = torch.squeeze(grad_tensor)

                            if param.grad is not None:
                                param.grad += grad_tensor
                            else:
                                param.grad = grad_tensor

        # Pass on the calculated gradients to the attached module
        if self.attached_module is not None:
            # pass on the calculated gradients and call the attached module's backward pass
            # HACK: we don't have a way to know which gradient outputs are tied to which gradient inputs
            # of the attached module. For now, just attach the first one since we are doing this only for
            # the loss module (which will have only one gradient output) and the model will need only one
            # gradient output to be passed to the loss module.
            assert len(self.gradient_outputs) == 1, "Passing gradients not properly implemented yet"
            assert len(self.attached_module.gradient_inputs) == 1, "Passing gradients not properly implemented yet"
            self.attached_module.gradient_inputs[0] = self.gradient_outputs[0]
            self.attached_module.backward()

        return self.gradient_outputs

    def training(self) -> bool:
        return self.fwd_compiled_graph_state.graph.training()

    def optimizer_on_device(self) -> bool:
        return self.opt_compiled_graph_state is not None

    def step(self) -> None:
        assert self.fwd_compiled_graph_state.graph.training(), "Model not compiled for training."
        assert self.opt_compiled_graph_state is not None, "Optimizer graph should be present for training."
        assert self.bwd_compiled_graph_state is not None, "Backward graph should be present for training."

        inputs = [
            *self.gradient_outputs,
        ]

        logger.info(
            f"Running optimizer step on model {self.framework_module.get_name()} {self.opt_compiled_graph_state.graph.get_name()} on device..."
        )

        self.runtime_model_state.run_program(ProgramType.Optimizer, inputs)
        out_params = self.runtime_model_state.get_outputs(ProgramType.Optimizer)

        update_param: Dict[str, CTensor] = {}
        for idx, param in enumerate(self.opt_compiled_graph_state.ordered_output_names):
            update_param[param] = out_params[idx]

        for weight_update_name in self.opt_compiled_graph_state.aliased_outputs:
            weight_name = self.opt_compiled_graph_state.aliased_outputs[weight_update_name]
            # self.opt_compiled_graph_state.get_parameter_tensor(weight_name).data = update_param[weight_update_name].data
            self.tensor_pool.update_tensor(
                weight_name, update_param[weight_update_name]
            )  # get_tensor(weight_name).set_tensor_data(update_param[weight_update_name])

            # Sanity check - assert that the parameter tensors in framework module are the same as the ones in our runtime.
            assert isinstance(
                self.framework_module, PyTorchModule
            ), "For now only PyTorchModule is supported in training"
            for torch_name, val in self.framework_module.module.named_parameters():
                if torch_name == weight_name:
                    assert self.opt_compiled_graph_state.get_parameter_tensor(
                        weight_name
                    ) is self.fwd_compiled_graph_state.get_parameter_tensor(weight_name)
                    assert self.fwd_compiled_graph_state.get_parameter_tensor(weight_name) is val

        self.gradient_outputs = []

    def save(self, output_path: str) -> None:
        """
        Save the model to a flatbuffer file.

        Parameters
        ----------
        output_path: str
            Path to the file where the flatbuffer will be saved.
        """
        if not output_path.endswith(".ttnn"):
            raise ValueError("The flatbuffer file must have a '.ttnn' extension.")
        self.compiled_binary.store(output_path)
        logger.info(f"Saved model as flatbuffer file to {output_path}")

    def export_to_cpp(self, export_path: str) -> None:
        """
        Export the model to a cpp file.

        Parameters
        ----------
        export_path: str
            Path to the file where the model c++ code will be exported.
        """

        logger.info(f"Exporting model {self.framework_module.get_name()} to cpp file...")
        cpp_code = run_mlir_compiler_to_cpp(self.forge_graph_module)

        with open(export_path, "w") as f:
            f.write(cpp_code)

        logger.info(f'Exported model as cpp file to "{export_path}"')
        logger.info(
            f"To compile and run this code, one can utilize the ttnn-standalone tool within the tt-mlir project. \
            It provides all the necessary build and run scripts. Copy the contents of the .cpp to ttnn-standalone.cpp \
            and use `./run` to compile&run the code."
        )
        logger.info("    Tool: https://github.com/tenstorrent/tt-mlir/tree/main/tools/ttnn-standalone")
        logger.info("    Docs: https://docs.tenstorrent.com/tt-mlir/ttnn-standalone.html")

    def export_to_shared_object(self) -> str:
        """
        Export the model to a shared object file.

        Parameters
        ----------
        export_path: str
            Path to the file where the model shared object code will be exported.
        """

        logger.info(f"Exporting model {self.framework_module.get_name()} to shared object file...")
        path_to_so = run_mlir_compiler_to_shared_object(self.forge_graph_module)

        logger.info(f'Exported model as shared object file to "{path_to_so}"')

        return path_to_so

    def update_host_weights(self):
        for name, param in self.framework_module.module.named_parameters():
            if param.requires_grad:
                weight = self.tensor_pool.get_tensor(name)
                weight.update_host_data()

    def remove_weights_from_device(self):
        for name, param in self.framework_module.module.named_parameters():
            if param.requires_grad:
                self.tensor_pool.get_tensor(name).detach_from_device()
