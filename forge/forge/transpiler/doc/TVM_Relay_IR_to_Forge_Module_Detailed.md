# TVM Relay IR to Forge Module: Detailed Line-by-Line Analysis

## Executive Summary

This document provides a comprehensive, line-by-line analysis of how framework models (PyTorch, ONNX, TensorFlow, etc.) are converted to TVM Relay IR and then transformed into executable Forge Python modules. This is the core code generation pipeline used by the Forge compiler.

---

## Table of Contents

1. [High-Level Flow](#high-level-flow)
2. [Phase 1: Framework Model → TVM Relay IR](#phase-1-framework-model--tvm-relay-ir)
3. [Phase 2: TVM Relay IR → JSON Graphs](#phase-2-tvm-relay-ir--json-graphs)
4. [Phase 3: JSON Graphs → Python Forge Module](#phase-3-json-graphs--python-forge-module)
5. [Phase 4: Python Module → ForgeModule Instance](#phase-4-python-module--forgemodule-instance)
6. [Key Data Structures](#key-data-structures)
7. [Operation Mapping System](#operation-mapping-system)
8. [Code Examples](#code-examples)

---

## High-Level Flow

```
Framework Model (PyTorch/ONNX/TensorFlow/etc.)
    ↓
[Phase 1] Convert to TVM Relay IR
    ↓
[Phase 2] Compile & Partition TVM Relay IR
    ↓
[Phase 3] Extract JSON Graphs
    ↓
[Phase 4] Generate Python Code
    ↓
[Phase 5] Import & Instantiate ForgeModule
```

---

## Phase 1: Framework Model → TVM Relay IR

### Entry Point: `generate_forge_module()` (`tvm_to_python.py:1808`)

**Purpose:** Main entry point for generating Forge modules from framework models.

**Function Signature:**
```python
def generate_forge_module(
    framework_mod,      # Wrapped framework module (PyTorchModule, OnnxModule, etc.)
    inputs,              # Sample input tensors
    compiler_cfg=None,   # Compiler configuration
    graph_name=None,     # Name for the graph
    verify_cfg=None,     # Verification configuration
    clean_later=False,   # Whether to clean up generated files
    input_names=[],      # Optional input names
):
```

**Line-by-Line Analysis:**

```python
# Lines 1819-1824: Initialize configurations
if compiler_cfg is None:
    compiler_cfg = CompilerConfig()  # Default compiler config
if verify_cfg is None:
    verify_cfg = _get_global_verify_config()  # Get global verification config

# Line 1825: Convert inputs to PyTorch tensors
pytorch_inputs = to_pt_tensors(inputs)  # Normalize all inputs to torch.Tensor

# Lines 1827-1828: Set graph name
if graph_name is None:
    graph_name = framework_mod.name  # Use module name if not provided

# Lines 1830-1834: Handle module reloading (for testing/debugging)
reload = bool(int(os.environ.get("FORGE_RELOAD_GENERATED_MODULES", "0")))
if reload and not compiler_cfg.retain_tvm_python_files:
    compiler_cfg.retain_tvm_python_files = True  # Keep files if reloading
    if not os.path.exists(metadata_path(graph_name)):
        reload = False  # Disable reload if metadata doesn't exist

# Lines 1836-1837: Get framework outputs for verification
if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
    framework_outputs = framework_mod.cpu_eval_forward(*pytorch_inputs)
    # Run framework model to get golden outputs for comparison

# Lines 1839-1851: Compile TVM to Python or reload
if not reload:
    module_name = graph_name if counter == 0 else f"{graph_name}_{counter}"
    # Call the main compilation function
    module_writers, flattened_inputs = compile_tvm_to_python(
        framework_mod,      # Framework module
        graph_name,         # Graph name
        pytorch_inputs,     # Input tensors
        module_name=module_name,  # Generated module name
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
        input_names=input_names,
    )
else:
    # Reload previously generated modules
    module_writers, flattened_inputs = load_writers_metadata(graph_name, inputs)

counter += 1  # Increment global counter for unique module names
sys.path.append(".")  # Add current directory to Python path

# Lines 1856-1879: Load and instantiate generated modules
forge_mods = []
devices = []
for writer in module_writers:
    # Load the generated Python module dynamically
    module_name = writer.module_name
    file_path = os.path.join(writer.module_directory, writer.filename)
    module = import_from_path(module_name, file_path)  # Dynamic import
    
    # Get the generated class
    TestClass = getattr(module, writer.class_name)
    
    devices.append(writer.dev)  # Track device type (TTDevice or CPUDevice)
    
    if writer.dev == "CPUDevice":
        # For CPU fallback, wrap in PyTorchModule
        forge_mod = forge.PyTorchModule(writer.module_name, TestClass())
        forge_mod.module.process_framework_parameters(framework_mod.module)
    else:
        # For TTDevice, instantiate directly
        forge_mod = TestClass(writer.module_name)
        forge_mod.process_framework_parameters(framework_mod.module)
        
        # Verify all parameters were loaded
        assert not any(
            [param.value() is None for param in forge_mod.get_parameters()]
        ), f"Could not retrieve parameters from framework and tvm"
    
    forge_mods.append(forge_mod)
    
    # Cleanup temporary files if not retaining
    if not compiler_cfg.retain_tvm_python_files:
        global generated_files
        generated_files.append(writer.filename)
        param_filename = os.path.join(writer.module_directory, writer.module_name + "_params.pt")
        if os.path.exists(param_filename):
            generated_files.append(os.path.abspath(param_filename))
        
        if not clean_later:
            cleanup_temporary_files()

# Lines 1891-1894: Convert inputs to appropriate format
if devices[0] == "CPUDevice":
    forge_inputs = forge.tensor.to_pt_tensors(flattened_inputs)
else:
    forge_inputs = forge.tensor.to_forge_tensors(flattened_inputs)

# Lines 1896-1898: Verify generated module against framework
if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
    forge_outputs = get_forge_outputs(forge_mods, devices, forge_inputs)
    verify_framework_vs_forge_codegen(framework_outputs, forge_outputs, verify_cfg=verify_cfg)

return forge_mods, devices, forge_inputs
```

### Core Compilation: `compile_tvm_to_python()` (`tvm_to_python.py:1903`)

**Purpose:** Convert framework model to TVM Relay IR and generate Python code.

**Line-by-Line Analysis:**

```python
# Lines 1912-1913: Initialize compiler config
if compiler_cfg is None:
    compiler_cfg = CompilerConfig()

# Line 1915: Determine if in training mode
is_training = False if verify_cfg == None else verify_cfg.test_kind.is_training()

# Line 1917: Detect framework type
framework = get_framework(framework_mod)  # Returns "pytorch", "onnx", "tensorflow", etc.

# Lines 1918-1924: Set model to eval/train mode
if framework in ["pytorch", "paddle"]:
    if is_training:
        framework_mod.module.train()
        verify_cfg.verify_tvm_compile = False
        logger.warning("Cannot verify TVM output vs. framework in training mode.")
    else:
        framework_mod.module.eval()  # Set to evaluation mode

# Lines 1926-1931: Get model path for ONNX/TFLite
path = None
if isinstance(framework_mod, OnnxModule):
    path = framework_mod.onnx_path
elif isinstance(framework_mod, TFLiteModule):
    path = framework_mod.tflite_path

# Line 1934: Import TVM graph loader (lazy import to avoid unnecessary TVM dependency)
from forge.tvm_calls.forge_compile import load_tvm_graph

# Lines 1936-1945: Load TVM graph (converts framework → TVM Relay IR → JSON)
json_graphs, flattened_pytorch_inputs, weights = load_tvm_graph(
    inputs,                    # Input tensors
    framework_mod.module,      # Framework model (torch.nn.Module, onnx.ModelProto, etc.)
    compiler_cfg,              # Compiler configuration
    graph_name,                # Graph name
    framework,                 # Framework type string
    path=path,                 # Path to model file (for ONNX/TFLite)
    verify_cfg=verify_cfg,     # Verification config
    input_names=input_names,   # Input names
)
# Returns:
# - json_graphs: List of JSON-serialized graphs (CPU pre, device, CPU post)
# - flattened_pytorch_inputs: Flattened input tensors
# - weights: Dictionary mapping weight names to (tensor, requires_grad) tuples
```

### Framework-Specific Conversion: `load_tvm_graph()` (`tvm_calls/forge_compile.py:79`)

**Purpose:** Route to framework-specific TVM compilation.

**Line-by-Line Analysis:**

```python
# Lines 114-126: Compile TVM graph or load from cache
json_graphs, flattened_inputs = compile_tvm_graph(
    inputs,
    module,
    compiler_cfg,
    graph_name=graph_name,
    input_names=input_names,
    path=path,
    verify_cfg=verify_cfg,
    framework=framework,
)

# Lines 128-130: Format weights from framework model
flattened_pytorch_inputs, weights = format_tvm_graph_weights(
    flattened_inputs, module, compiler_cfg, framework=framework
)

# Line 132: Serialize and store TVM graph (for caching)
serialize_and_store_tvm_graph(json_graphs, compiler_cfg, framework=framework)

return json_graphs, flattened_pytorch_inputs, weights
```

### ONNX-Specific Conversion: `compile_onnx_for_forge()` (`tvm_calls/forge_compile.py:769`)

**Purpose:** Convert ONNX model to TVM Relay IR.

**Line-by-Line Analysis:**

```python
# Lines 770-782: Setup ONNX Runtime session for verification
import onnxruntime as ort

so = ort.SessionOptions()
so.inter_op_num_threads = 2
so.intra_op_num_threads = 2

if onnx_path is not None:
    onnx_model = onnx_path
else:
    onnx_model = onnx_mod.SerializeToString()  # Serialize in-memory model

onnx_session = ort.InferenceSession(onnx_model, sess_options=so, providers=["CPUExecutionProvider"])

# Lines 784-792: Extract input information
input_names = []
for inp in onnx_session.get_inputs():
    input_names.append(inp.name)

input_dict = {}
input_shape_dict = {}
for name, tensor in zip(input_names, inputs):
    input_dict[name] = tensor
    input_shape_dict[name] = tensor.shape

# Line 794: Validate input count
assert len(input_names) == len(inputs), "Number of input names must match number of inputs"

# Lines 796-803: Get framework outputs for verification
framework_outputs = extract_framework_model_outputs(
    framework="onnx",
    model=onnx_mod,
    inputs=inputs,
    verify_tvm_compile=verify_cfg.verify_tvm_compile,
    input_dict=input_dict,
    onnx_session=onnx_session,
)

# Lines 805-807: Cleanup large objects
del onnx_session
del onnx_model

# Lines 809-815: Check TVM graph cache
graph_hash = hashlib.sha256()
if is_tvm_cache_enabled():
    graph_string = str(onnx_mod).encode("utf-8")
    graph_hash.update(graph_string)
    cached_graphs = load_serialized_tvm_graph(compiler_cfg, graph_hash.hexdigest(), framework="onnx")
    if cached_graphs is not None:
        return cached_graphs, inputs  # Return cached graphs if available

# Line 817: Convert ONNX to TVM Relay IR
mod, params = relay.frontend.from_onnx(onnx_mod, input_shape_dict, freeze_params=False)
# This is the KEY conversion:
# - onnx_mod: ONNX ModelProto
# - input_shape_dict: Dictionary mapping input names to shapes
# - freeze_params=False: Keep parameters as variables (not constants)
# Returns:
#   - mod: TVM IRModule containing Relay IR
#   - params: Dictionary of parameter tensors

# Line 818: Convert dynamic shapes to static
mod = relay.transform.DynamicToStatic()(mod)
# TVM Relay supports dynamic shapes, but Forge needs static shapes
# This pass infers and fixes all dynamic dimensions

# Line 819: Record execution stage
record_execution(ExecutionStage.FAILED_TVM_RELAY_IR_TRANSFORMATION)

# Lines 821-828: Handle constant propagation
if not compiler_cfg.enable_tvm_constant_prop:
    # Bind empty params (parameters remain as variables)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
else:
    # Convert framework params to TVM and constant fold
    assert compiler_cfg.convert_framework_params_to_tvm, \
        "Cannot use constant prop without converting framework params to relay"
    propped_params = {k: (v, True) for k, v in params.items()}
    mod = tvm.IRModule.from_expr(
        tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params)
    )

# Lines 830-840: Compile TVM Relay IR for Forge
partitioned_mod, forge_params = compile_tvm_for_forge(
    mod,                    # TVM IRModule
    params,                 # Parameter dictionary
    input_dict,             # Input dictionary
    framework_outputs,      # Framework outputs for verification
    input_names=input_names,
    graph_name=graph_name,
    return_params=True,
    compiler_cfg=compiler_cfg,
    verify_cfg=verify_cfg,
)
# This function:
# 1. Runs Relay optimization passes
# 2. Runs Forge-specific passes
# 3. Partitions graph into CPU and device regions
# 4. Returns partitioned module and parameters

# Lines 842-845: Extract JSON graphs from partitioned module
weight_names = [weight.name for weight in onnx_mod.graph.initializer]
json_graphs = extract_graphs(
    partitioned_mod, forge_params, input_names, weight_names, graph_hash=graph_hash.hexdigest()
)
# Converts TVM Relay IRModule to JSON representation

return json_graphs, inputs
```

### TVM Relay IR Compilation: `compile_for_forge()` (`tvm_calls/relay/op/forge.py:1017`)

**Purpose:** Compile TVM Relay IR with Forge-specific optimizations.

**Line-by-Line Analysis:**

```python
# Lines 1027-1037: Validate and normalize input
if not isinstance(relay_module, (IRModule, _function.Function)):
    raise ValueError("Type of input parameter mod must be tvm.IRModule")

if isinstance(relay_module, _function.Function):
    # Legacy support: convert Function to IRModule
    if params:
        relay_module = bind_params_by_name(relay_module, params)
    relay_module = IRModule.from_expr(relay_module)
    logger.warning("Please use input parameter mod (tvm.IRModule) instead of deprecated parameter func")

# Lines 1039-1044: Setup TVM compilation context
tophub_context = tvm.autotvm.utils.EmptyContext()

with tophub_context, tvm.transform.PassContext(opt_level=5):
    logger.trace("Before Compiling")
    logger.trace(relay_module.functions)
    dump_graph(relay_module, graph_name, "before_compiling")

# Line 1046: Run Relay optimization passes
relay_module = run_relay_compile_passes(relay_module)
# This includes:
# - Dead code elimination
# - Constant folding
# - Operator fusion
# - Type inference
dump_graph(relay_module, graph_name, "after_relay_passes")
record_execution(ExecutionStage.FAILED_TVM_PATTERN_CALLBACKS)

# Lines 1050-1053: Run Forge-specific compilation passes
compiled_relay_module = run_forge_compile_passes(
    relay_module, params, inputs, target, framework_outputs, verify_cfg
)
# Forge-specific passes include:
# - Operator lowering (convert unsupported ops)
# - Data format conversions
# - Memory layout optimizations
dump_graph(compiled_relay_module, graph_name, "after_forge_passes")
record_execution(ExecutionStage.FAILED_TVM_GRAPH_PARTITIONING)

# Lines 1056-1057: Warn about integer comparisons
warn_of_int_comparisons(compiled_relay_module)
# Integer comparisons may lead to incorrect results on hardware

return compiled_relay_module, params
```

### Graph Partitioning: `partition_for_forge()` (`tvm_calls/relay/op/forge.py:1989`)

**Purpose:** Partition graph into CPU and device (TT) regions.

**Line-by-Line Analysis:**

```python
# Line 1990: Initialize CPU fallback ops
initialize_forge_cpudevice_ops(mod, compiler_cfg)
# Marks certain ops as CPU-only based on compiler config

with tvm.transform.PassContext(opt_level=5):
    # Line 1996: Infer types
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    
    # Line 2000: Merge composite patterns
    mod = tvm.transform.Sequential([transform.MergeComposite(pattern_table())])(mod)
    # Combines multiple ops into single composite ops
    
    # Line 2004: Add NOPs to passthrough nodes
    mod["main"] = AddNopsToPassthrough().visit(mod["main"])
    # Ensures proper data flow
    
    # Line 2008: Infer types again
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    
    # Line 2012: Fold constants
    mod = tvm.transform.Sequential([transform.FoldConstant()])(mod)
    # Evaluates constant expressions at compile time
    
    # Line 2016: Enumerate nodes
    mod["main"] = EnumerateNodes().visit(mod["main"])
    # Assigns unique IDs to nodes
    
    # Line 2020: Infer types again
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    
    # Lines 2024-2025: Handle CPU fallback
    if compiler_cfg.enable_tvm_cpu_fallback:
        fallback_on_cpu(mod, compiler_cfg, input_names)
        # Marks unsupported ops to run on CPU
    
    # Line 2027: Annotate targets
    mod = tvm.transform.Sequential([transform.AnnotateTarget(["forge_cpudevice", "forge"])])(mod)
    # Annotates each node with target device (CPU or TT)
    
    # Line 2031: Merge compiler regions
    mod = tvm.transform.Sequential([transform.MergeCompilerRegions()])(mod)
    # Merges adjacent nodes with same target
    
    # Line 2035: Partition graph
    mod = tvm.transform.Sequential([transform.PartitionGraph(bind_constants=True)])(mod)
    # Splits graph into separate functions for each partition
    # bind_constants=True: Binds constants to their partitions
    
    # Lines 2039-2141: Process partitions and update parameters
    # ... (complex logic for handling partitioned modules)
    
    # Lines 2144-2149: Convert NaN attributes to zeros
    for partition_key, partition_val in params.items():
        for param_key, param_val in params[partition_key].items():
            param_val_np = param_val.asnumpy()
            if np.isnan(param_val_np).all():
                zero_mtx = tvm.nd.array(np.zeros(param_val_np.shape).astype(param_val.dtype))
                params[partition_key][param_key] = zero_mtx
    
    dump_graph(mod, graph_name, "after_forge_partition")
    
    return mod, params
```

---

## Phase 2: TVM Relay IR → JSON Graphs

### Graph Extraction: `extract_graphs()` (`tvm_calls/forge_compile.py:268`)

**Purpose:** Convert partitioned TVM Relay IRModule to JSON graphs.

**Line-by-Line Analysis:**

```python
# Lines 269-272: Extract main graph and set hash
mod = partitioned_mod["main"]
main_graph = str(mod.astext())  # Convert Relay IR to text representation
cpu_json_graph["hash"] = graph_hash
dev_json_graph["hash"] = graph_hash

# Lines 274-281: Validate function counts
assert len(cpu_json_graph["functions"].keys()) <= 2, \
    "At most two cpu functions should exist (one pre and one post)"
cpu_functions = list(cpu_json_graph["functions"].keys())
assert len(dev_json_graph["functions"].keys()) <= 1, \
    "At most one device function should exist"
device_function = list(dev_json_graph["functions"].keys())[0]

# Lines 283-308: Identify CPU pre and post functions
cpu_pre_function = None
cpu_post_function = None

func_callnodes = extract_function_callnodes(partitioned_mod["main"], partitioned_mod.get_global_vars())
for node in func_callnodes:
    if node.op.name_hint == device_function:
        # Find CPU function that feeds into device function
        for arg in node.args:
            op = None
            if isinstance(arg, tvm.relay.expr.TupleGetItem):
                op = arg.tuple_value.op
            elif isinstance(arg, tvm.relay.expr.Call):
                op = arg.op
            if op is not None:
                if op.name_hint != cpu_pre_function:
                    if not cpu_pre_function:
                        cpu_pre_function = op.name_hint
                    else:
                        assert op.name_hint == cpu_pre_function, \
                            "There is more than one cpu pre function"

if cpu_pre_function is not None:
    cpu_functions.remove(cpu_pre_function)
assert len(cpu_functions) <= 1, \
    "There is more than one cpu post function"
cpu_post_function = cpu_functions[0] if len(cpu_functions) else None

# Lines 310-335: Build CPU pre JSON graph
if cpu_pre_function is not None:
    cpu_pre_json_graph = copy.deepcopy(cpu_json_graph)
    cpu_pre_json_graph["graph"] = cpu_json_graph["functions"][cpu_pre_function]
    
    # Remove other functions
    functions_to_remove = []
    for function in cpu_pre_json_graph["functions"]:
        if function != cpu_pre_function:
            functions_to_remove.append(function)
    for func in functions_to_remove:
        del cpu_pre_json_graph["functions"][func]
    
    # Extract parameters for this function
    cpu_pre_json_graph["params"] = {}
    for function_name in forge_params.keys():
        if function_name == cpu_pre_function:
            cpu_pre_json_graph["params"].update({
                name: v.numpy()
                for (k, v), name in zip(
                    forge_params[function_name].items(),
                    cpu_pre_json_graph["param_names"][function_name]
                )
            })
else:
    cpu_pre_json_graph = {"graph": ""}

# Lines 337-349: Build device JSON graph
dev_json_graph["graph"] = dev_json_graph["functions"][device_function]

dev_json_graph["params"] = {}
for function_name in forge_params.keys():
    if function_name in dev_json_graph["param_names"]:
        dev_json_graph["params"].update({
            name: v.numpy()
            for (k, v), name in zip(
                forge_params[function_name].items(),
                dev_json_graph["param_names"][function_name]
            )
        })

# Lines 351-376: Build CPU post JSON graph (similar to CPU pre)
if cpu_post_function is not None:
    cpu_post_json_graph = copy.deepcopy(cpu_json_graph)
    cpu_post_json_graph["graph"] = cpu_json_graph["functions"][cpu_post_function]
    # ... (similar processing)
else:
    cpu_post_json_graph = {"graph": ""}

# Lines 378-408: Build final JSON graph list
json_graphs = []
if cpu_pre_function is not None:
    save_nid_to_input_idx(input_names, cpu_pre_json_graph)
    cpu_pre_json_graph["num_forge_inputs"] = len(input_names)
    json_graphs.append(
        copy.deepcopy(
            clean_names(
                json_graph=cpu_pre_json_graph,
                forge_params=forge_params,
                param_name_lookup=param_name_lookup
            )
        )
    )
else:
    save_nid_to_input_idx(input_names, dev_json_graph)
    dev_json_graph["num_forge_inputs"] = len(input_names)

json_graphs.append(
    copy.deepcopy(
        clean_names(
            json_graph=dev_json_graph,
            forge_params=forge_params,
            param_name_lookup=param_name_lookup
        )
    )
)

if cpu_post_json_graph["graph"] != "":
    json_graphs.append(
        copy.deepcopy(
            clean_names(
                json_graph=cpu_post_json_graph,
                forge_params=forge_params,
                param_name_lookup=param_name_lookup
            )
        )
    )

return json_graphs
```

**JSON Graph Structure:**
```json
{
    "graph": "<JSON string of graph structure>",
    "device": "tt" | "cpu",
    "params": {
        "param_name": <numpy array>
    },
    "param_names": {
        "function_name": ["param1", "param2", ...]
    },
    "functions": {
        "function_name": "<JSON string>"
    },
    "hash": "<graph hash>",
    "num_forge_inputs": <number>
}
```

---

## Phase 3: JSON Graphs → Python Forge Module

### JSON Graph Processing: `compile_tvm_to_python()` (continued)

**Line-by-Line Analysis (Processing JSON Graphs):**

```python
# Lines 1964-1965: Initialize module list
modules = []
for graph_index, json_graph in enumerate(json_graphs):
    # Process each JSON graph (CPU pre, device, CPU post)
    
    # Line 1966: Parse JSON graph
    graph = json.loads(json_graph["graph"])
    # Graph structure:
    # {
    #   "nodes": [list of node dictionaries],
    #   "arg_nodes": [list of input node indices],
    #   "heads": [list of output node indices],
    #   "node_row_ptr": [pointer array for node storage]
    # }
    
    # Line 1968: Check if this is CPU pre-processing
    is_cpu_pre = graph_index == 0 and json_graph["device"] == "cpu"
    
    # Line 1970: Extract output nodes
    output_nodes = [head[0] for head in graph["heads"]]
    
    # Lines 1972-1979: Define helper to detect no-op reshapes
    def is_nop_reshape(nid):
        node = graph["nodes"][nid]
        if node["name"] != "reshape":
            return False
        input_shape = graph["nodes"][node["inputs"][0][0]]["attrs"]["shape"]
        node_shape = node["attrs"]["shape"]
        return input_shape == node_shape  # Same shape = no-op
    
    # Line 1981: Extract input nodes
    input_nodes = graph["arg_nodes"]
    
    # Lines 1983-1991: Initialize data structures
    graph_input_names = {}      # Maps input index → name
    params = {}                 # Maps node_id → (name, shape, requires_grad, dtype)
    constants = {}              # Maps node_id → (name, shape, dtype)
    ops = {}                    # Maps node_id → Operation object
    returns = {}                # Maps output_node_id → output_name
    returns_requiring_batch_dim_fix = []
    forge_inputs = [None] * len(flattened_pytorch_inputs)
    params_from_tvm = {}
    node_name_to_node_type = {}
    
    # Lines 1993-2006: Helper to make names parser-friendly
    def make_parser_friendly_name(node, node_type):
        # Replace special characters that aren't valid in Python identifiers
        if framework == "tensorflow" or framework == "tf_graphdef" or framework == "jax":
            node["forge_name"] = node["forge_name"].replace(".", "_").replace("/", "_")
        elif framework == "pytorch":
            node["forge_name"] = node["forge_name"].replace(".", "_")
        elif framework == "onnx":
            node["forge_name"] = node["forge_name"].replace(".", "_").replace("/", "_").replace(":", "_")
        
        # Prevent names starting with digits
        if node["forge_name"][0] in [f"{n}" for n in range(10)]:
            node["forge_name"] = f"{node_type}_{node['name']}"
    
    # Lines 2008-2009: Initialize forge_name from name
    for node in graph["nodes"]:
        node["forge_name"] = node["name"]
    
    # Lines 2011-2029: Check for unsupported ops
    has_unsupported_ops = False
    json_graph_op_map = tvm_to_forge_op_map if json_graph["device"] == "tt" else tvm_to_pytorch_op_map
    for nid, node in enumerate(graph["nodes"]):
        if node["op"] != "kernel":
            continue
        if node["name"] not in json_graph_op_map:
            has_unsupported_ops = True
            unsupported_msg = f"Encountered unsupported op node type: {node['name']}, on device: {json_graph['device']}"
            logger.warning(unsupported_msg) if compiler_cfg.enable_tvm_unsupported_ops else logger.error(unsupported_msg)
    
    if has_unsupported_ops:
        assert compiler_cfg.enable_tvm_unsupported_ops, \
            "Encountered unsupported op types. Check error logs for more details"
    
    # Lines 2031-2263: Process each node in the graph
    for nid, node in enumerate(graph["nodes"]):
        node["nid"] = nid  # Assign node ID
        node["users"] = []  # Initialize users list
        
        # Extract shape
        shape = node["attrs"]["shape"][0][0]
        node["forge_shape"] = tuple(shape)
        
        # Process based on node type
        if node["op"] == "input":
            # Input node (either model input or parameter)
            if node["name"] not in weights:
                # Model input (not a weight)
                make_parser_friendly_name(node, "input_")
                inp_idx = nid
                if "nid_to_input_idx" in json_graph.keys() and len(json_graph["nid_to_input_idx"]) != 0:
                    inp_idx = json_graph["nid_to_input_idx"][nid]
                    forge_inputs[inp_idx] = flattened_pytorch_inputs[inp_idx]
                
                graph_input_names[inp_idx] = node["forge_name"]
                node_name_to_node_type[node["forge_name"]] = NodeType.Activation
                node["op"] = "*"  # Mark as input
            else:
                # Weight/parameter node
                tensor, requires_grad = weights[node["name"]]
                tensor.requires_grad = requires_grad
                
                if (requires_grad or json_graph["device"] == "cpu") and len(tensor.shape) > 0:
                    # Trainable parameter or CPU parameter
                    params[node["nid"]] = (
                        node["forge_name"],
                        node["forge_shape"],
                        requires_grad,
                        _determine_node_dtype(node),
                    )
                    node["op"] = "parameter"
                    node_name_to_node_type[node["forge_name"]] = NodeType.Parameter
                else:
                    # Constant (non-trainable)
                    if torch.numel(tensor) == 1 and len(tensor.shape) == 0:
                        tensor = tensor.reshape((1,))
                    if len(tensor.shape) > 4 and all([x == 1 for x in tensor.shape[0:-4]]):
                        tensor = tensor.reshape(tensor.shape[-4:])
                    if requires_grad:
                        params[node["nid"]] = (
                            node["forge_name"],
                            tensor.shape,
                            requires_grad,
                            _determine_node_dtype(node),
                        )
                        node_name_to_node_type[node["forge_name"]] = NodeType.Parameter
                    else:
                        constants[node["nid"]] = (
                            node["forge_name"],
                            tensor.shape,
                            _determine_node_dtype(node)
                        )
                        node_name_to_node_type[node["forge_name"]] = NodeType.Constant
        
        elif node["op"] == "const":
            # Constant node from TVM
            if isinstance(json_graph["params"][node["name"]], np.ndarray):
                tensor = torch.from_numpy(json_graph["params"][node["name"]])
            else:
                tensor = torch.tensor(json_graph["params"][node["name"]])
            
            requires_grad = node["attrs"]["is_param"] != "0"
            
            if requires_grad and len(tensor.shape) > 0:
                # Parameter
                if tensor.dtype == torch.bool:
                    requires_grad = False
                    node["attrs"]["is_param"] = "0"
                params_from_tvm[node["forge_name"]] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
                params[node["nid"]] = (
                    node["forge_name"],
                    node["forge_shape"],
                    requires_grad,
                    _determine_node_dtype(node),
                )
                node["op"] = "parameter"
                node_name_to_node_type[node["forge_name"]] = NodeType.Parameter
            else:
                # Constant
                if torch.numel(tensor) == 1 and len(tensor.shape) == 0:
                    tensor = tensor.reshape((1,))
                if len(tensor.shape) > 4 and all([x == 1 for x in tensor.shape[0:-4]]):
                    tensor = tensor.reshape(tensor.shape[-4:])
                params_from_tvm[node["forge_name"]] = tensor
                node["op"] = "constant"
                constants[node["nid"]] = (node["forge_name"], tensor.shape, _determine_node_dtype(node))
                node_name_to_node_type[node["forge_name"]] = NodeType.Constant
        
        elif node["op"] == "kernel":
            # Operation node
            op_map = tvm_to_forge_op_map if json_graph["device"] == "tt" else tvm_to_pytorch_op_map
            if node["name"] in op_map:
                op_type = op_map[node["name"]]  # Map TVM op → Forge op
            else:
                op_type = "unsupported"
            node["op"] = op_type
            
            # Get function name
            function_map = (
                forge_op_to_function_name if json_graph["device"] == "tt" else pytorch_op_to_function_name
            )
            function_name = function_map[op_type]  # e.g., "forge.op.Conv2d"
            node["forge_name"] = op_type + f"_{nid}"  # e.g., "conv2d_5"
            
            # Extract operation arguments
            args = ()
            argument_getter = (
                forge_ops_needing_arguments if json_graph["device"] == "tt" else pytorch_ops_needing_arguments
            )
            if op_type in argument_getter:
                if op_type == "dropout" and json_graph["device"] != "tt":
                    if is_training:
                        logger.warning("Dropout op cannot be cpu fallback in training mode...")
                    args = argument_getter[op_type](graph=graph, nid=nid, training=is_training)
                else:
                    args = argument_getter[op_type](graph=graph, nid=nid, compiler_cfg=compiler_cfg)
                assert args is not None
            
            if args == () and json_graph["device"] == "cpu" and op_type not in argument_getter:
                _, args = _populate_torch_init_args(graph, nid)
            
            # Extract input information
            input_names = []
            input_shapes = []
            input_dtypes = []
            input_node_types = []
            for input_port in range(int(node["attrs"]["num_inputs"])):
                input_nid = node["inputs"][input_port][0]
                input_node = graph["nodes"][input_nid]
                if "users" not in input_node:
                    input_node["users"] = []
                input_node["users"].append(nid)
                input_names.append(input_node["forge_name"])
                input_shapes.append(input_node["forge_shape"])
                input_dtypes.append(_determine_node_dtype(input_node))
                if input_nid in params.keys():
                    input_node_types.append(NodeType.Parameter)
                elif input_nid in constants.keys():
                    input_node_types.append(NodeType.Constant)
                else:
                    input_node_types.append(NodeType.Activation)
            
            # Create Operation object
            node_name_to_node_type[node["forge_name"]] = NodeType.Activation
            ops[node["nid"]] = Operation(
                function_name=function_name,      # e.g., "forge.op.Conv2d"
                output_name=node["forge_name"],    # e.g., "conv2d_5"
                input_names=input_names,           # List of input tensor names
                args=args,                         # Operation-specific arguments
                src_layer=span_to_src_layer(node), # Source layer name (for debugging)
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                input_node_types=input_node_types,
            )
    
    # Lines 2265-2286: Process output nodes
    if any([input is None for input in forge_inputs]):
        forge_inputs = flattened_pytorch_inputs
    
    for output_nid in output_nodes:
        output_node = graph["nodes"][output_nid]
        returns[output_nid] = output_node["forge_name"]
        if len(output_node["forge_shape"]) == 0:
            returns_requiring_batch_dim_fix.append(output_node["forge_name"])
        elif output_node["forge_shape"][0] != 1:
            returns_requiring_batch_dim_fix.append(output_node["forge_name"])
        
        # Add output node to graph
        new_output_nid = len(graph["nodes"])
        graph["nodes"].append({
            "forge_name": output_node["forge_name"] + "_output",
            "op": "output",
            "nid": new_output_nid,
            "inputs": [[output_nid, 0, 0]],
            "attrs": {"num_inputs": "1"},
        })
    
    # Lines 2420-2433: Create code writer
    if json_graph["device"] == "tt":
        delete_inputs = not verify_cfg.enable_op_level_comparision
        if not delete_inputs:
            logger.warning("Preserving Intermediate tensor values in ForgeModule forward may cause out-of-memory issues")
        writer = ForgeWriter(
            current_module_name,
            framework,
            contains_incompatible_np_floats=contains_incompatible_np_floats,
            delete_inputs=delete_inputs,
        )
    else:
        writer = PyTorchWriter(current_module_name, source_framework=framework)
    
    # Line 2435: Write header (imports)
    writer.write_header()
    
    # Lines 2437-2447: Write class definition
    if submodule:
        writer.write_class_definition(matched_params, matched_consts, class_name="Submodel", is_submodel=True)
        # ... (submodule handling)
    else:
        writer.write_class_definition(params, constants)
    
    # Lines 2510-2513: Write forward method
    if is_cpu_pre:
        writer.write_forward(ops, graph_input_names, returns, returns_requiring_batch_dim_fix)
    else:
        writer.write_forward(ops, graph_input_names, returns)
    
    # Lines 2515-2521: Write parameter parser
    param_file_name = None
    if len(params_from_tvm):
        param_file_name = os.path.join(writer.module_directory, writer.module_name + "_params.pt")
        torch.save(params_from_tvm, param_file_name)
    
    param_names.update(const_names)
    writer.write_param_parser(param_names, param_file_name)
    
    # Line 2523: Close file
    writer.close_file()
    
    modules.append(writer)
```

### Code Generation: `ForgeWriter` (`python_codegen.py:119`)

**Purpose:** Write Python code for Forge modules.

**Key Methods:**

#### `write_header()` (`python_codegen.py:142`)

```python
def write_header(self, include_pytest_imports=False):
    self.wl("import forge")
    self.wl("import forge.op")
    self.wl("from forge import ForgeModule")
    self.wl("")
    self.wl("from loguru import logger")
    self.wl("import torch")
    # ... (framework-specific imports)
```

#### `write_class_definition()` (`python_codegen.py:176`)

```python
def write_class_definition(self, params, constants, class_name=None, num_submodels=0, is_submodel=False):
    if class_name is None:
        class_name = self.class_name
    self.num_submodels = num_submodels
    self.wl("")
    self.wl(f"class {class_name}(ForgeModule):")
    self.indent += 1
    self.wl("def __init__(self, name):")
    self.indent += 1
    self.wl(f"super().__init__(name)")
    
    # Add parameters
    for param in params.values():
        name, shape, requires_grad, dtype = param
        if name in self.param_names:
            continue
        self.param_names.append(name)
        self.wl(
            f'self.add_parameter("{name}", forge.Parameter(*{shape}, requires_grad={requires_grad}, dev_data_format={forge_df_from_str(dtype, name)}))'
        )
    
    # Add constants
    for const in constants.values():
        name = const[0]
        shape = tuple(const[1])
        dtype = pytorch_df_from_str(const[2], name)
        self.const_names.append(name)
        self.wl(f'self.add_constant("{name}", shape={shape}, dtype={dtype})')
    
    self.indent = 0
    self.wl("")
```

#### `write_forward()` (`python_codegen.py:233`)

```python
def write_forward(self, ops, inputs, outputs):
    self.indent = 1
    activation_names = "".join([", " + name for name in [inputs[key] for key in sorted(inputs)]])
    self.wl("def forward(self" + activation_names + "):")
    self.indent += 1
    
    for key in sorted(ops):
        input_names = self.get_op_input_names(ops[key])
        activation_names = "".join([", " + name for name in input_names])
        
        if len(ops[key].args) == 0:
            arg_text = ""
        else:
            arg_text = "".join([", " + argument + "=" + value for argument, value in ops[key].args])
        
        set_src_layer = ""
        if ops[key].src_layer:
            set_src_layer = f'.set_src_layer("{ops[key].src_layer}")'
        
        # Write operation call
        self.wl(
            f'{ops[key].output_name} = {ops[key].function_name}("{ops[key].node_name}"{activation_names}{arg_text}){set_src_layer}'
        )
        
        # Delete inputs if needed (memory optimization)
        if self.delete_inputs:
            for name_to_del in ops[key].inputs_to_delete:
                self.wl(f"{name_to_del}._value = None")
    
    # Write return statement
    outputs = list(outputs.values())
    if len(outputs) == 1:
        output_names = outputs[0]
    else:
        output_names = ", ".join(outputs)
    
    self.wl(f"return {output_names}")
    self.indent = 0
    self.wl("")
```

**Generated Python Code Example:**
```python
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

class GeneratedModule(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("weight", forge.Parameter(*(3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_constant("bias", shape=(3,), dtype=torch.float32)
    
    def forward(self, input):
        conv2d_5 = forge.op.Conv2d("conv2d_5", input, self.get_parameter("weight"), kernel_size=(3, 3))
        relu_6 = forge.op.Relu("relu_6", conv2d_5)
        return relu_6
```

---

## Phase 4: Python Module → ForgeModule Instance

### Dynamic Import: `import_from_path()` (`tvm_to_python.py:37`)

```python
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Could not load module {module_name} from {file_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
```

### Parameter Loading: `process_framework_parameters()` (Generated in Python code)

```python
def process_framework_parameters(self, model):
    named_parameters = dict(model.state_dict().items())
    named_buffers = dict(model.named_buffers())
    named_parameters.update(named_buffers)
    
    for name, param in named_parameters.items():
        if name in self.param_names:
            self.get_parameter(name).set_value(param)
```

---

## Key Data Structures

### Operation Object (`tvm_unique_op_generation.py`)

```python
class Operation:
    function_name: str           # e.g., "forge.op.Conv2d"
    output_name: str            # e.g., "conv2d_5"
    input_names: List[str]       # List of input tensor names
    args: Tuple                 # Operation-specific arguments
    src_layer: Optional[str]    # Source layer name
    input_shapes: List[Tuple]    # Input shapes
    input_dtypes: List[str]      # Input dtypes
    input_node_types: List[NodeType]  # Parameter/Constant/Activation
    inputs_to_delete: List[str]  # Inputs to delete after use
    is_submodule_call: bool     # Whether this is a submodule call
    loop_with: List[Operation]   # Operations to loop with
```

### NodeType Enum

```python
class NodeType(Enum):
    Parameter = 1    # Trainable parameter
    Constant = 2     # Non-trainable constant
    Activation = 3   # Intermediate activation
```

---

## Operation Mapping System

### TVM to Forge Op Map (`tvm_to_python.py:1481`)

```python
tvm_to_forge_op_map = {
    "nn.conv2d": "conv2d",
    "nn.matmul": "matmul",
    "nn.relu": "relu",
    "add": "add",
    "multiply": "multiply",
    # ... (many more mappings)
}
```

### Forge Op to Function Name (`tvm_to_python.py:1561`)

```python
forge_op_to_function_name = {
    "conv2d": "forge.op.Conv2d",
    "matmul": "forge.op.Matmul",
    "relu": "forge.op.Relu",
    "add": "forge.op.Add",
    "multiply": "forge.op.Multiply",
    # ... (many more mappings)
}
```

### Operation Argument Extractors (`tvm_to_python.py:1636`)

```python
forge_ops_needing_arguments = {
    "conv2d": populate_conv2d_args,
    "matmul": populate_matmul_args,
    "relu": populate_relu_args,
    # ... (many more extractors)
}
```

Each extractor function takes `(graph, nid, compiler_cfg)` and returns a tuple of `(argument_name, argument_value)` pairs.

---

## Code Examples

### Example 1: Simple Conv2d Model

**Input (PyTorch):**
```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=3)
    
    def forward(self, x):
        return self.conv(x)
```

**TVM Relay IR (simplified):**
```python
def @main(%input: Tensor[(1, 1, 28, 28), float32], %weight: Tensor[(3, 1, 3, 3), float32]) {
  %0 = nn.conv2d(%input, %weight, padding=[0, 0, 0, 0], strides=[1, 1])
  %0
}
```

**Generated Forge Module:**
```python
import forge
import forge.op
from forge import ForgeModule

class SimpleModel(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("weight", forge.Parameter(*(3, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
    
    def forward(self, input):
        conv2d_0 = forge.op.Conv2d("conv2d_0", input, self.get_parameter("weight"), kernel_size=(3, 3), padding=(0, 0), stride=(1, 1))
        return conv2d_0
    
    def process_framework_parameters(self, model):
        named_parameters = dict(model.state_dict().items())
        for name, param in named_parameters.items():
            if name == "conv.weight":
                self.get_parameter("weight").set_value(param)
```

### Example 2: Multi-Layer Model

**Input (ONNX):**
- Conv2d → ReLU → MaxPool2d → Flatten → Linear

**JSON Graph Structure:**
```json
{
    "nodes": [
        {"op": "input", "name": "input", "attrs": {"shape": [[1, 1, 28, 28]]}},
        {"op": "kernel", "name": "nn.conv2d", "attrs": {"num_inputs": "2"}},
        {"op": "kernel", "name": "nn.relu", "attrs": {"num_inputs": "1"}},
        {"op": "kernel", "name": "nn.max_pool2d", "attrs": {"num_inputs": "1"}},
        {"op": "kernel", "name": "reshape", "attrs": {"num_inputs": "1"}},
        {"op": "kernel", "name": "nn.dense", "attrs": {"num_inputs": "2"}}
    ],
    "arg_nodes": [0],
    "heads": [[5, 0, 0]]
}
```

**Generated Forge Module:**
```python
class MultiLayerModel(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("conv_weight", forge.Parameter(*(3, 1, 3, 3), ...))
        self.add_parameter("linear_weight", forge.Parameter(*(10, 507), ...))
        self.add_parameter("linear_bias", forge.Parameter(*(10,), ...))
    
    def forward(self, input):
        conv2d_1 = forge.op.Conv2d("conv2d_1", input, self.get_parameter("conv_weight"), ...)
        relu_2 = forge.op.Relu("relu_2", conv2d_1)
        max_pool2d_3 = forge.op.MaxPool2d("max_pool2d_3", relu_2, ...)
        reshape_4 = forge.op.Reshape("reshape_4", max_pool2d_3, shape=(-1, 507))
        dense_5 = forge.op.Matmul("dense_5", reshape_4, self.get_parameter("linear_weight"))
        add_6 = forge.op.Add("add_6", dense_5, self.get_parameter("linear_bias"))
        return add_6
```

---

## Summary

The TVM Relay IR to Forge Module pipeline consists of four main phases:

1. **Framework → TVM Relay IR**: Convert framework models to TVM Relay IR using framework-specific frontends
2. **TVM Relay IR → JSON Graphs**: Compile and partition Relay IR, then extract JSON representations
3. **JSON Graphs → Python Code**: Process JSON graphs and generate Python Forge module code
4. **Python Code → ForgeModule**: Dynamically import and instantiate generated modules

Each phase involves complex transformations, optimizations, and code generation to produce efficient, executable Forge modules that can run on Tenstorrent de