# ONNX Transpiler Test Suite

This directory contains comprehensive test cases for the ONNX to TIR transpiler.

## Test Structure

The test suite is organized into subdirectories:

### Root Directory
- `test_utils.py`: Utility functions for creating ONNX models programmatically and comparing results
- `conftest.py`: Pytest configuration and fixtures
- `README.md`: This file

### `ops/` Directory
Operator-level tests for individual ONNX operations:
- `test_add.py`: Addition operator tests
- `test_arithmetic.py`: Arithmetic operations tests
- `test_avgpool.py`: Average pooling tests
- `test_clip.py`: Clip operator tests
- `test_concat.py`: Concatenation tests
- `test_conv.py`: Convolution tests
- `test_dropout.py`: Dropout tests
- `test_flatten.py`: Flatten tests
- `test_gemm.py`: General Matrix Multiplication tests
- `test_logsoftmax.py`: LogSoftmax tests
- `test_maxpool.py`: Max pooling tests
- `test_pad.py`: Padding tests
- `test_reduction_ops.py`: Reduction operations (ReduceSum, ReduceMean, ReduceMax)
- `test_relu.py`: ReLU activation tests
- `test_reshape.py`: Reshape tests
- `test_sigmoid.py`: Sigmoid tests
- `test_softmax.py`: Softmax tests
- `test_squeeze.py`: Squeeze tests
- `test_tanh.py`: Tanh tests
- `test_transpose.py`: Transpose tests
- `test_unsqueeze.py`: Unsqueeze tests

### `models/` Directory
End-to-end model tests:
- `test_mnist.py`: MNIST model transpilation tests

## Running Tests

### Run all operator tests:
```bash
pytest forge/forge/transpiler/frontends/onnx/tests/ops/ -v
```

### Run all model tests:
```bash
pytest forge/forge/transpiler/frontends/onnx/tests/models/ -v
```

### Run a specific operator test:
```bash
pytest forge/forge/transpiler/frontends/onnx/tests/ops/test_reduction_ops.py -v
```

### Run a specific test case:
```bash
pytest forge/forge/transpiler/frontends/onnx/tests/ops/test_reduction_ops.py::TestReduceSum::test_reducesum_basic -v
```

### Run with debug output:
```bash
pytest forge/forge/transpiler/frontends/onnx/tests/ops/test_reduction_ops.py -v -s
```

## Test Coverage

Each test case:
1. Creates an ONNX model programmatically with specific configurations
2. Transpiles the model to TIR with debug mode enabled
3. Verifies the TIR graph structure matches the ONNX model
4. Compares TIR execution outputs with ONNX Runtime outputs
5. Validates correctness across different:
   - Input shapes
   - Data types
   - Opset versions
   - Operation attributes

## Current Test Coverage

### Reduction Operations
- **ReduceSum**: Tests various input shapes, axes, keepdims, and opset versions (1, 11, 13)
- **ReduceMean**: Tests basic configurations
- **ReduceMax**: Tests basic configurations

## Adding New Tests

### Adding Operator Tests

To add tests for a new operation, create a test file in the `ops/` directory:

1. Create a test file: `ops/test_newop.py`
2. Import from parent directory: `from ..test_utils import ...` and `from ..engine import ...`
3. Create a test class following the pattern in `ops/test_reduction_ops.py`
4. Use `create_onnx_model()` from `test_utils.py` to create test models
5. Use `verify_tir_graph_structure()` to verify graph structure
6. Use `compare_tir_with_onnx()` to compare execution results
7. Add parametrized test cases for different configurations

Example:
```python
from ..engine import ONNXToForgeTranspiler
from ..test_utils import create_onnx_model, compare_tir_with_onnx

class TestNewOp:
    @pytest.mark.parametrize("input_shape", [(2, 3), (1, 4, 5)])
    def test_newop_basic(self, input_shape):
        onnx_model = create_onnx_model(...)
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        # ... verify and compare
```

### Adding Model Tests

To add tests for a new model, create a test file in the `models/` directory:

1. Create a test file: `models/test_modelname.py`
2. Import from parent directory: `from ..test_utils import ...` and `from ..engine import ...`
3. Follow the pattern in `models/test_mnist.py` for end-to-end model testing

