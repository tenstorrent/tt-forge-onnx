"""
Test cases for ONNX MaxPool operations (1D, 2D, and 3D).
Tests different input shapes, kernel sizes, attributes, opset versions, and edge cases.
"""
import pytest
import numpy as np
import onnx
import torch
from loguru import logger

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from forge.transpiler.frontends.onnx.tests.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
    verify_tir_graph_structure,
    print_onnx_model,
    print_tir_graph
)


# ============================================================================
# HELPER METHODS FOR CREATING MAXPOOL MODELS
# ============================================================================

def _create_maxpool1d_model(
    opset_version: int,
    input_shape: tuple,
    kernel_shape: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    auto_pad: str = 'NOTSET',
    dtype: int = onnx.TensorProto.FLOAT
) -> onnx.ModelProto:
    """Create an ONNX MaxPool model for 1D pooling."""
    if stride is None:
        stride = kernel_shape
    
    attrs = {
        'kernel_shape': [kernel_shape],
    }
    
    if stride != 1:
        attrs['strides'] = [stride]
    
    if auto_pad == 'NOTSET' and padding != 0:
        if isinstance(padding, int):
            attrs['pads'] = [padding, padding]
        elif isinstance(padding, (list, tuple)) and len(padding) == 2:
            attrs['pads'] = list(padding)
        else:
            attrs['pads'] = [0, 0]
    
    if auto_pad != 'NOTSET':
        attrs['auto_pad'] = auto_pad
    
    if opset_version >= 10:
        if dilation != 1:
            attrs['dilations'] = [dilation]
    
    if opset_version >= 10:
        attrs['ceil_mode'] = 1 if ceil_mode else 0
    
    return create_onnx_model(
        op_type='MaxPool',
        input_shapes=[input_shape],
        input_dtypes=[dtype],
        output_shapes=[input_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name='maxpool_node',
        input_names=['input_0'],
        output_names=['output_0']
    )


def _create_maxpool2d_model(
    opset_version: int,
    input_shape: tuple,
    kernel_shape: tuple,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    auto_pad: str = 'NOTSET',
    dtype: int = onnx.TensorProto.FLOAT
) -> onnx.ModelProto:
    """Create an ONNX MaxPool model for 2D pooling."""
    if stride is None:
        stride = kernel_shape[0] if isinstance(kernel_shape, (list, tuple)) else kernel_shape
    
    attrs = {
        'kernel_shape': list(kernel_shape),
    }
    
    if stride != 1:
        if isinstance(stride, int):
            attrs['strides'] = [stride, stride]
        else:
            attrs['strides'] = list(stride) if isinstance(stride, (list, tuple)) else [stride, stride]
    
    if auto_pad == 'NOTSET' and padding != 0:
        if isinstance(padding, int):
            attrs['pads'] = [padding, padding, padding, padding]
        elif isinstance(padding, (list, tuple)) and len(padding) == 4:
            attrs['pads'] = list(padding)
        else:
            attrs['pads'] = [0, 0, 0, 0]
    
    if auto_pad != 'NOTSET':
        attrs['auto_pad'] = auto_pad
    
    if opset_version >= 10:
        if isinstance(dilation, int):
            if dilation != 1:
                attrs['dilations'] = [dilation, dilation]
        elif isinstance(dilation, (list, tuple)):
            attrs['dilations'] = list(dilation)
    
    if opset_version >= 10:
        attrs['ceil_mode'] = 1 if ceil_mode else 0
    
    return create_onnx_model(
        op_type='MaxPool',
        input_shapes=[input_shape],
        input_dtypes=[dtype],
        output_shapes=[input_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name='maxpool_node',
        input_names=['input_0'],
        output_names=['output_0']
    )


def _create_maxpool3d_model(
    opset_version: int,
    input_shape: tuple,
    kernel_shape: tuple,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    auto_pad: str = 'NOTSET',
    dtype: int = onnx.TensorProto.FLOAT
) -> onnx.ModelProto:
    """Create an ONNX MaxPool model for 3D pooling."""
    if stride is None:
        stride = kernel_shape[0] if isinstance(kernel_shape, (list, tuple)) else kernel_shape
    
    attrs = {
        'kernel_shape': list(kernel_shape),
    }
    
    if stride != 1:
        if isinstance(stride, int):
            attrs['strides'] = [stride, stride, stride]
        else:
            attrs['strides'] = list(stride) if isinstance(stride, (list, tuple)) else [stride, stride, stride]
    
    if auto_pad == 'NOTSET' and padding != 0:
        if isinstance(padding, int):
            attrs['pads'] = [padding, padding, padding, padding, padding, padding]
        elif isinstance(padding, (list, tuple)) and len(padding) == 6:
            attrs['pads'] = list(padding)
        else:
            attrs['pads'] = [0, 0, 0, 0, 0, 0]
    
    if auto_pad != 'NOTSET':
        attrs['auto_pad'] = auto_pad
    
    if opset_version >= 10:
        if isinstance(dilation, int):
            if dilation != 1:
                attrs['dilations'] = [dilation, dilation, dilation]
        elif isinstance(dilation, (list, tuple)):
            attrs['dilations'] = list(dilation)
    
    if opset_version >= 10:
        attrs['ceil_mode'] = 1 if ceil_mode else 0
    
    return create_onnx_model(
        op_type='MaxPool',
        input_shapes=[input_shape],
        input_dtypes=[dtype],
        output_shapes=[input_shape],
        output_dtypes=[dtype],
        attrs=attrs,
        opset_version=opset_version,
        node_name='maxpool_node',
        input_names=['input_0'],
        output_names=['output_0']
    )


# ============================================================================
# MAXPOOL 1D TESTS
# ============================================================================

class TestMaxPool1d:
    """Comprehensive test cases for MaxPool1d operation."""
    
    # ========================================================================
    # BASIC TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [1, 8, 10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32), 3),
        ((1, 3, 16), 2),
        ((2, 1, 8), 2),
    ])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_maxpool1d_basic(self, opset_version, input_shape, kernel_shape, stride):
        """Test basic MaxPool1d operation."""
        onnx_model = _create_maxpool1d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # PADDING TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [1, 8, 10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32), 3),
        ((1, 3, 16), 2),
    ])
    @pytest.mark.parametrize("padding", [0, 1, (1, 1)])
    def test_maxpool1d_padding(self, opset_version, input_shape, kernel_shape, padding):
        """Test MaxPool1d with explicit padding."""
        if isinstance(padding, int):
            pad_w = padding
        elif isinstance(padding, (list, tuple)) and len(padding) == 2:
            pad_w = padding[0] + padding[1]
        else:
            pad_w = 0
        
        if pad_w > kernel_shape // 2:
            pytest.skip(f"Invalid padding: padding={padding}, kernel_shape={kernel_shape}. "
                       f"PyTorch requires padding <= kernel_size/2")
        
        onnx_model = _create_maxpool1d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=2,
            padding=padding,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # CEIL_MODE TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32), 3),
        ((1, 3, 31), 3),
        ((1, 3, 16), 2),
    ])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_maxpool1d_ceil_mode(self, opset_version, input_shape, kernel_shape, ceil_mode):
        """Test MaxPool1d with ceil_mode attribute."""
        onnx_model = _create_maxpool1d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=2,
            padding=0,
            ceil_mode=ceil_mode,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # AUTO_PAD TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32), 3),
        ((1, 3, 31), 3),
        ((1, 3, 16), 2),
    ])
    @pytest.mark.parametrize("auto_pad", ['SAME_UPPER', 'SAME_LOWER', 'VALID'])
    @pytest.mark.parametrize("stride", [2])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_maxpool1d_auto_pad(self, opset_version, input_shape, kernel_shape, auto_pad, stride, ceil_mode):
        """Test MaxPool1d with different auto_pad modes."""
        if opset_version < 10:
            pytest.skip(f"ceil_mode not available in opset {opset_version}")
        
        onnx_model = _create_maxpool1d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            auto_pad=auto_pad,
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version < 22 else 1e-4
        atol = 1e-4 if opset_version < 22 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # DILATION TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32), 3),
        ((1, 3, 16), 2),
    ])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_maxpool1d_dilation(self, opset_version, input_shape, kernel_shape, dilation):
        """Test MaxPool1d with dilation attribute."""
        onnx_model = _create_maxpool1d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=2,
            padding=0,
            dilation=dilation,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # EDGE CASES
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool1d_kernel_size_1(self, opset_version):
        """Test MaxPool1d with kernel_size=1."""
        onnx_model = _create_maxpool1d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 32),
            kernel_shape=1,
            stride=1,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 32).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool1d_stride_larger_than_kernel(self, opset_version):
        """Test MaxPool1d with stride larger than kernel size."""
        onnx_model = _create_maxpool1d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 16),
            kernel_shape=2,
            stride=4,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 16).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool1d_small_input(self, opset_version):
        """Test MaxPool1d with very small input."""
        onnx_model = _create_maxpool1d_model(
            opset_version=opset_version,
            input_shape=(1, 1, 4),
            kernel_shape=3,
            stride=1,
            padding=1,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 1, 4).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"


# ============================================================================
# MAXPOOL 2D TESTS
# ============================================================================

class TestMaxPool2d:
    """Comprehensive test cases for MaxPool2d operation."""
    
    # ========================================================================
    # BASIC TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [1, 8, 10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32, 32), (3, 3)),
        ((1, 3, 16, 16), (2, 2)),
        ((2, 1, 8, 8), (2, 2)),
    ])
    @pytest.mark.parametrize("stride", [1, 2, (2, 2)])
    def test_maxpool2d_basic(self, opset_version, input_shape, kernel_shape, stride):
        """Test basic MaxPool2d operation."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # PADDING TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [1, 8, 10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32, 32), (3, 3)),
        ((1, 3, 16, 16), (2, 2)),
    ])
    @pytest.mark.parametrize("padding", [0, 1, (1, 1), (1, 1, 1, 1)])
    def test_maxpool2d_padding(self, opset_version, input_shape, kernel_shape, padding):
        """Test MaxPool2d with explicit padding."""
        if isinstance(padding, int):
            pad_h, pad_w = padding, padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:
                pad_h, pad_w = padding[0], padding[1]
            elif len(padding) == 4:
                pad_h = padding[0] + padding[2]
                pad_w = padding[1] + padding[3]
            else:
                pad_h, pad_w = 0, 0
        else:
            pad_h, pad_w = 0, 0
        
        if pad_h > kernel_shape[0] // 2 or pad_w > kernel_shape[1] // 2:
            pytest.skip(f"Invalid padding: padding={padding}, kernel_shape={kernel_shape}. "
                       f"PyTorch requires padding <= kernel_size/2")
        
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=2,
            padding=padding,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # CEIL_MODE TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32, 32), (3, 3)),
        ((1, 3, 31, 31), (3, 3)),
        ((1, 3, 16, 16), (2, 2)),
    ])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_maxpool2d_ceil_mode(self, opset_version, input_shape, kernel_shape, ceil_mode):
        """Test MaxPool2d with ceil_mode attribute."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=2,
            padding=0,
            ceil_mode=ceil_mode,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # AUTO_PAD TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32, 32), (3, 3)),
        ((1, 3, 31, 31), (3, 3)),
        ((1, 3, 16, 16), (2, 2)),
    ])
    @pytest.mark.parametrize("auto_pad", ['SAME_UPPER', 'SAME_LOWER', 'VALID'])
    @pytest.mark.parametrize("stride", [2, (2, 2)])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_maxpool2d_auto_pad(self, opset_version, input_shape, kernel_shape, auto_pad, stride, ceil_mode):
        """Test MaxPool2d with different auto_pad modes."""
        if opset_version < 10:
            pytest.skip(f"ceil_mode not available in opset {opset_version}")
        
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            auto_pad=auto_pad,
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version < 22 else 1e-4
        atol = 1e-4 if opset_version < 22 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # DILATION TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 32, 32), (3, 3)),
        ((1, 3, 16, 16), (2, 2)),
    ])
    @pytest.mark.parametrize("dilation", [1, 2, (2, 2)])
    def test_maxpool2d_dilation(self, opset_version, input_shape, kernel_shape, dilation):
        """Test MaxPool2d with dilation attribute."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=2,
            padding=0,
            dilation=dilation,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # EDGE CASES
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool2d_kernel_size_1(self, opset_version):
        """Test MaxPool2d with kernel_size=1."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 32, 32),
            kernel_shape=(1, 1),
            stride=1,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 32, 32).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool2d_stride_larger_than_kernel(self, opset_version):
        """Test MaxPool2d with stride larger than kernel size."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 16, 16),
            kernel_shape=(2, 2),
            stride=4,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 16, 16).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool2d_small_input(self, opset_version):
        """Test MaxPool2d with very small input."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 1, 4, 4),
            kernel_shape=(3, 3),
            stride=1,
            padding=1,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 1, 4, 4).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    def test_maxpool2d_asymmetric_stride(self, opset_version):
        """Test MaxPool2d with asymmetric strides."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 32, 32),
            kernel_shape=(3, 3),
            stride=(1, 2),
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 32, 32).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    def test_maxpool2d_rectangular_kernel(self, opset_version):
        """Test MaxPool2d with rectangular kernels."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 32, 32),
            kernel_shape=(3, 5),
            stride=2,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 32, 32).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    def test_maxpool2d_asymmetric_dilation(self, opset_version):
        """Test MaxPool2d with asymmetric dilation."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 32, 32),
            kernel_shape=(3, 3),
            stride=2,
            padding=0,
            dilation=(1, 2),
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 32, 32).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    def test_maxpool2d_input_equals_kernel(self, opset_version):
        """Test MaxPool2d when input size equals kernel size."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 5, 5),
            kernel_shape=(5, 5),
            stride=1,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 5, 5).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    def test_maxpool2d_input_smaller_than_kernel_with_padding(self, opset_version):
        """Test MaxPool2d when input is smaller than kernel."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 3, 3),
            kernel_shape=(5, 5),
            stride=1,
            padding=2,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 3, 3).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    def test_maxpool2d_large_kernel(self, opset_version):
        """Test MaxPool2d with large kernel size."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 32, 32),
            kernel_shape=(7, 7),
            stride=2,
            padding=3,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 32, 32).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    @pytest.mark.parametrize("kernel_shape", [(2, 2), (3, 3), (5, 5)])
    def test_maxpool2d_stride_1_various_kernels(self, opset_version, kernel_shape):
        """Test MaxPool2d with stride=1 and various kernel sizes."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 16, 16),
            kernel_shape=kernel_shape,
            stride=1,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 16, 16).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool2d_combined_edge_cases(self, opset_version):
        """Test MaxPool2d with combined edge cases."""
        onnx_model = _create_maxpool2d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 31, 31),
            kernel_shape=(3, 3),
            stride=2,
            padding=0,
            dilation=2,
            ceil_mode=True,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 31, 31).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool2d_auto_pad_with_stride_1(self, opset_version):
        """Test MaxPool2d with auto_pad and stride=1."""
        for auto_pad in ['SAME_UPPER', 'SAME_LOWER']:
            onnx_model = _create_maxpool2d_model(
                opset_version=opset_version,
                input_shape=(1, 3, 32, 32),
                kernel_shape=(3, 3),
                stride=1,
                padding=0,
                ceil_mode=False,
                auto_pad=auto_pad,
                dtype=onnx.TensorProto.FLOAT
            )
            
            input_data = {
                'input_0': np.random.randn(1, 3, 32, 32).astype(np.float32)
            }
            
            transpiler = ONNXToForgeTranspiler(debug=False)
            tir_graph = transpiler.transpile(onnx_model)
            
            verify_tir_graph_structure(tir_graph, onnx_model)
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
            
            assert len(comparison['errors']) == 0, f"Comparison errors for {auto_pad}: {comparison['errors']}"
            assert all(comparison['matches'].values()), f"Output values don't match for {auto_pad}"


# ============================================================================
# MAXPOOL 3D TESTS
# ============================================================================

class TestMaxPool3d:
    """Comprehensive test cases for MaxPool3d operation."""
    
    # ========================================================================
    # BASIC TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [1, 8, 10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 8, 8, 8), (3, 3, 3)),
        ((1, 3, 4, 4, 4), (2, 2, 2)),
        ((2, 1, 4, 4, 4), (2, 2, 2)),
    ])
    @pytest.mark.parametrize("stride", [1, 2, (2, 2, 2)])
    def test_maxpool3d_basic(self, opset_version, input_shape, kernel_shape, stride):
        """Test basic MaxPool3d operation."""
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # PADDING TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [1, 8, 10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 8, 8, 8), (3, 3, 3)),
        ((1, 3, 4, 4, 4), (2, 2, 2)),
    ])
    @pytest.mark.parametrize("padding", [0, 1, (1, 1, 1)])
    def test_maxpool3d_padding(self, opset_version, input_shape, kernel_shape, padding):
        """Test MaxPool3d with explicit padding."""
        if isinstance(padding, int):
            pad_d = pad_h = pad_w = padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 3:
                pad_d, pad_h, pad_w = padding[0], padding[1], padding[2]
            elif len(padding) == 6:
                pad_d = padding[0] + padding[3]
                pad_h = padding[1] + padding[4]
                pad_w = padding[2] + padding[5]
            else:
                pad_d = pad_h = pad_w = 0
        else:
            pad_d = pad_h = pad_w = 0
        
        if pad_d > kernel_shape[0] // 2 or pad_h > kernel_shape[1] // 2 or pad_w > kernel_shape[2] // 2:
            pytest.skip(f"Invalid padding: padding={padding}, kernel_shape={kernel_shape}. "
                       f"PyTorch requires padding <= kernel_size/2")
        
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=2,
            padding=padding,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # CEIL_MODE TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 8, 8, 8), (3, 3, 3)),
        ((1, 3, 7, 7, 7), (3, 3, 3)),
        ((1, 3, 4, 4, 4), (2, 2, 2)),
    ])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_maxpool3d_ceil_mode(self, opset_version, input_shape, kernel_shape, ceil_mode):
        """Test MaxPool3d with ceil_mode attribute."""
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=2,
            padding=0,
            ceil_mode=ceil_mode,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # AUTO_PAD TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 8, 8, 8), (3, 3, 3)),
        ((1, 3, 7, 7, 7), (3, 3, 3)),
        ((1, 3, 4, 4, 4), (2, 2, 2)),
    ])
    @pytest.mark.parametrize("auto_pad", ['SAME_UPPER', 'SAME_LOWER', 'VALID'])
    @pytest.mark.parametrize("stride", [2, (2, 2, 2)])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_maxpool3d_auto_pad(self, opset_version, input_shape, kernel_shape, auto_pad, stride, ceil_mode):
        """Test MaxPool3d with different auto_pad modes."""
        if opset_version < 10:
            pytest.skip(f"ceil_mode not available in opset {opset_version}")
        
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            auto_pad=auto_pad,
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        rtol = 1e-3 if opset_version < 22 else 1e-4
        atol = 1e-4 if opset_version < 22 else 1e-5
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=rtol, atol=atol)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # DILATION TESTS
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    @pytest.mark.parametrize("input_shape, kernel_shape", [
        ((1, 3, 8, 8, 8), (3, 3, 3)),
        ((1, 3, 4, 4, 4), (2, 2, 2)),
    ])
    @pytest.mark.parametrize("dilation", [1, 2, (2, 2, 2)])
    def test_maxpool3d_dilation(self, opset_version, input_shape, kernel_shape, dilation):
        """Test MaxPool3d with dilation attribute."""
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=2,
            padding=0,
            dilation=dilation,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(*input_shape).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    # ========================================================================
    # EDGE CASES
    # ========================================================================
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool3d_kernel_size_1(self, opset_version):
        """Test MaxPool3d with kernel_size=1."""
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 8, 8, 8),
            kernel_shape=(1, 1, 1),
            stride=1,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 8, 8, 8).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool3d_stride_larger_than_kernel(self, opset_version):
        """Test MaxPool3d with stride larger than kernel size."""
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 4, 4, 4),
            kernel_shape=(2, 2, 2),
            stride=4,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 4, 4, 4).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [11, 22])
    def test_maxpool3d_small_input(self, opset_version):
        """Test MaxPool3d with very small input."""
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=(1, 1, 4, 4, 4),
            kernel_shape=(3, 3, 3),
            stride=1,
            padding=1,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 1, 4, 4, 4).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    def test_maxpool3d_asymmetric_stride(self, opset_version):
        """Test MaxPool3d with asymmetric strides."""
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 8, 8, 8),
            kernel_shape=(3, 3, 3),
            stride=(1, 2, 2),
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 8, 8, 8).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    def test_maxpool3d_rectangular_kernel(self, opset_version):
        """Test MaxPool3d with rectangular kernels."""
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 8, 8, 8),
            kernel_shape=(3, 3, 5),
            stride=2,
            padding=0,
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 8, 8, 8).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"
    
    @pytest.mark.parametrize("opset_version", [10, 11, 22])
    def test_maxpool3d_asymmetric_dilation(self, opset_version):
        """Test MaxPool3d with asymmetric dilation."""
        onnx_model = _create_maxpool3d_model(
            opset_version=opset_version,
            input_shape=(1, 3, 8, 8, 8),
            kernel_shape=(3, 3, 3),
            stride=2,
            padding=0,
            dilation=(1, 2, 2),
            auto_pad='NOTSET',
            dtype=onnx.TensorProto.FLOAT
        )
        
        input_data = {
            'input_0': np.random.randn(1, 3, 8, 8, 8).astype(np.float32)
        }
        
        transpiler = ONNXToForgeTranspiler(debug=False)
        tir_graph = transpiler.transpile(onnx_model)
        
        verify_tir_graph_structure(tir_graph, onnx_model)
        comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data, rtol=1e-4, atol=1e-5)
        
        assert len(comparison['errors']) == 0, f"Comparison errors: {comparison['errors']}"
        assert all(comparison['matches'].values()), "Output values don't match"

