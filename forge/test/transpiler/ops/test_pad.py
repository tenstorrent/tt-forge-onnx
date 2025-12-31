# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Pad operation.
Tests different input shapes, dtypes, opset versions, padding modes, and edge cases.
"""
import pytest
import numpy as np
import onnx
import torch

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    create_onnx_model,
    compare_tir_with_onnx,
    verify_tir_graph_structure,
)


@pytest.mark.transpiler
class TestPad:
    """Restructured Pad tests organized by opset version and parameter combinations."""

    # Helper method to create Pad ONNX model based on opset version
    @staticmethod
    def _create_pad_model(
        opset_version, input_shape, pads, mode="constant", constant_value=0.0, axes=None, dtype=onnx.TensorProto.FLOAT
    ):
        """Helper to create Pad ONNX model with proper opset-specific structure."""
        np_dtype = np.float32 if dtype == onnx.TensorProto.FLOAT else np.int64
        attrs = {"mode": mode}
        initializers = {}
        input_names = ["data"]

        # Opset 1: pads as 'paddings' attribute, value as 'value' attribute
        if opset_version == 1:
            attrs["paddings"] = pads
            attrs["value"] = constant_value

        # Opsets 2-10: pads as 'pads' attribute, value as 'value' attribute
        elif opset_version <= 10:
            attrs["pads"] = pads
            attrs["value"] = constant_value

        # Opsets 11-17: pads as input, constant_value as input (for constant mode)
        elif opset_version <= 17:
            input_names.append("pads")
            initializers["pads"] = np.array(pads, dtype=np.int64)
            if mode == "constant":
                input_names.append("constant_value")
                initializers["constant_value"] = np.array(constant_value, dtype=np_dtype)

        # Opsets 18+: pads as input, constant_value/axes as inputs based on mode
        # IMPORTANT: ONNX Pad input order is: [data, pads, constant_value?, axes?]
        # For non-constant modes, we still need to provide constant_value as the 3rd input
        # (it will be ignored but must have the correct type T to match data)
        else:  # opset >= 18
            input_names.append("pads")
            initializers["pads"] = np.array(pads, dtype=np.int64)
            # Always provide constant_value as 3rd input (required by input order)
            # For non-constant modes, it's ignored but must have correct type
            input_names.append("constant_value")
            initializers["constant_value"] = np.array(constant_value, dtype=np_dtype)
            # Axes is 4th input (if provided)
            if axes is not None:
                input_names.append("axes")
                initializers["axes"] = np.array(axes, dtype=np.int64)

        # Calculate output shape
        output_shape = list(input_shape)
        if axes is None:
            # Pad all dimensions
            num_dims = len(input_shape)
            for i in range(num_dims):
                output_shape[i] += pads[i] + pads[i + num_dims]
        else:
            # Pad only selected axes
            for idx, axis in enumerate(axes):
                normalized_axis = axis if axis >= 0 else len(input_shape) + axis
                output_shape[normalized_axis] += pads[2 * idx] + pads[2 * idx + 1]

        # For opset 18+, we need to provide input shapes and dtypes for all inputs
        # The create_onnx_model function will skip creating input_value_infos for initializers
        # Input order: [data, pads, constant_value, axes?]
        if opset_version >= 18:
            input_shapes_list = [input_shape]
            input_dtypes_list = [dtype]

            # Add pads input info (int64, shape is [len(pads)])
            # Note: create_onnx_model will skip creating value_info for initializers
            input_shapes_list.append((len(pads),))
            input_dtypes_list.append(onnx.TensorProto.INT64)

            # Always add constant_value input info (same dtype as data, scalar)
            # Required by ONNX Pad input order, even for non-constant modes
            input_shapes_list.append(())  # Scalar
            input_dtypes_list.append(dtype)

            # Add axes input info if present (int64, shape is [len(axes)])
            if axes is not None:
                input_shapes_list.append((len(axes),))
                input_dtypes_list.append(onnx.TensorProto.INT64)

            return create_onnx_model(
                op_type="Pad",
                input_shapes=input_shapes_list,
                input_dtypes=input_dtypes_list,
                output_shapes=[tuple(output_shape)],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="pad_test",
                input_names=input_names,
                initializers=initializers,
            )
        else:
            # For opsets < 18, pads and constant_value are attributes or handled differently
            return create_onnx_model(
                op_type="Pad",
                input_shapes=[input_shape],
                input_dtypes=[dtype],
                output_shapes=[tuple(output_shape)],
                output_dtypes=[dtype],
                attrs=attrs,
                opset_version=opset_version,
                node_name="pad_test",
                input_names=input_names,
                initializers=initializers,
            )

    # ========================================================================
    # OPSET 1-2 TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [1, 2])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (5,),  # 1D
            (3, 4),  # 2D
            (2, 3, 4),  # 3D
            (1, 2, 3, 4),  # 4D
            (1, 2, 3, 4, 5),  # 5D
        ],
    )
    @pytest.mark.parametrize("mode", ["constant", "reflect", "edge"])
    def test_opset_1_2_modes(self, opset_version, input_shape, mode):
        """Test Pad opsets 1-2 with all modes (constant, reflect, edge)."""
        # Skip 1D non-constant modes - PyTorch limitation
        if len(input_shape) == 1 and mode != "constant":
            pytest.skip(f"PyTorch doesn't support non-constant padding for 1D tensors")

        # For non-constant modes, PyTorch can only pad the last N dimensions:
        # - 2D: last 1 dim, 3D: last 1-2 dims, 4D: last 2-3 dims, 5D: last 3 dims
        # For constant mode, we can pad all dimensions
        input_rank = len(input_shape)
        if mode == "constant":
            # Pad all dimensions
            pads = [1, 1] * input_rank
        else:
            # For non-constant modes, pad only the allowed last N dimensions
            # ONNX pads format: [dim0_begin, dim1_begin, ..., dim0_end, dim1_end, ...]
            if input_rank == 2:
                # 2D: pad only last 1 dimension (dim 1)
                pads = [0, 1, 0, 1]
            elif input_rank == 3:
                # 3D: pad last 2 dimensions (dims 1, 2)
                pads = [0, 1, 1, 0, 1, 1]
            elif input_rank == 4:
                # 4D: pad last 3 dimensions (dims 1, 2, 3) - PyTorch allows last 2-3 dims (Pad2d or Pad3d)
                pads = [0, 1, 1, 1, 0, 1, 1, 1]
            elif input_rank == 5:
                # 5D: pad last 3 dimensions (dims 2, 3, 4)
                pads = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
            else:
                pytest.skip(f"Unsupported input rank {input_rank} for non-constant mode")

        onnx_model = self._create_pad_model(opset_version, input_shape, pads, mode, 0.0)

        transpiler = ONNXToForgeTranspiler()
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["Pad"])
        assert len(tir_graph.nodes) == 1
        assert tir_graph.nodes[0].op_type == "Pad"

        # Verify mode conversion
        pad_node = tir_graph.nodes[0]
        expected_mode = "replicate" if mode == "edge" else mode
        assert pad_node.attrs.get("mode") == expected_mode

        # Test execution - verify TIR output for both opsets
        input_data = {"data": np.random.randn(*input_shape).astype(np.float32)}
        input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
        tir_outputs = tir_graph.run(input_dict)

        # Verify output exists - use the actual output name from run results
        # The graph outputs might use sanitized names, so use the first available output
        assert len(tir_outputs) > 0, "No outputs from TIR graph run"
        output_name = list(tir_outputs.keys())[0]
        tir_output = tir_outputs[output_name]

        # Verify output shape
        if mode == "constant":
            expected_output_shape = tuple(s + 2 for s in input_shape)  # +2 because pads = [1, 1] * rank
        else:
            # For non-constant modes, calculate expected shape based on pads
            expected_output_shape = list(input_shape)
            num_dims = len(pads) // 2
            for i in range(num_dims):
                dim_idx = input_rank - num_dims + i
                if dim_idx >= 0:
                    expected_output_shape[dim_idx] += pads[i] + pads[i + num_dims]
            expected_output_shape = tuple(expected_output_shape)

        assert tuple(tir_output.shape) == expected_output_shape, (
            f"Output shape mismatch: expected {expected_output_shape}, got {tuple(tir_output.shape)}\n"
            f"Test params: opset={opset_version}, shape={input_shape}, mode={mode}"
        )

        # Verify output dtype
        assert tir_output.dtype == torch.float32, f"Output dtype mismatch: expected float32, got {tir_output.dtype}"

        # For opset 2, also compare with ONNX Runtime (opset 1 not supported by ONNX Runtime)
        if opset_version == 2:
            comparison = compare_tir_with_onnx(
                tir_graph=tir_graph, onnx_model=onnx_model, input_data=input_data, rtol=1e-5, atol=1e-6
            )
            assert len(comparison["errors"]) == 0, (
                f"Comparison errors: {comparison['errors']}\n"
                f"Test params: opset={opset_version}, shape={input_shape}, mode={mode}"
            )
            assert all(comparison["matches"].values()), (
                f"Output mismatch: {comparison}\n"
                f"Test params: opset={opset_version}, shape={input_shape}, mode={mode}"
            )

    @pytest.mark.parametrize("opset_version", [1, 2])
    @pytest.mark.parametrize("constant_value", [0.0, 1.5, -1.0, 42.0])
    def test_opset_1_2_constant_values(self, opset_version, constant_value):
        """Test Pad opsets 1-2 with different constant values."""
        input_shape = (3, 4)
        pads = [1, 1, 1, 1]  # Pad all dimensions

        onnx_model = self._create_pad_model(opset_version, input_shape, pads, "constant", constant_value)

        transpiler = ONNXToForgeTranspiler()
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["Pad"])
        pad_node = tir_graph.nodes[0]
        assert pad_node.attrs.get("value") == constant_value

        input_data = {"data": np.random.randn(*input_shape).astype(np.float32)}

        # Test execution - verify TIR output for both opsets
        input_dict = {name: torch.from_numpy(data) for name, data in input_data.items()}
        tir_outputs = tir_graph.run(input_dict)

        # Verify output exists - use the actual output name from run results
        assert len(tir_outputs) > 0, "No outputs from TIR graph run"
        output_name = list(tir_outputs.keys())[0]
        tir_output = tir_outputs[output_name]

        # Verify output shape
        expected_output_shape = tuple(s + 2 for s in input_shape)  # +2 because pads = [1, 1] * rank
        assert tuple(tir_output.shape) == expected_output_shape, (
            f"Output shape mismatch: expected {expected_output_shape}, got {tuple(tir_output.shape)}\n"
            f"Test params: opset={opset_version}, constant_value={constant_value}"
        )

        # Verify output dtype
        assert tir_output.dtype == torch.float32, f"Output dtype mismatch: expected float32, got {tir_output.dtype}"

        # Verify padding values (check corners are padded with constant_value)
        output_np = tir_output.detach().cpu().numpy()
        # For 2D: check corners [0,0], [0,-1], [-1,0], [-1,-1] should be constant_value
        assert np.isclose(
            output_np[0, 0], constant_value
        ), f"Top-left corner should be {constant_value}, got {output_np[0, 0]}"
        assert np.isclose(
            output_np[0, -1], constant_value
        ), f"Top-right corner should be {constant_value}, got {output_np[0, -1]}"
        assert np.isclose(
            output_np[-1, 0], constant_value
        ), f"Bottom-left corner should be {constant_value}, got {output_np[-1, 0]}"
        assert np.isclose(
            output_np[-1, -1], constant_value
        ), f"Bottom-right corner should be {constant_value}, got {output_np[-1, -1]}"

        # For opset 2, also compare with ONNX Runtime (opset 1 not supported by ONNX Runtime)
        if opset_version == 2:
            comparison = compare_tir_with_onnx(
                tir_graph=tir_graph, onnx_model=onnx_model, input_data=input_data, rtol=1e-5, atol=1e-6
            )
            assert len(comparison["errors"]) == 0, (
                f"Comparison errors: {comparison['errors']}\n"
                f"Test params: opset={opset_version}, constant_value={constant_value}"
            )
            assert all(comparison["matches"].values()), (
                f"Output mismatch: {comparison}\n"
                f"Test params: opset={opset_version}, constant_value={constant_value}"
            )

    # ========================================================================
    # OPSET 11-17 TESTS
    # ========================================================================

    @pytest.mark.parametrize("opset_version", [11, 13, 17])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (5,),  # 1D
            (3, 4),  # 2D
            (2, 3, 4),  # 3D
            (1, 2, 3, 4),  # 4D
            (1, 1, 2, 3, 4),  # 5D
        ],
    )
    @pytest.mark.parametrize("mode", ["constant", "reflect", "edge"])
    def test_opset_11_to_17_modes(self, opset_version, input_shape, mode):
        """Test Pad opsets 11-17 with different modes and dimensions."""
        # Skip 1D non-constant modes - PyTorch limitation
        if len(input_shape) == 1 and mode != "constant":
            pytest.skip(f"PyTorch doesn't support non-constant padding for 1D tensors")

        # For non-constant modes, PyTorch can only pad the last N dimensions:
        # - 2D: last 1 dim, 3D: last 1-2 dims, 4D: last 2-3 dims, 5D: last 3 dims
        # For constant mode, we can pad all dimensions
        input_rank = len(input_shape)
        if mode == "constant":
            # Pad all dimensions
            pads = [1, 1] * input_rank
        else:
            # For non-constant modes, pad only the allowed last N dimensions
            # ONNX pads format: [dim0_begin, dim1_begin, ..., dim0_end, dim1_end, ...]
            if input_rank == 2:
                # 2D: pad only last 1 dimension (dim 1)
                pads = [0, 1, 0, 1]
            elif input_rank == 3:
                # 3D: pad last 2 dimensions (dims 1, 2)
                pads = [0, 1, 1, 0, 1, 1]
            elif input_rank == 4:
                # 4D: pad last 3 dimensions (dims 1, 2, 3) - PyTorch allows last 2-3 dims (Pad2d or Pad3d)
                pads = [0, 1, 1, 1, 0, 1, 1, 1]
            elif input_rank == 5:
                # 5D: pad last 3 dimensions (dims 2, 3, 4)
                pads = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
            else:
                pytest.skip(f"Unsupported input rank {input_rank} for non-constant mode")

        onnx_model = self._create_pad_model(opset_version, input_shape, pads, mode, 0.0)

        transpiler = ONNXToForgeTranspiler()
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["Pad"])
        assert len(tir_graph.nodes) == 1
        assert tir_graph.nodes[0].op_type == "Pad"

        # Verify mode conversion
        pad_node = tir_graph.nodes[0]
        expected_mode = "replicate" if mode == "edge" else mode
        assert pad_node.attrs.get("mode") == expected_mode

        # Test execution
        input_data = {"data": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(
            tir_graph=tir_graph, onnx_model=onnx_model, input_data=input_data, rtol=1e-5, atol=1e-6
        )
        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, shape={input_shape}, mode={mode}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, shape={input_shape}, mode={mode}"
        )

    @pytest.mark.parametrize("opset_version", [11, 13, 17])
    @pytest.mark.parametrize("constant_value", [0.0, 1.5, -1.0, 42.0])
    def test_opset_11_to_17_constant_values(self, opset_version, constant_value):
        """Test Pad opsets 11-17 with different constant values."""
        input_shape = (3, 4)
        pads = [1, 1, 1, 1]

        onnx_model = self._create_pad_model(opset_version, input_shape, pads, "constant", constant_value)

        transpiler = ONNXToForgeTranspiler()
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["Pad"])
        pad_node = tir_graph.nodes[0]
        assert pad_node.attrs.get("value") == constant_value

        input_data = {"data": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(
            tir_graph=tir_graph, onnx_model=onnx_model, input_data=input_data, rtol=1e-5, atol=1e-6
        )
        assert len(comparison["errors"]) == 0
        assert all(comparison["matches"].values())

    # # ========================================================================
    # # OPSET 18+ TESTS (with selective axes support)
    # # ========================================================================

    @pytest.mark.parametrize("opset_version", [18, 19, 21, 23])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (5,),  # 1D
            (3, 4),  # 2D
            (2, 3, 4),  # 3D
            (1, 2, 3, 4),  # 4D
            (1, 1, 2, 3, 4),  # 5D
        ],
    )
    @pytest.mark.parametrize("mode", ["constant", "reflect", "edge"])
    def test_opset_18_plus_modes(self, opset_version, input_shape, mode):
        """Test Pad opsets 18+ with different modes and dimensions."""
        # Skip 1D non-constant modes - PyTorch limitation
        if len(input_shape) == 1 and mode != "constant":
            pytest.skip(f"PyTorch doesn't support non-constant padding for 1D tensors")

        # For non-constant modes, PyTorch can only pad the last N dimensions:
        # - 2D: last 1 dim, 3D: last 1-2 dims, 4D: last 2-3 dims, 5D: last 3 dims
        # For constant mode, we can pad all dimensions
        input_rank = len(input_shape)
        if mode == "constant":
            # Pad all dimensions
            pads = [1, 1] * input_rank
        else:
            # For non-constant modes, pad only the allowed last N dimensions
            # ONNX pads format: [dim0_begin, dim1_begin, ..., dim0_end, dim1_end, ...]
            if input_rank == 2:
                # 2D: pad only last 1 dimension (dim 1)
                pads = [0, 1, 0, 1]
            elif input_rank == 3:
                # 3D: pad last 2 dimensions (dims 1, 2)
                pads = [0, 1, 1, 0, 1, 1]
            elif input_rank == 4:
                # 4D: pad last 3 dimensions (dims 1, 2, 3) - PyTorch allows last 2-3 dims (Pad2d or Pad3d)
                pads = [0, 1, 1, 1, 0, 1, 1, 1]
            elif input_rank == 5:
                # 5D: pad last 3 dimensions (dims 2, 3, 4)
                pads = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
            else:
                pytest.skip(f"Unsupported input rank {input_rank} for non-constant mode")

        onnx_model = self._create_pad_model(opset_version, input_shape, pads, mode, 0.0)

        transpiler = ONNXToForgeTranspiler()
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["Pad"])
        assert len(tir_graph.nodes) == 1
        assert tir_graph.nodes[0].op_type == "Pad"

        # Verify mode conversion
        pad_node = tir_graph.nodes[0]
        expected_mode = "replicate" if mode == "edge" else ("circular" if mode == "wrap" else mode)
        assert pad_node.attrs.get("mode") == expected_mode

        # Test execution
        input_data = {"data": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(
            tir_graph=tir_graph, onnx_model=onnx_model, input_data=input_data, rtol=1e-5, atol=1e-6
        )
        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, shape={input_shape}, mode={mode}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, shape={input_shape}, mode={mode}"
        )

    @pytest.mark.parametrize("opset_version", [18, 19, 21, 23])
    @pytest.mark.parametrize("mode", ["constant", "reflect", "edge", "wrap"])
    def test_opset_18_plus_wrap_mode(self, opset_version, mode):
        """Test wrap mode for opset 19+."""
        # Skip wrap mode for opset < 19 (must check before creating model)
        if mode == "wrap" and opset_version < 19:
            pytest.skip(f"Wrap mode only available in opset 19+, got opset {opset_version}")

        input_shape = (3, 4)
        # For non-constant modes, pad only the last dimension (PyTorch constraint for 2D)
        if mode == "constant":
            pads = [1, 1, 1, 1]  # Pad all dimensions
        else:
            pads = [0, 1, 0, 1]  # Pad only last dimension (dim 1)

        onnx_model = self._create_pad_model(opset_version, input_shape, pads, mode, 0.0)

        transpiler = ONNXToForgeTranspiler()
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["Pad"])
        assert len(tir_graph.nodes) == 1
        assert tir_graph.nodes[0].op_type == "Pad"

        pad_node = tir_graph.nodes[0]
        expected_mode = "replicate" if mode == "edge" else ("circular" if mode == "wrap" else mode)
        assert pad_node.attrs.get("mode") == expected_mode

        # Test execution
        input_data = {"data": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(
            tir_graph=tir_graph, onnx_model=onnx_model, input_data=input_data, rtol=1e-5, atol=1e-6
        )
        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n" f"Test params: opset={opset_version}, mode={mode}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, mode={mode}"
        )

    @pytest.mark.parametrize("opset_version", [18, 19, 21, 23])
    @pytest.mark.parametrize(
        "axes",
        [
            [1, 2],  # Pad middle dimensions
            [0, 3],  # Pad first and last
            [2],  # Pad single dimension
            [-1, -2],  # Negative indices
        ],
    )
    @pytest.mark.parametrize("mode", ["constant"])
    def test_opset_18_plus_selective_axes(self, opset_version, axes, mode):
        """Test selective axis padding for opsets 18+."""
        input_shape = (2, 3, 4, 5)

        # Create pads for selected axes only
        pads = []
        for _ in axes:
            pads.extend([1, 1])  # Pad each selected axis by 1 on each side

        onnx_model = self._create_pad_model(opset_version, input_shape, pads, mode, 0.0, axes)

        transpiler = ONNXToForgeTranspiler()
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["Pad"])
        assert len(tir_graph.nodes) == 1
        assert tir_graph.nodes[0].op_type == "Pad"

        # Test execution
        input_data = {"data": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(
            tir_graph=tir_graph, onnx_model=onnx_model, input_data=input_data, rtol=1e-5, atol=1e-6
        )
        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, axes={axes}, mode={mode}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, axes={axes}, mode={mode}"
        )

    @pytest.mark.parametrize("opset_version", [18, 19, 21, 23])
    @pytest.mark.parametrize(
        "axes",
        [
            [2, 3],  # Pad last 2 dimensions (valid for 4D non-constant, Pad2d style)
            [1, 2, 3],  # Pad last 3 dimensions (valid for 4D non-constant, Pad3d style)
            [-2, -1],  # Pad last 2 dimensions (valid for 4D non-constant, negative indices)
            [-3, -2, -1],  # Pad last 3 dimensions (valid for 4D non-constant, Pad3d style)
        ],
    )
    @pytest.mark.parametrize("mode", ["reflect", "edge", "wrap"])
    def test_opset_18_plus_selective_axes_nonconstant(self, opset_version, axes, mode):
        """Test selective axis padding for opsets 18+ with non-constant modes."""
        # Skip wrap mode for opset < 19
        if mode == "wrap" and opset_version < 19:
            pytest.skip(f"Wrap mode only available in opset 19+")

        input_shape = (2, 3, 4, 5)  # 4D tensor

        # For 4D tensors, PyTorch can pad last 2-3 dimensions (dims [2, 3] or [1, 2, 3])
        # Validate that axes are within allowed range
        normalized_axes = [a if a >= 0 else len(input_shape) + a for a in axes]
        if not all(axis in [1, 2, 3] for axis in normalized_axes):
            pytest.skip(
                f"For 4D tensors with {mode} mode, only dimensions [1, 2, 3] can be padded. "
                f"Requested axes: {axes} (normalized: {normalized_axes})"
            )
        # Also check that we're padding the last N dimensions (consecutive from the end)
        expected_dims = list(range(4 - len(normalized_axes), 4))
        if normalized_axes != sorted(normalized_axes) or normalized_axes != expected_dims:
            pytest.skip(
                f"For 4D tensors with {mode} mode, must pad consecutive last dimensions. "
                f"Requested axes: {axes} (normalized: {normalized_axes}), "
                f"expected: {expected_dims}"
            )

        # Create pads for selected axes only
        pads = []
        for _ in axes:
            pads.extend([1, 1])  # Pad each selected axis by 1 on each side

        onnx_model = self._create_pad_model(opset_version, input_shape, pads, mode, 0.0, axes)

        transpiler = ONNXToForgeTranspiler()
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["Pad"])
        assert len(tir_graph.nodes) == 1
        assert tir_graph.nodes[0].op_type == "Pad"

        # Verify mode conversion
        pad_node = tir_graph.nodes[0]
        expected_mode = "replicate" if mode == "edge" else ("circular" if mode == "wrap" else mode)
        assert pad_node.attrs.get("mode") == expected_mode

        # Test execution
        input_data = {"data": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(
            tir_graph=tir_graph, onnx_model=onnx_model, input_data=input_data, rtol=1e-5, atol=1e-6
        )
        assert len(comparison["errors"]) == 0, (
            f"Comparison errors: {comparison['errors']}\n"
            f"Test params: opset={opset_version}, axes={axes}, mode={mode}"
        )
        assert all(comparison["matches"].values()), (
            f"Output mismatch: {comparison}\n" f"Test params: opset={opset_version}, axes={axes}, mode={mode}"
        )

    @pytest.mark.parametrize("opset_version", [18, 19, 21, 23])
    @pytest.mark.parametrize("constant_value", [0.0, 1.5, -1.0, 42.0])
    def test_opset_18_plus_constant_values(self, opset_version, constant_value):
        """Test Pad opsets 18+ with different constant values."""
        input_shape = (3, 4)
        pads = [1, 1, 1, 1]

        onnx_model = self._create_pad_model(opset_version, input_shape, pads, "constant", constant_value)

        transpiler = ONNXToForgeTranspiler()
        tir_graph = transpiler.transpile(onnx_model)

        verify_tir_graph_structure(tir_graph, onnx_model, expected_op_types=["Pad"])
        pad_node = tir_graph.nodes[0]
        assert pad_node.attrs.get("value") == constant_value

        input_data = {"data": np.random.randn(*input_shape).astype(np.float32)}
        comparison = compare_tir_with_onnx(
            tir_graph=tir_graph, onnx_model=onnx_model, input_data=input_data, rtol=1e-5, atol=1e-6
        )
        assert len(comparison["errors"]) == 0
        assert all(comparison["matches"].values())
