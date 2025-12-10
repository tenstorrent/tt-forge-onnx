"""
Debug mode for comparing TIR outputs with ONNXRuntime.
"""
import io
from loguru import logger
import numpy as np
import onnx
import torch

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    ort = None


class DebugValidationError(Exception):
    """Exception raised when debug mode validation fails (e.g., shape mismatch, value mismatch)."""
    pass


def get_activation_value(onnx_model: onnx.ModelProto, inputs: list, activation_names: list):
    """
    Get activation value from ONNXRuntime for comparison.
    
    Args:
        onnx_model: ONNX model
        inputs: List of numpy arrays (inputs to the model)
        activation_names: List of activation names to extract
        
    Returns:
        List of numpy arrays (activation values)
    """
    if not ORT_AVAILABLE:
        raise ImportError("onnxruntime is required for debug mode. Install with: pip install onnxruntime")
    
    if not all(isinstance(x, np.ndarray) for x in inputs):
        inputs = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in inputs]
    
    if not isinstance(activation_names, (list, tuple)):
        activation_names = [activation_names]
    
    # Create a temporary model with only the desired outputs
    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(onnx_model)
    
    # Clear existing outputs
    while len(model_copy.graph.output):
        model_copy.graph.output.pop()
    
    # Add desired outputs
    for activation_name in activation_names:
        activation_value = onnx.helper.ValueInfoProto()
        activation_value.name = activation_name
        model_copy.graph.output.append(activation_value)
    
    # Save and load to ensure serialization
    buffer = io.BytesIO()
    onnx.save(model_copy, buffer)
    buffer.seek(0)
    model_serialized = onnx.load(buffer)
    
    # Create inference session
    sess = ort.InferenceSession(model_serialized.SerializeToString())
    
    # Get input names
    input_names = [x.name for x in sess.get_inputs()]
    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs_dict = dict(zip(input_names, inputs))
    
    # Run inference
    outputs = sess.run(None, inputs_dict)
    return outputs


def debug_node_output(
    onnx_model: onnx.ModelProto,
    inputs: list,
    node_outputs: dict,
    onnx_node,
    rtol: float = 1e-3,
    atol: float = 1e-4
):
    """
    Compare TIR node output with ONNXRuntime output.
    
    Args:
        onnx_model: Original ONNX model
        inputs: List of input tensors (numpy arrays) - model inputs only
        node_outputs: Dict of {output_name: torch.Tensor} from TIR
        onnx_node: ONNX node being compared
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    if not ORT_AVAILABLE:
        logger.warning("onnxruntime not available, skipping debug comparison")
        return
    
    try:
        # Get expected outputs from ONNXRuntime
        expected_outputs = get_activation_value(onnx_model, inputs, list(onnx_node.output))
        
        # Compare each output
        for i, output_name in enumerate(onnx_node.output):
            if output_name not in node_outputs:
                error_msg = f"Output {output_name} not found in node outputs"
                logger.error(error_msg)
                raise DebugValidationError(error_msg)
            
            predicted = node_outputs[output_name]
            expected = expected_outputs[i] if i < len(expected_outputs) else None
            
            if expected is None:
                error_msg = f"No expected output for {output_name}"
                logger.error(error_msg)
                raise DebugValidationError(error_msg)
            
            # Convert to numpy for comparison
            if isinstance(predicted, torch.Tensor):
                predicted_np = predicted.detach().cpu().numpy()
            else:
                predicted_np = np.array(predicted)
            
            expected_np = np.array(expected)
            
            # Check shapes
            if predicted_np.shape != expected_np.shape:
                error_msg = (
                    f"[{onnx_node.name}] Shape mismatch for {output_name}: "
                    f"predicted {predicted_np.shape} vs expected {expected_np.shape}"
                )
                logger.error(error_msg)
                # In debug mode, shape mismatches are critical - raise exception to stop execution
                raise DebugValidationError(error_msg)
            
            # Check values
            if np.allclose(predicted_np, expected_np, rtol=rtol, atol=atol):
                logger.info(
                    f"[{onnx_node.name}] ✓ {output_name} matches ONNXRuntime "
                    f"(shape: {predicted_np.shape})"
                )
            else:
                max_diff = np.abs(predicted_np - expected_np).max()
                mean_diff = np.abs(predicted_np - expected_np).mean()
                relative_diff = max_diff / (np.abs(expected_np).max() + 1e-8)
                error_msg = (
                    f"[{onnx_node.name}] ✗ {output_name} differs from ONNXRuntime "
                    f"(shape: {predicted_np.shape}, max diff: {max_diff:.6e}, "
                    f"mean diff: {mean_diff:.6e}, relative: {relative_diff:.6e})"
                )
                logger.error(error_msg)
                # In debug mode, value mismatches are critical - raise exception to stop execution
                raise DebugValidationError(error_msg)
                
    except Exception as e:
        logger.error(f"Error in debug comparison for node {onnx_node.name}: {e}", exc_info=True)

