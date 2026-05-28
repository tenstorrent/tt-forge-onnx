# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Debug mode validator — compare TIR outputs with ONNX Runtime.

Optimization: instead of running ORT once per ONNX node, the caller should
use ``collect_all_ort_outputs()`` to run ORT **once** for the whole model and
obtain a cache of every intermediate tensor.  ``compare_node_outputs_with_cache()``
then does pure in-process comparison against that cache with no further ORT
calls.

Legacy entry-point ``debug_node_output()`` is kept for backward compatibility
but internally calls ``get_activation_value()``, which runs ORT for a single
node — avoid using it in hot loops.
"""
import io
from typing import Dict, List

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

from forge.transpiler.utils.exceptions import DebugValidationError


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_ort_session(model_proto: onnx.ModelProto) -> "ort.InferenceSession":
    """
    Serialize *model_proto* and create an ORT session with noise-suppressing
    options (single-threaded, ERROR-only logging).
    """
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # 0=Verbose … 3=Error
    opts.inter_op_num_threads = 1  # no thread-pool → no affinity errors
    opts.intra_op_num_threads = 1
    buf = io.BytesIO()
    onnx.save(model_proto, buf)
    buf.seek(0)
    return ort.InferenceSession(buf.read(), sess_options=opts)


def _inputs_to_numpy(inputs) -> List[np.ndarray]:
    """Ensure every element in *inputs* is a NumPy array."""
    return [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x) for x in inputs]


def _build_inputs_dict(sess: "ort.InferenceSession", inputs: List[np.ndarray]) -> dict:
    """Zip session input names with numpy arrays."""
    input_names = [inp.name for inp in sess.get_inputs()]
    if not isinstance(inputs, list):
        inputs = [inputs]
    return dict(zip(input_names, inputs))


# ---------------------------------------------------------------------------
# Public API — optimised single-pass approach
# ---------------------------------------------------------------------------


def collect_all_ort_outputs(
    onnx_model: onnx.ModelProto,
    inputs: list,
    output_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Run ONNX Runtime **once** and return every requested intermediate tensor.

    This is the efficient alternative to calling ``get_activation_value()`` in
    a per-node loop.  The caller collects *all* ONNX node output names upfront,
    passes them here, and receives a ``{name: np.ndarray}`` cache that can be
    queried cheaply during the validation loop without any further ORT calls.

    Args:
        onnx_model   : Original ``onnx.ModelProto``.
        inputs       : Model inputs as a list of numpy arrays (or torch tensors).
        output_names : All intermediate tensor names to capture.  Empty or
                       optional output names (empty strings) are silently
                       ignored.

    Returns:
        Dictionary mapping each requested output name to its NumPy value.
        Names that ORT could not produce (e.g. graph inputs / initializers
        that were also listed) are omitted without raising an error.

    Raises:
        ImportError           : If ``onnxruntime`` is not installed.
        DebugValidationError  : If ORT fails to run the model.
    """
    if not ORT_AVAILABLE:
        raise ImportError("onnxruntime is required for debug mode.  " "Install with: pip install onnxruntime")

    # Deduplicate and filter out empty/blank names
    unique_names = list(dict.fromkeys(n for n in output_names if n and n.strip()))
    if not unique_names:
        return {}

    inputs_np = _inputs_to_numpy(inputs)

    # Build a model copy that exposes every requested tensor as an output.
    # We only add names that are not already outputs to avoid duplication.
    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(onnx_model)

    existing_output_names = {vi.name for vi in model_copy.graph.output}
    for name in unique_names:
        if name not in existing_output_names:
            vi = onnx.helper.ValueInfoProto()
            vi.name = name
            model_copy.graph.output.append(vi)

    try:
        sess = _make_ort_session(model_copy)
        inputs_dict = _build_inputs_dict(sess, inputs_np)
        # Fetch only the extra outputs we added (session outputs include the
        # originals too, so we can't index by position directly).
        output_meta = [o.name for o in sess.get_outputs()]
        raw_results = sess.run(None, inputs_dict)
        result_map = dict(zip(output_meta, raw_results))

        return {name: result_map[name] for name in unique_names if name in result_map}

    except DebugValidationError:
        raise
    except Exception as e:
        raise DebugValidationError(f"ORT single-pass execution failed: {e}") from e


def compare_node_outputs_with_cache(
    ort_cache: Dict[str, np.ndarray],
    tir_outputs: Dict[str, torch.Tensor],
    onnx_node,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """
    Compare TIR node outputs against a pre-computed ORT cache.

    No ORT session is created here — this is a pure in-process comparison
    intended to be called inside the TIR execution loop after
    ``collect_all_ort_outputs()`` has already been called once.

    Args:
        ort_cache   : ``{tensor_name: np.ndarray}`` produced by
                      ``collect_all_ort_outputs()``.
        tir_outputs : ``{output_name: torch.Tensor}`` collected from TIR nodes.
        onnx_node   : ``onnx.NodeProto`` of the ONNX node being validated.
        rtol        : Relative tolerance for ``np.allclose``.
        atol        : Absolute tolerance for ``np.allclose``.

    Raises:
        DebugValidationError : On shape, dtype, or value mismatch.
    """
    node_name = getattr(onnx_node, "name", "<unknown>")

    for output_name in onnx_node.output:
        if not output_name:
            continue  # skip optional empty outputs

        if output_name not in tir_outputs:
            raise DebugValidationError(
                f"[{node_name}] TIR did not produce output '{output_name}'. " f"Available: {list(tir_outputs.keys())}"
            )

        if output_name not in ort_cache:
            logger.warning(
                f"[{node_name}] '{output_name}' not found in ORT cache — skipping comparison. "
                "This may happen for graph inputs / initializers listed as node outputs."
            )
            continue

        predicted = tir_outputs[output_name]
        expected = ort_cache[output_name]

        # Convert TIR tensor to numpy
        if isinstance(predicted, torch.Tensor):
            predicted_np = predicted.detach().cpu().numpy()
        else:
            predicted_np = np.asarray(predicted)

        expected_np = np.asarray(expected)

        # 1. Shape check
        if predicted_np.shape != expected_np.shape:
            raise DebugValidationError(
                f"[{node_name}] Shape mismatch for '{output_name}': "
                f"TIR {predicted_np.shape} vs ORT {expected_np.shape}"
            )

        # 2. Dtype check (warn only — dtype promotion is sometimes intentional)
        if predicted_np.dtype != expected_np.dtype:
            logger.warning(
                f"[{node_name}] Dtype mismatch for '{output_name}': "
                f"TIR {predicted_np.dtype} vs ORT {expected_np.dtype} — casting for comparison"
            )
            expected_np = expected_np.astype(predicted_np.dtype)

        # 3. Value check
        if not np.allclose(predicted_np, expected_np, rtol=rtol, atol=atol):
            max_diff = np.abs(predicted_np - expected_np).max()
            mean_diff = np.abs(predicted_np - expected_np).mean()
            relative_diff = max_diff / (np.abs(expected_np).max() + 1e-8)
            raise DebugValidationError(
                f"[{node_name}] Value mismatch for '{output_name}' "
                f"(shape {predicted_np.shape}):  "
                f"max_diff={max_diff:.4e}  mean_diff={mean_diff:.4e}  "
                f"relative={relative_diff:.4e}  (atol={atol})"
            )


# ---------------------------------------------------------------------------
# Legacy helpers — kept for backward compatibility
# ---------------------------------------------------------------------------


def get_activation_value(
    onnx_model: onnx.ModelProto,
    inputs: list,
    activation_names: list,
) -> List[np.ndarray]:
    """
    Get intermediate activation values from ORT for the named tensors.

    .. deprecated::
        Prefer ``collect_all_ort_outputs()`` which runs ORT only once for
        the whole model instead of once per node.

    Args:
        onnx_model       : ONNX model.
        inputs           : Model inputs as a list of numpy arrays or tensors.
        activation_names : Names of the intermediate tensors to extract.

    Returns:
        List of numpy arrays in the same order as *activation_names*.
    """
    if not ORT_AVAILABLE:
        raise ImportError("onnxruntime is required for debug mode.  " "Install with: pip install onnxruntime")

    if not isinstance(activation_names, (list, tuple)):
        activation_names = [activation_names]

    inputs_np = _inputs_to_numpy(inputs)

    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(onnx_model)

    while len(model_copy.graph.output):
        model_copy.graph.output.pop()

    for name in activation_names:
        vi = onnx.helper.ValueInfoProto()
        vi.name = name
        model_copy.graph.output.append(vi)

    sess = _make_ort_session(model_copy)
    inputs_dict = _build_inputs_dict(sess, inputs_np)
    return sess.run(None, inputs_dict)


def debug_node_output(
    onnx_model: onnx.ModelProto,
    inputs: list,
    node_outputs: dict,
    onnx_node,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """
    Compare TIR node outputs against ORT by running ORT for this single node.

    .. deprecated::
        Prefer the ``collect_all_ort_outputs`` + ``compare_node_outputs_with_cache``
        pair, which runs ORT once for the entire model instead of once per node.

    Args:
        onnx_model   : Original ``onnx.ModelProto``.
        inputs       : Model inputs (list of numpy arrays / tensors).
        node_outputs : ``{output_name: tensor}`` from TIR execution.
        onnx_node    : ``onnx.NodeProto`` of the node being validated.
        rtol         : Relative tolerance.
        atol         : Absolute tolerance.
    """
    if not ORT_AVAILABLE:
        logger.warning("onnxruntime not available, skipping debug comparison")
        return

    try:
        expected_outputs = get_activation_value(onnx_model, inputs, list(onnx_node.output))
        ort_cache = dict(zip(onnx_node.output, expected_outputs))
        compare_node_outputs_with_cache(ort_cache, node_outputs, onnx_node, rtol=rtol, atol=atol)

    except DebugValidationError:
        raise
    except Exception as e:
        node_name = getattr(onnx_node, "name", None)
        error_msg = f"Unexpected error in debug comparison for node {node_name}: {e}"
        logger.error(error_msg, exc_info=True)
        raise DebugValidationError(error_msg, frontend_node_name=node_name) from e
