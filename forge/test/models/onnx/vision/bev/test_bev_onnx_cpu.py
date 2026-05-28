# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from test.models.onnx.vision.bev.model_utils.bev_utils import (
    INPUT_NAMES,
    assets_available,
    bev_paths,
    list_sequences,
    load_ground_truth_outputs,
    load_inputs,
)


ATOL = 1e-3
RTOL = 1e-3

_SEQUENCES = list_sequences() if assets_available() else []


@pytest.fixture(scope="module")
def bev_session():
    if not assets_available():
        paths = bev_paths()
        pytest.skip(
            f"BEV assets not found under {paths['root']}. "
            "Set BEV_ASSETS_DIR or populate model/input_samples/output_samples."
        )
    paths = bev_paths()
    onnx_model = onnx.load(str(paths["model"]))
    onnx.checker.check_model(onnx_model)

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(paths["model"]), sess_opts, providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in session.get_outputs()]
    return session, output_names


@pytest.mark.nightly
@pytest.mark.parametrize(
    "seq_id",
    _SEQUENCES if _SEQUENCES else [pytest.param("", marks=pytest.mark.skip(reason="BEV assets not found"))],
)
def test_bev_onnx_cpu_inference(bev_session, seq_id):
    session, output_names = bev_session

    input_tensors = load_inputs(seq_id)
    input_np = {name: t.numpy() for name, t in zip(INPUT_NAMES, input_tensors)}

    actual_outputs = session.run(output_names, input_np)
    gt = load_ground_truth_outputs(seq_id)

    failures = []
    for name, actual in zip(output_names, actual_outputs):
        expected = gt[name]
        if not np.allclose(actual, expected, atol=ATOL, rtol=RTOL):
            max_abs = float(np.abs(actual - expected).max())
            failures.append((name, max_abs))

    assert not failures, (
        f"[seq={seq_id}] {len(failures)}/{len(output_names)} output(s) exceeded tolerance "
        f"(atol={ATOL}, rtol={RTOL}):\n"
        + "\n".join(f"  {name}: max|Δ|={err:.3e}" for name, err in failures)
    )
