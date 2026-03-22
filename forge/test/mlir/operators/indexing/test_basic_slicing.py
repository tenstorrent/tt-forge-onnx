# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "input_tensor_slice",
    [
        pytest.param(
            (torch.arange(10, dtype=torch.float32), slice(2, 8, 2)),
            id="slice_with_step",
        ),
        pytest.param((torch.arange(10, dtype=torch.float32), slice(0, 5)), id="first_five_elements"),
        pytest.param((torch.arange(10, dtype=torch.float32), slice(None, -1)), id="all_but_last"),
        pytest.param((torch.arange(10, dtype=torch.float32), slice(None, None, 2)), id="every_second_element"),
        pytest.param((torch.arange(10, dtype=torch.float32), slice(3, None)), id="slice_from_index_three_to_end"),
    ],
)
@pytest.mark.push
def test_slicing(input_tensor_slice):
    input_tensor, slicing = input_tensor_slice

    class SlicingModule(nn.Module):
        def __init__(self, slicing):
            super().__init__()
            self.slicing = slicing

        def forward(self, x):
            # Apply slicing
            return x[self.slicing]

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = SlicingModule(slicing)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor_slicing",
    [
        pytest.param(
            (torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (slice(1, 3), slice(None), slice(None, None, 2))),
            id="slicing_across_multiple_dims",
        ),
        pytest.param(
            (torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (0, slice(None), [0, 2])),
            id="specific_rows",
            marks=pytest.mark.xfail(reason="Exception: warning unhandled case: <class 'NoneType'>"),
        ),
        pytest.param(
            (torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (0, 0, [0, 2])),
            id="specific_rows_columns",
        ),
        pytest.param(
            (torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (slice(None), slice(1, 3), slice(None))),
            id="slice_middle_columns",
        ),
        pytest.param(
            (torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (slice(0, 2), slice(1, 3))),
            id="rows_and_columns_subrange",
        ),
        pytest.param((torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (slice(None), None)), id="add_new_axis"),
        pytest.param((torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (..., 0)), id="ellipsis_first_dim"),
        pytest.param(
            (torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (slice(None), slice(0, 1))),
            id="keep_dimension_with_size_1",
        ),
        pytest.param(
            (torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (slice(-2, None), slice(-2, None))),
            id="negative_indexing_subrange",
        ),
        pytest.param(
            (torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), (slice(-1, None), slice(-1, None))),
            id="negative_indexing_single_element",
        ),
    ],
)
@pytest.mark.push
def test_multidimensional_slicing(input_tensor_slicing):
    input_tensor, slicing = input_tensor_slicing

    class SlicingModule(torch.nn.Module):
        def __init__(self, slicing):
            super().__init__()
            self.slicing = slicing

        def forward(self, x):
            return x[self.slicing]

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = SlicingModule(slicing)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)
