# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shape-inference mixins for all TIR operation nodes.

Design
------
* **Generic family classes** cover entire operation families where the shape
  rule is identical.  Nodes inherit directly from the appropriate class:

      class ReluNode(ElementwiseUnaryShape, TIRNode): ...
      class AddNode(BinaryBroadcastShape, TIRNode): ...
      class WhereNode(TernaryBroadcastShape, TIRNode): ...
      class ReduceSumNode(ReductionShape, TIRNode): ...

* **Specific classes** are created only when an operation needs custom logic
  that cannot be shared with any other operation family:

      MatMulShape, ConcatShape, PadShape, LayerNormShape,
      EmbeddingShape, IndexSelectShape, IndexShape,
      ConvShape (Conv1d/2d/3d — each node sets ``_ndim``),
      PoolingShape (MaxPool/AveragePool 1d/2d/3d — each node sets ``_ndim``),
      ReshapeShape, TransposeShape, SqueezeShape, UnsqueezeShape,
      BroadcastToShape, SplitShape, FullShape

All classes validate their inputs and return ``None`` on failure, letting
the engine fall back to fake-tensor execution.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def broadcast_shape(shape_a: Tuple, shape_b: Tuple) -> Optional[Tuple]:
    """NumPy-style pairwise broadcast. Returns ``None`` on incompatible shapes."""
    if shape_a is None or shape_b is None:
        return None
    len_a, len_b = len(shape_a), len(shape_b)
    max_len = max(len_a, len_b)
    a: List[int] = [1] * (max_len - len_a) + list(shape_a)
    b: List[int] = [1] * (max_len - len_b) + list(shape_b)
    out: List[int] = []
    for da, db in zip(a, b):
        if da != db and da != 1 and db != 1:
            return None
        out.append(max(da, db))
    return tuple(out)


def _conv_out_dim(in_size: int, kernel: int, stride: int, pad: int, dilation: int) -> int:
    return math.floor((in_size + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1)


def _pool_out_dim(in_size: int, kernel: int, stride: int, pad: int, dilation: int, ceil_mode: bool) -> int:
    numerator = in_size + 2 * pad - dilation * (kernel - 1) - 1
    return math.ceil(numerator / stride + 1) if ceil_mode else math.floor(numerator / stride + 1)


def _normalize_to_int_tuple(value, ndim: int) -> Optional[Tuple[int, ...]]:
    """Coerce int / list / tuple to a fixed-length int tuple; returns None for string padding."""
    if isinstance(value, str):
        return None
    if isinstance(value, int):
        return (value,) * ndim
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return (0,) * ndim
        if len(value) == 1:
            return (int(value[0]),) * ndim
        if len(value) >= ndim:
            return tuple(int(v) for v in value[:ndim])
        t = tuple(int(v) for v in value)
        return t + (t[-1],) * (ndim - len(t))
    return None


def _conv_infer_shape(node, tensor_shapes: Dict, ndim: int) -> Optional[Dict[str, Tuple]]:
    """Shared conv shape logic for Conv1d/2d/3d — (N, C_in, *spatial) → (N, C_out, *spatial_out)."""
    names_in = getattr(node, "input_names", None)
    names_out = getattr(node, "output_names", None)
    if not names_in or len(names_in) < 2 or not names_out:
        return None
    x = tensor_shapes.get(names_in[0])
    w = tensor_shapes.get(names_in[1])
    if (
        x is None
        or w is None
        or node._has_unknown_dimension(x)
        or node._has_unknown_dimension(w)
        or len(x) != ndim + 2
        or len(w) < ndim + 2
    ):
        return None
    attrs = getattr(node, "attrs", {})
    stride = _normalize_to_int_tuple(attrs.get("stride", 1), ndim)
    dilation = _normalize_to_int_tuple(attrs.get("dilation", 1), ndim)
    padding = _normalize_to_int_tuple(attrs.get("padding", 0), ndim)
    if stride is None or dilation is None or padding is None:
        return None
    spatial = [_conv_out_dim(x[2 + i], w[2 + i], stride[i], padding[i], dilation[i]) for i in range(ndim)]
    return {names_out[0]: (x[0], w[0]) + tuple(spatial)}


def _pool_infer_shape(node, tensor_shapes: Dict, ndim: int) -> Optional[Dict[str, Tuple]]:
    """Shared pool shape logic for Max/AveragePool — (N, C, *spatial) → (N, C, *spatial_out)."""
    names_in = getattr(node, "input_names", None)
    names_out = getattr(node, "output_names", None)
    if not names_in or not names_out:
        return None
    x = tensor_shapes.get(names_in[0])
    if x is None or node._has_unknown_dimension(x) or len(x) != ndim + 2:
        return None
    attrs = getattr(node, "attrs", {})
    kernel_raw = attrs.get("kernel_size")
    if kernel_raw is None:
        return None
    kernel = _normalize_to_int_tuple(kernel_raw, ndim)
    stride = _normalize_to_int_tuple(attrs.get("stride", kernel_raw), ndim)
    padding = _normalize_to_int_tuple(attrs.get("padding", 0), ndim)
    dilation = _normalize_to_int_tuple(attrs.get("dilation", 1), ndim)
    ceil_mode = bool(attrs.get("ceil_mode", False))
    if kernel is None or stride is None or padding is None or dilation is None:
        return None
    spatial = [_pool_out_dim(x[2 + i], kernel[i], stride[i], padding[i], dilation[i], ceil_mode) for i in range(ndim)]
    return {names_out[0]: (x[0], x[1]) + tuple(spatial)}


# ---------------------------------------------------------------------------
# Generic family classes
# ---------------------------------------------------------------------------


class ElementwiseUnaryShape:
    """
    Single-input, single-output element-wise op: output shape == input shape.

    Used by: Relu, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyRelu, Dropout,
             Sqrt, Erf, Reciprocal, Pow, Clip, Cast, Identity, …
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        in_shape = tensor_shapes.get(names_in[0])
        if in_shape is None or self._has_unknown_dimension(in_shape):  # type: ignore[attr-defined]
            return None
        return {names_out[0]: tuple(in_shape)}


class BinaryBroadcastShape:
    """
    Two-input, single-output broadcast op: output shape == broadcast(a, b).

    Used by: Add, Sub, Mul, Div, Equal, Greater, Less, GreaterOrEqual, LessOrEqual, …
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or len(names_in) < 2 or not names_out:
            return None
        a = tensor_shapes.get(names_in[0])
        b = tensor_shapes.get(names_in[1])
        if (
            a is None
            or b is None
            or self._has_unknown_dimension(a)  # type: ignore[attr-defined]
            or self._has_unknown_dimension(b)  # type: ignore[attr-defined]
        ):
            return None
        out = broadcast_shape(a, b)
        return {names_out[0]: out} if out is not None else None


class TernaryBroadcastShape:
    """
    Three-input, single-output broadcast op: output == broadcast(broadcast(a, b), c).

    Used by: Where (condition, x, y)
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or len(names_in) < 3 or not names_out:
            return None
        s0 = tensor_shapes.get(names_in[0])
        s1 = tensor_shapes.get(names_in[1])
        s2 = tensor_shapes.get(names_in[2])
        if s0 is None or s1 is None or s2 is None:
            return None
        if (
            self._has_unknown_dimension(s0)  # type: ignore[attr-defined]
            or self._has_unknown_dimension(s1)  # type: ignore[attr-defined]
            or self._has_unknown_dimension(s2)  # type: ignore[attr-defined]
        ):
            return None
        out = broadcast_shape(s0, s1)
        if out is None:
            return None
        out = broadcast_shape(out, s2)
        return {names_out[0]: out} if out is not None else None


class ReductionShape:
    """
    Single-input reduction op.  Reads ``dim`` and ``keepdim`` from attrs.

    Used by: ReduceSum, ReduceMean, ReduceMax, …

    ``compute_shape`` is a static method so callers can use it independently:
        ReductionShape.compute_shape(input_shape, dim, keepdim)
    """

    @staticmethod
    def compute_shape(input_shape: Tuple, dim, keepdim: bool) -> Optional[Tuple]:
        if input_shape is None:
            return None
        rank = len(input_shape)
        if dim is None:
            return tuple(1 for _ in input_shape) if keepdim else ()
        dims: List[int] = [dim] if isinstance(dim, int) else list(dim)
        dims = [(d + rank if d < 0 else d) for d in dims]
        if any(d < 0 or d >= rank for d in dims):
            return None
        if keepdim:
            return tuple(1 if i in dims else d for i, d in enumerate(input_shape))
        return tuple(d for i, d in enumerate(input_shape) if i not in dims)

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        in_shape = tensor_shapes.get(names_in[0])
        if in_shape is None or self._has_unknown_dimension(in_shape):  # type: ignore[attr-defined]
            return None
        attrs = getattr(self, "attrs", {})  # type: ignore[attr-defined]
        out = self.compute_shape(in_shape, attrs.get("dim"), bool(attrs.get("keepdim", False)))
        return {names_out[0]: out} if out is not None else None


# ---------------------------------------------------------------------------
# Specific classes — custom shape logic per operation
# ---------------------------------------------------------------------------


class MatMulShape:
    """
    torch.matmul shape rule (NumPy semantics).

    * Both 1-D  → dot product, scalar output.
    * One 1-D   → vector output (extra dim stripped after matmul promotion).
    * Both ≥ 2-D → batch dims are broadcast; last two dims are (m,k)×(k,n)→(m,n).
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or len(names_in) < 2 or not names_out:
            return None
        a = tensor_shapes.get(names_in[0])
        b = tensor_shapes.get(names_in[1])
        if (
            a is None
            or b is None
            or self._has_unknown_dimension(a)  # type: ignore[attr-defined]
            or self._has_unknown_dimension(b)  # type: ignore[attr-defined]
            or len(a) < 1
            or len(b) < 1
        ):
            return None
        a_vec, b_vec = len(a) == 1, len(b) == 1
        a2 = (1, a[0]) if a_vec else a
        b2 = (b[0], 1) if b_vec else b
        if a2[-1] != b2[-2]:
            return None
        batch = broadcast_shape(a2[:-2], b2[:-2])
        if batch is None:
            return None
        out = tuple(batch) + (a2[-2], b2[-1])
        if a_vec:
            out = out[:-2] + (out[-1],)
        if b_vec:
            out = out[:-1]
        return {names_out[0]: out}


class LayerNormShape:
    """
    torch.nn.LayerNorm.  Primary output Y has the same shape as X.
    Optional Mean and InvStdDev outputs have shape x_shape[:axis] + (1,)*(rank-axis).
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        x = tensor_shapes.get(names_in[0])
        if x is None or self._has_unknown_dimension(x):  # type: ignore[attr-defined]
            return None
        attrs = getattr(self, "attrs", {})  # type: ignore[attr-defined]
        axis = int(attrs.get("axis", -1))
        rank = len(x)
        if axis < 0:
            axis = rank + axis
        if axis < 0 or axis >= rank:
            return None
        result: Dict[str, Tuple] = {names_out[0]: tuple(x)}
        stats = tuple(x[:axis]) + (1,) * (rank - axis)
        if len(names_out) > 1:
            result[names_out[1]] = stats
        if len(names_out) > 2:
            result[names_out[2]] = stats
        return result


class EmbeddingShape:
    """
    torch.nn.functional.embedding — output: (*idx_shape, embed_dim).

    Inputs: [embedding_table (vocab, embed_dim), indices (*idx_shape)]
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or len(names_in) < 2 or not names_out:
            return None
        table = tensor_shapes.get(names_in[0])
        idx = tensor_shapes.get(names_in[1])
        if table is None or idx is None:
            return None
        if (
            self._has_unknown_dimension(table)  # type: ignore[attr-defined]
            or self._has_unknown_dimension(idx)  # type: ignore[attr-defined]
            or len(table) < 2
        ):
            return None
        return {names_out[0]: tuple(idx) + (table[1],)}


class IndexSelectShape:
    """
    torch.index_select(data, dim, index) — replaces data_shape[dim] with len(index).
    index must be 1-D.
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or len(names_in) < 2 or not names_out:
            return None
        data = tensor_shapes.get(names_in[0])
        idx = tensor_shapes.get(names_in[1])
        if data is None or idx is None:
            return None
        if (
            self._has_unknown_dimension(data)  # type: ignore[attr-defined]
            or self._has_unknown_dimension(idx)  # type: ignore[attr-defined]
            or len(idx) != 1
        ):
            return None
        attrs = getattr(self, "attrs", {})  # type: ignore[attr-defined]
        dim = int(attrs.get("dim", 0))
        rank = len(data)
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            return None
        out = list(data)
        out[dim] = idx[0]
        return {names_out[0]: tuple(out)}


class IndexShape:
    """
    Strided-slice: data[start:stop:stride] along axis.
    Reads axis, start, stop, stride from attrs.
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        shape = tensor_shapes.get(names_in[0])
        if shape is None or self._has_unknown_dimension(shape):  # type: ignore[attr-defined]
            return None
        attrs = getattr(self, "attrs", {})  # type: ignore[attr-defined]
        axis = int(attrs.get("axis", 0))
        stride = int(attrs.get("stride", 1))
        rank = len(shape)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank or stride == 0:
            return None
        sz = shape[axis]
        start = int(attrs.get("start", 0))
        stop = int(attrs.get("stop", sz))
        if start < 0:
            start += sz
        if stop < 0:
            stop += sz
        if stride > 0:
            start, stop = max(0, min(start, sz)), max(0, min(stop, sz))
            out_dim = max(0, (stop - start + stride - 1) // stride)
        else:
            start = max(0, min(start, sz - 1))
            stop = max(-1, min(stop, sz - 1))
            out_dim = max(0, (start - stop + abs(stride) - 1) // abs(stride))
        result = list(shape)
        result[axis] = out_dim
        return {names_out[0]: tuple(result)}


class ConvShape:
    """
    Shape mixin for all convolution operations (Conv1d / Conv2d / Conv3d).

    The spatial rank is declared as ``_ndim`` on each concrete node class:

        class Conv2dNode(ConvShape, TIRNode):
            _ndim = 2

    Input layout : ``(N, C_in,  *spatial_in)``
    Weight layout: ``(C_out, C_in/groups, *kernel)``
    Output layout: ``(N, C_out, *spatial_out)``
    """

    _ndim: int  # must be set to 1, 2, or 3 by the inheriting node class

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        return _conv_infer_shape(self, tensor_shapes, ndim=self._ndim)


class PoolingShape:
    """
    Shape mixin for all pooling operations (MaxPool and AveragePool in 1D/2D/3D).

    The spatial rank is declared as ``_ndim`` on each concrete node class:

        class MaxPool2dNode(PoolingShape, TIRNode):
            _ndim = 2

    Input / output layout: ``(N, C, *spatial_in)`` → ``(N, C, *spatial_out)``
    Channels are always preserved.
    """

    _ndim: int  # must be set to 1, 2, or 3 by the inheriting node class

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        return _pool_infer_shape(self, tensor_shapes, ndim=self._ndim)


class ReshapeShape:
    """
    torch.reshape.  Resolves a single ``-1`` wildcard dimension from
    total element count; returns ``None`` if ambiguous or indivisible.
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        in_shape = tensor_shapes.get(names_in[0])
        target = getattr(self, "attrs", {}).get("shape")  # type: ignore[attr-defined]
        if in_shape is None or target is None or self._has_unknown_dimension(in_shape):  # type: ignore[attr-defined]
            return None
        target = tuple(int(v) for v in target)
        if -1 not in target:
            return {names_out[0]: target}
        if target.count(-1) > 1:
            return None
        total = 1
        for d in in_shape:
            total *= d
        known = 1
        for d in target:
            if d != -1:
                known *= d
        if known == 0 or total % known != 0:
            return None
        return {names_out[0]: tuple(total // known if d == -1 else d for d in target)}


class TransposeShape:
    """torch.transpose — swaps dim0 and dim1 (both from attrs)."""

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        shape = tensor_shapes.get(names_in[0])
        if shape is None or self._has_unknown_dimension(shape):  # type: ignore[attr-defined]
            return None
        attrs = getattr(self, "attrs", {})  # type: ignore[attr-defined]
        rank = len(shape)
        dim0, dim1 = int(attrs["dim0"]), int(attrs["dim1"])
        if dim0 < 0:
            dim0 += rank
        if dim1 < 0:
            dim1 += rank
        if not (0 <= dim0 < rank and 0 <= dim1 < rank):
            return None
        out = list(shape)
        out[dim0], out[dim1] = out[dim1], out[dim0]
        return {names_out[0]: tuple(out)}


class SqueezeShape:
    """
    torch.squeeze.  Removes size-1 dims listed in ``dim`` attr.
    Non-size-1 dims are left intact (squeeze on a non-1 dim is a no-op).
    When dim is absent, all size-1 dims are removed.
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        shape = tensor_shapes.get(names_in[0])
        if shape is None or self._has_unknown_dimension(shape):  # type: ignore[attr-defined]
            return None
        attrs = getattr(self, "attrs", {})  # type: ignore[attr-defined]
        dim = attrs.get("dim")
        if dim is None:
            return {names_out[0]: tuple(d for d in shape if d != 1)}
        dims = list(dim) if isinstance(dim, (list, tuple)) else [int(dim)]
        rank = len(shape)
        norm = sorted({(d + rank if d < 0 else d) for d in dims}, reverse=True)
        out = list(shape)
        for d in norm:
            if d < 0 or d >= len(out):
                return None
            if out[d] == 1:
                out.pop(d)
        return {names_out[0]: tuple(out)}


class UnsqueezeShape:
    """torch.unsqueeze — inserts a size-1 dim at position ``dim`` (from attrs)."""

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        shape = tensor_shapes.get(names_in[0])
        if shape is None or self._has_unknown_dimension(shape):  # type: ignore[attr-defined]
            return None
        dim = getattr(self, "attrs", {}).get("dim")  # type: ignore[attr-defined]
        if dim is None:
            return None
        dim = int(dim)
        out = list(shape)
        rank_out = len(out) + 1
        if dim < 0:
            dim = rank_out + dim
        if dim < 0 or dim > len(out):
            return None
        out.insert(dim, 1)
        return {names_out[0]: tuple(out)}


class BroadcastToShape:
    """torch.broadcast_to — reads target shape from ``output_shape`` attr."""

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_out:
            return None
        shape = getattr(self, "attrs", {}).get("output_shape")  # type: ignore[attr-defined]
        if shape is None:
            return None
        return {names_out[0]: tuple(shape)}


class SplitShape:
    """
    torch.split — returns a per-output shape dict.
    Reads ``dim`` and ``split_sizes`` from attrs; divides evenly when sizes absent.
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        shape = tensor_shapes.get(names_in[0])
        if shape is None or self._has_unknown_dimension(shape):  # type: ignore[attr-defined]
            return None
        attrs = getattr(self, "attrs", {})  # type: ignore[attr-defined]
        dim = int(attrs.get("dim", 0))
        rank = len(shape)
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            return None
        split_sizes = attrs.get("split_sizes")
        n = len(names_out)
        if split_sizes is None:
            sz = shape[dim]
            if sz % n != 0:
                return None
            split_sizes = [sz // n] * n
        if len(split_sizes) != n:
            return None
        result: Dict[str, Tuple] = {}
        for name, size in zip(names_out, split_sizes):
            out = list(shape)
            out[dim] = size
            result[name] = tuple(out)
        return result


class ConcatShape:
    """
    torch.cat — sums sizes along the ``dim`` axis; all other dims must match.
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        shapes = [tensor_shapes.get(n) for n in names_in]
        if any(s is None or self._has_unknown_dimension(s) for s in shapes):  # type: ignore[attr-defined]
            return None
        rank = len(shapes[0])
        if any(len(s) != rank for s in shapes):
            return None
        attrs = getattr(self, "attrs", {})  # type: ignore[attr-defined]
        dim = int(attrs.get("dim", 0))
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            return None
        out = list(shapes[0])
        out[dim] = 0
        for s in shapes:
            for i in range(rank):
                if i != dim and s[i] != out[i]:
                    return None
            out[dim] += s[dim]
        return {names_out[0]: tuple(out)}


class PadShape:
    """
    F.pad — pad tuple format: (last_dim_begin, last_dim_end, …) from last dim inward.
    Each pair grows the corresponding dimension by begin+end.
    """

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_in = getattr(self, "input_names", None)  # type: ignore[attr-defined]
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_in or not names_out:
            return None
        in_shape = tensor_shapes.get(names_in[0])
        if in_shape is None or self._has_unknown_dimension(in_shape):  # type: ignore[attr-defined]
            return None
        pad = getattr(self, "attrs", {}).get("pad")  # type: ignore[attr-defined]
        if not pad:
            return {names_out[0]: tuple(in_shape)}
        ndim = len(in_shape)
        out = list(in_shape)
        for i in range(len(pad) // 2):
            d = ndim - 1 - i
            if d < 0:
                return None
            out[d] = in_shape[d] + pad[2 * i] + pad[2 * i + 1]
        return {names_out[0]: tuple(out)}


class FullShape:
    """torch.full — output shape is the ``shape`` attr set at node creation."""

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        names_out = getattr(self, "output_names", None)  # type: ignore[attr-defined]
        if not names_out:
            return None
        shape = getattr(self, "attrs", {}).get("shape")  # type: ignore[attr-defined]
        return {names_out[0]: tuple(shape)} if shape is not None else None
