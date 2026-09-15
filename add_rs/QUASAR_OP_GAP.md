# Quasar op gap, measured (2026-09-09)

Method: one single-op ONNX model per op, one process each (the simulator-vs-hardware
choice is process-wide, and a hang must not take the rest of the probe down), bf16,
against the installed build carrying only the Add + to_layout dispatch.
Harness: `add_rs/probe_exec_one_op.py`. Log: `add_rs/logs/probe_exec_ops.log`.

## Result

| ONNX node | ttnn op emitted | Outcome on Quasar |
|---|---|---|
| `Reshape`   | `ttnn.reshape`   | refused — `kernel.hpp:418`; **now PASS** `pcc=0.999996` via PR #9292 |
| `Transpose` | `ttnn.permute`   | refused — `kernel.hpp:418` |
| `MatMul`    | `ttnn.matmul`    | refused — `kernel.hpp:418` |
| `Relu`      | `ttnn.relu`      | refused — `kernel.hpp:418` |
| `Mul`       | `ttnn.multiply`  | refused — `kernel.hpp:418` |

Every one is the same assertion:

```
TT_FATAL: DataMovementKernel is not supported on Quasar.
          Use QuasarDataMovementKernel instead.
```

So the gap is uniform. It is not that these ops are individually broken on Quasar —
it is that every mainline program factory constructs a `DataMovementKernel`, and that
constructor refuses on a Quasar cluster. Enabling an op means routing it to Quasar's
own op library, which builds `QuasarDataMovementKernel`s instead.

## Two frontend surprises worth recording

Compile-only probe (`add_rs/probe_next_ops.py`, log `logs/probe_next_ops.log`):

- ONNX `Transpose` lowers to **`ttnn.permute`**, not `ttnn.transpose`. The
  `transpose.cpp` dispatch site is never reached from an ONNX frontend, so routing
  it would be dead code.
- ONNX `Cast` to bf16 is **folded away** — the stream is just `to_layout` +
  `deallocate`, no `ttnn.typecast`. Same conclusion.

That is why `reshape` was picked as the next op rather than transpose or typecast:
it is the only one of the three that forge actually emits.

## Substitutability, from the real call sites

| Op | Runtime call | Quasar entry point | Same arguments? |
|---|---|---|---|
| `reshape` | `reshape.cpp:25` `(in, shape, memcfg)` | `quasar::reshape`, `ttsl::Span<const int32_t>` overload | yes |
| `permute` | `permute.cpp:26` `(in, perm, memcfg, padValue)` | none — needs decomposition into transposes | no |
| `matmul` | `matmul.cpp:75` | `quasar::matmul::matmul` | no — takes a Quasar-specific `MatmulProgramConfig` |
| `relu` | unary dispatch | none — Quasar binds no unary ops | no |

`reshape` is the only one that is a like-for-like substitution today. It is now
routed and passing on both Quasar and Wormhole — tt-mlir PR #9292 (draft, stacked
on #9287). Logs: `logs/reshape_final_quasar.log`, `logs/reshape_final_wormhole.log`.
