# BEV Config Sweep Report
Generated : 2026-05-04 15:38:11
Baseline  : opt_level=1, consteval=True

---
## Results Table
| ID | Config | Status | compile_s | mean_infer_ms | std_ms | samples_sec | Δ fps | pcc_worst | PCC | Decision | Log |
|--|--|--|--|--|--|--|--|--|--|--|--|
| baseline | opt_level=1, consteval=True  [BASELINE] | **FAIL** | — | — | — | — | — | — | — | — | [log](baseline.log) |
| A1 | A1 — enable_trace=True | **FAIL** | — | — | — | — | — | — | — | — | [log](A1.log) |
| B1 | B1 — l1_interleaved_fallback=True | **FAIL** | — | — | — | — | — | — | — | — | [log](B1.log) |
| B2 | B2 — tensor-l1-usage-cap=1.0 | **FAIL** | — | — | — | — | — | — | — | — | [log](B2.log) |
| B3 | B3 — memory_layout_analysis=True (DFSharding) | **FAIL** | — | — | — | — | — | — | — | — | [log](B3.log) |
| B4 | B4 — MLA=True, policy=GreedyL1Interleaved | **FAIL** | — | — | — | — | — | — | — | — | [log](B4.log) |
| B5 | B5 — MLA=True, policy=BFInterleaved | **FAIL** | — | — | — | — | — | — | — | — | [log](B5.log) |
| C1 | C1 — math_fidelity=HiFi2 | **FAIL** | — | — | — | — | — | — | — | — | [log](C1.log) |
| C2 | C2 — math_fidelity=HiFi3 | **FAIL** | — | — | — | — | — | — | — | — | [log](C2.log) |
| C3 | C3 — math_fidelity=LoFi | **FAIL** | — | — | — | — | — | — | — | — | [log](C3.log) |
| C4 | C4 — fp32_dest_acc=True | **FAIL** | — | — | — | — | — | — | — | — | [log](C4.log) |
| D1 | D1 — weight_dtype=Bfp8_b | **FAIL** | — | — | — | — | — | — | — | — | [log](D1.log) |
| D2 | D2 — weight_dtype=Bfp4_b | **FAIL** | — | — | — | — | — | — | — | — | [log](D2.log) |
| E1 | E1 — d2m_fusing=True | **FAIL** | — | — | — | — | — | — | — | — | [log](E1.log) |
| E2 | E2 — permute_matmul_fusion=True | **FAIL** | — | — | — | — | — | — | — | — | [log](E2.log) |
| E3 | E3 — fusing=False [ref] | **FAIL** | — | — | — | — | — | — | — | — | [log](E3.log) |
| E4 | E4 — conv2d_multiply_fusing=False [ref] | **FAIL** | — | — | — | — | — | — | — | — | [log](E4.log) |
| F1 | F1 — dram_space_saving=True | **FAIL** | — | — | — | — | — | — | — | — | [log](F1.log) |
| F2 | F2 — remove_dead_values=True | **FAIL** | — | — | — | — | — | — | — | — | [log](F2.log) |
| F3 | F3 — erase_inverse_ops=False [ref] | **FAIL** | — | — | — | — | — | — | — | — | [log](F3.log) |
| F4 | F4 — implicit_broadcast_folding=False [ref] | **FAIL** | — | — | — | — | — | — | — | — | [log](F4.log) |
| G1 | G1 — row_major=True | **FAIL** | — | — | — | — | — | — | — | — | [log](G1.log) |
| G2 | G2 — max_legal_layouts=16 | **FAIL** | — | — | — | — | — | — | — | — | [log](G2.log) |
| G3 | G3 — max_legal_layouts=4 | **FAIL** | — | — | — | — | — | — | — | — | [log](G3.log) |
| H1 | H1 — consteval_inputs_to_system_memory=False | **FAIL** | — | — | — | — | — | — | — | — | [log](H1.log) |
| H2 | H2 — cpu_hoisted_consteval=False | **FAIL** | — | — | — | — | — | — | — | — | [log](H2.log) |
| I1 | I1 — enable-greedy-optimizer=false | **FAIL** | — | — | — | — | — | — | — | — | [log](I1.log) |
| I2 | I2 — tensor-l1-usage-cap=1.0 (custom_config) | **FAIL** | — | — | — | — | — | — | — | — | [log](I2.log) |
| I3 | I3 — tensor-l1-usage-cap=0.8 | **FAIL** | — | — | — | — | — | — | — | — | [log](I3.log) |
| I4 | I4 — max-fallback-attempts=100 | **FAIL** | — | — | — | — | — | — | — | — | [log](I4.log) |
| I5 | I5 — enable-quant-dequant-conversion-pass=false | **FAIL** | — | — | — | — | — | — | — | — | [log](I5.log) |
| J1 | J1 — decomposition_workaround=False [risky] | **FAIL** | — | — | — | — | — | — | — | — | [log](J1.log) |

---
## Retained Configs (PCC OK + FPS improved)

_None — no config improved FPS while keeping PCC ≥ 0.99_

---
## Failed / Timed-out Tests

- **baseline** — FAIL (wall=8s)  opt_level=1, consteval=True  [BASELINE]
- **A1** — FAIL (wall=7s)  A1 — enable_trace=True
- **B1** — FAIL (wall=7s)  B1 — l1_interleaved_fallback=True
- **B2** — FAIL (wall=7s)  B2 — tensor-l1-usage-cap=1.0
- **B3** — FAIL (wall=9s)  B3 — memory_layout_analysis=True (DFSharding)
- **B4** — FAIL (wall=9s)  B4 — MLA=True, policy=GreedyL1Interleaved
- **B5** — FAIL (wall=8s)  B5 — MLA=True, policy=BFInterleaved
- **C1** — FAIL (wall=7s)  C1 — math_fidelity=HiFi2
- **C2** — FAIL (wall=9s)  C2 — math_fidelity=HiFi3
- **C3** — FAIL (wall=9s)  C3 — math_fidelity=LoFi
- **C4** — FAIL (wall=8s)  C4 — fp32_dest_acc=True
- **D1** — FAIL (wall=10s)  D1 — weight_dtype=Bfp8_b
- **D2** — FAIL (wall=9s)  D2 — weight_dtype=Bfp4_b
- **E1** — FAIL (wall=8s)  E1 — d2m_fusing=True
- **E2** — FAIL (wall=16s)  E2 — permute_matmul_fusion=True
- **E3** — FAIL (wall=10s)  E3 — fusing=False [ref]
- **E4** — FAIL (wall=17s)  E4 — conv2d_multiply_fusing=False [ref]
- **F1** — FAIL (wall=13s)  F1 — dram_space_saving=True
- **F2** — FAIL (wall=9s)  F2 — remove_dead_values=True
- **F3** — FAIL (wall=7s)  F3 — erase_inverse_ops=False [ref]
- **F4** — FAIL (wall=7s)  F4 — implicit_broadcast_folding=False [ref]
- **G1** — FAIL (wall=7s)  G1 — row_major=True
- **G2** — FAIL (wall=7s)  G2 — max_legal_layouts=16
- **G3** — FAIL (wall=8s)  G3 — max_legal_layouts=4
- **H1** — FAIL (wall=7s)  H1 — consteval_inputs_to_system_memory=False
- **H2** — FAIL (wall=7s)  H2 — cpu_hoisted_consteval=False
- **I1** — FAIL (wall=7s)  I1 — enable-greedy-optimizer=false
- **I2** — FAIL (wall=9s)  I2 — tensor-l1-usage-cap=1.0 (custom_config)
- **I3** — FAIL (wall=8s)  I3 — tensor-l1-usage-cap=0.8
- **I4** — FAIL (wall=8s)  I4 — max-fallback-attempts=100
- **I5** — FAIL (wall=7s)  I5 — enable-quant-dequant-conversion-pass=false
- **J1** — FAIL (wall=9s)  J1 — decomposition_workaround=False [risky]

## Individual Logs

- [baseline](baseline.log) — opt_level=1, consteval=True  [BASELINE]
- [A1](A1.log) — A1 — enable_trace=True
- [B1](B1.log) — B1 — l1_interleaved_fallback=True
- [B2](B2.log) — B2 — tensor-l1-usage-cap=1.0
- [B3](B3.log) — B3 — memory_layout_analysis=True (DFSharding)
- [B4](B4.log) — B4 — MLA=True, policy=GreedyL1Interleaved
- [B5](B5.log) — B5 — MLA=True, policy=BFInterleaved
- [C1](C1.log) — C1 — math_fidelity=HiFi2
- [C2](C2.log) — C2 — math_fidelity=HiFi3
- [C3](C3.log) — C3 — math_fidelity=LoFi
- [C4](C4.log) — C4 — fp32_dest_acc=True
- [D1](D1.log) — D1 — weight_dtype=Bfp8_b
- [D2](D2.log) — D2 — weight_dtype=Bfp4_b
- [E1](E1.log) — E1 — d2m_fusing=True
- [E2](E2.log) — E2 — permute_matmul_fusion=True
- [E3](E3.log) — E3 — fusing=False [ref]
- [E4](E4.log) — E4 — conv2d_multiply_fusing=False [ref]
- [F1](F1.log) — F1 — dram_space_saving=True
- [F2](F2.log) — F2 — remove_dead_values=True
- [F3](F3.log) — F3 — erase_inverse_ops=False [ref]
- [F4](F4.log) — F4 — implicit_broadcast_folding=False [ref]
- [G1](G1.log) — G1 — row_major=True
- [G2](G2.log) — G2 — max_legal_layouts=16
- [G3](G3.log) — G3 — max_legal_layouts=4
- [H1](H1.log) — H1 — consteval_inputs_to_system_memory=False
- [H2](H2.log) — H2 — cpu_hoisted_consteval=False
- [I1](I1.log) — I1 — enable-greedy-optimizer=false
- [I2](I2.log) — I2 — tensor-l1-usage-cap=1.0 (custom_config)
- [I3](I3.log) — I3 — tensor-l1-usage-cap=0.8
- [I4](I4.log) — I4 — max-fallback-attempts=100
- [I5](I5.log) — I5 — enable-quant-dequant-conversion-pass=false
- [J1](J1.log) — J1 — decomposition_workaround=False [risky]
