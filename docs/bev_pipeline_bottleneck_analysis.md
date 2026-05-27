# BEV Pipeline Bottleneck Analysis

Config: `opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled`

## Full Pipeline Timing

| Block | Block Name | Inference ms | Total/frame ms | FPS | % of pipeline |
|-------|------------|-------------|----------------|-----|---------------|
| A | Deformed Backbone | 339.9 | 358.7 | 2.79 | **56.4%** |
| C | Cylinder Backbone | 113.0 | 118.5 | 8.44 | **18.6%** |
| B | BEV Transform | 61.9 | 81.7 | 12.24 | 12.9% |
| E | BEV Aggregator | 33.6 | 37.2 | 26.86 | 5.9% |
| D | Cylinder BEV Transform | 19.1 | 22.9 | 43.76 | 3.6% |
| F | Output Heads | 12.2 | 16.7 | 59.90 | 2.6% |
| **Total** | | **580.0** | **635.6** | **1.57** | **100%** |

## Memory Health (TTNNCollectPerfMetrics)

| Block | total_shardable | sharded% | eff_sharded% | spilled% | notes |
|-------|----------------|----------|--------------|----------|-------|
| A | 408 | 52.0% | 24.5% | 24.5% | 72 conv2d spills |
| B | 60 | 40.0% | 26.7% | 33.3% | |
| C | 113 | 54.0% | 25.7% | 24.8% | 16 conv2d spills |
| D | 15 | 33.3% | 26.7% | 40.0% | |
| E | 70 | 42.9% | 28.6% | 31.4% | |
| F | 30 | 66.7% | 30.0% | 6.7% | healthiest |

- **eff_sharded%** = sharded ops whose output stays in L1 (not immediately evicted)
- **spilled%** = sharded ops whose output is immediately `to_memory_config(DRAM)` before the consumer

## FPS Impact of Halving Each Block

| Block | Current ms | Halved ms | New pipeline FPS | Gain |
|-------|-----------|-----------|-----------------|------|
| A | 359 | 179 | 2.19 | **+0.62** |
| C | 118 | 59 | 1.73 | +0.16 |
| B | 82 | 41 | 1.68 | +0.11 |
| E | 37 | 19 | 1.62 | +0.05 |
| D | 23 | 11 | 1.60 | +0.03 |
| F | 17 | 8 | 1.59 | +0.02 |

## Attack Priority

1. **Block A** — 56% of pipeline, 72 conv2d DRAM spills, +0.62 FPS if halved. Highest leverage.
2. **Block C** — 18.6% of pipeline, similar spill pattern to A (16 spills), +0.16 FPS.
3. Blocks B, E, D, F — diminishing returns (<13% each, low priority).

## Target FPS Progression

| Milestone | Assumption | Expected FPS |
|-----------|-----------|-------------|
| Baseline | Current | 1.57 |
| Fix Block A spills (50% reduction) | Remove 72 DRAM bounces | ~2.0 |
| Fix Block C spills | Similar pattern to A | ~2.1 |
| 30 FPS target | All blocks need ~10× improvement | far |
