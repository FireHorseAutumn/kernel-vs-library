# kernel-vs-library

Custom CUDA against the vendor libraries, one small artifact at a time.
Each benchmark is "hand-written CUDA beats cuBLAS/cuSOLVER/CUTLASS at X",
with the literature anchor and the kill criterion stated up front. The
winnable territory is what libraries are structurally bad at: batched tiny
matrices (launch-overhead-bound) and fused kernels where the library path
would materialize a temporary. Hand-written kernels only; no composed
library calls count as wins.

## Status

- Two of six benchmarks published with full win/loss tables: batched tiny
  Cholesky (btl, n<=32, vs cuSOLVER batched) and batched tiny FP8 GEMM
  (vs cuBLASLt). Four planned with anchors stated.
- Recorded on an RTX 5060 Laptop GPU (`sm_120`), CUDA 13.2, Eigen 5.0.1;
  GPU times are CUDA-event medians with warmup.

## Headline: batched tiny Cholesky vs cuSOLVER

Fused Cholesky factor + solve, SPD batches, ms per batched call, best of
the two btl variants against `potrfBatched` + `potrsBatched`. Selected
cells from the full 60-config sweep:

| precision | n | batch | best btl | cuSOLVER | btl vs cuSOLVER | winner |
|---|--:|--:|--:|--:|--:|---|
| f32 | 2 | 65536 | 0.014 | 2.32 | 160.10x | btl |
| f32 | 8 | 1024 | 0.015 | 0.082 | 5.49x | btl |
| f32 | 32 | 1024 | 0.098 | 0.179 | 1.82x | btl |
| f64 | 32 | 64 | 0.145 | 0.126 | 0.87x | cusolver |
| f64 | 32 | 1024 | 0.745 | 0.660 | 0.89x | cusolver |
| f64 | 32 | 65536 | 60.9 | 90.2 | 1.48x | btl |

## Key numbers

- Cholesky: btl wins 58/60 configs, 1.04x to 160x vs cuSOLVER, up to ~826x
  vs single-thread Eigen CPU. The two losses are n=32 float64 at batch 64
  and 1024, published per the kill criterion.
- FP8 GEMM: the hand-written warp-per-matrix kernel wins n=16 at every
  batch (0.65 to 0.87x cuBLASLt time); cuBLASLt wins n=32 and n=64
  everywhere, up to 18.1x.
- Known anomaly: the btl warp-per-matrix variant loses to
  thread-per-matrix for float64 at batch 65536 (n=12: 7.22 ms vs 1.07
  ms): shuffle-heavy factor path plus register pressure on 64-bit lanes.

## Honest limits

- The kill criterion is shared: any library win gets published. cuSOLVER
  takes n=32 float64 at batch 64/1024; cuBLASLt takes every n=32/64 FP8
  cell. Those losses are in the tables, not the fine print.
- The FP8 n=16 batch=65536 cell is bimodal even governor-pinned (kernel
  reps span 707 to 2282 µs; one of four pinned runs lost 1.55x there).
- e5m2: cuBLASLt offers no algo for the row-major strided batched
  configuration on this stack. FP4 e2m1 exists as an accuracy appendix
  only (11.1 to 11.3x the e4m3 error), never as the kernel bet.
- Four of six benchmarks (Joseph WMMA, Hamming 1-bit kNN, DLT
  triangulation, fused reprojection GEMV) are planned; no result yet.

## Build and test

Each benchmark ships alone: a driver in `src/`, a library-path baseline, a
tolerance-checked correctness test against library output in `tests/`, and
a timing harness that persists results to JSON in `benchmarks/`.

## Repository layout

- `include/kvl/`: kernel headers, one per benchmark
- `src/`: benchmark drivers (.cu), library-path baselines
- `tests/`: correctness vs library output (tolerance-checked)
- `benchmarks/`: timing harness, results persisted to JSON

## Deeper reading

- `docs/RUNGS.md`: the full record: ground rule and hand-crafted-wins
  requirement, both published result tables verbatim, the anomalies, the
  bimodal cells, and the planned benchmarks with their anchors.
