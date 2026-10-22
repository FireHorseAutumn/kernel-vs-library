# kernel-vs-library (portfolio benchmarks — custom CUDA vs cuBLAS/cuSOLVER/CUTLASS)

Standalone collection of small, self-contained, honestly-measurable artifacts:
each is "custom CUDA beats the library call at X", with the published
literature anchor and the kill criterion stated up front.

## Ground rule
There are no large square GEMMs in this problem class — the matrices are tiny
(3x3, 6x6, 18x18) or the solves are fused accumulate-and-solve patterns. So
"beat cuBLAS at dgemm" has no target. The winnable territory is what libraries
are structurally bad at: batched tiny matrices (launch-overhead-bound) and
fused kernels where the library path would materialize a temporary.

## Hard requirement: hand-crafted wins only (added 2026-09-27)
Every claimed win must come from a **hand-written kernel** — the point of
this project is demonstrating hand-craftsmanship against vendor libraries,
so no composing library calls into a "custom" path, no CUTLASS collectives
with a custom epilogue counting as a win, no thin wrappers. The baseline is
the library's best honest configuration; the challenger is hand-rolled CUDA
down to the epilogue. This makes the wins slower to earn and worth more:
a hand-crafted kernel that beats cuSOLVER/CUTLASS under a stated condition
is a credibility artifact; a cleverly-composed library call is marketing.
Consequence: this project is long-horizon by design — it will take the time
it takes.

## Benchmarks (each ships alone, each has a kill criterion)
1. **Batched tiny Cholesky** (n<=32) — vs cuSOLVER batched. Anchor:
   Haidar/Abdelfattah/Dongarra 2-4x. **Landed in ../batched-tiny-linalg:**
   last recorded table has 58/60 btl wins and both cuSOLVER win conditions.
2. **WMMA/tensor-core Joseph form** — 18x18 (I-KH)P(I-KH)^T sandwich, symmetric
   upper-triangle only, vs CUTLASS grouped GEMM. Planned; no result.
3. **Hamming kNN as 1-bit GEMM** — XOR-popcount (DP4A/IMMA or b1 tensor-core)
   with fused top-2 + Lowe-ratio epilogue, vs CUTLASS 1-bit GEMM + separate
   matcher. Planned for ../drift-estimator; no result.
4. **Batched DLT triangulation** — per-warp closed-form 4x4 symmetric eigen,
   vs cuSOLVER batched gesvdj (notoriously slow at this size). Planned.
5. **Fused reprojection GEMV + robust loss** — Huber in epilogue, vs
   cuBLAS gemvStridedBatched + separate loss pass. Planned.
6. **Batched tiny FP8 GEMM** — FP8 storage, FP32 accumulation, hand-written
   warp-per-matrix CUDA vs cuBLASLt. **Spec-only pause in ../fp8-tiny-gemm;**
   no implementation or benchmark result yet.

## Kill criterion (shared)
If the library path wins under any tested condition, publish that condition.
Library humility is the credibility — e.g. "GPU preprocessing only wins at
batch >= 4" is the kind of number that makes the wins believable.

## Layout
- `include/kvl/` — kernel headers, one per benchmark
- `src/`         — benchmark drivers (.cu), library-path baselines
- `tests/`       — correctness vs library output (tolerance-checked)
- `benchmarks/`  — timing harness, results persisted to JSON

## TODO (in order)
1. [ ] `include/kvl/joseph_wmma.cuh` — WMMA Joseph sandwich (18x18)
2. [ ] `src/bench_joseph.cu` — vs CUTLASS grouped GEMM; JSON out
3. [ ] `include/kvl/hamming_1bit.cuh` — bit-pack + XOR-popcount + epilogue
4. [ ] `src/bench_hamming.cu` — vs CUTLASS b1 GEMM + CPU matcher
5. [ ] `include/kvl/dlt.cuh` — closed-form batched triangulation
6. [ ] `src/bench_dlt.cu` — vs cuSOLVER batched gesvdj
7. [ ] `include/kvl/reproj.cuh` — fused GEMV + Huber epilogue
8. [ ] `src/bench_reproj.cu` — vs cuBLAS gemvStridedBatched
9. [ ] README results table, win/loss per benchmark, conditions stated
