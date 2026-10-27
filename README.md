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
   Haidar/Abdelfattah/Dongarra 2-4x. **Published in ../batched-tiny-linalg
   (table below):** btl wins 58/60 configs, 1.04x–160x vs cuSOLVER; cuSOLVER
   wins n=32 f64 at batch 64 and 1024 — both losses published per the kill
   criterion.
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
   warp-per-matrix CUDA vs cuBLASLt. **Published in ../fp8-tiny-gemm (table
   below):** win zone is n=16 only (0.75–0.85x cuBLASLt at every batch);
   cuBLASLt wins n=32/64 everywhere, up to 10.9x.

## Kill criterion (shared)
If the library path wins under any tested condition, publish that condition.
Library humility is the credibility — e.g. "GPU preprocessing only wins at
batch >= 4" is the kind of number that makes the wins believable.

## Published results

### 1. Batched tiny Cholesky (from ../batched-tiny-linalg, 2026-10-10, RTX 5060 Laptop, sm_120, CUDA 13.2, Eigen 5.0.1)

Fused Cholesky factor + solve (one n-vector rhs per matrix), full row-major
storage, SPD batches A = B Bᵀ + nI. GPU times are CUDA-event medians (20
iters, 5 warmup); Eigen CPU is fixed-size `Eigen::LLT` single-thread.
cuSOLVER = `potrfBatched` + `potrsBatched`; its in-place A/rhs restores and
pointer-array setup are excluded from its timing (generous to cuSOLVER).

float32 (ms, median of 20 GPU / 5-11 CPU iters, warmup first):

| n | batch | btl thread/matrix | btl warp/matrix | cuSOLVER batched | Eigen CPU | best btl vs cuSOLVER | winner |
|---|-------|------------------:|----------------:|-----------------:|----------:|---------------------:|--------|
| 2 | 64 | 0.014 | 0.015 | 0.070 | 0.003 | 4.84x | eigen_cpu |
| 2 | 1024 | 0.014 | 0.014 | 0.071 | 0.050 | 4.97x | btl |
| 2 | 65536 | 0.014 | 0.029 | 2.32 | 3.34 | 160.10x | btl |
| 3 | 64 | 0.015 | 0.014 | 0.069 | 0.008 | 4.95x | eigen_cpu |
| 3 | 1024 | 0.014 | 0.014 | 0.076 | 0.128 | 5.34x | btl |
| 3 | 65536 | 0.030 | 0.043 | 2.69 | 8.76 | 90.27x | btl |
| 4 | 64 | 0.014 | 0.014 | 0.069 | 0.026 | 4.80x | btl |
| 4 | 1024 | 0.018 | 0.014 | 0.089 | 0.407 | 6.22x | btl |
| 4 | 65536 | 0.064 | 0.062 | 4.27 | 27.9 | 69.19x | btl |
| 6 | 64 | 0.015 | 0.014 | 0.069 | 0.054 | 4.93x | btl |
| 6 | 1024 | 0.021 | 0.014 | 0.087 | 0.907 | 6.19x | btl |
| 6 | 65536 | 0.113 | 0.093 | 3.81 | 58.3 | 40.85x | btl |
| 8 | 64 | 0.016 | 0.014 | 0.071 | 0.091 | 4.91x | btl |
| 8 | 1024 | 0.031 | 0.015 | 0.082 | 1.54 | 5.49x | btl |
| 8 | 65536 | 0.192 | 0.128 | 4.46 | 106 | 34.77x | btl |
| 9 | 64 | 0.021 | 0.021 | 0.069 | 0.164 | 3.29x | btl |
| 9 | 1024 | 0.033 | 0.017 | 0.084 | 3.12 | 4.79x | btl |
| 9 | 65536 | 0.254 | 0.235 | 5.03 | 190 | 21.35x | btl |
| 12 | 64 | 0.021 | 0.017 | 0.070 | 0.226 | 4.13x | btl |
| 12 | 1024 | 0.050 | 0.021 | 0.087 | 3.85 | 4.08x | btl |
| 12 | 65536 | 0.393 | 0.447 | 7.63 | 264 | 19.43x | btl |
| 16 | 64 | 0.027 | 0.020 | 0.073 | 0.341 | 3.58x | btl |
| 16 | 1024 | 0.084 | 0.027 | 0.098 | 5.74 | 3.56x | btl |
| 16 | 65536 | 0.769 | 1.19 | 9.20 | 393 | 11.97x | btl |
| 24 | 64 | 0.060 | 0.022 | 0.103 | 0.690 | 4.61x | btl |
| 24 | 1024 | 0.198 | 0.051 | 0.154 | 11.1 | 3.00x | btl |
| 24 | 65536 | 3.92 | 5.41 | 24.9 | 742 | 6.35x | btl |
| 32 | 64 | 0.147 | 0.031 | 0.113 | 1.96 | 3.63x | btl |
| 32 | 1024 | 0.506 | 0.098 | 0.179 | 30.1 | 1.82x | btl |
| 32 | 65536 | 14.9 | 24.5 | 27.6 | 2037 | 1.86x | btl |

float64 (ms, median of 20 GPU / 5-11 CPU iters, warmup first):

| n | batch | btl thread/matrix | btl warp/matrix | cuSOLVER batched | Eigen CPU | best btl vs cuSOLVER | winner |
|---|-------|------------------:|----------------:|-----------------:|----------:|---------------------:|--------|
| 2 | 64 | 0.015 | 0.016 | 0.070 | 0.005 | 4.62x | eigen_cpu |
| 2 | 1024 | 0.014 | 0.016 | 0.080 | 0.079 | 5.59x | btl |
| 2 | 65536 | 0.045 | 0.091 | 3.45 | 5.27 | 76.19x | btl |
| 3 | 64 | 0.014 | 0.019 | 0.069 | 0.009 | 4.76x | eigen_cpu |
| 3 | 1024 | 0.019 | 0.018 | 0.090 | 0.142 | 4.96x | btl |
| 3 | 65536 | 0.075 | 0.256 | 5.30 | 9.59 | 70.97x | btl |
| 4 | 64 | 0.018 | 0.024 | 0.069 | 0.038 | 3.79x | btl |
| 4 | 1024 | 0.029 | 0.025 | 0.130 | 0.621 | 5.20x | btl |
| 4 | 65536 | 0.108 | 0.346 | 5.51 | 39.9 | 50.79x | btl |
| 6 | 64 | 0.020 | 0.025 | 0.071 | 0.064 | 3.47x | btl |
| 6 | 1024 | 0.038 | 0.039 | 0.124 | 1.09 | 3.26x | btl |
| 6 | 65536 | 0.212 | 1.30 | 8.29 | 71.2 | 39.07x | btl |
| 8 | 64 | 0.033 | 0.037 | 0.072 | 0.104 | 2.20x | btl |
| 8 | 1024 | 0.058 | 0.058 | 0.154 | 1.85 | 2.67x | btl |
| 8 | 65536 | 0.380 | 2.15 | 11.5 | 126 | 30.25x | btl |
| 9 | 64 | 0.029 | 0.035 | 0.072 | 0.174 | 2.51x | btl |
| 9 | 1024 | 0.068 | 0.102 | 0.175 | 3.19 | 2.58x | btl |
| 9 | 65536 | 0.556 | 4.83 | 15.6 | 200 | 28.13x | btl |
| 12 | 64 | 0.043 | 0.045 | 0.074 | 0.286 | 1.72x | btl |
| 12 | 1024 | 0.113 | 0.111 | 0.200 | 4.40 | 1.80x | btl |
| 12 | 65536 | 1.07 | 7.22 | 18.7 | 278 | 17.38x | btl |
| 16 | 64 | 0.072 | 0.063 | 0.078 | 0.377 | 1.25x | btl |
| 16 | 1024 | 0.188 | 0.160 | 0.241 | 6.28 | 1.51x | btl |
| 16 | 65536 | 2.68 | 11.5 | 24.8 | 418 | 9.28x | btl |
| 24 | 64 | 0.186 | 0.094 | 0.114 | 0.806 | 1.21x | btl |
| 24 | 1024 | 0.456 | 0.457 | 0.472 | 13.6 | 1.04x | btl |
| 24 | 65536 | 15.1 | 42.6 | 50.0 | 850 | 3.31x | btl |
| 32 | 64 | 0.387 | 0.145 | 0.126 | 2.14 | 0.87x | cusolver |
| 32 | 1024 | 1.05 | 0.745 | 0.660 | 34.1 | 0.89x | cusolver |
| 32 | 65536 | 60.9 | 71.0 | 90.2 | 2201 | 1.48x | btl |

Kill criterion verdict: **cuSOLVER batched wins at n=32 float64, batch ∈
{64, 1024}** (0.126 ms vs btl-warp 0.145 ms; 0.660 ms vs 0.745 ms).
Published per the kill criterion. Condition: fp64 + largest n in range +
small/medium batch, where btl's launch-latency advantage is amortized away
and cuSOLVER's tuned double pipelines dominate. btl wins the other 58/60
configs, by 1.04x–160x vs cuSOLVER and up to ~826x vs single-thread Eigen
CPU.

Known anomaly: the btl warp-per-matrix variant is slower than
thread-per-matrix for float64 at batch = 65536 (e.g. n=12: 7.22 ms vs
1.07 ms) — shuffle-heavy factor path plus register pressure on 64-bit
lanes. Thread variant wins those cells; warp variant wins at small batch ×
large n. Both kept; optimization deferred.

### 6. Batched tiny FP8 GEMM (from ../fp8-tiny-gemm, 2026-10-25, RTX 5060 Laptop, sm_120, CUDA 13.2)

µs per batched call, median of 21 with [min, max]. e4m3 storage, FP32
accumulation, warp-per-matrix hand-written kernel vs cuBLASLt batched FP8
(the library's best honest configuration).

| n | batch | kernel | cuBLASLt | kernel/lt | winner |
|--:|--:|--|--|--|--|
| 16 | 1 | 7.4 [7.0, 9.4] | 9.9 [9.5, 12.1] | 0.75 | kernel |
| 16 | 64 | 8.0 [7.4, 8.7] | 10.0 [9.8, 12.0] | 0.80 | kernel |
| 16 | 1024 | 16.0 [14.0, 18.1] | 20.3 [19.0, 21.6] | 0.79 | kernel |
| 16 | 65536 | 833.3 [828.8, 836.0] | 981.2 [916.7, 989.0] | 0.85 | kernel |
| 32 | 1 | 28.0 [26.7, 31.0] | 10.8 [10.2, 12.3] | 2.59 | cublaslt |
| 32 | 64 | 24.1 [22.8, 27.5] | 15.7 [13.5, 16.1] | 1.53 | cublaslt |
| 32 | 1024 | 81.7 [79.4, 84.0] | 71.5 [70.8, 71.9] | 1.14 | cublaslt |
| 32 | 65536 | 5931.9 [5306.2, 6067.9] | 4614.2 [4418.5, 5964.7] | 1.29 | cublaslt |
| 64 | 1 | 108.3 [107.9, 111.6] | 10.0 [9.4, 12.2] | 10.85 | cublaslt |
| 64 | 64 | 112.5 [111.3, 115.5] | 14.0 [12.6, 15.6] | 8.06 | cublaslt |
| 64 | 1024 | 672.8 [656.1, 695.2] | 104.1 [103.7, 106.1] | 6.46 | cublaslt |
| 64 | 65536 | 48336.0 [47824.1, 75029.8] | 6542.5 [6455.1, 6895.5] | 7.39 | cublaslt |

The launch-tax hypothesis survives only at the smallest size tested: at
n=16 the kernel wins every batch point (0.75–0.85x vendor time); at n=32
and n=64 cuBLASLt's tensor-core path wins everywhere, up to 10.9x. Launch-
overhead exhibit (batch=1 row): the kernel's fixed cost is ~7.4 µs vs ~10
µs for cuBLASLt — the ~2.5 µs setup tax the tiny regime amortizes.

**e5m2:** cuBLASLt's heuristic offers no algo for the row-major strided
batched e5m2 configuration on this stack (recorded as `null` cells in the
JSON); kernel-side e5m2 times are in the sweep and track e4m3 to ~2%.
FP4 e2m1 exists as an accuracy appendix only (measured 11.1–11.3x the e4m3
error — above the 4x mantissa-count prediction; absolute-grid dominated on
the U(-1,1) inputs), never as the kernel bet.

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
