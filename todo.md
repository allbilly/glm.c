# TODO: Metal TPS Optimization Plan (Malfet + yalm)

## Objective
- Move from correctness-first Metal backend to throughput-first backend for `infer.m` + `infer.metal`.
- Replace hybrid CPU-forward behavior with real Metal-forward orchestration.
- Improve decode TPS using a profiling-first, kernel-iteration workflow.

## Context
- Current backend is still hybrid: `infer.m` calls `glm_cpu_forward_token`.
- Naive kernels exist in `infer.metal` and are functionally correct, but not throughput-optimized.
- CPU and Metal benchmark targets already exist (`bench-tps-cpu`, `bench-tps-metal`, `bench-tps-omp`).

## Targets
- [ ] Metal decode TPS >= CPU decode TPS at 16 and 128 tokens.
- [ ] 2x current Metal TPS at 128 tokens.
- [ ] Reach >=50% of `llama.cpp` Metal TPS on matched prompts/tokens.
- [ ] No parity regressions (`temp=0 --seed 0`) and no debug-checkpoint regressions.

## Phase 1: Measurement Foundation (yalm-style)
- [ ] Add explicit benchmark modes: prefill-only, decode-only, full run.
- [ ] Add warmup run before timed run for all benchmarks.
- [ ] Add per-token latency stats (p50/p95) in benchmark output.
- [ ] Add active-bytes and estimated bandwidth reporting.
- [ ] Add microbench targets per kernel family:
  - [ ] `matvec_q80_rows`
  - [ ] attention scores/context
  - [ ] `rmsnorm_f32`
  - [ ] `rope_inplace`
- [ ] Store benchmark artifacts under `/tmp` with stable naming.

## Phase 2: Metal Runtime Cleanup
- [ ] One command buffer per token, one synchronization point per token.
- [ ] Reuse pipeline states and avoid repeated lookup in hot path.
- [ ] Remove hot-path per-call `newBufferWithBytesNoCopy` churn.
- [ ] Pre-allocate persistent `MTLBuffer` objects for activations and argument blocks.
- [ ] Replace repeated `setBytes` usage with reusable constant/argument buffers where possible.

## Phase 3: GEMV Optimization Path (Malfet-inspired)

### Step 3.1: SIMD baseline
- [ ] Convert scalar matvec loop to vectorized loads (`char4`/`float4`) and vector dot accumulation.
- [ ] Unroll inner loop by fixed factor and verify numerical parity.

### Step 3.2: SIMD-group reduction
- [ ] Move from naive one-thread-per-row to simdgroup-cooperative accumulation.
- [ ] Use simdgroup reduction primitives for partial sums.
- [ ] Sweep threadgroup sizes (64/128/256) and store best config per shape.

### Step 3.3: Memory pattern optimization
- [ ] Tile and cache hot `x` slices in threadgroup memory.
- [ ] Improve coalesced access for quantized weights and scales.
- [ ] Introduce row tiling (multi-row per threadgroup) where beneficial.

## Phase 4: Attention Path Optimization
- [ ] Fuse attention score + softmax + context kernels where profitable.
- [ ] Keep numerically stable softmax (max-subtract + finite checks).
- [ ] Consider fusing RoPE + KV write operations when safe.
- [ ] Evaluate cache layout alternatives for head-major coalescing.
- [ ] Benchmark short vs long context separately (256 / 2k / 8k).

## Phase 5: FFN and MoE Path Optimization
- [ ] Keep router/top-k on CPU for first perf pass to preserve behavior.
- [ ] Batch selected expert matvecs to reduce dispatch count.
- [ ] Fuse `silu_mul_inplace` with adjacent ops where safe.
- [ ] Add timing counters for shared expert vs routed expert paths.

## Phase 6: BLAS/MPS Strategy
- [ ] Keep Accelerate/OpenBLAS in `infer.c` for dense float CPU ops.
- [ ] Evaluate MPS for selected float ops only when overhead is amortized.
- [ ] Keep quantized Q8 core operations in custom Metal kernels.
- [ ] Document per-op rule: custom Metal kernel vs MPS op.

## Phase 7: Validation Gates (per optimization wave)
- [ ] Gate P1: deterministic CPU vs Metal parity (`temp=0`).
- [ ] Gate P2: debug tensor sums within tolerance at key checkpoints.
- [ ] Gate P3: no NaN/Inf for >=128 generated tokens.
- [ ] Gate P4: no CLI regression (`-i`, `-f`, chat/completion, context limits, seed).
- [ ] Gate P5: TPS gain confirmed at 16, 128, and 512 tokens.

## Tooling and Profiling Workflow
- [ ] Add `MTL_CAPTURE_ENABLED=1` capture recipe to README.
- [ ] Add Xcode `.gputrace` workflow section and artifact naming.
- [ ] Add Metal API validation instructions for debug builds.

## Deliverables
- [ ] `infer.m`: full Metal-forward orchestration with no CPU-forward delegation.
- [ ] `infer.metal`: optimized kernel variants + reference kernels.
- [ ] `Makefile`: reproducible microbench + throughput + capture targets.
- [ ] `README`: benchmark methodology and profiling/debug guide.
- [ ] `/tmp` benchmark snapshots + summary TPS table for each phase.
