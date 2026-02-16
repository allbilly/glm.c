# Full GPU Forward Implementation - Status Report

## Completed: Phase 1 - Foundation ✅

### Infrastructure Implemented
1. **GPU Weight Upload System** (`glm_metal_upload_weights()`)
   - Uploads all 47 layers of model weights to GPU buffers
   - Handles attention weights (Q_a, Q_b, KV_a, K_b, V_b, output)
   - Handles FFN/MoE weights (shared experts, gate, up, down)
   - Handles base model weights (token embed, output norm, LM head)

2. **Extended Buffer Management**
   - `MetalLayerBuffers` struct for per-layer GPU buffers
   - `MetalRunStateBuffers` extended with all activation buffers
   - Persistent GPU-resident buffers for KV cache

3. **Skeleton Native Forward Function**
   - `glm_metal_forward_token_native()` entry point created
   - Falls back to hybrid mode until fully implemented
   - Ready for Phase 2/3 integration

4. **New Fused Kernel Architecture**
   - `transformer_layer_fused` kernel added to Metal library
   - Designed to perform full layer in single dispatch
   - Works within Metal's 30-buffer limit

### Current Performance
```
CPU (OMP=4):     5.6 tok/s  (baseline)
Metal (hybrid):  4.3 tok/s  (-24% vs CPU)
llama.cpp Metal: 26.0 tok/s (target)
```

## Next Steps: Phase 2-4 Implementation

### Phase 2: Layer-Wise Kernels (Week 1)

**File: `infer.metal`**

Implement `transformer_layer_fused` kernel operations:

```metal
// Step 1: RMSNorm for attention
float ss = 0;
for (uint k = 0; k < args.dim; k++) ss += x[k] * x[k];
float inv = rsqrt(ss / args.dim + 1e-5f);
float x_norm = x[idx] * inv * attn_norm[idx];

// Step 2: Q_a projection (matvec_q80)
// Step 3: Q_b projection (matvec_q80) 
// Step 4: KV_a projection (matvec_q80)
// Step 5: Apply RoPE to q_pe and k_pe
// Step 6: Compute attention scores
// Step 7: Softmax (numerically stable)
// Step 8: Attention context (matvec)
// Step 9: Output projection (matvec_q80)
// Step 10: Residual add
// Step 11-15: FFN operations
```

**Key Optimizations:**
- Use threadgroup memory for caching hot vectors
- SIMD-group reduction for matvec partial sums
- Vectorized loads (char4/float4)

### Phase 3: End-to-End Forward (Week 1-2)

**File: `infer.m`**

Implement `glm_metal_forward_token_native()`:

```objc
int glm_metal_forward_token_native(const Runtime *rt, RunState *st, 
                                    int token, int pos, int debug_mode) {
    // 1. Token embedding lookup on GPU
    encode_embed_lookup(token, g_run.x);
    
    // 2. One command buffer for all layers
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    
    // 3. Encode all 47 layers
    for (int l = 0; l < 47; l++) {
        encode_layer_dispatch(encoder, l, pos);
    }
    
    // 4. Final RMSNorm + LM head
    encode_final_norm_and_lm_head(encoder);
    
    // 5. Single synchronization point!
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    
    // 6. Read logits back to CPU
    memcpy(st->logits, g_run.logits.contents, vocab_size * sizeof(float));
    
    return 0;
}
```

**Impact:** Reduces from ~1000 dispatches to ~50 dispatches per token

### Phase 4: Optimization (Week 2)

1. **SIMD-Group Optimizations**
   - Use `matvec_q80_rows_simdgroup` variant (2.9x speedup demonstrated)
   - Apply to all matvec operations in layer kernel

2. **Memory Layout Tuning**
   - Transpose weights for coalesced access
   - Tile weight matrices for better cache utilization

3. **Occupancy Tuning**
   - Sweep threadgroup sizes (64/128/256)
   - Benchmark each configuration

4. **FlashAttention-Style Fusion**
   - Further fuse attention operations
   - Reduce memory traffic for KV cache

## Expected Timeline & Performance

| Phase | Work | Expected TPS | Dispatch Count |
|-------|------|--------------|----------------|
| Current | Hybrid | 4.3 | 1000 |
| Phase 2 | Layer kernels | ~10 | 1000 |
| Phase 3 | End-to-end | ~15 | 50 |
| Phase 4 | Optimized | ~20-25 | 50 |

**Target: 20-25 tok/s (match llama.cpp CPU, approach llama.cpp Metal)**

## Files Modified

- `infer.m`: GPU infrastructure, weight upload, native forward skeleton
- `infer.metal`: Fused layer kernel architecture
- `todo.md`: Updated progress tracking

## Validation Strategy

1. **Parity Testing**
   - Run with `temp=0 --seed 0`
   - Compare output token-by-token with CPU backend
   - Must be bit-exact

2. **Numerical Stability**
   - Generate 128+ tokens
   - Check for NaN/Inf in all intermediate buffers
   - Debug tensor sums at key checkpoints

3. **Performance Benchmarking**
   - `make bench-tps` after each phase
   - Track dispatches/token using Metal GPU capture
   - Profile memory bandwidth utilization

## Risk Mitigation

- **CPU fallback preserved**: `--backend cpu` still works
- **Incremental validation**: Test each layer independently first
- **GPU capture ready**: `MTL_CAPTURE_ENABLED=1` workflow documented
- **Memory monitoring**: Buffer sizes validated against M4 limits

## Command Reference

```bash
# Build and test
make build-metal
./glm4.7-flash ./model/GLM-4.7-Flash.bin --backend metal -n 16 -i "1+1="

# Benchmark
make bench-tps
make bench-tps-omp

# Profile
MTL_CAPTURE_ENABLED=1 ./glm4.7-flash ...

# Compare with CPU
./glm4.7-flash ... --backend cpu
```

## Next Immediate Actions

1. **Implement attention operations** in `transformer_layer_fused` kernel
   - Start with RMSNorm + QKV projections
   - Add RoPE application
   - Implement attention scores + softmax

2. **Add FFN operations** to same kernel
   - Dense FFN for layer 0
   - Shared + routed experts for layers 1-46

3. **Wire up native forward** in `glm_metal_forward_token_native()`
   - Single command buffer for all layers
   - One synchronization point
   - Return logits to CPU

4. **Optimize and tune**
   - SIMD-group variants
   - Threadgroup size sweeps
   - Memory layout improvements

---

**Status: Phase 1 complete, ready for Phase 2 implementation**
**Estimated completion: 1-2 weeks**
**Target performance: 20-25 tok/s**
