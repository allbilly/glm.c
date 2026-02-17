Done — I implemented all checklist items in todo.md and ran the benchmarks to verify.                                                                                                    
                                                                                                                                                                                              
     - Switched Metal compile defaults to optimized mode with debug/fast variants in Makefile:12, Makefile:143, Makefile:150, Makefile:154 (METAL_OPT=release|debug|fast, with build-metal-   
     debug / build-metal-fast targets).                                                                                                                                                       
     - Added staged attention helper kernels in infer.metal:185 and infer.metal:204:                                                                                                          
       - rope_inplace_lut (RoPE via lookup tables, no per-token pow/cos/sin)                                                                                                                  
       - prepare_q_head (head slice prep + RoPE application)                                                                                                                                  
     - Refactored native-fast execution from single fused dispatch to staged multi-dispatch path in infer.m:1110 (no transformer_layer_fused dispatch anymore), with parallelized GPU         
     stages for:                                                                                                                                                                              
       - q_abs projection                                                                                                                                                                     
       - score computation over sequence positions                                                                                                                                            
       - context accumulation                                                                                                                                                                 
       - output projection                                                                                                                                                                    
     - Added reusable staged matvec dispatch helper in infer.m:480 and wired it through the native path (infer.m:1332, infer.m:1449, infer.m:1525, infer.m:1636).                             
     - Added precomputed RoPE tables on prepare (infer.m:1080) and consumed them in native forward (infer.m:1389, infer.m:1437).                                                              
     - Moved static weight uploads to private GPU memory with one-time blit upload in infer.m:339 (MTLResourceStorageModePrivate + blit copy).                                                
                                                                                                                                                                                              
     I also marked all todo checkboxes complete and recorded latest measurements in todo.md:13 and todo.md:36.                                                                                
                                                                                                                                                                                              
     Validation I ran:                                                                                                                                                                        
     - make build-metal                                                                                                                                                                       
     - make bench-native-fast-profile                                                                                                                                                         
     - make bench-native-fast-matrix                                                                                                                                                          
     - Native parity sanity check:                                                                                                                                                            
       - GLM_METAL_NATIVE=1 GLM_METAL_NATIVE_UNSAFE=1 GLM_METAL_NATIVE_PARITY=1 GLM_METAL_NATIVE_PARITY_LAYERS=1 GLM_METAL_NATIVE_PARITY_STAGE=attn ./glm4.7-flash ...                        
                                                                                                                                                                                              
     Latest measured throughput:                                                                                                                                                              
     - Native-fast first decode: ~9.25 tok/s                                                                                                                                                  
     - Native-fast steady decode estimate: ~9.19 tok/s                                                                                                                                        
     - Progress to 20 tok/s: 45.9%             


     =====

# Native-Fast Performance Improvement Plan

Current issue: native-fast throughput (`~0.006 tok/s`) indicates the path is still running like a correctness/reference GPU implementation rather than an optimized one.

## Key bottlenecks identified

- `Makefile`: Metal compilation uses `-O0` (debug optimization level).
- `infer.metal`: fused layer currently forced to a single threadgroup.
- `infer.metal`: heavy MLA attention work is gated by `if (lid == 0u)` (effectively serial).

## Priority actions

- [x] Switch Metal build to optimized mode for benchmark/release (`-O3` or `-Ofast`), while keeping a debug option.
- [x] Parallelize the current `lid == 0u` attention path across threads/simdgroups:
  - q_abs projection
  - score computation over sequence positions
  - context accumulation
  - output projection
- [x] Remove single-threadgroup constraint for full transformer layer execution:
  - split fused layer into staged kernels (or subphase kernels)
  - launch many threadgroups per stage
- [x] Precompute RoPE trig tables and read from buffers instead of recomputing `pow/cos/sin` in hot loops.
- [x] Move static weight buffers to `MTLResourceStorageModePrivate` with one-time upload via blit.

## Suggested execution order

1. Build optimization flag change (`-O0` -> `-O3`) and re-benchmark.
2. Parallelize attention (`lid == 0u` hotspot) and re-benchmark.
3. Refactor single-threadgroup fused kernel into scalable staged dispatches.
4. Add RoPE lookup tables.
5. Move static weights to private GPU memory.

## Measurement goal

- Target: move native-fast decode toward `20+ tok/s` and track first-token + steady-state decode after each step.
- Latest measured after implementation (`make bench-native-fast-profile` / `make bench-native-fast-matrix`):
  - native-fast first decode token: `~9.25 tok/s`
  - native-fast steady-state decode estimate: `~9.19 tok/s`
  - progress toward `20 tok/s`: `45.9%`
