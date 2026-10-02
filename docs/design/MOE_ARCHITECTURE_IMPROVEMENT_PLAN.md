# MoE Architecture Improvement Plan

Date: 2026-07-01

Research direction: **Memory-Bounded Sparse Transformers for Microcontroller Language Modeling**

This note reviews the current NanoLLM MoE architecture and identifies improvements for:

- Faster inference
- More parameters or longer context under the same MCU budget
- ESP32-S3-specific optimization
- SD card and other storage-backed model layouts

## Current Architecture Summary

NanoLLM currently implements a compact decoder-only transformer with optional FFN-only MoE:

- Attention remains dense multi-head attention.
- MoE replaces only the feed-forward path.
- Routing is token-level top-k, usually top-1.
- A shared expert can be added to every MoE block.
- Deployment modes include:
  - PROGMEM embedded weights and vocab for no-PSRAM Cardputer builds.
  - SPIFFS-loaded dense/MoE binary artifacts for filesystem builds.
- Current no-PSRAM target budgets are approximately:
  - ~200 KiB working RAM
  - ~7 MiB firmware app partition
  - 128-token context target

Important existing bottleneck:

```text
working_bytes ~= 4 * (6 * S * D + S^2 + V)
```

The `S^2` attention buffer and `V` logits buffer dominate RAM before flash is exhausted.

## Critical Correctness and Speed Finding

The ESP32 `generate()` path currently computes one full forward pass over the prompt, computes logits once, and then appends multiple tokens from the same logits without recomputing model state for each generated token.

This means serious inference optimization should start by adding a correct incremental decoding path. Otherwise token/s measurements will not reflect real autoregressive generation.

Recommended first milestone:

1. Implement `forward_last_token()` or `prefill() + decode_one()`.
2. Recompute or cache model state per generated token.
3. Validate generated token parity against Python for fixed prompts.
4. Only then benchmark speed and energy/token.

## Highest-Impact Improvements

### 1. Replace Full `S*S` Attention Scratch With Streaming Attention

Current runtime allocates `attn_scores` as `S * S` float32. At 128 context this is 64 KiB by itself.

Recommendation:

- Compute attention one query row at a time.
- Reuse a single `S`-length score row instead of an `S*S` matrix.
- For autoregressive decode, compute only the newest token's attention against cached K/V.

Expected effect:

- Working memory changes from:

```text
4 * (6*S*D + S^2 + V)
```

to roughly:

```text
4 * (hidden/residual buffers + K/V cache + S + V)
```

For no-cache full-sequence mode, this alone saves ~64 KiB at `S=128`.

This memory can be spent on:

- Longer context
- Larger vocabulary
- Larger `d_model`
- More layers

Priority: **Very high**

### 2. Add KV Cache for Incremental Decoding

Current inference recomputes Q/K/V for the whole sequence during the prompt pass and does not implement true token-by-token decode.

Recommendation:

- Add per-layer K/V cache with shape `[n_layers, max_seq_len, d_model]` or a reduced GQA layout.
- During prefill, compute and store K/V for all prompt tokens.
- During decode, compute Q/K/V only for the new token and attend to cached K/V.

Tradeoff:

- KV cache increases persistent activation memory.
- But it converts per-token decode from repeated full-context transformer passes into a one-token pass.

For tiny models, KV cache should be optional:

- `NANOLLM_DECODE_RECOMPUTE`: lower RAM, slower
- `NANOLLM_DECODE_KV_CACHE`: higher RAM, much faster

Priority: **Very high for speed**, medium for no-PSRAM RAM budget unless paired with GQA or quantized cache.

### 3. Use Grouped-Query or Multi-Query Attention

Current Q/K/V dimensions are all `d_model`. With multi-head attention, K/V are not reduced.

Recommendation:

- Add `n_kv_heads` to the model config.
- Support MQA (`n_kv_heads=1`) or GQA (`n_kv_heads < n_heads`).
- Export separate K/V shapes and update ESP32/C++ loaders.

Benefits:

- Smaller K/V projections
- Smaller KV cache
- Less flash for K/V weights
- Faster attention

For microcontrollers, MQA or small-GQA is more valuable than full MHA.

Priority: **High**

### 4. Remove Top-1 Router Overhead

For `top_k=1`, softmax over the selected expert always produces gate 1.0. Current ESP32 MoE paths still build vectors, partial-sort experts, and compute exponentials.

Recommendation:

- Add a fast path when `top_k == 1`:
  - Compute router logits.
  - Use linear scan argmax.
  - Skip `partial_sort`, `expf`, and gate normalization.
  - Run only the selected expert.

Expected effect:

- Lower per-token latency.
- Lower heap churn.
- Simpler code path for the default Cardputer model.

Priority: **High and low-risk**

### 5. Preallocate MoE Scratch Buffers

The embedded MoE path and SPIFFS MoE path allocate temporary `std::vector`s inside `feed_forward*()` calls. Even when these are per-call rather than per-token, this is undesirable on ESP32.

Recommendation:

Move MoE scratch into persistent model buffers:

- `router_logits`
- `topk_indices`
- `topk_logits`
- `topk_gates`
- `expert_hidden`
- `expert_output`
- `shared_hidden`

Allocate them once in `allocateBuffers()` based on config.

Benefits:

- Less heap fragmentation
- More deterministic latency
- Easier memory accounting

Priority: **High and low-risk**

### 6. Replace Float Dequantized Linear With Int8 Kernels

Current `linear()` and `linearFromProgmem()` convert each int8 weight to float inside the innermost loop.

Recommendation:

- Quantize activations to int8 or int16 per token.
- Use int8 dot products with int32 accumulation.
- Dequantize only the output vector.
- Use ESP-DSP / ESP-NN kernels where possible, or add hand-written ESP32-S3-optimized kernels.

Potential kernel layout:

```text
int8 weight [out_dim, in_dim]
int8 activation [in_dim]
int32 accumulator [out_dim]
float or int16 output [out_dim]
```

Benefits:

- Much faster matrix-vector multiply
- Lower flash bandwidth pressure
- Lower float conversion cost

Priority: **High, but requires parity work**

### 7. Use Faster Activations

Current GELU uses `tanhf`, which is expensive on MCU.

Options:

- Replace GELU with ReLU or squared-ReLU during training.
- Use SwiGLU/GEGLU only if quality justifies extra parameters.
- Use a lookup table or low-order approximation for GELU.

Best MCU path:

- Train with ReLU or squared-ReLU variants and export that activation in config.

Priority: **Medium-high**

### 8. Avoid Duplicating Tied Embeddings in Export

Python ties `lm_head.weight` to `token_embedding.weight`, but current export stores both token embedding and LM head.

Recommendation:

- Add a config flag `tie_lm_head=true`.
- In embedded and binary formats, either:
  - reference token embedding data for LM head, or
  - omit LM head when tied and reuse token embedding pointer.

Benefits:

- Saves `vocab_size * d_model` int8 bytes.
- At `V=1024, D=44`, saves ~45 KiB.
- At larger vocab, savings become more important.

Priority: **Medium, easy flash win**

## Packing More Parameters

### Best Near-Term Strategy

Use saved RAM from streaming attention to increase either:

1. context length, or
2. width/vocab, or
3. number of layers.

Because no-PSRAM RAM is the hard limiter, flash headroom alone is less useful until activation memory is reduced.

Recommended model families to compare:

| Family | Goal |
|---|---|
| Dense-small | Baseline under same RAM/flash |
| MoE-top1 | More total parameters, same active FFN cost |
| MoE-top1-shared | Stability and common knowledge path |
| GQA-MoE | More context/less KV memory |
| Streaming-attn MoE | Longer context without `S*S` scratch |

### Expert Sizing

For top-1 MoE, increasing expert count increases stored parameters without proportional active compute. On MCU, the practical limiter is storage bandwidth and pointer/metadata overhead.

Recommended defaults:

- `moe_top_k=1`
- `moe_n_experts=4` or `8`
- shared expert enabled only if quality gain is measurable
- keep router small and always resident

## Increasing Context Length

The current context blocker is the `S*S` attention score buffer.

Recommended roadmap:

1. Replace `S*S` scores with one-row streaming softmax.
2. Add optional KV cache.
3. Add MQA/GQA to reduce KV cache size.
4. Consider int8 or int16 KV cache.
5. Replace learned position embedding with RoPE or ALiBi to avoid fixed position table scaling.

Potential outcomes:

- 128 -> 192/256 context on no-PSRAM if streaming attention removes full `S*S` buffer.
- Larger gains on PSRAM builds using KV cache in PSRAM and hot buffers in internal SRAM.

## ESP32-S3-Specific Optimizations

### Memory Placement

Use heap capabilities explicitly:

- Internal SRAM for hot buffers:
  - current token hidden state
  - router logits
  - attention score row
  - expert hidden/output scratch
- PSRAM, if present, for colder/larger buffers:
  - KV cache
  - SD expert cache
  - full prompt hidden-state history

Use APIs such as:

```cpp
heap_caps_malloc(size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
```

### Flash Access

`pgm_read_byte` inside innermost matmul loops is simple but slow.

Options:

- Tile weight rows into small internal SRAM blocks before dot products.
- Align arrays to cache-friendly boundaries.
- Group expert tensors contiguously by layer/expert.
- Benchmark direct PROGMEM read vs tiled read for FFN sizes used by NanoLLM.

### CPU Kernels

Use or evaluate:

- ESP-DSP for vector operations.
- ESP-NN style int8 matrix/vector kernels.
- `IRAM_ATTR` for hot kernels if instruction-cache behavior becomes a bottleneck.
- Approximate math for `expf`, `tanhf`, and softmax.

### Runtime Instrumentation

Add `esp_timer_get_time()` spans around:

- embedding
- each block attention
- each block MoE FFN
- LM head
- sampling
- SD cache misses

This is required for publishable speed claims.

## SD Card and Other Storage Devices

### Current State

The repo has a host-side SD staging script, but ESP32 runtime model loading is hardcoded to SPIFFS for file reads. There is no runtime SD model loader yet.

### Recommended Storage Abstraction

Introduce an FS-agnostic loader:

```cpp
bool load(fs::FS& fs, const char* weights_path, const char* config_path);
```

Then support:

- `SPIFFS`
- `SD`
- `SD_MMC` if the board wiring supports it
- future LittleFS/FFat variants

Main boot order should become:

1. embedded PROGMEM
2. SD card `/nanollm/model.bin`
3. SPIFFS `/model.bin`

### SD-Backed Expert Paging

For large MoE models, do not load all expert weights into heap. Instead:

- Keep embeddings, attention, norms, router, and LM head resident.
- Store routed expert FFN tensors as separately indexed blobs on SD.
- Build an expert index:

```json
{
  "layers": [
    {
      "experts": [
        {"id": 0, "offset": 123456, "length": 8192},
        {"id": 1, "offset": 131648, "length": 8192}
      ]
    }
  ]
}
```

Runtime cache:

- Small LRU expert cache in PSRAM if available.
- One or two expert slots in internal SRAM for no-PSRAM builds.
- Read contiguous expert blobs with aligned block sizes.
- Measure cold-cache and warm-cache latency separately.

### Best SD Format

Prefer a single contiguous expert pack file:

```text
model_core.bin       resident core tensors
experts.pack         routed expert tensors
experts_index.json   layer/expert offsets, lengths, scales
model_config.json
vocab.json
```

This avoids many small-file opens and reduces FAT overhead.

## Recommended Implementation Order

### Phase 0: Correctness and Measurement

1. Fix autoregressive generation with `prefill()` + `decode_one()`.
2. Add per-stage timing with `esp_timer_get_time()`.
3. Add parity tests against Python for fixed token prompts.

### Phase 1: Low-Risk Speed Wins

1. Top-1 router fast path.
2. Preallocated MoE scratch buffers.
3. Faster GELU or train-time activation switch.
4. Tied LM-head export reuse.

### Phase 2: Memory-Bounded Attention

1. One-row streaming attention.
2. Optional KV cache.
3. GQA/MQA support.
4. Longer context experiments.

### Phase 3: ESP32-S3 Kernels

1. Int8 activation quantization for matvec.
2. ESP-DSP/ESP-NN benchmarking.
3. Weight tiling from PROGMEM/SD to internal SRAM.

### Phase 4: SD-Backed MoE

1. FS-agnostic loader.
2. SD boot fallback.
3. Expert pack/index format.
4. LRU expert cache.
5. Router-informed prefetch.

## Research Contribution Fit

These improvements directly support the paper thesis:

> Memory-bounded sparse transformers require joint optimization of activation memory, logits memory, sparse FFN capacity, and storage bandwidth.

The strongest publishable experiment is a Pareto frontier:

- dense vs MoE
- same SRAM budget
- same flash or SD storage budget
- same latency target
- quality vs active compute vs memory

The most novel systems result would be:

> SD-backed sparse experts let an ESP32-S3 access more total parameters than fit in firmware while preserving bounded active compute through top-1 routing and expert caching.
