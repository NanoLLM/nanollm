# MoE on Cardputer: Research, Feasibility, and Execution Plan

Date: 2026-06-30

## TL;DR

MoE can help this NanoLLM project if it is implemented as **FFN-only sparse experts** with **top-1 routing** and (optionally) a **shared expert**. This can increase total parameter capacity while keeping active per-token compute closer to the dense baseline.

For ESP32-S3 specifically:

- Weights live in **firmware flash** (PROGMEM) via `partitions_embedded.csv` — not copied to RAM.
- **Working RAM** (~200 KiB) caps `d_model` / `vocab_size` before flash does.
- Validated: **128-token context**, **vocab=1024**, **44L, d=44** (~4.2M params, ~67% flash). Vocab above 1024 hits working RAM before flash.
- At 64-token context the flash-optimal config was d=112 / 13L (~6.5M params); longer context requires narrower width.

```bash
python scripts/moe_tradeoff_estimator.py --cardputer-optimal
./scripts/train_moe_pipeline.sh --profile quick --export   # smoke test
python scripts/verify_cardputer_serial.py --reset
```

## Why this applies to this repo

Current code already has the right decomposition:

- FFN path is isolated in `feed_forward` in `esp32_m5stack/src/model_esp32.cpp`.
- Weight export format is centralized in `python/export_weights.py`.
- Working-memory pressure comes mostly from activations and attention scratch, not just stored weights.

So MoE can target the FFN path without rewriting the entire model stack.

## What recent MoE research contributes here

### 1) Sparse FFN routing is still the core win

- Switch Transformers (top-1 routing) established that sparse activation can scale parameters without proportional compute growth.
- For edge, top-1 is usually preferable to top-2 because it cuts routing and data movement.

### 2) Shared experts are practical for small-device stability

- DeepSeekMoE-style separation of shared and routed experts is useful for tiny models too.
- A shared expert can preserve common knowledge while routed experts specialize.

### 3) Better load balancing matters during training

- New balancing techniques (including auxiliary-loss-free variants) reduce router collapse.
- This mainly affects training quality and stability, not embedded runtime directly.

### 4) Ultra-compressed MoE (e.g., QMoE direction) is conceptually useful

- Very aggressive compression demonstrates that MoE weights can be shrunk dramatically.
- But practical implementations often depend on GPU-specific decode kernels, so this is not directly portable to ESP32-S3.

## ESP32-S3 memory mapping facts that matter

### Supported MMU targets

Per ESP-IDF memory-management docs, dynamic mapping supports:

- SPI flash
- PSRAM

Key APIs include:

- `esp_mmu_map`
- `esp_mmu_unmap`
- `esp_mmu_map_get_max_consecutive_free_block_size`

### SD card limitation

SD/FATFS access is via VFS/FatFs file APIs (`read`, `lseek`, etc.).
This is block/file IO, not MMU virtual-address mapping of file contents.

Implication:

- Treat SD as a backing store for expert pages.
- Stage needed experts into RAM/PSRAM cache buffers.

### PSRAM caveats

- PSRAM helps capacity but has lower/variable effective bandwidth.
- Large streaming accesses can thrash shared cache and hurt code/rodata locality.
- Use PSRAM for large expert buffers and keep hot routing/scratch in internal RAM.

## Recommended architecture for NanoLLM on Cardputer

### Phase 1 (recommended first): In-RAM MoE baseline

Implement FFN-MoE with small expert count (e.g., 4) and top-1 routing:

- Keep attention dense.
- Keep model dims conservative (especially `max_seq_len`).
- Keep all experts loaded (SPIFFS/embedded) for correctness baseline.

Goal: validate quality and latency tradeoff with minimal systems risk.

### Phase 2: SD-backed expert paging

Add expert storage file(s) on SD and runtime cache:

- Build an expert index table: `{expert_id, offset, length, scales}`.
- Read contiguous expert blobs with aligned block sizes.
- Maintain small LRU cache in PSRAM for hot experts.

Goal: increase total parameter count beyond what can stay resident.

### Phase 3: Throughput tuning

- Prefetch predicted next experts (router lookahead heuristic).
- Group experts physically on disk by co-occurrence.
- Reduce random seeks and maximize sequential reads.

## Practical SOTA-inspired choices (edge-friendly)

Use now:

- FFN-only MoE
- top-1 routing
- one shared expert (optional but recommended)
- int8 weights (existing pipeline already aligned)

Defer unless needed:

- top-2 routing
- many experts per layer (>8)
- complex balancing logic in inference
- GPU-kernel-specific sparse formats

## Vocabulary vs context tradeoff (128-token Cardputer)

Working RAM (~200 KiB) caps vocabulary **before** flash does when using flash-backed
(PROGMEM) weights. The dominant terms are the **S² attention buffer** and the
**V-sized logits buffer**:

```
working_bytes = 4 × (6×S×D + S² + V)
```

At **S=128, D=44** (current Cardputer defaults):

| Component | Size |
|-----------|------|
| Hidden + Q/K/V buffers (`6×S×D`) | ~132 KiB |
| Attention scores (`S²`) | **64 KiB** |
| Logits buffer (`4×V`) | 4 bytes × V |
| **Total at V=0** | **~196 KiB** |
| **Headroom for logits** | **~4 KiB → V ≤ 1024** |

**`VOCAB_SIZE=1024` is the hard RAM ceiling at d_model=44 / 128 context.** Flash
still has headroom (~47–58% of the 7.1 MiB app partition depending on layer count).

### What larger vocab buys (and does not)

- **Larger vocab** → fewer BPE tokens per word → more words fit in 128 slots.
- **Does not replace long context** — coherence over distant references still needs
  longer `max_seq_len` (which forces smaller `d_model` or PSRAM).

Rough effective word capacity vs vocab 512 baseline:

| Vocab | Requires | Effective words in 128 slots |
|-------|----------|------------------------------|
| 1024 | d=44 (current max) | ~140–145 vs 512-vocab baseline |
| 2048 | d≤40 | ~155–160 |
| 4096 | d≤40 | ~160–170 |

### Configs if you need more vocab

| Target vocab | Suggested shape | Params (approx) |
|--------------|-----------------|-----------------|
| 1024 | d=44, L=44, d_ff=176 | ~3.5M |
| 2048 | d=40, L=44, d_ff=160 | ~3.0M |
| 4096 | d=40, L=44, d_ff=160 | ~3.1M |

```bash
# Check max vocab for a given (S, D):
python3 -c "from scripts.moe_tradeoff_estimator import estimate_working_memory_bytes; \
B=200*1024; S,D=128,44; print((B-estimate_working_memory_bytes(S,D,0))//4)"

# Example: vocab=2048 (requires smaller d_model)
VOCAB_SIZE=2048 D_MODEL=40 D_FF=160 MOE_SHARED_D_FF=80 ./scripts/train_moe_pipeline.sh --export
```

## Memory and latency risk checklist

1. Keep activation memory bounded first (`max_seq_len` dominates quadratic attention scratch).
2. Avoid repeated SD reads of the same expert in one generation step.
3. Cache router weights and small metadata in internal RAM.
4. Use contiguous file allocation for expert blobs when possible.
5. Benchmark cold-cache and warm-cache token latency separately.

## Concrete next coding steps for this repo

1. Python model:
   - Add `MoEFeedForward` module in `python/model.py` behind a config flag.
2. Exporter:
   - Extend `python/export_weights.py` with expert metadata + per-expert tensors.
3. ESP32 runtime:
   - Add MoE branch in `esp32_m5stack/src/model_esp32.cpp` `feed_forward` path.
4. Config:
   - Extend model config JSON with fields such as:
     - `use_moe`
     - `n_experts`
     - `top_k`
     - `shared_expert`
5. Benchmark:
   - Compare dense vs MoE on tokens/sec and output quality on fixed prompts.

## Suggested initial config for experimentation

- `d_model=32`
- `n_layers=1`
- `max_seq_len=32`
- `n_experts=4`
- `top_k=1`
- `shared_expert=true`

Then scale one axis at a time.

## References used

- Switch Transformers: arXiv:2101.03961
- Mixtral of Experts: arXiv:2401.04088
- DeepSeekMoE: arXiv:2401.06066
- Auxiliary-loss-free balancing: arXiv:2408.15664
- QMoE: arXiv:2310.16795
- ESP-IDF ESP32-S3 Memory Management (`esp_mmu_map` docs)
- ESP-IDF ESP32-S3 External RAM guide
- ESP-IDF ESP32-S3 FatFs/storage guide
