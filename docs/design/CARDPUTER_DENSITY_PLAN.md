# Cardputer Native LLM — Maximizing Parameter & Information Density

Date: 2026-07-02
Scope: `esp32_m5stack/` runtime + `python/` export/quant + `scripts/moe_tradeoff_estimator.py`
Target: `m5stack_cardputer_nopsram` (ESP32-S3, no PSRAM), embedded PROGMEM weights + vocab.

## TL;DR

The Cardputer build is **still RAM-bound** as of Phase 1.2 (streamed attention + activation trimming). The frontier has **shifted from depth to width**:

- **Before**: d=44, L=44 (depth-constrained)
- **After (P1.1)**: d=64, L=35 (still depth-limited at ~201 KiB RAM)
- **After (P1.2)**: d=128, L=11 (width-limited at ~142 KiB RAM, 29 KiB headroom left)

Current `--cardputer-optimal` reaches **d=128, L=11, V=512, ~6.53M params** (27% more parameters than original), using only **71% of the 200 KiB working-RAM budget**. The max-flash search now reaches **~6.51M params** at **98.2%** of the 7104 KiB app partition.

Next levers in priority order:

1. **Use freed RAM headroom** to scale vocab (512→1024) or context (128→256) without exceeding 200 KiB.
2. **Add RoPE** to decouple learned position embeddings from fixed sequence length (saves embedding matrix memory once context grows).
3. **Spend the remaining flash more efficiently** with **sub-8-bit weight packing** (int4, then ternary/BitNet-b1.58) and **cross-layer weight sharing** (MobileLLM-LS style).
4. **Make it fast** with int8 kernels, MQA/GQA, and a top-1 router fast path.

This document supersedes the runtime sections of `MOE_ARCHITECTURE_IMPROVEMENT_PLAN.md`
by tying every change to the measured RAM/flash bottleneck.

---

## 1. Current State (measured)

### 1.1 Flash layout (`partitions_embedded.csv`)

| Region | Offset | Size | Notes |
|---|---|---|---|
| `nvs` | 0x9000 | 24 KiB | |
| `phy_init` | 0xf000 | 4 KiB | |
| `factory` (app) | 0x10000 | **0x6F0000 = 7104 KiB** | firmware **+ PROGMEM weights + vocab** |
| `spiffs` | 0x700000 | 1024 KiB | unused in embedded mode |

Weights live in the **factory app partition** (PROGMEM), read in place via
`pgm_read_byte` — never copied to heap. So flash pressure = firmware code +
int8 weights + vocab table.

### 1.2 Estimator output after Phase 1.2 (activation trimming)

Post-Phase 1.1 (streaming) + Phase 1.2 (trimmed Q, temp buffers):

```
d_model=128  n_layers=11  n_heads=4  d_ff=448
moe_n_experts=4  top_k=1  shared_d_ff=224
vocab=512  block_size=128
weight_bytes ≈ 6.59 MB   firmware ≈ 6.59 MB
working_bytes = 200704 B = 196.0 KiB
param_count ≈ 6.53M
```

**Shift in frontier**: The old optimal (d=64, L=35) is now suboptimal. The architecture went from **depth-limited (many layers, small width)** to **width-limited (few layers, large width)**. 

Max-flash search now reaches:

```
d_model=92  n_layers=15  n_heads=4  d_ff=252
moe_n_experts=8  top_k=1  shared_d_ff=126
vocab=512  block_size=128
weight_bytes ≈ 6.44 MB   firmware ≈ 6.98 MB (≈98.2% of app partition)
working_bytes = 144976 B = 141.6 KiB
param_count ≈ 6.51M
```

**Freed headroom**: 200.7 KiB - 141.6 KiB = **59 KiB of working RAM still available** for further optimizations or context scaling.

### 1.3 Where the 200 KiB goes

The runtime allocates (all float32) in `allocateBuffers()`
([model_esp32.cpp](esp32_m5stack/src/model_esp32.cpp#L446)):

```
working_bytes = 4 * (3*S*D  +  3*D  +  S  +  V)
                     └─────┘  └───┘   └──┘   └┘
         activations(hidden,K,V)  temp(Q,t1,t2)  scores logits
   (hidden,K,V)   (Q,temp1,temp2)
```

At `S=128, D=128, V=512` (current optimal after Phase 1.2):

| Buffer | floats | bytes |
|---|---|---|
| 3 × (S·D) = K,V,hidden | 49 152 | **192 KiB** |
| 3·D = Q, temp1, temp2 | 384 | 1.5 KiB |
| streamed attention scores | 128 | 0.5 KiB |
| V logits | 512 | 2 KiB |
| **total after Phase 1.2** | 50 176 | **196 KiB** |

Savings by phase:

- **Phase 1.1** (streaming): Removed `S²` scores buffer (was 64 KiB at S=128).
- **Phase 1.2** (activation trim): Reduced Q, temp1, temp2 from S·D each to D each (saved ~96 KiB from Phase 1.1 baseline).
- **Net**: Brought working RAM from ~330+ KiB → 196 KiB, unlocking d_model to scale from 44 to 128.

Key insight: We are now **29 KiB below the 200 KiB ceiling**, leaving room for:

- **Vocab scaling to 1024** (costs ~4 KiB) — feasible.
- **Context scaling to 256+** (costs proportionally more) — would overshoot.
- **int4 load-time dequant scratch** (temporary) — absorbs freed space during loading.

### 1.4 Runtime shape (correctness/speed caveats)

- **No KV cache / no incremental decode.** `attention()` still recomputes Q/K/V for the whole
  sequence each pass, but it now streams a single score row instead of materializing the
  full matrix ([model_esp32.cpp](esp32_m5stack/src/model_esp32.cpp#L667)).
- **Full MHA:** Q/K/V/O are all `d_model×d_model` — K/V not reduced.
- **Float dequant in the inner loop:** `linearFromProgmem` converts each int8 weight to
  float per MAC ([model_esp32.cpp](esp32_m5stack/src/model_esp32.cpp#L544)).
- **GELU via `tanhf`** — expensive on ESP32-S3.
- **Learned position table** `pos_embedding[max_seq_len, d_model]` — scales flash + RAM with context.
- **lm_head tied at train time** but the int8 export still stores a separate lm_head matrix
  (duplicate `V·D` bytes).

---

## 2. State-of-the-Art Techniques (and how they map here)

| Technique | Source | What it buys on Cardputer | Fit |
|---|---|---|---|
| **Deep-and-thin + embedding sharing + GQA** | MobileLLM (ICML'24, arXiv:2402.14905) | best quality-per-param at sub-1B scale; our Phase 1.2 shift to 11L/d=128 is now wide-and-shallow — _opposite_ of MobileLLM's deep-and-thin. Worth re-evaluating: retrain with deep-and-thin (e.g., 24L/d=48) + GQA might regain quality at lower depth cost. | ◐ investigate |
| **Block-wise weight sharing (MobileLLM-LS)** | MobileLLM | ~2× effective depth for **0** extra flash; small latency cost | ✅ high value (flash-free depth) |
| **Ternary 1.58-bit weights** | BitNet b1.58 (arXiv:2402.17764) | ~5× param density vs int8 in same flash; matmul becomes add/sub | ✅ big flash win, needs QAT |
| **int4 / NF4 weight packing** | QLoRA / GPTQ family | 2× density vs int8, modest quality loss, no retrain (PTQ) | ✅ fastest density win |
| **GQA / MQA** | Ainslie'23 / Shazeer'19 | smaller K/V weights + KV cache; enables cache at all on MCU | ✅ high value |
| **RoPE / ALiBi** | Su'21 / Press'21 | removes learned position table; unbounded context; less flash/RAM | ✅ recommended |
| **SwiGLU / squared-ReLU FFN** | Shazeer'20 | better quality-per-FFN-param; ReLU² is MCU-cheap | ◐ evaluate |
| **FFN-only top-1 MoE + shared expert** | Switch / DeepSeekMoE | more stored params at ~dense active compute | ✅ already implemented |
| **PEER / product-key expert retrieval** | He'24 | many tiny experts, sub-linear routing | ◐ future, complex |
| **SD-backed expert paging** | (systems) | params beyond flash via LRU expert cache | ◐ Phase 3, PSRAM-friendly |

Key takeaway from SOTA: for sub-billion **on-device** models, **architecture and
bit-width dominate raw parameter count.** The two changes with the best
density-per-effort here are **cross-layer weight sharing** (free depth) and
**sub-8-bit packing** (int4 now, ternary later).

---

## 3. Improvement Plan (prioritized)

### Phase 0 — Instrumentation & correctness (prereq for any claim)
- **0.1** Add `esp_timer` spans around embedding / attention / FFN / lm_head / sampling.
- **0.2** Add a correct incremental decode path (`prefill()` + `decode_one()`); validate
  token parity vs `scripts/verify_cardputer_serial.py`. Today generation appends tokens
  from a single logits vector, so tok/s numbers are not true autoregression.
- **0.3** Log real free internal SRAM after `allocateBuffers()` and after first forward.

### Phase 1 — Reclaim working RAM (raises the density ceiling)  ← biggest lever
- **1.1 Streaming attention.** Implemented. The runtime now uses a single reused
  `S`-length score row (compute row → softmax → accumulate into output), and the estimator
  matches that layout. This freed ~64 KiB at S=128 and shifted the search frontier to
  **d=64 / ~5.87M params**.
- **1.2 Trim the activation set.** `temp1/temp2/Q` need not be full-sequence; write
  attention output in place. Target `hidden(S·D) + K(S·D) + V(S·D) + O(D)` ≈ `3·S·D`.
  From the current post-1.1 state, this is the next major RAM win.
- **1.3 Spend the reclaimed ceiling.** Re-run the estimator; the freed ~125 KiB RAM +
  ~1.6 MB flash allow a larger `d_model` (e.g. 64–80), larger `vocab` (2048), or
  longer context (192–256). Pick the axis by ablation on held-out chat perplexity.
- **1.4 RoPE positions.** Replace the learned `pos_embedding` table with RoPE applied in
  `attention()`. Removes a `max_seq_len·d_model` table from flash **and** decouples
  context length from a fixed table.

### Phase 2 — Sub-8-bit flash packing (fills the freed flash with params)
- **2.1 int4 weight packing (PTQ first).** Add a `bits=4` path to
  [export_weights.py](python/export_weights.py) (group-wise scales, 2 weights/byte) and a
  `linearFromProgmem4` unpacking kernel. **2× params** in the same flash, no retrain.
  Gate with a config `weight_bits` flag; keep int8 as default until parity is verified.
- **2.2 De-dup tied lm_head.** With `tie_lm_head=true`, reference the token-embedding
  block for logits instead of storing a second `V·D` matrix. Saves `V·D` int8 bytes
  (~45 KiB at V=1024/D=44; grows with vocab).
- **2.3 Ternary (BitNet b1.58) track.** Add QAT in `python/` (ternary linears with
  learned scales) + a packed loader (5 trits/byte ≈ 1.6 bits). ~5× density vs int8.
  This is the long-horizon path to a much larger model in the same 7 MB; validate
  quality against int8 baseline before adopting.

### Phase 3 — Cross-layer weight sharing (free depth)
- **3.1 MobileLLM-LS block sharing.** Reuse each transformer block's weights across N
  adjacent layers (e.g. share every 2 layers). Doubles effective depth for **zero**
  extra flash; only per-layer norm/scale need to differ. Add `layer_share_group` to
  config and loader; the PROGMEM layout stores unique blocks + a share map.

### Phase 4 — Speed & compute density
- **4.1 int8×int8→int32 kernels.** Quantize activations per token, accumulate in int32,
  dequant once per output vector (replaces per-MAC float conversion). Evaluate ESP-NN /
  ESP-DSP; add `IRAM_ATTR` hot kernels.
- **4.2 MQA/GQA.** Add `n_kv_heads`; export reduced K/V. Shrinks K/V flash, K/V compute,
  and — critically — makes a **KV cache** small enough to consider on no-PSRAM.
- **4.3 Top-1 router fast path.** For `top_k==1`, argmax + single expert; skip
  `partial_sort`, `expf`, and gate normalization.
- **4.4 Cheap activation.** Train/export ReLU² (or GELU LUT) instead of `tanhf` GELU.

### Phase 5 — Beyond-flash capacity (optional, PSRAM/SD)
- **5.1 KV cache** for real incremental decode. Note: a full multi-layer fp32 KV cache is
  **infeasible in 200 KiB internal SRAM** for 35 layers (~1.5 MB MHA / ~385 KiB MQA-fp32).
  Enable only with **int8+MQA cache**, reduced context, or on the PSRAM build.
- **5.2 SD-backed expert paging** with an LRU expert cache (Phase 2/3 of
  `MOE_CARDPUTER_RESEARCH.md`) to store total params beyond the 7 MB partition.

---

## 4. Projected density gains (order-of-magnitude)

Current post-Phase-1.1 baseline: **~5.87M int8 params**, 90.2% flash, 196.5 KiB RAM
(optimal quality point), or **~6.48M params** at 98.2% flash (max-fill point).

| Change | Mechanism | Effect | Cumulative param budget |
|---|---|---|---|
| P1.2 activation trimming | frees more of the 6·S·D buffer set | raises width/vocab/context ceiling again | beyond the current **~5.9M–6.5M int8** frontier |
| P2.1 int4 packing | 2× density/flash | 2× params same flash | **~12–13M** effective |
| P2.3 ternary b1.58 | ~5× density/flash | replaces int4 track | **~25–30M** effective |
| P3 block sharing (×2) | free depth | 2× layers, 0 flash | +100% effective depth |

These are capacity estimates; **quality per parameter must be validated** at each step
against the current chat checkpoint (perplexity + `verify_cardputer_serial.py`).

---

## 5. Invariants to preserve (per `AGENTS.md`)

Any change touching the weight format must land **atomically** across:
`python/export_weights.py` → `cpp/` loader → ESP32 loaders (`model_esp32.cpp`,
`model_embedded.cpp`, `model_weights_types.h`) → sanity checks → tests.
Gate incomplete paths with hard errors. Keep `lm_head`↔`token_embedding` tying and
tokenizer parity across Python/C++/ESP32.

## 6. Recommended sequencing

`P0.1–0.2` → `P1.1` ✅ → `P1.2` ✅ → `P1.3` (pick growth axis) → `P2.2` (easy
flash) → `P4.3/4.4` (cheap speed) → `P2.1 int4` → `P4.1 int8 kernels` → `P4.2 MQA` →
`P3 sharing` → `P2.3 ternary` → `P5` (PSRAM/SD).

**Status**:
- `P1.1 streaming attention` ✅ **LANDED** (2026-07-02): Removed S² attention scratch, compiles cleanly.
- `P1.2 activation trimming` ✅ **LANDED** (2026-07-02): Reduced temp_buffer1/temp2/attn_q from S·D to D, freed ~96 KiB, compiles cleanly.
- `P1.3 growth axis` — **NEXT**: Use the 29 KiB headroom to decide between (a) vocab 512→1024, (b) context 128→256, or (c) slight depth increase. Retrain to pick best quality frontier.
