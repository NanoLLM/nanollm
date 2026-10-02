# Flash-Backed Precomputed KV Cache for Larger Initial Context (ESP32-S3)

Research & viability note. Date: 2026-10-01.

**Question:** Can we use the ESP32-S3's free flash to store a precomputed KV cache
(a fixed system prompt) and thereby increase NanoLLM's initial context size on the
Cardputer?

**Short answer:** Yes, it is technically feasible and it is a clean extension of the
existing int8-KV architecture. But it gives a **real yet bounded** context increase —
roughly one extra context token per token of system prompt you bake in — and it does
**not** remove the fundamental ~200 KiB working-RAM ceiling. It trades RAM for
per-token flash bandwidth and requires a bit-exact offline precompute pipeline. The
higher-lever, more durable wins are a smaller KV head dimension and (on the R8N16
board the user owns) PSRAM.

---

## 1. Ground truth: the deployed model and its memory budget

All numbers below come from the actual release artifact
`releases/cardputer_mqa_ctx224_chat_v1/deploy/model_config.json`, not the aspirational
snippet in `AGENTS.md` §4 (that `d=32 / L=124 / seq=512` line is a different, larger
research config; the shipped Cardputer model is the one below).

```
vocab_size  2048
d_model     80
n_layers    18
n_heads     4
n_kv_heads  1        <- multi-query attention (MQA)
kv_dim      20       = n_kv_heads * (d_model / n_heads) = 1 * (80 / 4)
d_ff        320
moe         4 experts, top-1, shared d_ff=160
max_seq_len 224
quantized   int8 (weights + KV cache)
```

Working-RAM accounting (mirrors `allocateBuffers()` in `esp32_m5stack/src/model_esp32.cpp`
and `estimate_working_memory_bytes()` in `scripts/moe_tradeoff_estimator.py`):

| Component                              | Bytes    | KiB     |
|----------------------------------------|----------|---------|
| single-token scratch (`4*d + S` floats) | 2,176    | 2.1     |
| int8 K/V cache @ 224 (`2·L·S·kv_dim`)   | 161,280  | 157.5   |
| float32 K/V scales (`2·L·S`)            | 32,256   | 31.5    |
| **Total working RAM**                    | **~191 K** | **~191**  |

The 200 KiB budget (`kSramWarningThresholdBytes`) is **~99% consumed by the KV cache
alone**. This is the single most important fact for this research: **on this model the
context length is KV-RAM-bound, not flash-bound and not compute-bound.** Per-token KV
cost is `2·L·kv_dim + 2·L·4 = 720 + 144 = 864 B/token`, so 200 KiB holds ~234 tokens —
which is exactly why `max_seq_len` is pinned at 224.

---

## 2. How the KV cache works today (the thing we'd be reusing)

`model_esp32.cpp` → `NanoLLM::decodeStep()` (the MQA/int8 path, taken when
`n_kv_heads != n_heads`) computes, per layer, per token:

1. `k = W_k @ x`, `v = W_v @ x` (int8 weights from PROGMEM, float activations).
2. RoPE applied at the **absolute token position** (`apply_heads(..., pos, ...)`).
3. Per-token int8 quantization: `quantize_cache()` takes the max-abs over the token's
   `kv_dim` values, sets `scale = 127/max_abs`, clamps to [-127,127], and stores
   `int8` values + one float `scale` per token.
4. Stores into persistent `std::vector` buffers:
   - `kv_key_cache[layer][pos][kv_dim]` (int8)
   - `kv_value_cache[layer][pos][kv_dim]` (int8)
   - `kv_key_scales[layer][pos]`, `kv_value_scales[layer][pos]` (float32)
5. Attention: for the new token, dot `q` against every cached `K[t]` (dequantizing
   `K[t] / scale[t]`), softmax, then accumulate `V[t] / scale[t]`.

Consequences that make flash-KV viable:

- **Deterministic.** Same model + same prompt → bit-identical int8 K/V and scales.
  The artifact is reproducible offline.
- **Position-anchored.** RoPE is applied at the absolute position. A system prompt that
  always occupies positions `[0, P)` can have its RoPE baked in once; new tokens
  continue at `[P, …)` and stay consistent.
- **Compact.** 864 B/token (int8 KV + float scales). A 128-token system prompt is
  ~108 KiB; 224 tokens is ~189 KiB — comfortably inside the **1 MiB SPIFFS partition
  that is currently unused** in the `m5stack_cardputer_nopsram` build
  (`partitions_embedded.csv`: `spiffs, data, spiffs, 0x700000, 0x100000`).

Also relevant: there is **no persistent system prompt today.**
`main.cpp → generateResponse()` rebuilds `"User: {prompt}\nAssistant:"` fresh on every
turn and `resetCache()` clears `decode_history`. So every turn re-prefills from scratch.
A flash KV cache would *introduce* a fixed, reusable prefix for the first time.

---

## 3. What flash-KV actually buys (the honest math)

Let `P` = number of system-prompt tokens baked into flash, and let the RAM still hold
KV for `S_ram` tokens. With the current model, `S_ram ≈ 234`.

- **Today:** the whole 224-token window is one turn; user prompt + response share it.
- **With flash system KV:** the system prefix's KV lives in flash (0 RAM), so the RAM
  only needs to hold KV for `max_seq - P` tokens. You can raise
  `max_seq_len ≈ P + 234`.

| System prompt `P` (flash KV size) | New max context ≈ `P + 234` | Gain vs 224 |
|-----------------------------------|------------------------------|-------------|
| 0  (none, today)                  | 224                          | —           |
| 64  (~55 KiB flash)               | ~298                         | +74         |
| 128 (~108 KiB flash)              | ~362                         | +138        |
| 192 (~162 KiB flash)              | ~426                         | +202        |
| 256 (~216 KiB flash)              | ~490                         | +266        |

**So yes — it is a genuine increase in the initial/max context, roughly `+P` tokens for
a `P`-token baked system prompt.** The system prompt stops competing for the 200 KiB
RAM budget, and that budget is redirected to the user turn + generated tokens.

**The catch (why it is not a free lunch):** attention must read **every** cached K/V
position on **every** decode step. Moving the system KV to flash does not stop the
reads — it just changes *where* the bytes come from. Each generated token must read
`P · n_layers · kv_dim` int8 K bytes + `P · n_layers · kv_dim` int8 V bytes and dequant
them. For `P=128, L=18, kv_dim=20`: ~90 KiB of flash reads **per generated token**
(`P·L·kv_dim` int8 K + same for V), plus per-element int8→float dequant and the score
multiplies. On the ESP32-S3,
`pgm_read_byte` inside inner attention loops is already a known cost (see
`docs/design/MOE_ARCHITECTURE_IMPROVEMENT_PLAN.md`, "Flash Access"). So flash-KV
**trades RAM for per-token flash bandwidth and dequant compute** — the longer the baked
prefix, the *slower* each decoded token becomes.

---

## 4. Why it is NOT the biggest lever

The repo's own design thesis (improvement plan, "Packing More Parameters"):

> Because no-PSRAM RAM is the hard limiter, flash headroom alone is less useful until
> activation memory is reduced.

Flash-KV is a *clever* use of flash that the thesis predicts gives only a **bounded**
win. Two alternatives dominate it:

**(a) Smaller KV head dimension (training-time, permanent, no bandwidth penalty).**
Context scales as `1 / per_token_KV`. At `L=18, d=80`:

| kv_dim | per-token KV | max context @ 200 KiB |
|--------|-------------|------------------------|
| 20 (now) | 864 B | ~234 |
| 10       | 504 B | ~402 |
| 8        | 432 B | ~469 |
| 4        | 288 B | ~703 |

Cutting `kv_dim` from 20 → 8 raises context ~2x with **zero** extra flash reads and no
new pipeline. (MQA at `kv_dim=8` i.e. 4-dim keys/values is a real, quality-impactful
change — must be retrained/benchmarked.) This is the durable structural fix.

**(b) PSRAM.** The user's Waveshare board is `ESP32-S3R8N16` (8 MB PSRAM). The repo
already has a `m5stack_cardputer` (PSRAM) env and the `system1` build uses PSRAM for
activations. Putting the KV cache in PSRAM removes the 200 KiB wall entirely — context
is then limited by flash/weights, not RAM. This is the largest single lever and it is
available on the actual hardware.

---

## 5. Viability assessment

| Dimension | Assessment |
|-----------|------------|
| **Technical feasibility** | High. Reuses the exact int8 KV layout + per-token scales + RoPE-at-position the runtime already produces. |
| **Where to store it** | 1 MiB SPIFFS partition (unused in nopsram build) holds up to ~1,180 tokens of KV. Alternatively a blob in the app partition. |
| **Context gain** | Real but bounded: `+P` tokens for a `P`-token baked prefix. Does not break the 200 KiB RAM ceiling. |
| **Latency cost** | Per-decode-token flash reads + dequant scale with `P`. Long prefixes make decoding *slower*. |
| **Generality** | Low. Works only for a *fixed* system prompt + *fixed* weights. Any change to either invalidates the artifact and requires re-precompute. |
| **Correctness risk** | The precompute must be **bit-exact** with `decodeStep`'s int8 quantization (per-token max-abs → `scale=127/max_abs`, clamp, round) and the same RoPE, or Python/C++/ESP32 parity breaks. This is the hardest part. |
| **Effort** | Medium-high: offline precompute tool (Python) + firmware `prefillFromFlash()` + cache-layout plumbing + parity tests. |

**Verdict: viable as a targeted, prompt-specific optimization, not as a general
context-scaling mechanism.** A good first experiment: bake a fixed 64–128 token system
prompt into SPIFFS, keep `max_seq ≈ 290–360`, and measure (i) quality vs. an
equivalent-RAM retrain with smaller `kv_dim`, and (ii) tokens/sec penalty from the
flash reads. If the latency penalty is acceptable for the product (e.g. a fixed
assistant persona that never changes), it is a clean win. If general long context is the
goal, prefer smaller `kv_dim` and/or PSRAM.

---

## 6. If pursued: concrete implementation sketch

**Offline (Python, new `python/export_prefill_cache.py`):**
1. Load the exported checkpoint + the exact int8 quantization used by `decodeStep`.
2. Run the fixed system prompt through the model; for each layer/position compute
   `k`, `v`, apply RoPE at the absolute position, then quantize identically to the C++
   `quantize_cache` (max-abs → `127/max_abs`, clamp, round; keep the float scale).
3. Emit a binary blob in the `[layer][pos][kv_dim]` layout with a small header:
   `{magic, P, n_layers, kv_dim, n_kv_heads, d_model, model_hash}`, then K int8,
   V int8, K scales, V scales (or interleaved per-token to match the runtime exactly).
4. Record the SHA of the weights + the exact prompt so firmware can verify freshness.

**Firmware (`model_esp32.cpp`):**
1. Add `prefillFromFlash(path)`: read the blob, verify model/prompt hash, populate
   `kv_key_cache/kv_value_cache/kv_key_scales/kv_value_scales` for positions `[0, P)`,
   and set `decode_history` length to `P`.
2. Extend `decodeStep`'s attention loop so that for `t < P` it reads K/V **from the
   flash-backed (or SPIFFS-file) source** and dequantizes, and for `t >= P` it reads
   from the in-RAM cache as today. (Simplest first cut: copy the baked prefix into the
   in-RAM cache at `max_seq - P` capacity at boot; the flash then only serves *boot
   time*, not the hot loop — see §7.)
3. Raise `max_seq_len` to `P + S_ram` in the config.

**Parity:** extend `scripts/verify_cardputer_serial.py` with a fixed system-prompt
prompt and assert the first generated tokens match Python exactly (the precompute must
reproduce the runtime's int8 KV bit-for-bit).

---

## 7. Recommended first step (cheap, de-risking)

Before building the streaming-attention flash read, do the **boot-time copy** variant:

- Precompute the system KV offline and write it to SPIFFS.
- At boot, read it **once** into the in-RAM KV buffers (this is just a fast bulk
  SPIFFS read, ~100 KiB, one-time).
- Decode then reads the system KV from **RAM** exactly as today — **zero** per-token
  flash-bandwidth penalty.
- Net effect: the system prompt occupies flash at rest but RAM at run time; the win is
  purely that the *rest of the turn* (user + response) gets the full 200 KiB KV budget
  while the persona stays "always on" without re-prefilling each turn.

This variant answers the user's actual question ("increase initial context") with the
smallest change and no hot-loop flash reads. The per-step flash-read variant (true
zero-RAM system KV, enabling `P` beyond the RAM budget) is a follow-up only if the boot
copy is insufficient.

---

## 8. Bottom line

- **Is it possible?** Yes. The int8, per-token-scaled, RoPE-positioned KV cache is an
  ideal, reproducible artifact to store in flash; there is 1 MiB of free SPIFFS.
- **Does it increase initial context?** Yes, by roughly the length of the baked system
  prompt (~+P tokens), because that prefix stops consuming the 200 KiB working RAM.
- **Is it the best way to do it?** No. It is a bounded, prompt-specific, latency-costing
  win. Smaller `kv_dim` (training-time) and PSRAM (on the R8N16 board) are the larger,
  more durable levers. Use flash-KV as a targeted optimization for a fixed persona, and
  do the boot-time-copy variant first to avoid per-token flash reads.

## Sources (repo)
- `releases/cardputer_mqa_ctx224_chat_v1/deploy/model_config.json` — deployed model dims.
- `releases/cardputer_mqa_ctx224_chat_v1/manifest.json` — artifacts, `cardputer_profile`.
- `esp32_m5stack/src/model_esp32.cpp` — `allocateBuffers()`, `decodeStep()`, `attention()`, `quantize_cache`, `resetCache()`.
- `esp32_m5stack/src/model_esp32.h` — `kSramWarningThresholdBytes`, `kMaxInferenceSeqLenNoPsram`, KV buffer members.
- `esp32_m5stack/src/model_embedded.cpp` — PROGMEM weight load.
- `esp32_m5stack/src/main.cpp:238` — `generateResponse()` (no persistent system prompt).
- `esp32_m5stack/partitions_embedded.csv` — 7 MiB app + 1 MiB SPIFFS partitions.
- `esp32_m5stack/platformio.ini` — `m5stack_cardputer_nopsram` build flags (embedded weights+vocab, no PSRAM).
- `scripts/moe_tradeoff_estimator.py` — working-memory model used for the numbers above.
- `docs/design/MOE_ARCHITECTURE_IMPROVEMENT_PLAN.md` — prior art: KV cache, GQA/MQA, int8 KV, flash-access costs, storage abstraction, "RAM is the hard limiter."
- `AGENTS.md` §4 — (note: the `d=32/L=124/seq=512` config there is an alternate research
  config, not the deployed model; the deployed model is the 18-layer/d=80 one above).
