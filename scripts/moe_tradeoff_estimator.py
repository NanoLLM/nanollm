#!/usr/bin/env python3
"""
Estimate dense vs MoE tradeoffs for NanoLLM-style models on constrained hardware.

Includes Cardputer (ESP32-S3, no PSRAM) sizing: int8 weight RAM, inference scratch,
and a search for the largest layer/expert count that still fits.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


# ESP32-S3 StampS3 / M5Stack Cardputer (no PSRAM) planning constants.
# See esp32_m5stack/src/model_esp32.h (kSramWarningThresholdBytes) and
# esp32_m5stack/src/model_esp32.cpp (SPIFFS weight load into heap).
CARDPUTER_SRAM_BYTES = 512 * 1024
CARDPUTER_SYSTEM_RESERVE_BYTES = 120 * 1024  # firmware, display, tokenizer, stack headroom
CARDPUTER_WORKING_BUDGET_BYTES = 200 * 1024  # matches kSramWarningThresholdBytes
CARDPUTER_INFERENCE_SEQ_CAP = 256  # kMaxInferenceSeqLenNoPsram in esp32 model_esp32.h
CARDPUTER_WEIGHT_BUDGET_BYTES = 240 * 1024   # int8 weights loaded from SPIFFS into RAM
CARDPUTER_MOE_SCRATCH_BUDGET_BYTES = 8 * 1024
CARDPUTER_SPIFFS_BUDGET_BYTES = 512 * 1024   # conservative SPIFFS file budget
# partitions_embedded.csv factory app partition (0x6F0000).
CARDPUTER_APP_PARTITION_BYTES = 0x6F0000
# Measured firmware minus PROGMEM model+vocab (718 KiB build with ~180 KiB weights).
CARDPUTER_FIRMWARE_CODE_BYTES = 490_000
CARDPUTER_FLASH_MARGIN_BYTES = 128_000
CARDPUTER_VOCAB_FLASH_BASE_BYTES = 8_000
CARDPUTER_VOCAB_FLASH_PER_TOKEN_BYTES = 105


@dataclass(frozen=True)
class CardputerConfig:
    vocab_size: int = 256
    d_model: int = 32
    n_layers: int = 1
    n_heads: int = 2
    n_kv_heads: int = 1
    d_ff: int = 64
    max_seq_len: int = 256
    moe_n_experts: int = 2
    moe_top_k: int = 1
    moe_shared_d_ff: int = 16


@dataclass(frozen=True)
class CardputerSizing:
    config: CardputerConfig
    weight_bytes: int
    working_bytes: int
    moe_scratch_bytes: int
    total_heap_bytes: int
    score: int
    firmware_bytes: int = 0


def estimate_vocab_flash_bytes(vocab_size: int) -> int:
    return CARDPUTER_VOCAB_FLASH_BASE_BYTES + vocab_size * CARDPUTER_VOCAB_FLASH_PER_TOKEN_BYTES


def cardputer_max_weight_bytes(vocab_size: int) -> int:
    """Max int8 PROGMEM weight bytes given app partition minus code, vocab, margin."""
    return (
        CARDPUTER_APP_PARTITION_BYTES
        - CARDPUTER_FIRMWARE_CODE_BYTES
        - estimate_vocab_flash_bytes(vocab_size)
        - CARDPUTER_FLASH_MARGIN_BYTES
    )


def estimate_firmware_flash_bytes(cfg: CardputerConfig, weight_bytes: int) -> int:
    return (
        CARDPUTER_FIRMWARE_CODE_BYTES
        + weight_bytes
        + estimate_vocab_flash_bytes(cfg.vocab_size)
    )


def estimate_param_count(cfg: CardputerConfig) -> int:
    """Rough trainable-parameter count (embedding tied to lm_head)."""
    d = cfg.d_model
    layers = cfg.n_layers
    experts = cfg.moe_n_experts
    ff = cfg.d_ff
    shared = cfg.moe_shared_d_ff
    embed = cfg.vocab_size * d * 2
    kv_dim = cfg.n_kv_heads * (d // cfg.n_heads)
    attention = (2 * d * d) + (2 * d * kv_dim)
    per_layer = attention + experts * (2 * d * ff) + (2 * d * shared if shared > 0 else 0)
    return embed + layers * per_layer


def estimate_dense_param_count(cfg: CardputerConfig) -> int:
    """Exact NanoLLM dense parameter count with tied token/output embeddings."""
    d = cfg.d_model
    embeddings = (cfg.vocab_size + cfg.max_seq_len) * d
    kv_dim = cfg.n_kv_heads * (d // cfg.n_heads)
    attention = (2 * d * d) + (2 * d * kv_dim)
    norms = 4 * d
    ffn = dense_ffn_params_per_layer(d, cfg.d_ff)
    return embeddings + cfg.n_layers * (attention + norms + ffn) + 2 * d


def estimate_working_memory_bytes(max_seq_len: int, d_model: int, vocab_size: int,
                                  n_layers: int = 1, n_heads: int = 1,
                                  n_kv_heads: int = 1) -> int:
    """
    Mirrors ESP32 runtime buffer allocation in allocateBuffers().

    Cached decode uses int8 K/V per layer, two float scales per token/layer,
    and float32 single-token hidden/norm/output scratch plus S attention scores.
    """
    s = min(max_seq_len, CARDPUTER_INFERENCE_SEQ_CAP)
    d = d_model
    _ = vocab_size
    kv_dim = n_kv_heads * (d // n_heads)
    single_token_float_bytes = (4 * d + s) * 4
    kv_cache_bytes = 2 * n_layers * s * kv_dim
    kv_scale_bytes = 2 * n_layers * s * 4
    return single_token_float_bytes + kv_cache_bytes + kv_scale_bytes


def estimate_generate_peak_memory_bytes(max_seq_len: int, d_model: int, vocab_size: int,
                                        n_layers: int = 1, n_heads: int = 1,
                                        n_kv_heads: int = 1) -> int:
    """Alias kept for callers that want explicit peak naming."""
    return estimate_working_memory_bytes(max_seq_len, d_model, vocab_size,
                                         n_layers, n_heads, n_kv_heads)


def estimate_moe_scratch_bytes(
    n_experts: int,
    expert_d_ff: int,
    d_model: int,
    shared_d_ff: int,
    top_k: int = 1,
) -> int:
    """Per-layer MoE feed_forward temporary vectors (see model_esp32.cpp)."""
    return (
        n_experts * 4
        + top_k * (4 + 4 + 4)
        + n_experts * 4
        + expert_d_ff * 4
        + d_model * 4
        + shared_d_ff * 4
    )


def _quantized_linear_bytes(rows: int, cols: int, bias_len: int = 0) -> int:
    """One export_weights quantized linear record (no bias stub vs MoE bias)."""
    size = 4 + 8 + rows * cols
    if bias_len > 0:
        size += 4 + 4 + bias_len
    else:
        size += 4  # legacy dense stub (uint32 size=0)
    return size


def _moe_linear_bytes(in_dim: int, out_dim: int) -> int:
    """MoE linear record always includes bias metadata."""
    return 4 + 8 + out_dim * in_dim + 4 + 4 + out_dim


def _quantized_layernorm_bytes(size: int) -> int:
    return 4 + 4 + size


def estimate_moe_bin_bytes(cfg: CardputerConfig) -> int:
    """
    Estimate int8 model.bin size for NLMO export format.
    Calibrated against checkpoints/moe_small_chat_20260630 (~32 KiB, 1 layer, 2 experts).
    """
    total = 4 + 4 + 1 + 3 + 4 + (4 if cfg.n_kv_heads != cfg.n_heads else 0)

    total += _quantized_linear_bytes(cfg.vocab_size, cfg.d_model)
    total += _quantized_linear_bytes(cfg.max_seq_len, cfg.d_model)

    for _ in range(cfg.n_layers):
        kv_dim = cfg.n_kv_heads * (cfg.d_model // cfg.n_heads)
        total += _quantized_linear_bytes(cfg.d_model, cfg.d_model) * 2  # q, o
        total += _quantized_linear_bytes(kv_dim, cfg.d_model) * 2  # k, v
        total += _quantized_layernorm_bytes(cfg.d_model) * 4  # norm1 w/b, norm2 w/b
        total += 13  # moe header: III?
        total += _moe_linear_bytes(cfg.d_model, cfg.moe_n_experts)  # router + bias
        for _expert in range(cfg.moe_n_experts):
            total += _moe_linear_bytes(cfg.d_model, cfg.d_ff)
            total += _moe_linear_bytes(cfg.d_ff, cfg.d_model)
        if cfg.moe_shared_d_ff > 0:
            total += _moe_linear_bytes(cfg.d_model, cfg.moe_shared_d_ff)
            total += _moe_linear_bytes(cfg.moe_shared_d_ff, cfg.d_model)

    total += _quantized_layernorm_bytes(cfg.d_model) * 2  # final norm w/b
    # lm_head is tied to token_embedding at train time and omitted from export.
    total += 4  # lm_head bias stub when present in legacy artifacts
    return total


def estimate_dense_bin_bytes(cfg: CardputerConfig) -> int:
    """Estimate the legacy dense int8 model.bin layout."""
    total = 8 if cfg.n_kv_heads == cfg.n_heads else 20
    total += 12 + cfg.vocab_size * cfg.d_model
    total += 12 + cfg.max_seq_len * cfg.d_model
    for _ in range(cfg.n_layers):
        kv_dim = cfg.n_kv_heads * (cfg.d_model // cfg.n_heads)
        total += _quantized_linear_bytes(cfg.d_model, cfg.d_model) * 2
        total += _quantized_linear_bytes(kv_dim, cfg.d_model) * 2
        total += (16 + 2 * cfg.d_model) * 2  # two LayerNorm weight/bias pairs
        total += _quantized_linear_bytes(cfg.d_ff, cfg.d_model, cfg.d_ff)
        total += _quantized_linear_bytes(cfg.d_model, cfg.d_ff, cfg.d_model)
    total += 16 + 2 * cfg.d_model  # final LayerNorm
    total += 16  # tied lm_head stub
    return total


def dense_ffn_params_per_layer(d_model: int, d_ff: int) -> int:
    return (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)


def moe_ffn_params_per_layer(
    d_model: int,
    n_experts: int,
    expert_d_ff: int,
    shared_d_ff: int,
    add_router_bias: bool,
) -> int:
    expert_params = n_experts * (
        (d_model * expert_d_ff + expert_d_ff) + (expert_d_ff * d_model + d_model)
    )
    shared_params = 0
    if shared_d_ff > 0:
        shared_params = (d_model * shared_d_ff + shared_d_ff) + (shared_d_ff * d_model + d_model)
    router_params = (d_model * n_experts) + (n_experts if add_router_bias else 0)
    return expert_params + shared_params + router_params


def dense_ffn_active_macs_per_token(d_model: int, d_ff: int) -> int:
    return 2 * d_model * d_ff


def moe_ffn_active_macs_per_token(
    d_model: int,
    top_k: int,
    expert_d_ff: int,
    shared_d_ff: int,
    n_experts: int,
) -> int:
    router = d_model * n_experts
    routed = top_k * (2 * d_model * expert_d_ff)
    shared = 2 * d_model * shared_d_ff if shared_d_ff > 0 else 0
    return router + routed + shared


def cardputer_flash_fits(cfg: CardputerConfig) -> Tuple[bool, CardputerSizing]:
    weight_bytes = estimate_moe_bin_bytes(cfg)
    working_bytes = estimate_working_memory_bytes(
        cfg.max_seq_len, cfg.d_model, cfg.vocab_size,
        cfg.n_layers, cfg.n_heads, cfg.n_kv_heads)
    scratch_bytes = estimate_moe_scratch_bytes(
        cfg.moe_n_experts, cfg.d_ff, cfg.d_model, cfg.moe_shared_d_ff, cfg.moe_top_k
    )
    firmware_bytes = estimate_firmware_flash_bytes(cfg, weight_bytes)
    total_heap = working_bytes + CARDPUTER_SYSTEM_RESERVE_BYTES
    weight_budget = cardputer_max_weight_bytes(cfg.vocab_size)

    ok = (
        weight_bytes <= weight_budget
        and working_bytes <= CARDPUTER_WORKING_BUDGET_BYTES
        and scratch_bytes <= CARDPUTER_MOE_SCRATCH_BUDGET_BYTES
        and firmware_bytes <= CARDPUTER_APP_PARTITION_BYTES
    )

    params = estimate_param_count(cfg)
    expert_penalty = max(0, cfg.moe_n_experts - 4) * 50_000
    score = params - expert_penalty

    sizing = CardputerSizing(
        config=cfg,
        weight_bytes=weight_bytes,
        working_bytes=working_bytes,
        moe_scratch_bytes=scratch_bytes,
        total_heap_bytes=total_heap,
        score=score,
        firmware_bytes=firmware_bytes,
    )
    return ok, sizing


def _flash_search_grid(base: CardputerConfig) -> Iterable[CardputerConfig]:
    vocab_choices = (base.vocab_size,) if base.vocab_size != 512 else (512, 1024, 2048)
    d_model_choices = (base.d_model,) if base.d_model != 32 else tuple(range(16, 129, 4))
    for vocab_size in vocab_choices:
        for d_model in d_model_choices:
            for n_heads in (4, 8):
                if d_model % n_heads != 0:
                    continue
                for d_ff in range(d_model, min(d_model * 4, 512) + 1, 8):
                    for layers in range(1, 257):
                        for experts in (4, 6, 8):
                            for shared in (
                                max(16, d_ff // 4),
                                d_ff // 2,
                            ):
                                yield CardputerConfig(
                                    vocab_size=vocab_size,
                                    d_model=d_model,
                                    n_layers=layers,
                                    n_heads=n_heads,
                                    n_kv_heads=base.n_kv_heads,
                                    d_ff=d_ff,
                                    max_seq_len=base.max_seq_len,
                                    moe_n_experts=experts,
                                    moe_top_k=base.moe_top_k,
                                    moe_shared_d_ff=shared,
                                )


def find_cardputer_flash_optimal(
    base: Optional[CardputerConfig] = None,
    max_experts: int = 4,
) -> Tuple[CardputerSizing, List[CardputerSizing]]:
    """Best quality config: maximize parameter count under RAM + flash limits."""
    if base is None:
        base = CardputerConfig()

    candidates: List[CardputerSizing] = []
    for cfg in _flash_search_grid(base):
        ok, sizing = cardputer_flash_fits(cfg)
        if ok:
            candidates.append(sizing)

    if not candidates:
        raise RuntimeError("No flash-backed Cardputer configuration found in search grid.")

    pool = [s for s in candidates if s.config.moe_n_experts <= max_experts] or candidates
    best = max(pool, key=lambda s: (s.score, s.config.n_layers, s.config.d_model))
    candidates.sort(key=lambda s: (-s.score, -s.config.n_layers, -s.config.d_model))
    return best, candidates


def find_cardputer_flash_max_fill(
    base: Optional[CardputerConfig] = None,
) -> Tuple[CardputerSizing, List[CardputerSizing]]:
    """Largest int8 weight footprint that still fits the app partition + RAM."""
    if base is None:
        base = CardputerConfig()

    candidates: List[CardputerSizing] = []
    for cfg in _flash_search_grid(base):
        ok, sizing = cardputer_flash_fits(cfg)
        if ok:
            candidates.append(sizing)

    if not candidates:
        raise RuntimeError("No flash-backed Cardputer configuration found in search grid.")

    best = max(candidates, key=lambda s: (s.weight_bytes, s.score))
    candidates.sort(key=lambda s: (-s.weight_bytes, -s.score))
    return best, candidates


def cardputer_spiffs_fits(cfg: CardputerConfig) -> Tuple[bool, CardputerSizing]:
    weight_bytes = estimate_moe_bin_bytes(cfg)
    working_bytes = estimate_working_memory_bytes(
        cfg.max_seq_len, cfg.d_model, cfg.vocab_size,
        cfg.n_layers, cfg.n_heads, cfg.n_kv_heads)
    scratch_bytes = estimate_moe_scratch_bytes(
        cfg.moe_n_experts, cfg.d_ff, cfg.d_model, cfg.moe_shared_d_ff, cfg.moe_top_k
    )
    total_heap = weight_bytes + working_bytes + CARDPUTER_SYSTEM_RESERVE_BYTES

    ok = (
        weight_bytes <= CARDPUTER_WEIGHT_BUDGET_BYTES
        and working_bytes <= CARDPUTER_WORKING_BUDGET_BYTES
        and scratch_bytes <= CARDPUTER_MOE_SCRATCH_BUDGET_BYTES
        and weight_bytes <= CARDPUTER_SPIFFS_BUDGET_BYTES
        and total_heap <= CARDPUTER_SRAM_BYTES
    )

    expert_penalty = max(0, cfg.moe_n_experts - 4) * 5
    score = cfg.n_layers * 20 + min(cfg.moe_n_experts, 4) * 8 - expert_penalty

    sizing = CardputerSizing(
        config=cfg,
        weight_bytes=weight_bytes,
        working_bytes=working_bytes,
        moe_scratch_bytes=scratch_bytes,
        total_heap_bytes=total_heap,
        score=score,
    )
    return ok, sizing


def find_cardputer_optimal(
    base: CardputerConfig,
    max_layers: int = 8,
    max_experts: int = 8,
    max_experts_recommended: int = 4,
) -> Tuple[CardputerSizing, List[CardputerSizing]]:
    """Return best SPIFFS/RAM-feasible config (weights loaded into heap)."""
    feasible: List[CardputerSizing] = []
    for layers in range(1, max_layers + 1):
        for experts in range(2, max_experts + 1):
            cfg = CardputerConfig(
                vocab_size=base.vocab_size,
                d_model=base.d_model,
                n_layers=layers,
                n_heads=base.n_heads,
                n_kv_heads=base.n_kv_heads,
                d_ff=base.d_ff,
                max_seq_len=base.max_seq_len,
                moe_n_experts=experts,
                moe_top_k=base.moe_top_k,
                moe_shared_d_ff=base.moe_shared_d_ff,
            )
            ok, sizing = cardputer_spiffs_fits(cfg)
            if ok:
                feasible.append(sizing)

    if not feasible:
        raise RuntimeError("No feasible Cardputer MoE configuration found in search grid.")

    recommended_pool = [
        s for s in feasible if s.config.moe_n_experts <= max_experts_recommended
    ] or feasible
    best = max(recommended_pool, key=lambda s: (s.score, s.config.n_layers, s.config.moe_n_experts))
    feasible.sort(key=lambda s: (-s.score, -s.config.n_layers, -s.config.moe_n_experts))
    return best, feasible


def cardputer_fits(cfg: CardputerConfig) -> Tuple[bool, CardputerSizing]:
    """Default fit check: flash-backed deployment (weights not copied to RAM)."""
    return cardputer_flash_fits(cfg)


def fmt_bytes(n: int) -> str:
    kib = n / 1024
    mib = kib / 1024
    return f"{n} B ({kib:.1f} KiB, {mib:.3f} MiB)"


def print_cardputer_report(
    best: CardputerSizing,
    candidates: Iterable[CardputerSizing],
    *,
    flash_backed: bool = True,
) -> None:
    cfg = best.config
    print("== Cardputer MoE Sizing (ESP32-S3, no PSRAM) ==")
    if flash_backed:
        print(
            f"Deployment: PROGMEM / firmware flash (weights not copied to RAM at load)"
        )
        print(
            f"Budgets: app partition={CARDPUTER_APP_PARTITION_BYTES // 1024} KiB, "
            f"code≈{CARDPUTER_FIRMWARE_CODE_BYTES // 1024} KiB, "
            f"max weights≈{cardputer_max_weight_bytes(cfg.vocab_size) // 1024} KiB (vocab={cfg.vocab_size}), "
            f"working RAM<={CARDPUTER_WORKING_BUDGET_BYTES // 1024} KiB"
        )
    else:
        print(
            f"Budgets: weights<={CARDPUTER_WEIGHT_BUDGET_BYTES // 1024} KiB, "
            f"working<={CARDPUTER_WORKING_BUDGET_BYTES // 1024} KiB, "
            f"total heap<={CARDPUTER_SRAM_BYTES // 1024} KiB "
            f"(includes {CARDPUTER_SYSTEM_RESERVE_BYTES // 1024} KiB system reserve)"
        )
    print()
    print("Recommended training defaults:")
    print(f"  N_LAYERS={cfg.n_layers}")
    print(f"  N_HEADS={cfg.n_heads}")
    print(f"  N_KV_HEADS={cfg.n_kv_heads}")
    print(f"  MOE_N_EXPERTS={cfg.moe_n_experts}")
    print(f"  MOE_TOP_K={cfg.moe_top_k}")
    print(f"  MOE_SHARED_D_FF={cfg.moe_shared_d_ff}")
    print(f"  D_MODEL={cfg.d_model}  D_FF={cfg.d_ff}  BLOCK_SIZE={cfg.max_seq_len}  VOCAB_SIZE={cfg.vocab_size}")
    print()
    print("Estimated runtime footprint:")
    weight_label = "int8 weights (PROGMEM flash)" if flash_backed else "int8 weights (SPIFFS -> RAM)"
    print(f"  {weight_label}: {fmt_bytes(best.weight_bytes)}")
    print(f"  inference working buffers:    {fmt_bytes(best.working_bytes)}")
    print(f"  MoE scratch (per layer):      {fmt_bytes(best.moe_scratch_bytes)}")
    if flash_backed and best.firmware_bytes:
        print(
            f"  est. firmware total:          {fmt_bytes(best.firmware_bytes)} "
            f"({100 * best.firmware_bytes / CARDPUTER_APP_PARTITION_BYTES:.1f}% of app partition)"
        )
        print(f"  est. parameter count:         {estimate_param_count(cfg):,}")
    if not flash_backed:
        print(f"  total heap (weights+working+system): {fmt_bytes(best.total_heap_bytes)}")
    print()
    print("Top feasible configs:")
    for sizing in list(candidates)[:8]:
        c = sizing.config
        print(
            f"  d={c.d_model} L={c.n_layers} E={c.moe_n_experts} V={c.vocab_size}  "
            f"weights={sizing.weight_bytes / 1024:.1f} KiB  "
            f"working={sizing.working_bytes / 1024:.1f} KiB  "
            f"params≈{estimate_param_count(c):,}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate dense vs MoE tradeoffs for NanoLLM")
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--n-kv-heads", type=int, default=1,
                   help="Key/value heads (1 enables MQA)")
    p.add_argument("--d-ff", type=int, default=64, help="Routed expert FFN hidden size")
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=256)

    p.add_argument("--moe-experts", type=int, default=4)
    p.add_argument("--moe-top-k", type=int, default=1)
    p.add_argument("--moe-shared-d-ff", type=int, default=16)
    p.add_argument("--router-no-bias", action="store_true")
    p.add_argument(
        "--dense-model",
        action="store_true",
        help="Size the explicit dense model instead of the MoE binary",
    )

    p.add_argument("--sram-budget-kb", type=int, default=200)
    p.add_argument(
        "--cardputer-optimal",
        action="store_true",
        help="Search for best-quality MoE config (max params) with PROGMEM flash weights",
    )
    p.add_argument(
        "--cardputer-max-flash",
        action="store_true",
        help="Search for largest weight footprint that fills the app flash partition",
    )
    p.add_argument(
        "--spiffs-optimal",
        action="store_true",
        help="Search for optimal config when weights are loaded from SPIFFS into RAM",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="With sizing search flags, print recommended config as JSON",
    )

    args = p.parse_args()

    if args.cardputer_optimal or args.cardputer_max_flash or args.spiffs_optimal:
        flash_backed = not args.spiffs_optimal
        base = CardputerConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            moe_top_k=args.moe_top_k,
            moe_shared_d_ff=args.moe_shared_d_ff,
        )
        if flash_backed:
            if args.cardputer_max_flash:
                best, candidates = find_cardputer_flash_max_fill(base)
            else:
                best, candidates = find_cardputer_flash_optimal(base)
        else:
            best, candidates = find_cardputer_optimal(base)
        if args.json:
            c = best.config
            print(
                json.dumps(
                    {
                        "n_layers": c.n_layers,
                        "n_heads": c.n_heads,
                        "n_kv_heads": c.n_kv_heads,
                        "moe_n_experts": c.moe_n_experts,
                        "moe_top_k": c.moe_top_k,
                        "moe_shared_d_ff": c.moe_shared_d_ff,
                        "d_model": c.d_model,
                        "d_ff": c.d_ff,
                        "max_seq_len": c.max_seq_len,
                        "vocab_size": c.vocab_size,
                        "weight_bytes": best.weight_bytes,
                        "working_bytes": best.working_bytes,
                        "firmware_bytes": best.firmware_bytes,
                        "param_count": estimate_param_count(c),
                        "total_heap_bytes": best.total_heap_bytes,
                        "flash_backed": flash_backed,
                        "app_partition_bytes": CARDPUTER_APP_PARTITION_BYTES,
                    },
                    indent=2,
                )
            )
        else:
            print_cardputer_report(best, candidates, flash_backed=flash_backed)
        return

    dense_layer_params = dense_ffn_params_per_layer(args.d_model, args.d_ff)
    moe_layer_params = moe_ffn_params_per_layer(
        d_model=args.d_model,
        n_experts=args.moe_experts,
        expert_d_ff=args.d_ff,
        shared_d_ff=args.moe_shared_d_ff,
        add_router_bias=not args.router_no_bias,
    )

    dense_total_ffn_params = dense_layer_params * args.n_layers
    moe_total_ffn_params = moe_layer_params * args.n_layers

    dense_macs = dense_ffn_active_macs_per_token(args.d_model, args.d_ff)
    moe_macs = moe_ffn_active_macs_per_token(
        d_model=args.d_model,
        top_k=args.moe_top_k,
        expert_d_ff=args.d_ff,
        shared_d_ff=args.moe_shared_d_ff,
        n_experts=args.moe_experts,
    )

    work_mem = estimate_working_memory_bytes(
        args.max_seq_len, args.d_model, args.vocab_size,
        args.n_layers, args.n_heads, args.n_kv_heads)
    budget_bytes = args.sram_budget_kb * 1024

    cfg = CardputerConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        moe_n_experts=args.moe_experts,
        moe_top_k=args.moe_top_k,
        moe_shared_d_ff=args.moe_shared_d_ff,
    )
    if args.dense_model:
        weight_bytes = estimate_dense_bin_bytes(cfg)
        firmware_bytes = estimate_firmware_flash_bytes(cfg, weight_bytes)
        ok = (
            work_mem <= budget_bytes
            and weight_bytes <= cardputer_max_weight_bytes(cfg.vocab_size)
            and firmware_bytes <= CARDPUTER_APP_PARTITION_BYTES
        )
        sizing = CardputerSizing(
            config=cfg,
            weight_bytes=weight_bytes,
            working_bytes=work_mem,
            moe_scratch_bytes=0,
            total_heap_bytes=work_mem + CARDPUTER_SYSTEM_RESERVE_BYTES,
            score=estimate_dense_param_count(cfg),
            firmware_bytes=firmware_bytes,
        )
    else:
        weight_bytes = estimate_moe_bin_bytes(cfg)
        ok, sizing = cardputer_fits(cfg)

    print("== NanoLLM Dense vs MoE Estimator ==")
    print(
        f"Model: vocab={args.vocab_size}, d_model={args.d_model}, "
        f"layers={args.n_layers}, seq={args.max_seq_len}"
    )
    print(f"Working activation memory estimate: {fmt_bytes(work_mem)}")
    print(f"SRAM budget: {fmt_bytes(budget_bytes)}")
    print(f"Within working budget: {'yes' if work_mem <= budget_bytes else 'no'}")
    print(f"Estimated int8 weight file: {fmt_bytes(weight_bytes)}")
    print(
        f"Estimated parameter count: "
        f"{estimate_dense_param_count(cfg) if args.dense_model else estimate_param_count(cfg):,}"
    )
    print(f"Cardputer feasible (flash-backed): {'yes' if ok else 'no'}")
    if ok:
        print(f"Cardputer working RAM estimate: {fmt_bytes(sizing.working_bytes)}")
        print(f"Cardputer flash weight budget: {fmt_bytes(cardputer_max_weight_bytes(cfg.vocab_size))}")
        if sizing.firmware_bytes:
            print(f"Cardputer est. firmware total: {fmt_bytes(sizing.firmware_bytes)}")
    print()

    print("Dense FFN")
    print(f"- d_ff: {args.d_ff}")
    print(f"- params per layer: {dense_layer_params:,}")
    print(f"- total FFN params: {dense_total_ffn_params:,}")
    print(f"- active FFN MACs/token/layer: {dense_macs:,}")
    print()

    print("MoE FFN")
    print(f"- experts: {args.moe_experts}")
    print(f"- top_k: {args.moe_top_k}")
    print(f"- expert_d_ff: {args.d_ff}")
    print(f"- shared_d_ff: {args.moe_shared_d_ff}")
    print(f"- params per layer: {moe_layer_params:,}")
    print(f"- total FFN params: {moe_total_ffn_params:,}")
    print(f"- active FFN MACs/token/layer: {moe_macs:,}")
    print()

    if dense_total_ffn_params > 0:
        cap_ratio = moe_total_ffn_params / dense_total_ffn_params
    else:
        cap_ratio = 0.0
    if dense_macs > 0:
        mac_ratio = moe_macs / dense_macs
    else:
        mac_ratio = 0.0

    print("Comparison")
    print(f"- FFN parameter capacity ratio (MoE/Dense): {cap_ratio:.2f}x")
    print(f"- Active FFN compute ratio (MoE/Dense): {mac_ratio:.2f}x")


if __name__ == "__main__":
    main()
