"""
Load model.bin and run inference with the same int8 math as C++/ESP32.

This is the Python reference runtime for cross-stack parity checks.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Optional

import numpy as np


@dataclass
class ModelConfig:
    vocab_size: int = 0
    d_model: int = 0
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    d_ff: int = 0
    max_seq_len: int = 0
    quantized: bool = True
    use_moe: bool = False
    moe_n_experts: int = 0
    moe_top_k: int = 1
    moe_shared_d_ff: int = 0
    use_rope: bool = False


@dataclass
class QuantizedLayer:
    weight: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    weight_scale: float = 1.0
    bias: Optional[np.ndarray] = None
    bias_scale: float = 1.0


@dataclass
class QuantizedBlock:
    attn_q: QuantizedLayer = field(default_factory=QuantizedLayer)
    attn_k: QuantizedLayer = field(default_factory=QuantizedLayer)
    attn_v: QuantizedLayer = field(default_factory=QuantizedLayer)
    attn_o: QuantizedLayer = field(default_factory=QuantizedLayer)
    norm1_weight: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    norm1_bias: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    norm1_weight_scale: float = 1.0
    norm1_bias_scale: float = 1.0
    norm2_weight: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    norm2_bias: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    norm2_weight_scale: float = 1.0
    norm2_bias_scale: float = 1.0
    ff1: Optional[QuantizedLayer] = None
    ff2: Optional[QuantizedLayer] = None
    use_moe: bool = False
    moe_n_experts: int = 0
    moe_top_k: int = 1
    moe_expert_d_ff: int = 0
    moe_has_shared: bool = False
    moe_router: Optional[QuantizedLayer] = None
    moe_ff1: List[QuantizedLayer] = field(default_factory=list)
    moe_ff2: List[QuantizedLayer] = field(default_factory=list)
    moe_shared_ff1: Optional[QuantizedLayer] = None
    moe_shared_ff2: Optional[QuantizedLayer] = None


class QuantizedNanoLLM:
    """Python mirror of cpp/model.cpp int8 inference."""

    def __init__(self):
        self.config = ModelConfig()
        self.token_embedding = np.array([], dtype=np.int8)
        self.pos_embedding = np.array([], dtype=np.int8)
        self.token_embedding_scale = 1.0
        self.pos_embedding_scale = 1.0
        self.blocks: List[QuantizedBlock] = []
        self.norm_weight = np.array([], dtype=np.int8)
        self.norm_bias = np.array([], dtype=np.int8)
        self.norm_weight_scale = 1.0
        self.norm_bias_scale = 1.0
        self.lm_head = np.array([], dtype=np.int8)
        self.lm_head_scale = 1.0
        self.rope_inv_freq: Optional[np.ndarray] = None
        self.reset_cache()

    @classmethod
    def load(cls, weights_path: str | Path, config_path: str | Path) -> "QuantizedNanoLLM":
        model = cls()
        model.config = _load_config(config_path)
        with open(weights_path, "rb") as f:
            _load_weights(model, f)
        if model.config.use_rope:
            d_k = model.config.d_model // model.config.n_heads
            half = d_k // 2
            model.rope_inv_freq = np.array(
                [1.0 / (10000 ** (i / half)) for i in range(half)],
                dtype=np.float32,
            )
        if not model.config.quantized:
            raise ValueError("Only quantized model.bin is supported by QuantizedNanoLLM")
        return model

    def reset_cache(self) -> None:
        """Clear the bounded per-layer int8 K/V decode cache."""
        self.cache_len = 0
        self._cache_history: List[int] = []
        self.k_cache: List[np.ndarray] = []
        self.v_cache: List[np.ndarray] = []
        self.k_cache_scales: List[np.ndarray] = []
        self.v_cache_scales: List[np.ndarray] = []

    resetCache = reset_cache

    def _ensure_cache(self) -> None:
        cfg = self.config
        kv_dim = (cfg.d_model // cfg.n_heads) * cfg.n_kv_heads
        if len(self.k_cache) == cfg.n_layers:
            return
        shape = (cfg.max_seq_len, kv_dim)
        self.k_cache = [np.zeros(shape, dtype=np.int8) for _ in range(cfg.n_layers)]
        self.v_cache = [np.zeros(shape, dtype=np.int8) for _ in range(cfg.n_layers)]
        self.k_cache_scales = [np.ones(cfg.max_seq_len, dtype=np.float32) for _ in range(cfg.n_layers)]
        self.v_cache_scales = [np.ones(cfg.max_seq_len, dtype=np.float32) for _ in range(cfg.n_layers)]

    def decode_step(self, token: int) -> np.ndarray:
        """Decode one token without allocating full-sequence activation scratch."""
        cfg = self.config
        self._ensure_cache()
        if self.cache_len >= cfg.max_seq_len:
            replay = self._cache_history[1:] + [token]
            self.reset_cache()
            logits = None
            for replay_token in replay:
                logits = self.decode_step(replay_token)
            assert logits is not None
            return logits
        pos = self.cache_len
        token = token if 0 <= token < cfg.vocab_size else 0
        self._cache_history.append(token)
        x = (self.token_embedding.reshape(-1, cfg.d_model)[token].astype(np.float32)
             / self.token_embedding_scale)
        if not cfg.use_rope:
            x += self.pos_embedding.reshape(-1, cfg.d_model)[pos].astype(np.float32) / self.pos_embedding_scale

        d_k = cfg.d_model // cfg.n_heads
        repeats = cfg.n_heads // cfg.n_kv_heads
        for layer_idx, block in enumerate(self.blocks):
            normed = np.empty(cfg.d_model, dtype=np.float32)
            _layer_norm(x, normed, block.norm1_weight, block.norm1_weight_scale,
                        block.norm1_bias, block.norm1_bias_scale)
            q = np.empty(cfg.d_model, dtype=np.float32)
            k = np.empty(d_k * cfg.n_kv_heads, dtype=np.float32)
            v = np.empty_like(k)
            _linear_layer(block.attn_q, normed, q)
            _linear_layer(block.attn_k, normed, k)
            _linear_layer(block.attn_v, normed, v)
            if cfg.use_rope and self.rope_inv_freq is not None:
                _rope_apply_heads(q, cfg.n_heads, d_k, pos, self.rope_inv_freq)
                _rope_apply_heads(k, cfg.n_kv_heads, d_k, pos, self.rope_inv_freq)
            kq, ks = _quantize_cache_vector(k)
            vq, vs = _quantize_cache_vector(v)
            self.k_cache[layer_idx][pos] = kq
            self.v_cache[layer_idx][pos] = vq
            self.k_cache_scales[layer_idx][pos] = ks
            self.v_cache_scales[layer_idx][pos] = vs

            values = np.zeros(cfg.d_model, dtype=np.float32)
            scores = np.empty(pos + 1, dtype=np.float32)
            for head in range(cfg.n_heads):
                kv_head = head // repeats
                q_slice = q[head * d_k:(head + 1) * d_k]
                kv_slice = slice(kv_head * d_k, (kv_head + 1) * d_k)
                for t in range(pos + 1):
                    key = self.k_cache[layer_idx][t, kv_slice].astype(np.float32)
                    key /= self.k_cache_scales[layer_idx][t]
                    scores[t] = np.dot(q_slice, key) / math.sqrt(d_k)
                _softmax_inplace(scores)
                out = values[head * d_k:(head + 1) * d_k]
                for t in range(pos + 1):
                    val = self.v_cache[layer_idx][t, kv_slice].astype(np.float32)
                    out += scores[t] * val / self.v_cache_scales[layer_idx][t]
            projected = np.empty(cfg.d_model, dtype=np.float32)
            _linear_layer(block.attn_o, values, projected)
            x += projected
            _layer_norm(x, normed, block.norm2_weight, block.norm2_weight_scale,
                        block.norm2_bias, block.norm2_bias_scale)
            ff = np.empty(cfg.d_model, dtype=np.float32)
            _feed_forward(normed, ff, block, cfg, 1)
            x += ff

        final = np.empty(cfg.d_model, dtype=np.float32)
        _layer_norm(x, final, self.norm_weight, self.norm_weight_scale,
                    self.norm_bias, self.norm_bias_scale)
        logits = np.empty(cfg.vocab_size, dtype=np.float32)
        _linear(self.lm_head, self.lm_head_scale, final, logits)
        self.cache_len += 1
        return logits

    def _next_token_logits(self, prompt: List[int]) -> np.ndarray:
        cfg = self.config
        context = prompt[-cfg.max_seq_len :] or [0]
        seq_len = len(context)

        hidden = np.zeros(seq_len * cfg.d_model, dtype=np.float32)
        temp1 = np.zeros_like(hidden)
        temp2 = np.zeros_like(hidden)

        for i in range(seq_len):
            token = context[i]
            if token < 0 or token >= cfg.vocab_size:
                token = 0
            pos = i
            base = i * cfg.d_model
            hidden[base : base + cfg.d_model] = (
                self.token_embedding[token * cfg.d_model : (token + 1) * cfg.d_model].astype(np.float32)
                / self.token_embedding_scale
            )
            if not cfg.use_rope:
                hidden[base : base + cfg.d_model] += (
                    self.pos_embedding[pos * cfg.d_model : (pos + 1) * cfg.d_model].astype(np.float32)
                    / self.pos_embedding_scale
                )

        for block_idx in range(cfg.n_layers):
            block = self.blocks[block_idx]
            for i in range(seq_len):
                _layer_norm(
                    hidden[i * cfg.d_model : (i + 1) * cfg.d_model],
                    temp1[i * cfg.d_model : (i + 1) * cfg.d_model],
                    block.norm1_weight,
                    block.norm1_weight_scale,
                    block.norm1_bias,
                    block.norm1_bias_scale,
                )
            _attention(temp1, temp2, block, cfg, seq_len, self.rope_inv_freq)
            hidden[: seq_len * cfg.d_model] += temp2[: seq_len * cfg.d_model]

            for i in range(seq_len):
                _layer_norm(
                    hidden[i * cfg.d_model : (i + 1) * cfg.d_model],
                    temp1[i * cfg.d_model : (i + 1) * cfg.d_model],
                    block.norm2_weight,
                    block.norm2_weight_scale,
                    block.norm2_bias,
                    block.norm2_bias_scale,
                )
            _feed_forward(temp1, temp2, block, cfg, seq_len)
            hidden[: seq_len * cfg.d_model] += temp2[: seq_len * cfg.d_model]

        for i in range(seq_len):
            _layer_norm(
                hidden[i * cfg.d_model : (i + 1) * cfg.d_model],
                temp1[i * cfg.d_model : (i + 1) * cfg.d_model],
                self.norm_weight,
                self.norm_weight_scale,
                self.norm_bias,
                self.norm_bias_scale,
            )

        logits = np.zeros(cfg.vocab_size, dtype=np.float32)
        _linear(
            self.lm_head,
            self.lm_head_scale,
            temp1[(seq_len - 1) * cfg.d_model : seq_len * cfg.d_model],
            logits,
        )
        return logits

    def generate(self, prompt: List[int], max_new_tokens: int = 1, temperature: float = 1.0) -> List[int]:
        generated = list(prompt) or [0]
        if self.config.n_kv_heads == self.config.n_heads:
            for _ in range(max_new_tokens):
                logits = self._next_token_logits(generated)
                if temperature != 0:
                    logits /= temperature
                _softmax_inplace(logits)
                next_token = int(np.argmax(logits))
                generated.append(next_token)
                if next_token == 3:
                    break
            return generated
        self.reset_cache()
        logits = None
        for token in generated[-self.config.max_seq_len:]:
            logits = self.decode_step(token)
        for _ in range(max_new_tokens):
            scaled = logits.copy()
            if temperature != 0:
                scaled /= temperature
            _softmax_inplace(scaled)
            next_token = int(np.argmax(scaled))
            generated.append(next_token)
            logits = self.decode_step(next_token)
            if next_token == 3:  # <EOS>, fixed by the tokenizer artifact contract
                break
        return generated

    def next_token(self, prompt: List[int]) -> int:
        generated = self.generate(prompt, max_new_tokens=1, temperature=1.0)
        prefix_len = max(1, len(prompt))
        if len(generated) <= prefix_len:
            return -1
        return generated[prefix_len]


def _load_config(config_path: str | Path) -> ModelConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelConfig(
        vocab_size=int(data["vocab_size"]),
        d_model=int(data["d_model"]),
        n_layers=int(data["n_layers"]),
        n_heads=int(data["n_heads"]),
        n_kv_heads=int(data.get("n_kv_heads") or data["n_heads"]),
        d_ff=int(data["d_ff"]),
        max_seq_len=int(data["max_seq_len"]),
        quantized=bool(data.get("quantized", True)),
        use_moe=bool(data.get("use_moe", False)),
        moe_n_experts=int(data.get("moe_n_experts", 0)),
        moe_top_k=int(data.get("moe_top_k", 1)),
        moe_shared_d_ff=int(data.get("moe_shared_d_ff", 0)),
        use_rope=bool(data.get("use_rope", False)),
    )


def _read_quantized_layer(f: BinaryIO) -> QuantizedLayer:
    scale = struct.unpack("f", f.read(4))[0]
    rows, cols = struct.unpack("II", f.read(8))
    weight = np.frombuffer(f.read(rows * cols), dtype=np.int8).reshape(rows, cols).copy()
    return QuantizedLayer(weight=weight, weight_scale=scale)


def _read_quantized_bias(f: BinaryIO) -> tuple[np.ndarray, float]:
    scale = struct.unpack("f", f.read(4))[0]
    size = struct.unpack("I", f.read(4))[0]
    if size == 0:
        return np.array([], dtype=np.int8), scale
    bias = np.frombuffer(f.read(size), dtype=np.int8).copy()
    return bias, scale


def _skip_bias_stub(f: BinaryIO) -> None:
    size = struct.unpack("I", f.read(4))[0]
    if size > 0:
        f.read(size)


def _read_moe_linear(f: BinaryIO) -> QuantizedLayer:
    layer = _read_quantized_layer(f)
    bias_scale = struct.unpack("f", f.read(4))[0]
    size = struct.unpack("I", f.read(4))[0]
    if size > 0:
        layer.bias = np.frombuffer(f.read(size), dtype=np.int8).copy()
        layer.bias_scale = bias_scale
    return layer


def _load_weights(model: QuantizedNanoLLM, f: BinaryIO) -> None:
    magic = f.read(4)
    is_moe = magic == b"NLMO"
    is_versioned_dense = magic == b"NLMD"
    if is_moe or is_versioned_dense:
        version = struct.unpack("I", f.read(4))[0]
        quantized_flag = struct.unpack("?", f.read(1))[0]
        f.read(3)
        n_layers = struct.unpack("I", f.read(4))[0]
        if version == 2:
            model.config.n_kv_heads = struct.unpack("I", f.read(4))[0]
        elif version != 1 or is_versioned_dense:
            kind = "MoE" if is_moe else "dense"
            raise ValueError(f"Unsupported {kind} format version: {version}")
        if not quantized_flag:
            raise ValueError("Expected quantized weights")
    else:
        f.seek(0)
        quantized_flag = struct.unpack("?", f.read(1))[0]
        f.read(3)
        n_layers = struct.unpack("I", f.read(4))[0]
        if not quantized_flag:
            raise ValueError("Expected quantized weights")

    model.config.n_layers = int(n_layers)

    tok = _read_quantized_layer(f)
    model.token_embedding = tok.weight.reshape(-1)
    model.token_embedding_scale = tok.weight_scale

    pos = _read_quantized_layer(f)
    if not model.config.use_rope:
        model.pos_embedding = pos.weight.reshape(-1)
        model.pos_embedding_scale = pos.weight_scale

    model.blocks = []
    for _ in range(n_layers):
        block = QuantizedBlock()
        block.attn_q = _read_quantized_layer(f)
        _skip_bias_stub(f)
        block.attn_k = _read_quantized_layer(f)
        _skip_bias_stub(f)
        block.attn_v = _read_quantized_layer(f)
        _skip_bias_stub(f)
        block.attn_o = _read_quantized_layer(f)
        _skip_bias_stub(f)

        block.norm1_weight, block.norm1_weight_scale = _read_quantized_bias(f)
        block.norm1_bias, block.norm1_bias_scale = _read_quantized_bias(f)
        block.norm2_weight, block.norm2_weight_scale = _read_quantized_bias(f)
        block.norm2_bias, block.norm2_bias_scale = _read_quantized_bias(f)

        if not is_moe:
            block.ff1 = _read_quantized_layer(f)
            block.ff1.bias, block.ff1.bias_scale = _read_quantized_bias(f)
            block.ff2 = _read_quantized_layer(f)
            block.ff2.bias, block.ff2.bias_scale = _read_quantized_bias(f)
        else:
            header = f.read(13)
            n_experts, top_k, expert_d_ff, has_shared = struct.unpack("III?", header[:13])
            block.use_moe = True
            block.moe_n_experts = int(n_experts)
            block.moe_top_k = int(top_k)
            block.moe_expert_d_ff = int(expert_d_ff)
            block.moe_has_shared = bool(has_shared)
            block.moe_router = _read_moe_linear(f)
            block.moe_ff1 = []
            block.moe_ff2 = []
            for _expert in range(block.moe_n_experts):
                block.moe_ff1.append(_read_moe_linear(f))
                block.moe_ff2.append(_read_moe_linear(f))
            if block.moe_has_shared:
                block.moe_shared_ff1 = _read_moe_linear(f)
                block.moe_shared_ff2 = _read_moe_linear(f)

        model.blocks.append(block)

    model.norm_weight, model.norm_weight_scale = _read_quantized_bias(f)
    model.norm_bias, model.norm_bias_scale = _read_quantized_bias(f)
    lm = _read_quantized_layer(f)
    if lm.weight.size == 0:
        model.lm_head = model.token_embedding
        model.lm_head_scale = model.token_embedding_scale
    else:
        model.lm_head = lm.weight
        model.lm_head_scale = lm.weight_scale
    _skip_bias_stub(f)


def _linear(
    weight: np.ndarray,
    weight_scale: float,
    input_vec: np.ndarray,
    output: np.ndarray,
    bias: Optional[np.ndarray] = None,
    bias_scale: float = 1.0,
) -> None:
    w_scale = weight_scale if abs(weight_scale) > 1e-9 else 1.0
    b_scale = bias_scale if bias is not None and abs(bias_scale) > 1e-9 else 1.0
    weight_2d = weight if weight.ndim == 2 else weight.reshape(-1, input_vec.size)
    out_dim, in_dim = weight_2d.shape
    for i in range(out_dim):
        s = 0.0
        row = weight_2d[i]
        for j in range(in_dim):
            s += float(input_vec[j]) * (float(row[j]) / w_scale)
        output[i] = s
        if bias is not None and bias.size > 0:
            output[i] += float(bias[i]) / b_scale


def _linear_layer(layer: QuantizedLayer, input_vec: np.ndarray, output: np.ndarray) -> None:
    _linear(layer.weight, layer.weight_scale, input_vec, output, layer.bias, layer.bias_scale)


def _layer_norm(
    input_vec: np.ndarray,
    output: np.ndarray,
    weight: np.ndarray,
    weight_scale: float,
    bias: np.ndarray,
    bias_scale: float,
) -> None:
    size = input_vec.size
    mean = float(np.sum(input_vec)) / size
    variance = float(np.sum((input_vec - mean) ** 2)) / size
    std_dev = math.sqrt(variance + 1e-5)
    w_scale = weight_scale if abs(weight_scale) > 1e-9 else 1.0
    b_scale = bias_scale if bias.size > 0 and abs(bias_scale) > 1e-9 else 1.0
    for i in range(size):
        gamma = float(weight[i]) / w_scale if weight.size > 0 else 1.0
        beta = float(bias[i]) / b_scale if bias.size > 0 else 0.0
        output[i] = ((float(input_vec[i]) - mean) / std_dev) * gamma + beta


def _gelu(x: np.ndarray) -> None:
    inv_sqrt_2 = 0.7071067811865475
    for i in range(x.size):
        xv = float(x[i])
        x[i] = 0.5 * xv * (1.0 + math.erf(xv * inv_sqrt_2))


def _softmax_inplace(x: np.ndarray) -> None:
    max_val = float(np.max(x))
    exp_vals = np.exp(x - max_val)
    total = float(np.sum(exp_vals))
    if total > 0:
        x[:] = exp_vals / total


def _quantize_cache_vector(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 cache quantization; scale converts q back with q / scale."""
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    scale = 127.0 / max_abs if max_abs > 0 else 1.0
    return np.clip(np.rint(x * scale), -127, 127).astype(np.int8), scale


def _rope_apply_head(vec: np.ndarray, d_k: int, position: int, inv_freq: np.ndarray) -> None:
    half = d_k // 2
    for i in range(half):
        angle = float(position) * float(inv_freq[i])
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x1 = vec[2 * i]
        x2 = vec[2 * i + 1]
        vec[2 * i] = x1 * cos_a - x2 * sin_a
        vec[2 * i + 1] = x1 * sin_a + x2 * cos_a


def _rope_apply_heads(vec: np.ndarray, n_heads: int, d_k: int, position: int, inv_freq: np.ndarray) -> None:
    for head in range(n_heads):
        _rope_apply_head(vec[head * d_k:(head + 1) * d_k], d_k, position, inv_freq)


def _attention(
    x: np.ndarray,
    output: np.ndarray,
    block: QuantizedBlock,
    cfg: ModelConfig,
    seq_len: int,
    rope_inv_freq: Optional[np.ndarray] = None,
) -> None:
    n_heads = cfg.n_heads if cfg.n_heads > 0 else 1
    if cfg.d_model % n_heads != 0:
        n_heads = 1
    d_k = cfg.d_model // n_heads
    n_kv_heads = cfg.n_kv_heads or n_heads
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_kv_heads must divide n_heads")
    kv_dim = n_kv_heads * d_k
    kv_repeats = n_heads // n_kv_heads
    inv_sqrt_dk = 1.0 / math.sqrt(float(d_k))

    q = np.zeros(seq_len * cfg.d_model, dtype=np.float32)
    k = np.zeros(seq_len * kv_dim, dtype=np.float32)
    v = np.zeros_like(k)

    for token in range(seq_len):
        inp = x[token * cfg.d_model : (token + 1) * cfg.d_model]
        _linear_layer(block.attn_q, inp, q[token * cfg.d_model : (token + 1) * cfg.d_model])
        _linear_layer(block.attn_k, inp, k[token * kv_dim : (token + 1) * kv_dim])
        _linear_layer(block.attn_v, inp, v[token * kv_dim : (token + 1) * kv_dim])
        if cfg.use_rope and rope_inv_freq is not None:
            _rope_apply_heads(k[token * kv_dim : (token + 1) * kv_dim], n_kv_heads, d_k, token, rope_inv_freq)

    attn_values = np.zeros(seq_len * cfg.d_model, dtype=np.float32)
    row_scores = np.zeros(seq_len, dtype=np.float32)
    for head in range(n_heads):
        head_offset = head * d_k
        kv_head_offset = (head // kv_repeats) * d_k
        for i in range(seq_len):
            q_base = i * cfg.d_model + head_offset
            if cfg.use_rope and rope_inv_freq is not None:
                _rope_apply_head(q[q_base:q_base + d_k], d_k, i, rope_inv_freq)
            for j in range(seq_len):
                if j > i:
                    row_scores[j] = -np.inf
                    continue
                k_base = j * kv_dim + kv_head_offset
                score = 0.0
                for dim in range(d_k):
                    score += q[q_base + dim] * k[k_base + dim]
                row_scores[j] = score * inv_sqrt_dk
            _softmax_inplace(row_scores)

            out_base = i * cfg.d_model + head_offset
            for dim in range(d_k):
                weighted = 0.0
                for t in range(seq_len):
                    weighted += row_scores[t] * v[t * kv_dim + kv_head_offset + dim]
                attn_values[out_base + dim] = weighted

    for token in range(seq_len):
        _linear_layer(
            block.attn_o,
            attn_values[token * cfg.d_model : (token + 1) * cfg.d_model],
            output[token * cfg.d_model : (token + 1) * cfg.d_model],
        )


def _feed_forward(x: np.ndarray, output: np.ndarray, block: QuantizedBlock, cfg: ModelConfig, seq_len: int) -> None:
    if not block.use_moe:
        for token in range(seq_len):
            inp = x[token * cfg.d_model : (token + 1) * cfg.d_model]
            out = output[token * cfg.d_model : (token + 1) * cfg.d_model]
            ff1_out = block.ff1.weight.shape[0]
            ff_hidden = np.zeros(ff1_out, dtype=np.float32)
            _linear_layer(block.ff1, inp, ff_hidden)
            _gelu(ff_hidden)
            _linear_layer(block.ff2, ff_hidden, out)
        return

    n_experts = max(1, block.moe_n_experts)
    top_k = min(max(1, block.moe_top_k), n_experts)
    expert_d_ff = max(1, block.moe_expert_d_ff)
    router_logits = np.zeros(n_experts, dtype=np.float32)
    expert_hidden = np.zeros(expert_d_ff, dtype=np.float32)
    expert_output = np.zeros(cfg.d_model, dtype=np.float32)
    shared_hidden = np.zeros(
        block.moe_shared_ff1.weight.shape[0] if block.moe_shared_ff1 is not None else expert_d_ff,
        dtype=np.float32,
    )

    for token in range(seq_len):
        inp = x[token * cfg.d_model : (token + 1) * cfg.d_model]
        out = output[token * cfg.d_model : (token + 1) * cfg.d_model]
        out[:] = 0.0

        _linear_layer(block.moe_router, inp, router_logits)
        order = sorted(range(n_experts), key=lambda idx: (-router_logits[idx], idx))[:top_k]
        topk_logits = router_logits[order]
        if top_k == 1:
            denom = float(np.sum(np.exp(router_logits - float(topk_logits[0]))))
            gates = np.array([1.0 / denom if denom > 0 else 1.0], dtype=np.float32)
        else:
            max_logit = float(np.max(topk_logits))
            gates = np.exp(topk_logits - max_logit)
            denom = float(np.sum(gates))
            if denom > 0:
                gates /= denom

        for k_idx, expert in enumerate(order):
            gate = float(gates[k_idx])
            _linear_layer(block.moe_ff1[expert], inp, expert_hidden)
            _gelu(expert_hidden)
            _linear_layer(block.moe_ff2[expert], expert_hidden, expert_output)
            out[:] += gate * expert_output

        if block.moe_has_shared and block.moe_shared_ff1 and block.moe_shared_ff2:
            _linear_layer(block.moe_shared_ff1, inp, shared_hidden)
            _gelu(shared_hidden)
            _linear_layer(block.moe_shared_ff2, shared_hidden, expert_output)
            out[:] += expert_output


def load_quantized_model(weights_path: str | Path, config_path: str | Path) -> QuantizedNanoLLM:
    """Convenience alias for QuantizedNanoLLM.load()."""
    return QuantizedNanoLLM.load(weights_path, config_path)
