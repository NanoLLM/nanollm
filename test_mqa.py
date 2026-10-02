import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "python"))

from export_weights import export_model
from model import NanoLLM
from quantized_runtime import QuantizedNanoLLM


def _config(n_kv_heads):
    return {
        "vocab_size": 19,
        "d_model": 8,
        "n_layers": 2,
        "n_heads": 2,
        "n_kv_heads": n_kv_heads,
        "d_ff": 12,
        "max_seq_len": 8,
        "dropout": 0.0,
    }


def _export(tmp_path, n_kv_heads):
    torch.manual_seed(7)
    config = _config(n_kv_heads)
    model = NanoLLM(**config)
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "model.bin"
    torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint)
    export_model(str(checkpoint), str(output))
    return output, output.with_name("model_config.json")


def test_mqa_projection_shapes_and_default_mha():
    mha = NanoLLM(**{k: v for k, v in _config(2).items() if k != "n_kv_heads"})
    mqa = NanoLLM(**_config(1))
    assert mha.blocks[0].attention.w_k.weight.shape == (8, 8)
    assert mqa.blocks[0].attention.w_k.weight.shape == (4, 8)
    logits, _ = mqa(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, 19)


def test_mqa_v2_export_and_cached_decode(tmp_path):
    weights, config = _export(tmp_path, 1)
    assert weights.read_bytes()[:4] == b"NLMD"
    runtime = QuantizedNanoLLM.load(weights, config)
    prompt = [1, 4, 2]
    full = runtime._next_token_logits(prompt)
    runtime.reset_cache()
    cached = None
    for token in prompt:
        cached = runtime.decode_step(token)
    np.testing.assert_allclose(cached, full, atol=0.08, rtol=0.08)
    assert runtime.k_cache[0].dtype == np.int8
    assert runtime.k_cache[0].shape == (8, 4)
    assert np.all(runtime.k_cache_scales[0][: len(prompt)] > 0)
    for token in range(12):
        runtime.decode_step(token % runtime.config.vocab_size)
    assert runtime.cache_len == runtime.config.max_seq_len


def test_mqa_cached_generate_matches_full_recompute(tmp_path):
    weights, config = _export(tmp_path, 1)
    runtime = QuantizedNanoLLM.load(weights, config)
    prompt = [1, 4, 2, 5]
    runtime.reset_cache()
    cached = runtime.generate(prompt, max_new_tokens=4)
    runtime.reset_cache()
    full = []
    generated = list(prompt)
    for _ in range(4):
        logits = runtime._next_token_logits(generated)
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        if next_token == 3:
            break
        full = generated
    assert cached == full


def test_legacy_mha_header_and_unknown_version_rejected(tmp_path):
    weights, config = _export(tmp_path, 2)
    assert weights.read_bytes()[:4] != b"NLMD"

    bad = tmp_path / "bad.bin"
    data = bytearray(weights.read_bytes())
    data[:8] = struct.pack("<4sI", b"NLMD", 99)
    bad.write_bytes(data)
    with pytest.raises(ValueError, match="Unsupported dense format version"):
        QuantizedNanoLLM.load(bad, config)
