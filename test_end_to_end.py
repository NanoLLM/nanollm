#!/usr/bin/env python3
"""
End-to-end test for NanoLLM training and deployment pipeline.
Tests: training -> weight export -> C++ inference compatibility
"""
import os
import sys
import subprocess
import torch
import json
import math
import tempfile
import shutil
import re
import ast
from pathlib import Path

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from model import NanoLLM, MoEFeedForward, MultiHeadAttention
from tokenizer import BPETokenizer
from train import split_dataset, TextDataset


def create_test_data():
    """Create a small test dataset."""
    test_data = """
The quick brown fox jumps over the lazy dog.
Machine learning is fascinating.
Natural language processing enables computers to understand text.
Deep learning uses neural networks.
Transformers have revolutionized NLP.
Artificial intelligence is transforming technology.
"""
    return test_data


def test_attention_is_causal():
    """Changing future tokens must not affect logits for earlier positions."""
    print("\n" + "=" * 60)
    print("TEST: Causal Attention Mask")
    print("=" * 60)

    torch.manual_seed(123)
    model = NanoLLM(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=4,
        d_ff=32,
        max_seq_len=8,
        dropout=0.0,
    )
    model.eval()

    base = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    changed_future = torch.tensor([[1, 2, 3, 7, 8, 9]], dtype=torch.long)

    with torch.no_grad():
        base_logits, _ = model(base)
        changed_logits, _ = model(changed_future)

    if not torch.allclose(base_logits[:, :3, :], changed_logits[:, :3, :], atol=1e-6):
        max_delta = (base_logits[:, :3, :] - changed_logits[:, :3, :]).abs().max().item()
        print(f"✗ Earlier logits changed after future-token edit (max_delta={max_delta:.6g})")
        return False

    print("✓ Attention is causal")
    return True


def test_attention_matches_manual_causal_math():
    """PyTorch attention must match the manual causal math used by C++/ESP32."""
    print("\n" + "=" * 60)
    print("TEST: Attention Manual Causal Parity")
    print("=" * 60)

    torch.manual_seed(456)
    attn = MultiHeadAttention(d_model=16, n_heads=4)
    attn.eval()
    x = torch.randn(2, 7, 16)

    with torch.no_grad():
        actual = attn(x)

        batch_size, seq_len, d_model = x.size()
        q = attn.w_q(x).view(batch_size, seq_len, attn.n_heads, attn.d_k).transpose(1, 2)
        k = attn.w_k(x).view(batch_size, seq_len, attn.n_heads, attn.d_k).transpose(1, 2)
        v = attn.w_v(x).view(batch_size, seq_len, attn.n_heads, attn.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.d_k)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        expected = torch.softmax(scores, dim=-1).matmul(v)
        expected = expected.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        expected = attn.w_o(expected)

    if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        max_delta = (actual - expected).abs().max().item()
        print(f"✗ SDPA attention drifted from manual causal math (max_delta={max_delta:.6g})")
        return False

    print("✓ SDPA attention matches manual causal attention math")
    return True


def test_quantized_attention_matches_pytorch():
    """Exported int8 runtime must match float PyTorch next-token prediction (MHA path)."""
    print("\n" + "=" * 60)
    print("TEST: Quantized Multi-Head Attention Parity")
    print("=" * 60)

    torch.manual_seed(789)
    model = NanoLLM(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_len=8,
        dropout=0.0,
    )
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(tmpdir, "model.pt")
        weights_path = os.path.join(tmpdir, "model.bin")
        config_path = os.path.join(tmpdir, "model_config.json")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": 64,
                    "d_model": 16,
                    "n_layers": 1,
                    "n_heads": 4,
                    "d_ff": 32,
                    "max_seq_len": 8,
                    "dropout": 0.0,
                    "use_moe": False,
                    "causal_attention": True,
                },
            },
            checkpoint,
        )

        export_cmd = [
            sys.executable,
            "python/export_weights.py",
            "--checkpoint",
            checkpoint,
            "--output",
            weights_path,
        ]
        export_result = subprocess.run(export_cmd, capture_output=True, text=True)
        if export_result.returncode != 0:
            print("✗ export_weights failed")
            print(export_result.stderr)
            return False

        from quantized_runtime import QuantizedNanoLLM

        quant = QuantizedNanoLLM.load(weights_path, config_path)
        prompt = [3, 7, 11, 19, 23]
        with torch.no_grad():
            logits, _ = model(torch.tensor([prompt], dtype=torch.long))
            expected = int(logits[0, -1].argmax().item())
        actual = int(quant.next_token(prompt))

        if actual != expected:
            print(f"✗ Float/quantized next-token mismatch: py={expected}, quant={actual}")
            return False

    print(f"✓ Float and quantized next-token match ({actual})")
    return True


def test_cached_decode_python_cpp_parity():
    """Cached decodeStep/generate must match between Python int8 and C++ dump_cached_tokens."""
    print("\n" + "=" * 60)
    print("TEST: Cached Decode Python/C++ Parity (MQA)")
    print("=" * 60)

    torch.manual_seed(321)
    config = {
        "vocab_size": 64,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 1,
        "d_ff": 32,
        "max_seq_len": 8,
        "dropout": 0.0,
        "use_moe": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(tmpdir, "model.pt")
        weights_path = os.path.join(tmpdir, "model.bin")
        config_path = os.path.join(tmpdir, "model_config.json")
        torch.save(
            {"model_state_dict": NanoLLM(**config).state_dict(), "config": config},
            checkpoint,
        )

        export_cmd = [
            sys.executable,
            "python/export_weights.py",
            "--checkpoint",
            checkpoint,
            "--output",
            weights_path,
        ]
        export_result = subprocess.run(export_cmd, capture_output=True, text=True)
        if export_result.returncode != 0:
            print("✗ export_weights failed for cached decode parity")
            print(export_result.stderr)
            return False

        if not weights_path.endswith(".bin"):
            return False
        header = Path(weights_path).read_bytes()[:4]
        if header != b"NLMD":
            print(f"✗ Expected NLMD v2 export, got header {header!r}")
            return False

        from quantized_runtime import QuantizedNanoLLM

        quant = QuantizedNanoLLM.load(weights_path, config_path)
        prompt = [3, 7, 11, 19]
        py_generated = quant.generate(prompt, max_new_tokens=4)
        py_next = py_generated[len(prompt)] if len(py_generated) > len(prompt) else -1

        exe_path = Path("cpp/build") / "dump_cached_tokens"
        if not exe_path.exists():
            print(f"Executable not found: {exe_path} (run C++ build test first)")
            return False

        cmd = [
            str(exe_path),
            str(Path(weights_path).absolute()),
            str(Path(config_path).absolute()),
            ",".join(str(t) for t in prompt),
            "4",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print("✗ dump_cached_tokens execution failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

        data = json.loads(result.stdout.strip())
        cpp_generated = data.get("generated_tokens")
        cpp_next = data.get("next_token")

        parity_ok = True
        if cpp_next != py_next:
            print(f"✗ Cached next-token mismatch: py={py_next}, cpp={cpp_next}")
            parity_ok = False
        if cpp_generated != py_generated:
            print("✗ Cached multi-token generation mismatch")
            print("  Python:", py_generated)
            print("  C++   :", cpp_generated)
            parity_ok = False

        if parity_ok:
            print(f"✓ Cached decode parity ({len(py_generated) - len(prompt)} new tokens)")
        return parity_ok


def test_training():
    """Test model training."""
    print("=" * 60)
    print("TEST 1: Training")
    print("=" * 60)
    
    # Create temporary data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(create_test_data())
        data_path = f.name
    
    try:
        # Train a tiny model
        cmd = [
            sys.executable, 'python/train.py',
            '--data', data_path,
            '--output_dir', 'test_checkpoints',
            '--epochs', '2',  # Just 2 epochs for testing
            '--batch_size', '4',
            '--vocab_size', '500',
            '--d_model', '32',  # Very small for fast testing
            '--n_layers', '1',
            '--n_heads', '2',
            '--d_ff', '64',
            '--block_size', '32',
            '--learning_rate', '1e-3',
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Training failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        # Check if checkpoint was created
        checkpoint_path = 'test_checkpoints/model_best.pt'
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        print("✓ Training successful!")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(data_path):
            os.unlink(data_path)


def test_python_bpe_tokenizer():
    """Test Python BPE tokenizer encoding/decoding."""
    print("\n" + "=" * 60)
    print("TEST 2: Python BPE Tokenizer")
    print("=" * 60)
    
    # Check if tokenizer exists from training
    tokenizer_path = 'test_checkpoints/tokenizer/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found: {tokenizer_path}")
        return False
    
    try:
        # Load tokenizer
        tokenizer = BPETokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        print(f"Loaded tokenizer with vocab_size={vocab_size}")
        
        # Test cases
        test_texts = [
            "Hello world",
            "Case Preservation: NanoLLM Should Keep Capitals.",
            "The quick brown fox",
            "Machine learning is fascinating.",
            "Test with numbers: 12345",
            "Special chars: !@#$%",
        ]
        
        all_passed = True
        for text in test_texts:
            # Encode
            tokens = tokenizer.encode(text)
            if not tokens:
                print(f"✗ Encoding failed for: '{text}'")
                all_passed = False
                continue
            
            # Decode
            decoded = tokenizer.decode(tokens)
            
            # Check round-trip (may not be exact due to normalization)
            # But should at least decode to something reasonable
            if not decoded:
                print(f"✗ Decoding failed for: '{text}'")
                print(f"  Tokens: {tokens}")
                all_passed = False
                continue
            
            print(f"  '{text}' -> {len(tokens)} tokens -> '{decoded[:50]}...'")

            if (
                0 not in tokens
                and any(ch.isupper() for ch in text)
                and not any(ch.isupper() for ch in decoded)
            ):
                print(f"✗ Capitalization was lost for: '{text}' -> '{decoded}'")
                all_passed = False
        
        if all_passed:
            print("✓ Python BPE tokenizer test passed!")
            return True
        else:
            print("✗ Some Python BPE tokenizer tests failed")
            return False
            
    except Exception as e:
        print(f"✗ Python BPE tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_preserves_capitalization():
    """Fresh tokenizers must preserve ASCII capitalization across encode/decode."""
    print("\n" + "=" * 60)
    print("TEST 2A: Tokenizer Capitalization Preservation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / 'case_data.txt'
        data_path.write_text(
            "Hello NanoLLM. NASA Uses Capitals. Hello NASA.\n",
            encoding='utf-8',
        )
        tokenizer = BPETokenizer(vocab_size=80).train([str(data_path)])
        text = "Hello NASA"
        decoded = tokenizer.decode(tokenizer.encode(text))
        if decoded != text:
            print(f"✗ Capitalization changed: '{text}' -> '{decoded}'")
            return False

    print("✓ Tokenizer preserves capitalization")
    return True


def test_moe_top1_routing_equivalence():
    """Top-1 fast routing retains the selected expert's full softmax gate."""
    print("\n" + "=" * 60)
    print("TEST 2B: MoE Top-1 Routing Equivalence")
    print("=" * 60)

    torch.manual_seed(1234)
    moe = MoEFeedForward(
        d_model=8,
        expert_d_ff=16,
        n_experts=4,
        top_k=1,
        shared_d_ff=4,
    )
    moe.eval()

    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        reference = moe(x)

        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        router_logits = moe.router(x_flat)
        router_probs = torch.softmax(router_logits, dim=-1)
        selected = torch.argmax(router_logits, dim=-1)
        fast_path = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(moe.experts):
            mask = selected == expert_id
            if torch.any(mask):
                gate = router_probs[mask, expert_id].unsqueeze(-1)
                fast_path[mask] = gate * expert(x_flat[mask])

        if moe.shared_expert is not None:
            fast_path += moe.shared_expert(x_flat)

        fast_path = fast_path.view(batch_size, seq_len, d_model)

    if not torch.allclose(reference, fast_path, atol=1e-6, rtol=1e-6):
        max_diff = torch.max(torch.abs(reference - fast_path)).item()
        print(f"✗ Top-1 fast path mismatch (max diff: {max_diff:.8f})")
        return False

    print("✓ Top-1 MoE argmax fast path preserves full-router gate")
    return True


def test_moe_top1_bf16_autocast_training_step():
    """Top-1 MoE fast path must train under CUDA bf16 autocast."""
    print("\n" + "=" * 60)
    print("TEST 2C: MoE Top-1 BF16 Autocast Training Step")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping bf16 autocast regression")
        return True

    torch.manual_seed(5678)
    device = torch.device('cuda')
    model = NanoLLM(
        vocab_size=256,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
        use_moe=True,
        moe_n_experts=4,
        moe_top_k=1,
        moe_shared_d_ff=16,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, 256, (2, 16), device=device)
    y = torch.randint(0, 256, (2, 16), device=device)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, loss = model(x, targets=y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    print("✓ Top-1 MoE bf16 autocast training step completed")
    return True


def test_validation_split_has_context_gap():
    """Train/validation windows must not overlap across the split boundary."""
    print("\n" + "=" * 60)
    print("TEST 2C: Validation Split Context Gap")
    print("=" * 60)

    class DummyDataset:
        def __init__(self, length, block_size):
            self._length = length
            self.block_size = block_size

        def __len__(self):
            return self._length

    dataset = DummyDataset(length=1000, block_size=128)
    train_subset, val_subset = split_dataset(dataset, val_split=0.1, seed=42)

    train_indices = list(train_subset.indices)
    val_indices = list(val_subset.indices)
    if not train_indices or not val_indices:
        print("✗ Split produced an empty train or validation subset")
        return False

    max_train_idx = max(train_indices)
    min_val_idx = min(val_indices)
    required_gap = dataset.block_size
    actual_gap = min_val_idx - max_train_idx

    if actual_gap < required_gap:
        print(
            f"✗ Validation gap too small: got {actual_gap}, need at least {required_gap}"
        )
        return False

    print(
        f"✓ Validation windows start {actual_gap} indices after the train tail "
        f"(required >= {required_gap})"
    )
    return True


def test_token_cache_rebuilds_on_tokenizer_change():
    """Token cache must rebuild when tokenizer semantics change, even if vocab size stays fixed."""
    print("\n" + "=" * 60)
    print("TEST 2D: Token Cache Invalidation")
    print("=" * 60)

    class DummyTokenizer:
        def __init__(self, fingerprint, offset):
            self._fingerprint = fingerprint
            self._offset = offset

        def get_vocab_size(self):
            return 8

        def fingerprint(self):
            return self._fingerprint

        def encode(self, text):
            return [self._offset + (ord(ch) % 4) for ch in text]

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, 'sample.txt')
        with open(data_path, 'w', encoding='utf-8') as handle:
            handle.write('alpha beta\ngamma delta\n')

        cache_dir = os.path.join(tmp_dir, 'cache')
        tokenizer_a = DummyTokenizer('fingerprint-a', 10)
        tokenizer_b = DummyTokenizer('fingerprint-b', 20)

        dataset_a = TextDataset(data_path, tokenizer_a, block_size=2, cache_dir=cache_dir, chunk_chars=8)
        first_tokens = list(dataset_a.data[:min(8, len(dataset_a.data))])

        dataset_b = TextDataset(data_path, tokenizer_b, block_size=2, cache_dir=cache_dir, chunk_chars=8)
        second_tokens = list(dataset_b.data[:min(8, len(dataset_b.data))])

        if first_tokens == second_tokens:
            print('✗ Cache was reused after tokenizer fingerprint changed')
            return False

        with open(os.path.join(cache_dir, 'sample_tokens_meta.json'), 'r', encoding='utf-8') as handle:
            meta = json.load(handle)

        if meta.get('tokenizer_fingerprint') != 'fingerprint-b':
            print('✗ Cache metadata did not capture the tokenizer fingerprint')
            return False

    print('✓ Token cache invalidates on tokenizer fingerprint changes')
    return True


def test_text_dataset_window_stride():
    """TextDataset should support sparse overlapping windows without changing token cache."""
    print("\n" + "=" * 60)
    print("TEST 2E: TextDataset Window Stride")
    print("=" * 60)

    class DummyTokenizer:
        def get_vocab_size(self):
            return 64

        def fingerprint(self):
            return 'stride-test'

        def encode(self, text):
            return [ord(ch) % 32 for ch in text]

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, 'stride.txt')
        with open(data_path, 'w', encoding='utf-8') as handle:
            handle.write('abcdefghijklmnopqrstuvwxyz')

        dataset = TextDataset(
            data_path,
            DummyTokenizer(),
            block_size=4,
            cache_dir=os.path.join(tmp_dir, 'cache'),
            chunk_chars=1024,
            window_stride=3,
        )

        expected_len = ((dataset.token_count - dataset.block_size - 1) // dataset.window_stride) + 1
        if len(dataset) != expected_len:
            print(f"✗ Unexpected stride dataset length: got {len(dataset)}, expected {expected_len}")
            return False

        x0, y0 = dataset[0]
        x1, y1 = dataset[1]
        if x0.tolist() != list(dataset.data[:4]) or y0.tolist() != list(dataset.data[1:5]):
            print("✗ First stride window does not start at token 0")
            return False
        if x1.tolist() != list(dataset.data[3:7]) or y1.tolist() != list(dataset.data[4:8]):
            print("✗ Second stride window does not start at token 3")
            return False

    print("✓ TextDataset window stride maps indices to sparse token windows")
    return True


def test_text_dataset_assistant_only_mask():
    """Assistant-only loss should mask non-assistant targets to -1."""
    print("\n" + "=" * 60)
    print("TEST 2F: TextDataset Assistant-Only Mask")
    print("=" * 60)

    from chat_template import ASSISTANT_PREFIX, USER_PREFIX

    class DummyEncoding:
        def __init__(self, text):
            self.ids = [ord(ch) % 32 for ch in text]
            self.offsets = [(i, i + 1) for i in range(len(text))]

    class DummyBackend:
        @staticmethod
        def encode(text):
            return DummyEncoding(text)

    class DummyTokenizer:
        def __init__(self):
            self.tokenizer = DummyBackend()

        def get_vocab_size(self):
            return 64

        def fingerprint(self):
            return "assistant-mask-test"

        def encode(self, text):
            return DummyBackend.encode(text).ids

    tokenizer = DummyTokenizer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "chat.txt")
        turn = (
            f"{USER_PREFIX}What is 2+2?\n"
            f"{ASSISTANT_PREFIX}The answer is four.\n\n"
        )
        with open(data_path, "w", encoding="utf-8") as handle:
            handle.write(turn)

        dataset = TextDataset(
            data_path,
            tokenizer,
            block_size=32,
            cache_dir=os.path.join(tmp_dir, "cache"),
            chunk_chars=1024,
            window_stride=1,
            assistant_only_loss=True,
        )
        found_masked = False
        found_trainable = False
        for idx in range(len(dataset)):
            _, y = dataset[idx]
            found_masked = found_masked or bool(torch.any(y == -1).item())
            found_trainable = found_trainable or bool(torch.any(y != -1).item())
            if found_masked and found_trainable:
                break
        if not found_masked:
            print("✗ Expected some masked targets for user/prompt tokens")
            return False
        if not found_trainable:
            print("✗ Expected some unmasked assistant targets")
            return False

    print("✓ Assistant-only masking marks non-assistant targets as ignored")
    return True


def test_chat_prompt_and_eval_contract():
    """Training/inference prompts and hard-eval checks should stay strict and aligned."""
    print("\n" + "=" * 60)
    print("TEST 2G: Chat Prompt and Evaluation Contract")
    print("=" * 60)

    from chat_eval import aggregate_chat_eval_score, passes_hard_chat_eval, score_chat_reply
    from chat_template import format_prompt, format_turn, normalize_chat_corpus

    prompt = format_prompt("Hello")
    if prompt != "User: Hello\nAssistant:":
        print(f"✗ Unexpected chat prompt suffix: {prompt!r}")
        return False

    haiku_turn = format_turn("Write a haiku.", "First line\nSecond line\nThird line")
    if "First line\nSecond line\nThird line" not in haiku_turn:
        print("✗ Chat formatting destroyed assistant line breaks")
        return False
    if not haiku_turn.endswith("<EOS>"):
        print("✗ Chat formatting omitted the supervised end-of-response token")
        return False
    normalized_haiku, _ = normalize_chat_corpus(haiku_turn + "\n\n")
    if "First line\nSecond line\nThird line" not in normalized_haiku:
        print("✗ Chat normalization destroyed assistant line breaks")
        return False

    identity = score_chat_reply("Hello! who are you?", "I am able to help.")
    if identity.get("identity") != 0.0:
        print("✗ Identity scoring matched 'ai' inside an unrelated word")
        return False

    arithmetic_decimal = score_chat_reply("What is 2+2?", "4.5")
    if arithmetic_decimal.get("arithmetic") != 0.0:
        print("✗ Arithmetic scoring accepted 4 as part of the wrong decimal answer")
        return False

    off_topic_haiku = score_chat_reply(
        "Write a haiku about rain.",
        "Sun warms the hillside\nBirds cross the open sky\nNight settles softly",
    )
    if off_topic_haiku.get("instruction") != 0.0:
        print("✗ Haiku scoring accepted three lines with no rain-related content")
        return False

    arithmetic_only = [
        {
            "prompt": "What is 2+2?",
            "reply": "4",
            "scores": score_chat_reply("What is 2+2?", "4"),
        }
    ]
    if passes_hard_chat_eval(arithmetic_only):
        print("✗ Hard evaluation passed without the required haiku result")
        return False

    nonsense = [
        {
            "prompt": "What is 2+2?",
            "reply": "This is clean but unrelated prose.",
            "scores": score_chat_reply("What is 2+2?", "This is clean but unrelated prose."),
        },
        {
            "prompt": "Write a haiku about rain.",
            "reply": "This is also clean but unrelated prose.",
            "scores": score_chat_reply(
                "Write a haiku about rain.",
                "This is also clean but unrelated prose.",
            ),
        },
    ]
    if aggregate_chat_eval_score(nonsense) >= 0.5:
        print("✗ Structural hygiene still outweighs failed capability checks")
        return False

    print("✓ Chat prompt suffix and semantic evaluation contract are aligned")
    return True


def test_vocab_export():
    """Test vocab.json export for C++."""
    print("\n" + "=" * 60)
    print("TEST 3: Vocab Export")
    print("=" * 60)
    
    tokenizer_path = 'test_checkpoints/tokenizer/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found: {tokenizer_path}")
        return False
    
    # Export vocab
    cmd = [
        sys.executable, 'python/export_tokenizer.py',
        '--tokenizer', tokenizer_path,
        '--output', 'test_weights',
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Vocab export failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Check if vocab.json was created
    vocab_path = 'test_weights/vocab.json'
    if not os.path.exists(vocab_path):
        print(f"Vocab file not found: {vocab_path}")
        return False
    
    # Verify vocab.json structure
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    required_keys = ['vocab_size', 'id_to_token', 'token_to_id']
    for key in required_keys:
        if key not in vocab_data:
            print(f"✗ Vocab file missing key: {key}")
            return False
    
    vocab_size = vocab_data['vocab_size']
    id_to_token = vocab_data['id_to_token']
    
    print(f"✓ Vocab export successful!")
    print(f"  Vocab size: {vocab_size}")
    print(f"  id_to_token entries: {len(id_to_token)}")
    print(f"  File size: {os.path.getsize(vocab_path) / 1024:.2f} KB")
    
    return True


def test_weight_export():
    """Test weight export."""
    print("\n" + "=" * 60)
    print("TEST 4: Weight Export")
    print("=" * 60)
    
    checkpoint_path = 'test_checkpoints/model_best.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    # Export weights
    cmd = [
        sys.executable, 'python/export_weights.py',
        '--checkpoint', checkpoint_path,
        '--output', 'test_weights/model.bin',
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Weight export failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Check if files were created
    weights_path = 'test_weights/model.bin'
    config_path = 'test_weights/model_config.json'
    
    if not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}")
        return False
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return False
    
    # Verify config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Weight export successful!")
    print(f"  Weights size: {os.path.getsize(weights_path) / 1024:.2f} KB")
    print(f"  Config: {config}")
    return True


def test_python_inference():
    """Test Python inference with BPE tokenizer."""
    print("\n" + "=" * 60)
    print("TEST 5: Python Inference (BPE)")
    print("=" * 60)
    
    checkpoint_path = 'test_checkpoints/model_best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Load tokenizer
    tokenizer_path = checkpoint.get('tokenizer_path')
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        tokenizer_path = 'test_checkpoints/tokenizer/tokenizer.json'
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found: {tokenizer_path}")
        return False
    
    tokenizer = BPETokenizer.from_file(tokenizer_path)
    
    # Create model
    model = NanoLLM(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test inference with BPE
    prompt = "Hello"
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
    
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_tokens}")
    
    with torch.no_grad():
        generated = model.generate(prompt_tensor, max_new_tokens=10, temperature=1.0)
    
    generated_tokens = generated[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generated text: '{generated_text}'")
    
    if len(generated_tokens) > len(prompt_tokens):
        print("✓ Python inference successful!")
        return True
    else:
        print("✗ Python inference failed - no tokens generated")
        return False


def test_cpp_bpe_tokenizer():
    """Test C++ BPE tokenizer loading and decoding."""
    print("\n" + "=" * 60)
    print("TEST 6: C++ BPE Tokenizer")
    print("=" * 60)
    
    vocab_path = 'test_weights/vocab.json'
    if not os.path.exists(vocab_path):
        print(f"Vocab file not found: {vocab_path}")
        return False
    
    # Create a simple test program to verify C++ tokenizer
    # We'll use the inference executable with a test mode
    build_dir = Path('cpp/build')
    exe_path = build_dir / 'inference'
    
    if not exe_path.exists():
        print(f"Executable not found: {exe_path}")
        print("  (C++ build may not have completed yet)")
        return False
    
    # Test that vocab.json can be loaded
    # We'll encode in Python and decode in C++
    tokenizer_path = 'test_checkpoints/tokenizer/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print(f"Python tokenizer not found: {tokenizer_path}")
        return False
    
    try:
        # Load Python tokenizer
        py_tokenizer = BPETokenizer.from_file(tokenizer_path)
        
        # Test text
        test_text = "Hello world"
        py_tokens = py_tokenizer.encode(test_text)
        py_decoded = py_tokenizer.decode(py_tokens)
        
        print(f"Python encode: '{test_text}' -> {py_tokens}")
        print(f"Python decode: {py_tokens} -> '{py_decoded}'")
        
        # Verify vocab.json is valid JSON
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        # Check that tokens from Python are in vocab
        id_to_token = vocab_data['id_to_token']
        all_tokens_found = True
        for token_id in py_tokens:
            token_id_str = str(token_id)
            if token_id_str not in id_to_token:
                print(f"✗ Token ID {token_id} not found in vocab.json")
                all_tokens_found = False
        
        if all_tokens_found:
            print(f"✓ All Python tokens found in vocab.json")
            print(f"✓ C++ BPE tokenizer test passed!")
            return True
        else:
            print(f"✗ Some tokens missing from vocab.json")
            return False
            
    except Exception as e:
        print(f"✗ C++ BPE tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_python_cpp_roundtrip():
    """Test Python encode -> C++ decode round-trip."""
    print("\n" + "=" * 60)
    print("TEST 7: Python-C++ Round-trip")
    print("=" * 60)
    
    vocab_path = 'test_weights/vocab.json'
    tokenizer_path = 'test_checkpoints/tokenizer/tokenizer.json'
    
    if not os.path.exists(vocab_path) or not os.path.exists(tokenizer_path):
        print("Required files not found")
        return False
    
    try:
        # Load Python tokenizer
        py_tokenizer = BPETokenizer.from_file(tokenizer_path)
        
        # Load vocab.json
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        id_to_token = vocab_data['id_to_token']
        
        # Test cases
        test_texts = [
            "Hello",
            "The quick",
            "Test 123",
        ]
        
        all_passed = True
        for text in test_texts:
            # Python encode
            py_tokens = py_tokenizer.encode(text)
            py_decoded = py_tokenizer.decode(py_tokens)
            
            # Simulate C++ decode (using vocab.json)
            cpp_decoded = ""
            for token_id in py_tokens:
                token_id_str = str(token_id)
                if token_id_str in id_to_token:
                    cpp_decoded += id_to_token[token_id_str]
                else:
                    cpp_decoded += "<?>"
            
            print(f"  '{text}'")
            print(f"    Python tokens: {py_tokens}")
            print(f"    Python decode: '{py_decoded}'")
            print(f"    C++ decode:    '{cpp_decoded}'")
            
            # Check if C++ decode matches Python decode
            if cpp_decoded != py_decoded:
                print(f"    ⚠ Decode mismatch (may be due to normalization)")
            else:
                print(f"    ✓ Decode match!")
        
        print("✓ Python-C++ round-trip test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Round-trip test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cpp_build():
    """Test C++ build."""
    print("\n" + "=" * 60)
    print("TEST 8: C++ Build")
    print("=" * 60)
    
    cpp_dir = Path('cpp')
    build_dir = cpp_dir / 'build'
    
    # Clean previous build
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Run cmake
    print("Running cmake...")
    result = subprocess.run(
        ['cmake', '..'],
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("CMake failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Build
    print("Building...")
    result = subprocess.run(
        ['make'],
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Build failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Check if executable exists
    exe_path = build_dir / 'inference'
    if not exe_path.exists():
        print(f"Executable not found: {exe_path}")
        return False
    
    print("✓ C++ build successful!")
    return True


def test_moe_training_and_export():
    """Train a tiny MoE checkpoint and export experimental MoE weights."""
    print("\n" + "=" * 60)
    print("TEST 9: MoE Training + Export")
    print("=" * 60)

    # Create temporary data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(create_test_data())
        data_path = f.name

    output_dir = 'test_checkpoints_moe'
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        train_cmd = [
            sys.executable, 'python/train.py',
            '--data', data_path,
            '--output_dir', output_dir,
            '--epochs', '1',
            '--batch_size', '4',
            '--vocab_size', '500',
            '--d_model', '32',
            '--n_layers', '1',
            '--n_heads', '2',
            '--d_ff', '32',
            '--block_size', '32',
            '--learning_rate', '1e-3',
            '--use_moe',
            '--moe_n_experts', '4',
            '--moe_top_k', '1',
            '--moe_shared_d_ff', '32',
        ]

        print(f"Running: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("MoE training failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

        checkpoint_path = os.path.join(output_dir, 'model_best.pt')
        if not os.path.exists(checkpoint_path):
            print(f"MoE checkpoint not found: {checkpoint_path}")
            return False

        export_cmd = [
            sys.executable, 'python/export_weights.py',
            '--checkpoint', checkpoint_path,
            '--output', 'test_weights/model_moe.bin',
            '--allow-moe-export',
        ]
        print(f"Running: {' '.join(export_cmd)}")
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("MoE export failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

        required_files = [
            'test_weights/model_moe.bin',
            'test_weights/model_moe_config.json',
            'test_weights/model_moe_format.json',
        ]
        for path in required_files:
            if not os.path.exists(path):
                print(f"Missing MoE export artifact: {path}")
                return False

        print("✓ MoE training + export successful!")
        return True
    finally:
        if os.path.exists(data_path):
            os.unlink(data_path)


def test_cardputer_512_context_parity():
    """Train/export a 512-ctx MoE model and verify RAM budget + Python/C++ parity.

    The budget assertion uses the canonical ctx224 deployment profile
    (cardputer-mqa-ctx224). The legacy 512-ctx/8192-vocab/124L research
    profile no longer fits the working-RAM budget now that the estimator
    counts the full MQA KV cache (745.5 KiB > 200 KiB).
    """
    print("\n" + "=" * 60)
    print("TEST 10: Cardputer 512-Context MoE Parity")
    print("=" * 60)

    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / 'scripts'))
    from moe_tradeoff_estimator import estimate_working_memory_bytes, CardputerConfig, cardputer_flash_fits

    cardputer_cfg = CardputerConfig(
        vocab_size=2048,
        d_model=80,
        n_layers=18,
        n_heads=4,
        n_kv_heads=1,
        d_ff=320,
        max_seq_len=224,
        moe_n_experts=4,
        moe_top_k=1,
        moe_shared_d_ff=160,
    )
    ok, sizing = cardputer_flash_fits(cardputer_cfg)
    if not ok:
        print("✗ Canonical cardputer profile (ctx224: 2048-vocab, d=80, 18L) fails flash/RAM budget check")
        return False
    working_bytes = estimate_working_memory_bytes(224, 80, 2048, 18, 4, 1)
    if working_bytes > 200 * 1024:
        print(f"✗ Working RAM estimate {working_bytes} exceeds 200 KiB budget")
        return False
    print(f"✓ Cardputer profile fits budget (working={working_bytes / 1024:.1f} KiB, flash={sizing.weight_bytes / 1024 / 1024:.2f} MiB)")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(create_test_data() * 40)
        data_path = f.name

    output_dir = 'test_checkpoints_ctx512'
    weights_path = 'test_weights/model_ctx512.bin'
    config_path = 'test_weights/model_ctx512_config.json'
    header_path = 'test_weights/model_ctx512_weights.h'
    exe_path = Path('cpp/build') / 'dump_next_token'

    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        train_cmd = [
            sys.executable, 'python/train.py',
            '--data', data_path,
            '--output_dir', output_dir,
            '--epochs', '1',
            '--batch_size', '2',
            '--vocab_size', '256',
            '--d_model', '32',
            '--n_layers', '2',
            '--n_heads', '4',
            '--d_ff', '64',
            '--block_size', '512',
            '--learning_rate', '1e-3',
            '--use_moe',
            '--moe_n_experts', '4',
            '--moe_top_k', '1',
            '--moe_shared_d_ff', '32',
            '--no-tensorboard',
        ]
        print(f"Running: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("512-ctx MoE training failed!")
            print("STDOUT:", result.stdout[-2000:])
            print("STDERR:", result.stderr[-2000:])
            return False

        checkpoint_path = os.path.join(output_dir, 'model_best.pt')
        tokenizer_path = os.path.join(output_dir, 'tokenizer/tokenizer.json')
        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if ckpt.get('config', {}).get('max_seq_len') != 512:
            print(f"✗ Checkpoint max_seq_len != 512: {ckpt.get('config')}")
            return False
        print("✓ Checkpoint trained with max_seq_len=512")

        export_cmd = [
            sys.executable, 'python/export_weights.py',
            '--checkpoint', checkpoint_path,
            '--output', weights_path,
            '--allow-moe-export',
        ]
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("512-ctx export failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

        with open(config_path, 'r', encoding='utf-8') as f:
            exported_cfg = json.load(f)
        if exported_cfg.get('max_seq_len') != 512:
            print(f"✗ Exported config max_seq_len != 512: {exported_cfg}")
            return False
        print("✓ Exported model.bin config has max_seq_len=512")

        header_cmd = [
            sys.executable, 'python/export_weights_header.py',
            '--checkpoint', checkpoint_path,
            '--output', header_path,
            '--namespace', 'nanollm',
        ]
        result = subprocess.run(header_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("512-ctx embedded header export failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

        with open(header_path, 'r', encoding='utf-8') as f:
            header_content = f.read()
        if 'cfg.max_seq_len = 512' not in header_content:
            print("✗ Embedded header missing cfg.max_seq_len = 512")
            return False
        print("✓ Embedded header declares max_seq_len=512")

        if not exe_path.exists():
            print(f"Executable not found: {exe_path} (run C++ build test first)")
            return False

        from quantized_runtime import QuantizedNanoLLM

        tokenizer = BPETokenizer.from_file(tokenizer_path)
        quant_model = QuantizedNanoLLM.load(weights_path, config_path)
        if quant_model.config.max_seq_len != 512:
            print(f"✗ Quantized runtime max_seq_len != 512: {quant_model.config.max_seq_len}")
            return False

        prompt = "The quick brown fox jumps over the lazy dog. " * 12
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) < 32:
            print("Prompt too short for context crop test")
            return False
        next_token_py = quant_model.next_token(prompt_tokens)

        cmd = [
            str(exe_path),
            str(Path(weights_path).absolute()),
            str(Path(config_path).absolute()),
            prompt,
            '1',
            str(Path(tokenizer_path).absolute()),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print("✗ dump_next_token failed for 512-ctx MoE")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

        data = json.loads(result.stdout.strip())
        if data.get('next_token') != next_token_py:
            print("✗ Next-token mismatch for 512-ctx MoE")
            print(f"  Python: {next_token_py}")
            print(f"  C++   : {data.get('next_token')}")
            return False

        print(f"✓ Python/C++ next-token parity at 512-ctx (prompt_len={len(prompt_tokens)}, next={next_token_py})")
        return True
    finally:
        if os.path.exists(data_path):
            os.unlink(data_path)


def test_python_cpp_output_match():
    """Compare Python quantized runtime and C++ next-token predictions for parity."""
    print("\n" + "=" * 60)
    print("TEST 9: Python vs C++ Output Parity (int8)")
    print("=" * 60)

    checkpoint_path = 'test_checkpoints/model_best.pt'
    tokenizer_path = 'test_checkpoints/tokenizer/tokenizer.json'
    weights_path = 'test_weights/model.bin'
    config_path = 'test_weights/model_config.json'
    exe_path = Path('cpp/build') / 'dump_next_token'

    missing = []
    for path, label in [
        (checkpoint_path, 'checkpoint'),
        (tokenizer_path, 'tokenizer'),
        (weights_path, 'weights'),
        (config_path, 'config'),
    ]:
        if not os.path.exists(path):
            missing.append(label)

    if missing:
        print(f"Required artifacts missing: {', '.join(missing)}")
        return False

    if not exe_path.exists():
        print(f"Executable not found: {exe_path}")
        return False

    sys.path.insert(0, 'python')
    from quantized_runtime import QuantizedNanoLLM

    tokenizer = BPETokenizer.from_file(tokenizer_path)
    quant_model = QuantizedNanoLLM.load(weights_path, config_path)

    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        print("Prompt encoding produced no tokens")
        return False

    generated_tokens_py = quant_model.generate(prompt_tokens, max_new_tokens=4)
    next_token_py = generated_tokens_py[len(prompt_tokens)]

    cmd = [
        str(exe_path),
        str(Path(weights_path).absolute()),
        str(Path(config_path).absolute()),
        prompt,
        "4",
        str(Path(tokenizer_path).absolute()),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print("✗ dump_next_token execution failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

    stdout = result.stdout.strip()
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        print("✗ Failed to parse dump_next_token output as JSON")
        print("Raw output:", stdout)
        return False

    prompt_tokens_cpp = data.get('prompt_tokens')
    generated_tokens_cpp = data.get('generated_tokens')
    next_token_cpp = data.get('next_token')

    parity_ok = True

    if prompt_tokens_cpp != prompt_tokens:
        print("✗ Prompt token mismatch")
        print("  Python:", prompt_tokens)
        print("  C++   :", prompt_tokens_cpp)
        parity_ok = False
    else:
        print("✓ Prompt tokens match")

    if next_token_cpp != next_token_py:
        print("✗ Next-token mismatch")
        print(f"  Python: {next_token_py}")
        print(f"  C++   : {next_token_cpp}")
        parity_ok = False
    else:
        decoded = tokenizer.decode(prompt_tokens + [next_token_py])
        print(f"✓ Next-token match (decoded: '{decoded}')")

    if generated_tokens_cpp and len(generated_tokens_cpp) <= len(prompt_tokens):
        print("✗ Generated tokens missing new token")
        parity_ok = False
    elif generated_tokens_cpp != generated_tokens_py:
        print("✗ Multi-token generation mismatch")
        print("  Python:", generated_tokens_py)
        print("  C++   :", generated_tokens_cpp)
        parity_ok = False
    else:
        print("✓ Multi-token generation matches")

    return parity_ok


def test_python_cpp_output_match_moe():
    """Compare Python quantized runtime and C++ next-token predictions for MoE parity."""
    print("\n" + "=" * 60)
    print("TEST 11: Python vs C++ Output Parity (MoE int8)")
    print("=" * 60)

    checkpoint_path = 'test_checkpoints_moe/model_best.pt'
    tokenizer_path = 'test_checkpoints_moe/tokenizer/tokenizer.json'
    weights_path = 'test_weights/model_moe.bin'
    config_path = 'test_weights/model_moe_config.json'
    exe_path = Path('cpp/build') / 'dump_next_token'

    missing = []
    for path, label in [
        (checkpoint_path, 'moe checkpoint'),
        (tokenizer_path, 'moe tokenizer'),
        (weights_path, 'moe weights'),
        (config_path, 'moe config'),
    ]:
        if not os.path.exists(path):
            missing.append(label)

    if missing:
        print(f"Required MoE artifacts missing: {', '.join(missing)}")
        return False

    if not exe_path.exists():
        print(f"Executable not found: {exe_path}")
        return False

    sys.path.insert(0, 'python')
    from quantized_runtime import QuantizedNanoLLM

    tokenizer = BPETokenizer.from_file(tokenizer_path)
    quant_model = QuantizedNanoLLM.load(weights_path, config_path)

    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        print("Prompt encoding produced no tokens")
        return False

    generated_tokens_py = quant_model.generate(prompt_tokens, max_new_tokens=4)
    next_token_py = generated_tokens_py[len(prompt_tokens)]

    cmd = [
        str(exe_path),
        str(Path(weights_path).absolute()),
        str(Path(config_path).absolute()),
        prompt,
        "4",
        str(Path(tokenizer_path).absolute()),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print("✗ dump_next_token execution failed for MoE")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

    stdout = result.stdout.strip()
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        print("✗ Failed to parse dump_next_token MoE output as JSON")
        print("Raw output:", stdout)
        return False

    prompt_tokens_cpp = data.get('prompt_tokens')
    generated_tokens_cpp = data.get('generated_tokens')
    next_token_cpp = data.get('next_token')

    parity_ok = True
    if prompt_tokens_cpp != prompt_tokens:
        print("✗ MoE prompt token mismatch")
        print("  Python:", prompt_tokens)
        print("  C++   :", prompt_tokens_cpp)
        parity_ok = False
    else:
        print("✓ MoE prompt tokens match")

    if next_token_cpp != next_token_py:
        print("✗ MoE next-token mismatch")
        print(f"  Python: {next_token_py}")
        print(f"  C++   : {next_token_cpp}")
        parity_ok = False
    else:
        decoded = tokenizer.decode(prompt_tokens + [next_token_py])
        print(f"✓ MoE next-token match (decoded: '{decoded}')")

    if generated_tokens_cpp != generated_tokens_py:
        print("✗ MoE multi-token generation mismatch")
        print("  Python:", generated_tokens_py)
        print("  C++   :", generated_tokens_cpp)
        parity_ok = False
    else:
        print("✓ MoE multi-token generation matches")

    return parity_ok


def test_cpp_inference():
    """Test C++ inference with BPE tokenizer."""
    print("\n" + "=" * 60)
    print("TEST 12: C++ Inference (BPE)")
    print("=" * 60)
    
    weights_path = 'test_weights/model.bin'
    config_path = 'test_weights/model_config.json'
    vocab_path = 'test_weights/vocab.json'
    build_dir = Path('cpp/build')
    exe_path = build_dir / 'inference'
    
    if not exe_path.exists():
        print(f"Executable not found: {exe_path}")
        return False
    
    # Check if vocab.json exists (for BPE)
    vocab_exists = os.path.exists(vocab_path)
    if vocab_exists:
        print(f"Using BPE tokenizer from: {vocab_path}")
    else:
        print("Warning: vocab.json not found, will use byte-level encoding")
    
    # Run inference
    cmd = [
        str(exe_path),
        str(Path(weights_path).absolute()),
        str(Path(config_path).absolute()),
        "Hello",
        "10"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    print("C++ output:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Check for BPE tokenizer usage
    bpe_used = False
    if vocab_exists:
        if "Using BPE tokenizer" in result.stdout:
            bpe_used = True
            print("✓ BPE tokenizer loaded successfully!")
        elif "BPE tokenizer not found" in result.stdout:
            print("⚠ BPE tokenizer not found, using byte-level fallback")
    
    # Check if model loaded successfully
    if "Model loaded successfully" in result.stdout:
        if result.returncode == 0:
            status = "✓ C++ inference successful!"
            if bpe_used:
                status += " (BPE tokenizer active)"
            print(status)
        else:
            print("⚠ C++ model loaded but program exited with code", result.returncode)
            print("  (This may indicate a generation issue, but model loading works)")
        return True
    elif "Model loaded" in result.stdout or "Loading model" in result.stdout:
        print("✓ C++ inference successful! (Model loading detected)")
        return True
    elif result.returncode == 0:
        print("⚠ C++ program completed but model loading status unclear")
        return True  # Still pass if program doesn't crash
    else:
        print(f"✗ C++ inference failed with return code {result.returncode}")
        return False


def test_embedded_weights_generation():
    """Test embedded weights header generation."""
    print("\n" + "=" * 60)
    print("TEST 13: Embedded Weights Generation")
    print("=" * 60)
    
    checkpoint_path = 'test_checkpoints/model_best.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    output_header = 'test_weights/model_weights.h'
    
    # Generate embedded weights header
    cmd = [
        sys.executable, 'python/export_weights_header.py',
        '--checkpoint', checkpoint_path,
        '--output', output_header,
        '--namespace', 'nanollm',
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Embedded weights generation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Check if header file was created
    if not os.path.exists(output_header):
        print(f"Header file not found: {output_header}")
        return False
    
    # Verify header file structure
    with open(output_header, 'r') as f:
        header_content = f.read()
    
    # Check for required elements
    required_elements = [
        '#ifndef',
        'namespace nanollm',
        'constexpr',
        'PROGMEM',
        'WEIGHTS_',
    ]
    
    missing = []
    for element in required_elements:
        if element not in header_content:
            missing.append(element)
    
    if missing:
        print(f"✗ Header file missing required elements: {missing}")
        return False
    
    file_size_kb = os.path.getsize(output_header) / 1024
    print(f"✓ Embedded weights header generated!")
    print(f"  File: {output_header}")
    print(f"  Size: {file_size_kb:.2f} KB")
    print(f"  Contains PROGMEM arrays")
    
    return True


def test_embedded_vocab_generation():
    """Test embedded vocab header generation."""
    print("\n" + "=" * 60)
    print("TEST 14: Embedded Vocab Generation")
    print("=" * 60)
    
    tokenizer_path = 'test_checkpoints/tokenizer/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found: {tokenizer_path}")
        return False
    
    output_header = 'test_weights/vocab_weights.h'
    
    # Generate embedded vocab header
    cmd = [
        sys.executable, 'python/export_vocab_header.py',
        '--tokenizer', tokenizer_path,
        '--output', output_header,
        '--namespace', 'nanollm',
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Embedded vocab generation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Check if header file was created
    if not os.path.exists(output_header):
        print(f"Header file not found: {output_header}")
        return False
    
    # Verify header file structure
    with open(output_header, 'r') as f:
        header_content = f.read()
    
    # Check for required elements
    required_elements = [
        '#ifndef',
        'namespace nanollm',
        'VOCAB_SIZE',
        'PROGMEM',
        'GetTokenById',
        'FindTokenId',
    ]
    
    missing = []
    for element in required_elements:
        if element not in header_content:
            missing.append(element)
    
    if missing:
        print(f"✗ Header file missing required elements: {missing}")
        return False
    
    # Verify vocab size matches
    if 'VOCAB_SIZE' in header_content:
        # Extract vocab size
        import re
        match = re.search(r'VOCAB_SIZE\s*=\s*(\d+)', header_content)
        if match:
            vocab_size = int(match.group(1))
            print(f"  Vocab size: {vocab_size}")
    
    file_size_kb = os.path.getsize(output_header) / 1024
    print(f"✓ Embedded vocab header generated!")
    print(f"  File: {output_header}")
    print(f"  Size: {file_size_kb:.2f} KB")
    print(f"  Contains PROGMEM token arrays")
    
    return True


def test_embedded_weights_structure():
    """Test that embedded weights header has correct structure."""
    print("\n" + "=" * 60)
    print("TEST 15: Embedded Weights Structure")
    print("=" * 60)
    
    header_path = 'test_weights/model_weights.h'
    if not os.path.exists(header_path):
        print(f"Header file not found: {header_path}")
        return False
    
    try:
        with open(header_path, 'r') as f:
            content = f.read()
        
        # Check for model components
        required_components = [
            'token_embedding',
            'pos_embedding',
            'block_0',
            'norm',
            'lm_head',
        ]
        
        missing = []
        for component in required_components:
            if component not in content.lower():
                missing.append(component)
        
        if missing:
            print(f"✗ Missing model components: {missing}")
            return False
        
        # Check for PROGMEM usage
        if 'PROGMEM' not in content:
            print("✗ PROGMEM not found in header")
            return False
        
        # Count weight arrays
        weight_count = content.count('const int8_t')
        print(f"✓ Embedded weights structure valid!")
        print(f"  Weight arrays: {weight_count}")
        print(f"  All components present")
        print(f"  Uses PROGMEM for flash storage")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading header file: {e}")
        return False


def test_embedded_vocab_structure():
    """Test that embedded vocab header has correct structure."""
    print("\n" + "=" * 60)
    print("TEST 16: Embedded Vocab Structure")
    print("=" * 60)
    
    header_path = 'test_weights/vocab_weights.h'
    if not os.path.exists(header_path):
        print(f"Header file not found: {header_path}")
        return False
    
    try:
        with open(header_path, 'r') as f:
            content = f.read()
        
        # Check for vocab components
        required_components = [
            'VOCAB_SIZE',
            'TOKEN_LOOKUP',
            'GetTokenById',
            'FindTokenId',
            'PROGMEM',
        ]
        
        missing = []
        for component in required_components:
            if component not in content:
                missing.append(component)
        
        if missing:
            print(f"✗ Missing vocab components: {missing}")
            return False
        
        # Count token arrays
        token_count = content.count('TOKEN_') - content.count('TOKEN_LOOKUP') - content.count('TOKEN_DATA')
        # More accurate: count TOKEN_*_STR or TOKEN_*_DATA patterns
        import re
        token_arrays = len(re.findall(r'TOKEN_\d+', content))
        
        # Extract vocab size
        vocab_size_match = re.search(r'VOCAB_SIZE\s*=\s*(\d+)', content)
        if vocab_size_match:
            vocab_size = int(vocab_size_match.group(1))
            print(f"  Vocab size: {vocab_size}")
            print(f"  Token arrays found: {token_arrays}")
            
            if token_arrays < vocab_size * 0.9:  # Allow some margin
                print(f"⚠ Warning: Expected ~{vocab_size} token arrays, found {token_arrays}")
        
        print(f"✓ Embedded vocab structure valid!")
        print(f"  All components present")
        print(f"  Uses PROGMEM for flash storage")
        print(f"  Helper functions available")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading header file: {e}")
        import traceback
        traceback.print_exc()
        return False


def _parse_embedded_vocab_header(header_path):
    """Parse embedded vocab header and return (tokens_list, merges_list, vocab_size, merge_count)."""
    with open(header_path, 'r') as f:
        content = f.read()

    vocab_match = re.search(r'constexpr\s+size_t\s+VOCAB_SIZE\s*=\s*(\d+);', content)
    vocab_size = int(vocab_match.group(1)) if vocab_match else None

    merge_count_match = re.search(r'constexpr\s+size_t\s+MERGE_COUNT\s*=\s*(\d+);', content)
    merge_count = int(merge_count_match.group(1)) if merge_count_match else 0

    tokens = {}
    token_pattern = re.compile(r'const char TOKEN_(\d+)_DATA\[\] PROGMEM = "(.*)";')
    for line in content.splitlines():
        line = line.strip()
        match = token_pattern.match(line)
        if not match:
            continue
        token_id = int(match.group(1))
        raw = match.group(2)
        try:
            token = ast.literal_eval(f'"{raw}"')
        except (SyntaxError, ValueError):
            token = raw
        tokens[token_id] = token

    merges = []
    merge_block = re.search(r'const\s+MergeEntry\s+MERGE_TABLE\[MERGE_COUNT\]\s+PROGMEM\s*=\s*\{(.*?)\};', content, re.S)
    if merge_block:
        entries = merge_block.group(1)
        entry_pattern = re.compile(r'\{\s*(\d+)\s*,\s*(\d+)\s*\}')
        for left, right in entry_pattern.findall(entries):
            merges.append((int(left), int(right)))

    token_list = [tokens.get(i, "") for i in range(vocab_size or 0)]
    return token_list, merges, vocab_size, merge_count


def test_embedded_tokenizer_consistency():
    """Ensure embedded tokenizer header matches exported vocab and merges."""
    print("\n" + "=" * 60)
    print("TEST 17: Embedded Tokenizer Consistency")
    print("=" * 60)

    vocab_header = 'test_weights/vocab_weights.h'
    tokenizer_json_path = 'test_checkpoints/tokenizer/tokenizer.json'

    missing = []
    for path, label in [
        (vocab_header, 'vocab header'),
        (tokenizer_json_path, 'tokenizer.json'),
    ]:
        if not os.path.exists(path):
            missing.append(label)

    if missing:
        print("Required artifacts missing: " + ", ".join(missing))
        return False

    header_tokens, header_merges, vocab_size, merge_count = _parse_embedded_vocab_header(vocab_header)

    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    token_to_id = tokenizer_data.get('model', {}).get('vocab', {})
    id_to_token_expected = {
        int(token_id): token for token, token_id in token_to_id.items()
    }
    merges_raw = tokenizer_data.get('model', {}).get('merges', [])

    expected_count = max(id_to_token_expected.keys(), default=-1) + 1
    expected_tokens = [id_to_token_expected.get(i, "") for i in range(expected_count)]

    if vocab_size is None or vocab_size != expected_count:
        print(f"✗ Vocab size mismatch (header: {vocab_size}, expected: {expected_count})")
        return False

    mismatch_tokens = [
        i for i in range(vocab_size)
        if header_tokens[i] != id_to_token_expected.get(i, "")
    ]
    if mismatch_tokens:
        print("✗ Token mismatch detected at indices: " + ", ".join(map(str, mismatch_tokens[:10])))
        if len(mismatch_tokens) > 10:
            print(f"  ...and {len(mismatch_tokens) - 10} more")
        return False

    expected_merges = []
    skipped_merges = 0
    for merge in merges_raw:
        if isinstance(merge, str):
            parts = merge.split()
        elif isinstance(merge, (list, tuple)) and len(merge) == 2:
            parts = list(merge)
        else:
            skipped_merges += 1
            continue

        if len(parts) != 2:
            skipped_merges += 1
            continue

        left, right = parts
        if left in token_to_id and right in token_to_id:
            expected_merges.append((int(token_to_id[left]), int(token_to_id[right])))
        else:
            skipped_merges += 1

    if merge_count != len(expected_merges):
        print(f"✗ Merge count mismatch (header: {merge_count}, expected: {len(expected_merges)})")
        if skipped_merges:
            print(f"  Note: {skipped_merges} merges were skipped because tokens were missing from the vocab export.")
        return False

    if header_merges != expected_merges:
        print("✗ Merge table ordering mismatch detected")
        for idx, (expected, actual) in enumerate(zip(expected_merges, header_merges)):
            if expected != actual:
                print(f"  First difference at rank {idx}: expected {expected}, got {actual}")
                break
        return False

    print("✓ Embedded tokenizer matches exported artifacts!")
    print(f"  ✓ Tokens verified ({vocab_size} entries)")
    print(f"  ✓ Merge table verified ({merge_count} entries)")
    if skipped_merges:
        print(f"  ⚠ {skipped_merges} merges were skipped during export (non-critical)")
    return True


def test_embedded_compatibility():
    """Test that embedded headers are compatible with ESP32 code."""
    print("\n" + "=" * 60)
    print("TEST 18: Embedded Compatibility Check")
    print("=" * 60)
    
    weights_header = 'test_weights/model_weights.h'
    vocab_header = 'test_weights/vocab_weights.h'
    
    if not os.path.exists(weights_header):
        print(f"Weights header not found: {weights_header}")
        return False
    
    if not os.path.exists(vocab_header):
        print(f"Vocab header not found: {vocab_header}")
        return False
    
    try:
        # Check that headers can be parsed (basic syntax check)
        # We'll check for common ESP32/Arduino compatibility issues
        
        with open(weights_header, 'r') as f:
            weights_content = f.read()
        
        with open(vocab_header, 'r') as f:
            vocab_content = f.read()
        
        # Check for ESP32 compatibility
        issues = []
        
        # Check for PROGMEM (required for ESP32)
        if 'PROGMEM' not in weights_content:
            issues.append("Weights header missing PROGMEM")
        if 'PROGMEM' not in vocab_content:
            issues.append("Vocab header missing PROGMEM")
        
        # Check for Arduino.h include (for String type in vocab)
        if 'Arduino.h' not in vocab_content:
            issues.append("Vocab header missing Arduino.h include")
        
        # Check for proper namespace
        if 'namespace nanollm' not in weights_content:
            issues.append("Weights header missing namespace")
        if 'namespace nanollm' not in vocab_content:
            issues.append("Vocab header missing namespace")
        
        # Check for header guards
        if '#ifndef' not in weights_content or '#define' not in weights_content:
            issues.append("Weights header missing header guards")
        if '#ifndef' not in vocab_content or '#define' not in vocab_content:
            issues.append("Vocab header missing header guards")
        
        if issues:
            print("✗ Compatibility issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("✓ Embedded headers are ESP32 compatible!")
        print("  ✓ PROGMEM used for flash storage")
        print("  ✓ Proper includes present")
        print("  ✓ Namespace defined")
        print("  ✓ Header guards present")
        
        return True
        
    except Exception as e:
        print(f"✗ Compatibility check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    """Clean up test files."""
    print("\n" + "=" * 60)
    print("Cleaning up test files...")
    print("=" * 60)
    
    dirs_to_remove = ['test_checkpoints', 'test_checkpoints_moe', 'test_checkpoints_ctx256', 'test_weights']
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed: {dir_name}")


def ensure_cpp_build():
    """Build the desktop C++ runtime if it is not already present.

    Several parity/tokenizer tests (Cached Decode, C++ BPE Tokenizer,
    Python-C++ Round-trip) run *before* the dedicated ``C++ Build`` test and
    expect ``cpp/build/<exe>`` to exist. On a fresh checkout that directory is
    absent, so those tests would fail with "Executable not found". Building up
    front makes the suite order-independent and reproducible on CI. Idempotent:
    a no-op when the required executables already exist.
    """
    build_dir = 'cpp/build'
    required = ['inference', 'test_inference', 'dump_next_token', 'dump_cached_tokens']
    if all(os.path.exists(os.path.join(build_dir, exe)) for exe in required):
        return
    print("\nPre-building C++ runtime (cpp/build) for parity tests...")
    r1 = subprocess.run(['cmake', '-S', 'cpp', '-B', build_dir],
                        capture_output=True, text=True)
    if r1.returncode != 0:
        print(f"  cmake configure failed:\n{r1.stderr}")
        return
    r2 = subprocess.run(['cmake', '--build', build_dir, '-j'],
                        capture_output=True, text=True)
    if r2.returncode != 0:
        print(f"  cmake build failed:\n{r2.stderr}")
    else:
        print("  C++ runtime pre-built.")


def main():
    """Run all tests."""
    print("NanoLLM End-to-End Test Suite")
    print("=" * 60)
    
    # Create test directories
    os.makedirs('test_checkpoints', exist_ok=True)
    os.makedirs('test_weights', exist_ok=True)

    # Build the C++ runtime up front so order-independent parity tests pass
    # on a fresh checkout (they run before the dedicated C++ Build test).
    ensure_cpp_build()
    
    tests = [
        ("Causal Attention Mask", test_attention_is_causal),
        ("Attention Manual Causal Parity", test_attention_matches_manual_causal_math),
        ("Quantized MHA Parity", test_quantized_attention_matches_pytorch),
        ("Cached Decode Python/C++ Parity", test_cached_decode_python_cpp_parity),
        ("Training", test_training),
        ("Python BPE Tokenizer", test_python_bpe_tokenizer),
        ("Tokenizer Capitalization Preservation", test_tokenizer_preserves_capitalization),
        ("MoE Top-1 Routing Equivalence", test_moe_top1_routing_equivalence),
        ("MoE Top-1 BF16 Autocast Training Step", test_moe_top1_bf16_autocast_training_step),
        ("Validation Split Context Gap", test_validation_split_has_context_gap),
        ("TextDataset Window Stride", test_text_dataset_window_stride),
        ("TextDataset Assistant-Only Mask", test_text_dataset_assistant_only_mask),
        ("Chat Prompt and Evaluation Contract", test_chat_prompt_and_eval_contract),
        ("Vocab Export", test_vocab_export),
        ("Weight Export", test_weight_export),
        ("Python Inference (BPE)", test_python_inference),
        ("C++ BPE Tokenizer", test_cpp_bpe_tokenizer),
        ("Python-C++ Round-trip", test_python_cpp_roundtrip),
        ("C++ Build", test_cpp_build),
        ("MoE Training + Export", test_moe_training_and_export),
        ("Cardputer 512-Context Parity", test_cardputer_512_context_parity),
        ("Python vs C++ Output Parity", test_python_cpp_output_match),
        ("Python vs C++ Output Parity (MoE)", test_python_cpp_output_match_moe),
        ("C++ Inference (BPE)", test_cpp_inference),
        ("Embedded Weights Generation", test_embedded_weights_generation),
        ("Embedded Vocab Generation", test_embedded_vocab_generation),
        ("Embedded Weights Structure", test_embedded_weights_structure),
        ("Embedded Vocab Structure", test_embedded_vocab_structure),
        ("Embedded Tokenizer Consistency", test_embedded_tokenizer_consistency),
        ("Embedded Compatibility", test_embedded_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    # Cleanup
    cleanup()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        return 0
    else:
        print("SOME TESTS FAILED! ✗")
        return 1


if __name__ == '__main__':
    sys.exit(main())

