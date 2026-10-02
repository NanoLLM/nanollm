#!/usr/bin/env python3
"""Compare Python QuantizedNanoLLM against C++ dump_next_token."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from quantized_runtime import QuantizedNanoLLM  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Compare Python int8 runtime vs C++ dump_next_token")
    parser.add_argument("--weights", default="weights/model.bin")
    parser.add_argument("--config", default="weights/model_config.json")
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json path")
    parser.add_argument("--cpp-bin", default="cpp/build/dump_next_token")
    parser.add_argument("--prompts", nargs="+", default=["hello", "Hello world"])
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Compare cached decode path (dump_cached_tokens) instead of full recompute generate",
    )
    args = parser.parse_args()

    weights = Path(args.weights)
    config = Path(args.config)
    cpp_bin = Path(args.cpp_bin)
    if args.cached:
        cpp_bin = Path(args.cpp_bin).with_name("dump_cached_tokens")
    if not weights.is_file() or not config.is_file():
        raise SystemExit(f"Missing weights/config: {weights}, {config}")
    if not cpp_bin.is_file():
        raise SystemExit(f"C++ binary not found: {cpp_bin}")

    tok_path = args.tokenizer
    if not tok_path:
        tok_path = str(PROJECT_ROOT / "checkpoints/moe_small_chat_20260630/tokenizer/tokenizer.json")
    tokenizer = BPETokenizer.from_file(tok_path)
    model = QuantizedNanoLLM.load(weights, config)

    all_ok = True
    for prompt in args.prompts:
        prompt_tokens = tokenizer.encode(prompt)
        py_generated = model.generate(prompt_tokens, max_new_tokens=args.max_new_tokens)
        py_next = py_generated[len(prompt_tokens)] if len(py_generated) > len(prompt_tokens) else -1
        cmd = [
            str(cpp_bin),
            str(weights),
            str(config),
            prompt,
            str(args.max_new_tokens),
            tok_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[FAIL] {prompt!r}: C++ failed\n{proc.stderr}")
            all_ok = False
            continue
        cpp_data = json.loads(proc.stdout.strip())
        cpp_next = cpp_data.get("next_token")
        cpp_tokens = cpp_data.get("prompt_tokens")
        cpp_generated = cpp_data.get("generated_tokens")
        ok = (
            py_next == cpp_next
            and cpp_tokens == prompt_tokens
            and cpp_generated == py_generated
        )
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {prompt!r}")
        print(f"  tokens: py={prompt_tokens} cpp={cpp_tokens}")
        print(f"  next:   py={py_next} cpp={cpp_next}")
        print(f"  generated: py={py_generated} cpp={cpp_generated}")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
