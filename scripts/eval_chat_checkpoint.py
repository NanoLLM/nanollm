#!/usr/bin/env python3
"""Run fixed chat prompts against a checkpoint for qualitative evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_eval import (  # noqa: E402
    DEFAULT_CHAT_PROMPTS,
    HELD_OUT_CHAT_PROMPTS,
    aggregate_chat_eval_score,
    print_chat_eval,
    run_chat_eval,
    run_structured_chat_eval,
    save_chat_eval,
)
from model import NanoLLM  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate chat quality on fixed prompts")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer path (default: checkpoint_dir/tokenizer/tokenizer.json)",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument(
        "--suite",
        choices=["default", "held_out", "both"],
        default="default",
        help="Prompt suite to evaluate",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional JSON file with a 'prompts' list",
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=None,
        help="Versioned JSON transfer suite with structured cases and rubrics",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint_data.get("config") or {}
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        tokenizer_path = checkpoint_path.parent / "tokenizer" / "tokenizer.json"
    tokenizer_path = tokenizer_path.expanduser().resolve()
    if not tokenizer_path.is_file():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = NanoLLM(
        vocab_size=config.get("vocab_size"),
        d_model=config.get("d_model"),
        n_layers=config.get("n_layers"),
        n_heads=config.get("n_heads"),
        n_kv_heads=config.get("n_kv_heads", config.get("n_heads")),
        d_ff=config.get("d_ff"),
        max_seq_len=config.get("max_seq_len", 128),
        dropout=config.get("dropout", 0.0),
        use_moe=config.get("use_moe", False),
        moe_n_experts=config.get("moe_n_experts", 4),
        moe_top_k=config.get("moe_top_k", 1),
        moe_shared_d_ff=config.get("moe_shared_d_ff", 0),
        use_rope=config.get("use_rope", False),
    )
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer.from_file(str(tokenizer_path))
    if args.cases_file is not None and args.prompts_file is not None:
        raise SystemExit("--cases-file and --prompts-file are mutually exclusive")

    if args.cases_file is not None:
        suite_payload = json.loads(args.cases_file.read_text(encoding="utf-8"))
        cases = list(suite_payload.get("cases") or [])
        if not cases:
            raise SystemExit(f"No cases found in {args.cases_file}")
        results = run_structured_chat_eval(
            model,
            tokenizer,
            device,
            cases=cases,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            greedy=args.greedy,
        )
        category_scores: dict[str, list[float]] = defaultdict(list)
        for row in results:
            value = float(row["scores"]["task_success"])
            category_scores[row["category"]].append(value)
            print(
                f"[transfer_eval] {row['id']} {row['prompt']!r} -> "
                f"{row['reply']!r} (task_success={value:.0f})"
            )
        score = sum(float(row["scores"]["task_success"]) for row in results) / len(results)
        payload = {
            "schema_version": 1,
            "suite": suite_payload.get("name", args.cases_file.stem),
            "suite_version": suite_payload.get("version"),
            "aggregate_score": score,
            "category_scores": {
                key: sum(values) / len(values) for key, values in sorted(category_scores.items())
            },
            "results": results,
        }
        print(f"[transfer_eval] aggregate_score={score:.3f}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.output}")
        return 0

    if args.prompts_file is not None:
        payload = json.loads(args.prompts_file.read_text(encoding="utf-8"))
        prompts = list(payload.get("prompts") or payload)
    elif args.suite == "held_out":
        prompts = list(HELD_OUT_CHAT_PROMPTS)
    elif args.suite == "both":
        prompts = list(DEFAULT_CHAT_PROMPTS) + list(HELD_OUT_CHAT_PROMPTS)
    else:
        prompts = list(DEFAULT_CHAT_PROMPTS)

    results = run_chat_eval(
        model,
        tokenizer,
        device,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        greedy=args.greedy,
    )
    score = aggregate_chat_eval_score(results)
    print_chat_eval(results, prefix="[chat_eval]", score=score)
    if args.output:
        save_chat_eval(results, args.output, score=score)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
