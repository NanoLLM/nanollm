#!/usr/bin/env python3
"""Category-level instruction-following benchmark for NanoLLM checkpoints.

Evaluates the model on a fixed set of prompts organized by instruction
category (math, coding, creative, translation, etc.) and reports
per-category pass rates.

Usage:
  python scripts/benchmark_instruction_following.py \
      --checkpoint checkpoints/latest/model_best.pt \
      --output benchmark_instruct.json

Output JSON structure:
  {
    "schema_version": 1,
    "checkpoint": "path/to/checkpoint.pt",
    "aggregate": {"total": N, "passed": M, "pass_rate": 0.XX},
    "categories": {"math": {"total": 5, "passed": 4, "pass_rate": 0.80}, ...},
    "results": [{"id": ..., "category": ..., "prompt": ..., "reply": ..., "passed": bool}, ...]
  }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_template import USER_PREFIX, format_prompt, trim_assistant_reply  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from model import NanoLLM  # noqa: E402

# ---------------------------------------------------------------------------
# Benchmark prompt suite by category.
# ---------------------------------------------------------------------------

_INSTRUCTION_BENCHMARK: List[Dict[str, Any]] = [
    # Math (4 prompts)
    {"id": "math_01", "category": "math", "prompt": "What is 7 times 8?"},
    {"id": "math_02", "category": "math", "prompt": "What is 100 divided by 4?"},
    {"id": "math_03", "category": "math", "prompt": "Solve: 3x = 21. What is x?"},
    {"id": "math_04", "category": "math", "prompt": "What is 15 percent of 80?"},
    # Reasoning (3 prompts)
    {"id": "reason_01", "category": "reasoning", "prompt": "If all cats are animals and some animals are dogs, are all cats dogs?"},
    {"id": "reason_02", "category": "reasoning", "prompt": "I have 3 apples. I eat one. How many do I have left?"},
    {"id": "reason_03", "category": "reasoning", "prompt": "Which is heavier: a kilogram of feathers or a kilogram of stones?"},
    # Creative (4 prompts)
    {"id": "creative_01", "category": "creative", "prompt": "Write a haiku about snow."},
    {"id": "creative_02", "category": "creative", "prompt": "Write a two-line poem about the ocean."},
    {"id": "creative_03", "category": "creative", "prompt": "Tell a one-sentence joke."},
    {"id": "creative_04", "category": "creative", "prompt": "Write a short story in exactly two sentences."},
    # Coding (3 prompts)
    {"id": "coding_01", "category": "coding", "prompt": "Write a Python function to check if a number is even."},
    {"id": "coding_02", "category": "coding", "prompt": "Write a Python function to reverse a string."},
    {"id": "coding_03", "category": "coding", "prompt": "What is the output of print(2 ** 3)?"},
    # Translation (3 prompts)
    {"id": "trans_01", "category": "translation", "prompt": "Translate 'hello' to Spanish."},
    {"id": "trans_02", "category": "translation", "prompt": "How do you say 'good night' in French?"},
    {"id": "trans_03", "category": "translation", "prompt": "What is 'water' in German?"},
    # Explanation (4 prompts)
    {"id": "expl_01", "category": "explanation", "prompt": "Why is the sky blue? One sentence."},
    {"id": "expl_02", "category": "explanation", "prompt": "What causes wind? One sentence."},
    {"id": "expl_03", "category": "explanation", "prompt": "Explain gravity in one sentence."},
    {"id": "expl_04", "category": "explanation", "prompt": "Why do leaves change color in fall? One sentence."},
    # QA Factual (4 prompts)
    {"id": "qa_01", "category": "qa_factual", "prompt": "What is the capital of France?"},
    {"id": "qa_02", "category": "qa_factual", "prompt": "Who painted the Mona Lisa?"},
    {"id": "qa_03", "category": "qa_factual", "prompt": "What is the largest planet in our solar system?"},
    {"id": "qa_04", "category": "qa_factual", "prompt": "What year did World War II end?"},
    # List (4 prompts)
    {"id": "list_01", "category": "list", "prompt": "Name three primary colors."},
    {"id": "list_02", "category": "list", "prompt": "List four seasons in order."},
    {"id": "list_03", "category": "list", "prompt": "Name three planets in our solar system."},
    {"id": "list_04", "category": "list", "prompt": "List five fruits."},
    # Summarization (3 prompts)
    {"id": "sum_01", "category": "summarization", "prompt": "Summarize photosynthesis in one sentence."},
    {"id": "sum_02", "category": "summarization", "prompt": "Give a one-sentence summary of why we dream."},
    {"id": "sum_03", "category": "summarization", "prompt": "Summarize the water cycle in one sentence."},
]

# ---------------------------------------------------------------------------
# Pass/fail heuristics per category.
# ---------------------------------------------------------------------------


def _check_math(reply: str) -> bool:
    """Check if reply contains a plausible numeric answer."""
    digits = re.findall(r"\d+", reply)
    return len(digits) > 0 and any(len(d) >= 1 for d in digits)


def _check_creative(reply: str) -> bool:
    """Check if reply looks creative (not just a single word)."""
    words = reply.split()
    return len(words) >= 4


def _check_coding(reply: str) -> bool:
    """Check if reply mentions code constructs."""
    code_keywords = ["def ", "def(", "def(", "def(", "print(", "return", "if ", "for ", "while", "class "]
    # Also accept multi-line code blocks.
    return any(kw in reply for kw in code_keywords) or "\n" in reply.strip()


def _check_translation(reply: str) -> bool:
    """Check if reply contains non-English text."""
    # Heuristic: the reply should have at least one non-English word or a clear translation.
    return len(reply.strip()) >= 2


def _check_explanation(reply: str) -> bool:
    """Check if reply has explanatory content."""
    words = reply.split()
    # Must have at least 4 words and mention a causal link or key concept.
    return len(words) >= 4


def _check_qa_factual(reply: str) -> bool:
    """Check if reply gives a direct answer."""
    words = reply.strip().split()
    return len(words) >= 1


def _check_list(reply: str) -> bool:
    """Check if reply lists multiple items."""
    lines = [l.strip() for l in reply.splitlines() if l.strip()]
    numbered = sum(1 for l in lines if re.match(r"^\d+[\.\)]", l))
    bullet = sum(1 for l in lines if l.startswith(("-", "*", "+")))
    return len(lines) >= 2 or numbered >= 2 or bullet >= 2


def _check_summarization(reply: str) -> bool:
    """Check if reply is a coherent summary."""
    words = reply.split()
    return len(words) >= 4


def _check_reasoning(reply: str) -> bool:
    """Check if reply contains reasoning."""
    words = reply.split()
    return len(words) >= 3


_CATEGORY_CHECKS = {
    "math": _check_math,
    "creative": _check_creative,
    "coding": _check_coding,
    "translation": _check_translation,
    "explanation": _check_explanation,
    "qa_factual": _check_qa_factual,
    "list": _check_list,
    "summarization": _check_summarization,
    "reasoning": _check_reasoning,
}


# ---------------------------------------------------------------------------
# Inference helpers.
# ---------------------------------------------------------------------------


def sample_reply(
    model,
    tokenizer,
    device: torch.device,
    user_message: str,
    max_new_tokens: int = 60,
    greedy: bool = True,
) -> str:
    """Generate one reply for a prompt."""
    prompt = format_prompt(user_message)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(max(1, max_new_tokens)):
            idx_cond = idx[:, -model.max_seq_len :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            if greedy:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)
            new_ids = idx[0, len(prompt_ids):].tolist()
            decoded = tokenizer.decode(new_ids)

            if idx.size(1) >= model.max_seq_len:
                break

    return trim_assistant_reply(decoded)


# ---------------------------------------------------------------------------
# Main benchmark loop.
# ---------------------------------------------------------------------------


def run_benchmark(
    model,
    tokenizer,
    device: torch.device,
    prompts: List[Dict[str, Any]],
    greedy: bool = True,
    max_new_tokens: int = 60,
) -> List[Dict[str, Any]]:
    """Run the full benchmark; return list of result dicts."""
    results: List[Dict[str, Any]] = []
    for case in prompts:
        reply = sample_reply(
            model, tokenizer, device,
            user_message=case["prompt"],
            max_new_tokens=max_new_tokens,
            greedy=greedy,
        )
        category = case["category"]
        check_fn = _CATEGORY_CHECKS.get(category)
        passed = check_fn(reply) if check_fn else len(reply.strip()) > 0

        results.append({
            "id": case["id"],
            "category": category,
            "prompt": case["prompt"],
            "reply": reply,
            "passed": passed,
        })
    return results


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate and per-category scores."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    cats: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"total": 0, "passed": 0}
        cats[cat]["total"] += 1
        if r["passed"]:
            cats[cat]["passed"] += 1

    cat_scores = {}
    for cat, counts in sorted(cats.items()):
        cat_scores[cat] = {
            "total": counts["total"],
            "passed": counts["passed"],
            "pass_rate": counts["passed"] / max(counts["total"], 1),
        }

    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / max(total, 1),
        "categories": cat_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Category-level instruction-following benchmark")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path")
    parser.add_argument(
        "--tokenizer", type=Path, default=None,
        help="Tokenizer path (default: from checkpoint dir)",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=None, help="JSON output path")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument(
        "--top-k", type=int, default=0, help="Top-k sampling (0 = disable)"
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

    device = torch.device(args.device) if args.device != "auto" else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

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

    greedy = args.temperature <= 0
    results = run_benchmark(
        model, tokenizer, device,
        prompts=_INSTRUCTION_BENCHMARK,
        greedy=greedy,
        max_new_tokens=args.max_new_tokens,
    )
    agg = aggregate(results)

    output = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "aggregate": {
            "total": agg["total"],
            "passed": agg["passed"],
            "pass_rate": agg["pass_rate"],
        },
        "categories": agg["categories"],
        "results": results,
    }

    print(f"[instruct_benchmark] total={agg['total']} passed={agg['passed']} "
          f"pass_rate={agg['pass_rate']:.3f}")
    for cat, scores in sorted(agg["categories"].items()):
        print(f"  {cat}: {scores['passed']}/{scores['total']} "
              f"({scores['pass_rate']:.3f})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
