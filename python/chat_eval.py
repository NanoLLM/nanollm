"""Fixed-prompt chat evaluation for qualitative training monitoring."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from chat_template import USER_PREFIX, format_prompt, find_turn_boundary, trim_assistant_reply

DEFAULT_CHAT_PROMPTS: Tuple[str, ...] = (
    "hello",
    "Hello! who are you?",
    "What is 2+2?",
    "Explain gravity in one sentence.",
    "Write a haiku about rain.",
)

# Prompts reserved for post-training transfer checks; keep out of SFT text.
HELD_OUT_CHAT_PROMPTS: Tuple[str, ...] = (
    "What is 9 plus 6?",
    "Name a primary color.",
    "Explain evaporation in one sentence.",
    "Write a haiku about snow.",
    "Who invented the telephone?",
)

_NLI_NOISE = (
    "hypothesis",
    "premise",
    "entailment",
    "contradiction",
    "given sentence",
    "extract the",
    "classify the",
)


def sample_chat_reply(
    model,
    tokenizer,
    device: torch.device,
    user_message: str,
    history: Optional[List[Tuple[str, str]]] = None,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    repetition_penalty: float = 1.12,
    repetition_window: int = 64,
    greedy: bool = False,
) -> str:
    """Generate one chat reply with turn-boundary trimming."""
    prompt = format_prompt(user_message, history=history)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    decoded_prefix = ""
    raw_tokenizer = getattr(tokenizer, "tokenizer", None)
    eos_token_id = raw_tokenizer.token_to_id("<EOS>") if raw_tokenizer is not None else None

    with torch.no_grad():
        for _ in range(max(1, max_new_tokens)):
            idx_cond = idx[:, -model.max_seq_len :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-3)

            if repetition_penalty > 1.0:
                recent = idx[0, -max(1, repetition_window) :].tolist()
                for token_id in set(recent):
                    logits[0, token_id] /= repetition_penalty

            if greedy:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    logits[logits < values[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)
            new_ids = idx[0, len(prompt_ids) :].tolist()
            decoded = tokenizer.decode(new_ids)
            if eos_token_id is not None and int(idx_next.item()) == eos_token_id:
                decoded_prefix = decoded
                break
            if find_turn_boundary(decoded) is not None:
                decoded = trim_assistant_reply(decoded)
                break
            decoded_prefix = decoded

            if idx.size(1) >= model.max_seq_len:
                break

    return trim_assistant_reply(decoded_prefix if decoded_prefix else tokenizer.decode(idx[0, len(prompt_ids) :].tolist()))


def score_chat_reply(prompt: str, reply: str) -> Dict[str, float]:
    """Score one prompt/reply pair for deterministic model selection."""
    scores: Dict[str, float] = {}
    text = reply.strip()
    lower = text.lower()
    prompt_lower = prompt.lower()

    scores["turn_boundary"] = 1.0 if (
        find_turn_boundary(text) is None
        and USER_PREFIX not in text
        and "Assistant:" not in text
    ) else 0.0

    words = re.findall(r"\b\w+\b", lower)
    if words:
        scores["repetition"] = len(set(words)) / len(words)
    else:
        scores["repetition"] = 0.0

    if any(phrase in lower for phrase in _NLI_NOISE):
        scores["nli_noise"] = 0.0
    else:
        scores["nli_noise"] = 1.0

    if "2+2" in prompt_lower or "2 + 2" in prompt_lower:
        scores["arithmetic"] = 1.0 if re.search(r"(?<![\d.])4(?![\d.])", text) else 0.0
    elif "9 plus 6" in prompt_lower or "9+6" in prompt_lower:
        scores["arithmetic"] = 1.0 if re.search(r"(?<![\d.])15(?![\d.])", text) else 0.0

    if "who are you" in prompt_lower:
        identity_pattern = re.compile(r"\b(?:assistant|ai|nanollm|model)\b|language model")
        scores["identity"] = (
            1.0
            if len(text) >= 8
            and scores["nli_noise"] > 0
            and identity_pattern.search(lower) is not None
            else 0.0
        )

    if "gravity" in prompt_lower:
        gravity_markers = ("gravity", "pull", "mass", "attract", "force")
        scores["explanation"] = (
            1.0
            if 12 <= len(text) <= 220
            and scores["nli_noise"] > 0
            and any(marker in lower for marker in gravity_markers)
            else 0.0
        )
    elif "evaporation" in prompt_lower:
        evap_markers = ("evaporat", "vapor", "vapour", "liquid", "gas", "heat", "water")
        scores["explanation"] = (
            1.0
            if 12 <= len(text) <= 220
            and scores["nli_noise"] > 0
            and any(marker in lower for marker in evap_markers)
            else 0.0
        )

    if "haiku" in prompt_lower:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        topic_markers = ("rain", "drop", "cloud", "puddle", "storm", "wet", "drizzle")
        if "snow" in prompt_lower:
            topic_markers = ("snow", "flake", "winter", "cold", "ice", "frost", "white")
        on_topic = True
        if "rain" in prompt_lower or "snow" in prompt_lower:
            on_topic = any(marker in lower for marker in topic_markers)
        if len(lines) == 3 and on_topic:
            scores["instruction"] = 1.0
        elif len(lines) == 2 and on_topic:
            scores["instruction"] = 0.5
        else:
            scores["instruction"] = 0.0

    if "primary color" in prompt_lower:
        scores["factual"] = (
            1.0
            if any(color in lower for color in ("red", "blue", "yellow"))
            and scores["nli_noise"] > 0
            else 0.0
        )

    if "telephone" in prompt_lower and ("invent" in prompt_lower or "who" in prompt_lower):
        scores["factual"] = (
            1.0
            if ("bell" in lower or "alexander" in lower)
            and scores["nli_noise"] > 0
            else 0.0
        )

    return scores


def score_structured_case(reply: str, rubric: Dict[str, Any]) -> Dict[str, float]:
    """Score an immutable transfer case with an explicit, data-only rubric."""
    text = reply.strip()
    lower = text.lower()
    checks: List[bool] = []

    min_chars = rubric.get("min_chars")
    if min_chars is not None:
        checks.append(len(text) >= int(min_chars))
    max_chars = rubric.get("max_chars")
    if max_chars is not None:
        checks.append(len(text) <= int(max_chars))

    required_any = [str(value).lower() for value in rubric.get("required_any", [])]
    if required_any:
        checks.append(any(value in lower for value in required_any))
    required_all = [str(value).lower() for value in rubric.get("required_all", [])]
    if required_all:
        checks.append(all(value in lower for value in required_all))
    forbidden = [str(value).lower() for value in rubric.get("forbidden", [])]
    if forbidden:
        checks.append(not any(value in lower for value in forbidden))

    expected_regex = rubric.get("expected_regex")
    if expected_regex:
        checks.append(re.search(str(expected_regex), text, flags=re.IGNORECASE) is not None)

    line_count = rubric.get("line_count")
    if line_count is not None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        checks.append(len(lines) == int(line_count))

    structural = score_chat_reply("", reply)
    return {
        **structural,
        "task_success": 1.0 if checks and all(checks) else 0.0,
    }


def aggregate_chat_eval_score(results: Sequence[Dict[str, Any]]) -> float:
    """Combine per-prompt scores into one selection metric."""
    if not results:
        return 0.0

    weights = {
        # Structural hygiene is useful, but must not outweigh task completion.
        "turn_boundary": 0.25,
        "repetition": 0.5,
        "nli_noise": 0.25,
        "arithmetic": 3.0,
        "identity": 3.0,
        "explanation": 3.0,
        "instruction": 3.0,
        "factual": 3.0,
    }
    total = 0.0
    weight_sum = 0.0
    for row in results:
        prompt_scores = row.get("scores") or score_chat_reply(row["prompt"], row["reply"])
        for key, weight in weights.items():
            if key in prompt_scores:
                total += weight * float(prompt_scores[key])
                weight_sum += weight
    return total / max(weight_sum, 1.0)


def passes_hard_chat_eval(results: Sequence[Dict[str, Any]]) -> bool:
    """Require task-critical prompts to pass before using chat eval for selection."""
    required: Dict[str, str] = {
        "What is 2+2?": "arithmetic",
        "Write a haiku about rain.": "instruction",
    }
    passed: set[str] = set()
    for row in results:
        prompt = row.get("prompt", "")
        if prompt not in required:
            continue
        key = required[prompt]
        scores = row.get("scores") or score_chat_reply(prompt, row.get("reply", ""))
        if float(scores.get(key, 0.0)) < 1.0:
            return False
        passed.add(prompt)
    return passed == set(required)


def run_chat_eval(
    model,
    tokenizer,
    device: torch.device,
    prompts: Optional[Sequence[str]] = None,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    greedy: bool = False,
) -> List[Dict[str, Any]]:
    """Run a small fixed prompt suite and return structured results."""
    prompts = list(prompts or DEFAULT_CHAT_PROMPTS)
    results: List[Dict[str, Any]] = []
    for prompt in prompts:
        reply = sample_chat_reply(
            model,
            tokenizer,
            device,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        scores = score_chat_reply(prompt, reply)
        results.append(
            {
                "prompt": prompt,
                "reply": reply,
                "reply_chars": len(reply),
                "scores": scores,
            }
        )
    return results


def run_structured_chat_eval(
    model,
    tokenizer,
    device: torch.device,
    cases: Sequence[Dict[str, Any]],
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    greedy: bool = False,
) -> List[Dict[str, Any]]:
    """Run versioned transfer cases without coupling them to training selection."""
    results: List[Dict[str, Any]] = []
    for case in cases:
        prompt = str(case["prompt"])
        reply = sample_chat_reply(
            model,
            tokenizer,
            device,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        results.append(
            {
                "id": str(case["id"]),
                "category": str(case["category"]),
                "prompt": prompt,
                "reply": reply,
                "reply_chars": len(reply),
                "scores": score_structured_case(reply, dict(case["rubric"])),
            }
        )
    return results


def print_chat_eval(
    results: Sequence[Dict[str, Any]],
    prefix: str = "  [chat_eval]",
    score: Optional[float] = None,
) -> None:
    for row in results:
        preview = row["reply"].replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:97] + "..."
        prompt_scores = row.get("scores") or {}
        score_bits = ", ".join(f"{k}={prompt_scores[k]:.2f}" for k in sorted(prompt_scores))
        print(f"{prefix} {row['prompt']!r} -> {preview!r} ({score_bits})")
    if score is not None:
        print(f"{prefix} aggregate_score={score:.3f}")


def save_chat_eval(
    results: Sequence[Dict[str, Any]],
    path: Path,
    score: Optional[float] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"results": list(results)}
    if score is not None:
        payload["aggregate_score"] = score
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
