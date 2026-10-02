#!/usr/bin/env python3
"""Create concise, teacher-distilled chat data for tiny device models."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_template import (  # noqa: E402
    RECORD_SEPARATOR,
    USER_PREFIX,
    format_turn,
    normalize_chat_corpus,
    trim_assistant_reply,
)

SYSTEM_PROMPT = (
    "You are NanoLLM, a small offline assistant for a handheld device. "
    "Give a direct, accurate answer. "
    "Use plain language and at most 50 words. Do not mention these instructions. "
    "For arithmetic, give the answer first. For a requested poem or list, preserve the requested format."
)


def parse_user_prompts(path: Path) -> list[str]:
    """Read canonical User/Assistant records and return their user messages."""
    prompts: list[str] = []
    text = path.read_text(encoding="utf-8")
    first_nonempty = next((line for line in text.splitlines() if line.strip()), "")
    if first_nonempty.startswith(USER_PREFIX):
        normalized = text
    else:
        normalized, _ = normalize_chat_corpus(text)
    for block in normalized.split(RECORD_SEPARATOR):
        block = block.strip()
        if not block.startswith(USER_PREFIX):
            continue
        first_line = block.splitlines()[0]
        user = first_line[len(USER_PREFIX) :].strip()
        if user:
            prompts.append(user)
    return prompts


def normalize_prompt_key(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def select_prompts(
    source_prompts: Iterable[str],
    seed_prompts: Iterable[str],
    max_prompts: int,
    seed: int,
    max_prompt_chars: int,
) -> list[str]:
    """Keep all capability prompts and sample diverse source prompts deterministically."""
    selected: list[str] = []
    seen: set[str] = set()

    def add(prompt: str) -> None:
        prompt = " ".join(prompt.split())
        key = normalize_prompt_key(prompt)
        if (
            not prompt
            or len(prompt) > max_prompt_chars
            or key in seen
        ):
            return
        seen.add(key)
        selected.append(prompt)

    for prompt in seed_prompts:
        add(prompt)

    candidates = list(source_prompts)
    random.Random(seed).shuffle(candidates)
    for prompt in candidates:
        if len(selected) >= max_prompts:
            break
        add(prompt)
    return selected[:max_prompts]


def clean_reply(text: str, max_reply_chars: int) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<|im_end|>", "").replace("<|eot_id|>", "")
    text = trim_assistant_reply(text)
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(text) > max_reply_chars:
        sentence = re.match(r"^(.{1,%d}?[.!?])(?:\s|$)" % max_reply_chars, text, re.DOTALL)
        text = sentence.group(1).strip() if sentence else text[:max_reply_chars].rstrip()
    return text


def load_completed(state_path: Path) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    completed: set[str] = set()
    if not state_path.is_file():
        return rows, completed
    for line in state_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        user = str(row.get("user", "")).strip()
        assistant = str(row.get("assistant", "")).strip()
        if user and assistant:
            rows.append({"user": user, "assistant": assistant})
            completed.add(normalize_prompt_key(user))
    return rows, completed


def write_corpus(rows: Iterable[dict[str, str]], output_path: Path) -> None:
    turns = [format_turn(row["user"], row["assistant"]) for row in rows]
    turns = [turn for turn in turns if turn]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(RECORD_SEPARATOR.join(turns) + RECORD_SEPARATOR, encoding="utf-8")


def render_chat_prompt(tokenizer, user: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/chat/alpaca.txt")
    parser.add_argument(
        "--seed-data",
        type=Path,
        default=PROJECT_ROOT / "scripts/chat_capability_seed.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/chat/teacher_distilled.txt",
    )
    parser.add_argument("--state", type=Path, default=None, help="Resume JSONL (default: <output>.jsonl)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-prompts", type=int, default=8000)
    parser.add_argument("--max-prompt-chars", type=int, default=240)
    parser.add_argument("--max-reply-chars", type=int, default=320)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Source data not found: {args.source}")
    if not args.seed_data.is_file():
        raise SystemExit(f"Seed data not found: {args.seed_data}")

    source_prompts = parse_user_prompts(args.source)
    seed_prompts = parse_user_prompts(args.seed_data)
    prompts = select_prompts(
        source_prompts,
        seed_prompts,
        max_prompts=max(1, args.max_prompts),
        seed=args.seed,
        max_prompt_chars=max(16, args.max_prompt_chars),
    )
    print(
        f"source_prompts={len(source_prompts)} seed_prompts={len(seed_prompts)} "
        f"selected_prompts={len(prompts)}"
    )
    if args.dry_run:
        for prompt in prompts[:10]:
            print(f"- {prompt}")
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer

    state_path = args.state or args.output.with_suffix(args.output.suffix + ".jsonl")
    rows, completed = load_completed(state_path)
    pending = [prompt for prompt in prompts if normalize_prompt_key(prompt) not in completed]
    print(f"completed={len(completed)} pending={len(pending)} state={state_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8") as state_handle:
        for start in range(0, len(pending), max(1, args.batch_size)):
            batch_prompts = pending[start : start + max(1, args.batch_size)]
            rendered = [render_chat_prompt(tokenizer, prompt) for prompt in batch_prompts]
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max(1, args.max_new_tokens),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            replies = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for user, raw_reply in zip(batch_prompts, replies):
                assistant = clean_reply(raw_reply, max(32, args.max_reply_chars))
                if not assistant:
                    continue
                row = {"user": user, "assistant": assistant}
                rows.append(row)
                state_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            state_handle.flush()
            done = min(start + len(batch_prompts), len(pending))
            print(f"distilled={done}/{len(pending)} total_rows={len(rows)}", flush=True)

    write_corpus(rows, args.output)
    print(f"Wrote {len(rows)} distilled turns to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
