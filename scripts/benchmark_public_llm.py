#!/usr/bin/env python3
"""Reproducible zero-shot multiple-choice benchmarks for tiny causal LMs."""

import argparse
import json
import math
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from model import NanoLLM  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

TASK_NAMES = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande")
TASK_DATASETS = {
    "hellaswag": (
        "Rowan/hellaswag", None, "validation", "218ec52e09a7e7462a5400043bb9a69a41d06b76"
    ),
    "arc_easy": (
        "allenai/ai2_arc", "ARC-Easy", "validation", "210d026faf9955653af8916fad021475a3f00453"
    ),
    "arc_challenge": (
        "allenai/ai2_arc", "ARC-Challenge", "validation", "210d026faf9955653af8916fad021475a3f00453"
    ),
    "piqa": (
        "ybisk/piqa", None, "validation", "21d9d8b65f6e607d60719f066527044e03028c04"
    ),
    "winogrande": (
        "allenai/winogrande", "winogrande_xl", "validation",
        "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    ),
}


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    radius /= denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def prepare_continuation(
    prompt_ids: list[int], full_ids: list[int], max_context: int
) -> tuple[list[int], int, bool]:
    """Crop a prompt+continuation while retaining all scoreable continuation tokens."""
    if len(full_ids) < 2:
        return (full_ids, 0, False)
    continuation_len = max(1, len(full_ids) - len(prompt_ids))
    max_full_tokens = max(2, max_context + 1)
    truncated = len(full_ids) > max_full_tokens
    if truncated:
        full_ids = full_ids[-max_full_tokens:]
    continuation_len = min(continuation_len, len(full_ids) - 1)
    return (full_ids, continuation_len, truncated)


class CausalLMAdapter(ABC):
    def __init__(self, device: torch.device):
        self.device = device

    @property
    @abstractmethod
    def metadata(self) -> dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_context(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def score_choices(self, prompt: str, continuations: Iterable[str]) -> tuple[int, list[float], bool]:
        prompt_ids = self.encode(prompt)
        prepared = []
        any_truncated = False
        for continuation in continuations:
            full_ids = self.encode(prompt + continuation)
            full_ids, continuation_len, truncated = prepare_continuation(
                prompt_ids, full_ids, self.max_context
            )
            if continuation_len <= 0:
                prepared.append(([self.pad_token_id, self.pad_token_id], 1))
            else:
                prepared.append((full_ids, continuation_len))
            any_truncated |= truncated

        max_input_len = max(len(ids) - 1 for ids, _ in prepared)
        batch = torch.full(
            (len(prepared), max_input_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(batch)
        for row, (ids, _) in enumerate(prepared):
            input_len = len(ids) - 1
            batch[row, :input_len] = torch.tensor(ids[:-1], dtype=torch.long, device=self.device)
            attention_mask[row, :input_len] = 1

        logits = self.logits(batch, attention_mask)
        scores = []
        for row, (ids, continuation_len) in enumerate(prepared):
            input_len = len(ids) - 1
            targets = torch.tensor(ids[1:], dtype=torch.long, device=self.device)
            token_log_probs = torch.log_softmax(logits[row, :input_len], dim=-1)
            selected = token_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            scores.append(-float(selected[-continuation_len:].mean().item()))
        prediction = min(range(len(scores)), key=scores.__getitem__)
        return prediction, scores, any_truncated


class NanoLLMAdapter(CausalLMAdapter):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = checkpoint["config"]
        self.model = NanoLLM(**self.config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        tokenizer_path = checkpoint.get("tokenizer_path")
        if not tokenizer_path or not Path(tokenizer_path).is_file():
            tokenizer_path = Path(checkpoint_path).parent / "tokenizer" / "tokenizer.json"
        if not Path(tokenizer_path).is_file():
            # Release bundles store the tokenizer at the release root, not next
            # to the checkpoint. Walk up ancestor dirs looking for it.
            base = Path(checkpoint_path).resolve()
            for parent in base.parents:
                candidate = parent / "tokenizer" / "tokenizer.json"
                if candidate.is_file():
                    tokenizer_path = str(candidate)
                    break
        self.tokenizer = BPETokenizer.from_file(str(tokenizer_path))
        self.checkpoint_path = str(checkpoint_path)

    @property
    def metadata(self) -> dict:
        return {
            "kind": "nanollm",
            "id": self.checkpoint_path,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "config": self.config,
        }

    @property
    def max_context(self) -> int:
        return int(self.config.get("max_seq_len", 128))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        return self.model(input_ids)[0]

    @property
    def pad_token_id(self) -> int:
        return 1


class HuggingFaceAdapter(CausalLMAdapter):
    def __init__(
        self,
        model_id: str,
        revision: str | None,
        device: torch.device,
        trust_remote_code: bool,
    ):
        super().__init__(device)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            use_fast=True,
        )
        model_args = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if trust_remote_code:
            model_args["use_safetensors"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_args).to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def metadata(self) -> dict:
        return {
            "kind": "huggingface",
            "id": self.model_id,
            "revision": self.revision,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "model_type": getattr(self.model.config, "model_type", None),
            "vocab_size": getattr(self.model.config, "vocab_size", None),
        }

    @property
    def max_context(self) -> int:
        values = [
            getattr(self.model.config, name, None)
            for name in ("max_position_embeddings", "n_positions", "max_sequence_length")
        ]
        return int(next((value for value in values if value), 512))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)


def load_task(task_name: str, limit: int):
    dataset_id, config_name, split, revision = TASK_DATASETS[task_name]
    split_expr = split if limit <= 0 else f"{split}[:{limit}]"
    return load_dataset(dataset_id, config_name, split=split_expr, revision=revision), {
        "dataset": dataset_id,
        "config": config_name,
        "split": split,
        "revision": revision,
    }


def format_sample(task_name: str, sample: dict) -> tuple[str, list[str], int] | None:
    if task_name == "hellaswag":
        return sample["ctx"].strip(), [" " + item.strip() for item in sample["endings"]], int(sample["label"])
    if task_name.startswith("arc_"):
        labels = [str(item) for item in sample["choices"]["label"]]
        answer = str(sample["answerKey"])
        if answer not in labels:
            return None
        prompt = sample["question"].strip() + "\nAnswer:"
        choices = [" " + item.strip() for item in sample["choices"]["text"]]
        return prompt, choices, labels.index(answer)
    if task_name == "piqa":
        prompt = "Question: " + sample["goal"].strip() + "\nAnswer:"
        return prompt, [" " + sample["sol1"].strip(), " " + sample["sol2"].strip()], int(sample["label"])
    if task_name == "winogrande":
        prefix, suffix = sample["sentence"].split("_", 1)
        choices = [
            " " + sample["option1"].strip() + suffix,
            " " + sample["option2"].strip() + suffix,
        ]
        return prefix.rstrip(), choices, int(sample["answer"]) - 1
    raise ValueError(f"Unsupported task: {task_name}")


def evaluate_task(adapter: CausalLMAdapter, task_name: str, limit: int) -> dict:
    dataset, dataset_metadata = load_task(task_name, limit)
    correct = 0
    total = 0
    truncated = 0
    started = time.time()
    for index, sample in enumerate(dataset, start=1):
        formatted = format_sample(task_name, sample)
        if formatted is None:
            continue
        prompt, choices, answer = formatted
        prediction, _, was_truncated = adapter.score_choices(prompt, choices)
        correct += int(prediction == answer)
        total += 1
        truncated += int(was_truncated)
        if index % 250 == 0:
            print(f"[{task_name}] {index}/{len(dataset)} accuracy={correct / total:.4f}", flush=True)
    low, high = wilson_interval(correct, total)
    return {
        "task": task_name,
        **dataset_metadata,
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_ci95_wilson": [low, high],
        "truncated_samples": truncated,
        "truncated_fraction": truncated / total if total else None,
        "elapsed_sec": round(time.time() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Run standard zero-shot tiny-LM benchmarks.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Native NanoLLM checkpoint")
    source.add_argument("--hf-model", help="Hugging Face causal LM model ID")
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--limit", type=int, default=0, help="Samples per task; 0 runs full split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.checkpoint:
        adapter = NanoLLMAdapter(args.checkpoint, device)
    else:
        adapter = HuggingFaceAdapter(
            args.hf_model,
            args.hf_revision,
            device,
            args.trust_remote_code,
        )

    results = {
        "schema_version": 2,
        "model": adapter.metadata,
        "device": str(device),
        "method": "zero-shot length-normalized continuation log-likelihood",
        "max_context": adapter.max_context,
        "sample_limit_per_task": args.limit,
        "tasks": [],
    }
    for task_name in args.tasks:
        item = evaluate_task(adapter, task_name, args.limit)
        results["tasks"].append(item)
        print(json.dumps(item, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
