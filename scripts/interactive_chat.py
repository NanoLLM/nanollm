#!/usr/bin/env python3
"""
Interactive terminal chat for qualitative NanoLLM testing.

Loads the newest checkpoint under checkpoints/ by default and runs a REPL
with adjustable sampling settings and optional multi-turn chat formatting.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_template import find_turn_boundary, format_prompt, trim_assistant_reply  # noqa: E402
from model import NanoLLM  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

CHECKPOINT_NAMES = ("model_best.pt", "model_latest.pt")


def _checkpoint_sort_key(path: Path) -> tuple:
    """Prefer model_best, then model_latest, then highest model_step, then mtime."""
    name = path.name
    if name == "model_best.pt":
        return (0, 0, -path.stat().st_mtime)
    if name == "model_latest.pt":
        return (1, 0, -path.stat().st_mtime)
    if name.startswith("model_step_") and name.endswith(".pt"):
        try:
            step = int(name[len("model_step_") : -len(".pt")])
        except ValueError:
            step = 0
        return (2, -step, -path.stat().st_mtime)
    return (3, 0, -path.stat().st_mtime)


def resolve_device(request: str = "auto", min_free_mib: int = 512) -> torch.device:
    """Pick cpu, explicit cuda:N, or the CUDA device with the most free memory."""
    request = request.strip().lower()
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if request.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{request} requested but CUDA is not available")
        return torch.device(request)

    if request != "auto":
        raise ValueError(f"Unknown device: {request!r} (use auto, cpu, cuda, or cuda:N)")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    min_free_bytes = min_free_mib * 1024 * 1024
    best_idx: Optional[int] = None
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(idx)
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as err:
            if "out of memory" in str(err).lower():
                print(
                    f"Warning: cuda:{idx} unavailable ({err}); skipping.",
                    file=sys.stderr,
                )
                continue
            raise
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx

    if best_idx is None:
        print("Warning: no usable CUDA device; using CPU.", file=sys.stderr)
        return torch.device("cpu")

    if best_free < min_free_bytes:
        print(
            f"Warning: no CUDA device has >= {min_free_mib} MiB free "
            f"(best: cuda:{best_idx} with {best_free // (1024 * 1024)} MiB). Using CPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")

    return torch.device(f"cuda:{best_idx}")


def move_model_to_device(model: torch.nn.Module, device: torch.device) -> torch.device:
    """Move model to device; fall back to CPU on CUDA OOM."""
    try:
        model.to(device)
        return device
    except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as err:
        if device.type != "cuda" or "out of memory" not in str(err).lower():
            raise
        print(
            f"CUDA OOM while loading model on {device} ({err}). Falling back to CPU.",
            file=sys.stderr,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu = torch.device("cpu")
        model.to(cpu)
        return cpu


def find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    """Return the most recently modified known checkpoint file."""
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    latest_link = checkpoints_dir / "latest"
    for name in CHECKPOINT_NAMES:
        candidate = latest_link / name
        if candidate.is_file():
            return candidate.resolve()

    candidates: List[Path] = []
    for name in CHECKPOINT_NAMES:
        candidates.extend(checkpoints_dir.rglob(name))

    # Also accept any *.pt directly under run subdirectories.
    for path in checkpoints_dir.glob("*/*.pt"):
        if path.name.startswith("model_"):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under {checkpoints_dir}. "
            "Train a model first or pass --checkpoint explicitly."
        )

    unique = sorted({path.resolve() for path in candidates}, key=_checkpoint_sort_key)
    return unique[0]


def checkpoint_quality_warnings(checkpoint_path: Path, checkpoint_data: dict) -> List[str]:
    """Return user-facing warnings for common checkpoint issues."""
    warnings: List[str] = []
    run_dir = checkpoint_path.parent
    pretrain_path = run_dir / "model_pretrain.pt"
    ckpt_epoch = int(checkpoint_data.get("epoch", 0) or 0)
    val_loss = checkpoint_data.get("val_loss", checkpoint_data.get("loss"))
    config = checkpoint_data.get("config") or {}

    if config.get("causal_attention") is not True:
        warnings.append(
            "Checkpoint was saved without causal_attention=True metadata. "
            "Older checkpoints may have been trained with future-token leakage; retrain before judging generation quality."
        )

    if checkpoint_path.name.startswith("model_step_"):
        step = checkpoint_path.stem.replace("model_step_", "")
        warnings.append(
            f"Mid-training snapshot at step {step} (epoch not finished; validation not run yet). "
            "Prefer model_best.pt after pretrain completes."
        )
        batch_loss = checkpoint_data.get("train_loss")
        if isinstance(batch_loss, (int, float)) and batch_loss < 0.5:
            warnings.append(
                f"Stored train_loss={batch_loss:.4f} is the last batch at save time, not epoch average — "
                "low batch loss does not mean good open-ended generation."
            )

    if is_pretrain_foundation_checkpoint(checkpoint_path, checkpoint_data):
        warnings.append(
            "FineWeb foundation pretrain only — not chat-tuned. "
            "Use /mode raw for completion-style prompts; run stage 2 chat fine-tune for dialogue. "
            f"Resume: OUTPUT_DIR={run_dir} ./scripts/train_moe_pipeline.sh --export"
        )

    if pretrain_path.is_file():
        try:
            pretrain_data = torch.load(pretrain_path, map_location="cpu", weights_only=False)
            pretrain_epoch = int(pretrain_data.get("epoch", 0) or 0)
            if ckpt_epoch <= pretrain_epoch and not any("foundation pretrain" in w for w in warnings):
                warnings.append(
                    "This looks like a pretrain-only checkpoint (FineWeb LM). "
                    "Chat mode usually produces repetitive garbage until chat fine-tune completes."
                )
        except (OSError, RuntimeError, KeyError):
            pass

    metrics_path = run_dir / "training_metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            data_path = str(metrics.get("run_config", {}).get("data", ""))
            if (
                "fineweb" in data_path
                and "chat" not in data_path
                and len(metrics.get("epochs", [])) == 0
                and not any("foundation pretrain" in w for w in warnings)
            ):
                warnings.append(
                    "Pretrain still in progress (no completed epochs logged). "
                    "Generation often loops on subwords (e.g. 'o', 'ist', ':') until training finishes."
                )
        except (OSError, json.JSONDecodeError):
            pass

    if isinstance(val_loss, (int, float)) and val_loss < 0.05:
        warnings.append(
            f"Very low val loss ({val_loss:.4f}) often means overfit on the pretrain sample; "
            "generation may collapse to repeating characters (e.g. ':' or 'o'). "
            "Try /mode raw, fewer pretrain epochs, or more diverse data."
        )

    return warnings


def resolve_tokenizer_path(
    checkpoint_path: Path,
    checkpoint_hint: Optional[str],
    override: Optional[str],
) -> Path:
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    if checkpoint_hint:
        candidates.append(Path(checkpoint_hint).expanduser())

    ckpt_dir = checkpoint_path.parent
    candidates.extend(
        [
            ckpt_dir / "tokenizer" / "tokenizer.json",
            ckpt_dir / "tokenizer.json",
            PROJECT_ROOT / "checkpoints" / "tokenizer" / "tokenizer.json",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Unable to locate tokenizer.json. Pass --tokenizer or ensure the checkpoint "
        "directory contains tokenizer/tokenizer.json."
    )


@dataclass
class GenerationSettings:
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: Optional[int] = 40
    greedy: bool = False
    repetition_penalty: float = 1.12
    repetition_window: int = 64


def is_pretrain_foundation_checkpoint(checkpoint_path: Path, checkpoint_data: dict) -> bool:
    """True when checkpoint is FineWeb LM pretrain without chat fine-tune."""
    run_dir = checkpoint_path.parent
    pretrain_path = run_dir / "model_pretrain.pt"
    ckpt_epoch = int(checkpoint_data.get("epoch", 0) or 0)

    if pretrain_path.is_file():
        try:
            pretrain_data = torch.load(pretrain_path, map_location="cpu", weights_only=False)
            pretrain_epoch = int(pretrain_data.get("epoch", 0) or 0)
            if ckpt_epoch <= pretrain_epoch:
                return True
        except (OSError, RuntimeError, KeyError):
            pass

    metrics_path = run_dir / "training_metrics.json"
    if not metrics_path.is_file():
        return checkpoint_path.name.startswith("model_step_")

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return checkpoint_path.name.startswith("model_step_")

    run_config = metrics.get("run_config", {})
    data_path = str(run_config.get("data", "")).lower()
    if "fineweb" not in data_path or "chat" in data_path:
        return False
    if len(metrics.get("epochs", [])) == 0:
        return True
    return ckpt_epoch <= int(run_config.get("epochs", 1) or 1)


class NanoLLMInteractive:
    """Load checkpoint + tokenizer and generate text in the terminal."""

    def __init__(
        self,
        checkpoint_path: Path,
        tokenizer_path: Path,
        device: torch.device,
        chat_mode: bool = True,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.chat_mode = chat_mode
        self.settings = GenerationSettings()
        self.history: List[Tuple[str, str]] = []
        self.show_debug = False
        self.stream_output = True

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" not in checkpoint_data:
            raise KeyError("Checkpoint missing 'model_state_dict'")
        config = checkpoint_data.get("config")
        if not config:
            raise KeyError("Checkpoint missing 'config' metadata")

        self.config = config
        self.model = NanoLLM(
            vocab_size=config.get("vocab_size"),
            d_model=config.get("d_model"),
            n_layers=config.get("n_layers"),
            n_heads=config.get("n_heads"),
            d_ff=config.get("d_ff"),
            max_seq_len=config.get("max_seq_len", 128),
            dropout=config.get("dropout", 0.0),
            use_moe=config.get("use_moe", False),
            moe_n_experts=config.get("moe_n_experts", 4),
            moe_top_k=config.get("moe_top_k", 1),
            moe_shared_d_ff=config.get("moe_shared_d_ff", 0),
        )
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.device = move_model_to_device(self.model, self.device)
        self.model.eval()

        self.tokenizer = BPETokenizer.from_file(str(tokenizer_path))
        self._bos_id = self._lookup_special_id("<BOS>", default=0)

    def _lookup_special_id(self, token: str, default: int) -> int:
        bpe_impl = getattr(self.tokenizer, "tokenizer", None)
        if bpe_impl is not None and hasattr(bpe_impl, "token_to_id"):
            token_id = bpe_impl.token_to_id(token)
            if token_id is not None:
                return int(token_id)
        return default

    def build_prompt(self, user_message: str) -> str:
        if not self.chat_mode:
            return user_message
        return format_prompt(user_message, history=self.history)

    def generate(self, prompt: str) -> Tuple[str, List[int], List[int]]:
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self._bos_id]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        new_ids: List[int] = []
        for _delta, token_id in self._stream_token_deltas(input_ids, self.settings.max_new_tokens):
            new_ids.append(token_id)
        text = trim_assistant_reply(self.tokenizer.decode(new_ids)) if new_ids else ""
        return text, prompt_ids, new_ids

    def stream_generate(self, prompt: str) -> Tuple[List[int], Iterator[Tuple[str, int]]]:
        """Return prompt token ids and an iterator of (text_delta, token_id) per step."""
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self._bos_id]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        return prompt_ids, self._stream_token_deltas(input_ids, self.settings.max_new_tokens)

    def _sample_next_token(self, idx: torch.Tensor) -> torch.Tensor:
        settings = self.settings
        idx_cond = idx[:, -self.model.max_seq_len :]
        logits, _ = self.model(idx_cond)
        logits = logits[:, -1, :] / max(settings.temperature, 1e-3)

        if settings.repetition_penalty > 1.0:
            window = max(1, settings.repetition_window)
            recent = idx[0, -window:].tolist()
            for token_id in set(recent):
                logits[0, token_id] /= settings.repetition_penalty

        if settings.greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)

        if settings.top_k is not None and settings.top_k > 0:
            top_k = min(settings.top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _stream_token_deltas(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
    ) -> Generator[Tuple[str, int], None, None]:
        """Yield (decoded_text_delta, token_id) for each generated token."""
        self.model.eval()
        prompt_len = idx.size(1)
        decoded_prefix = ""
        with torch.no_grad():
            for _ in range(max(1, max_new_tokens)):
                idx_next = self._sample_next_token(idx)
                token_id = int(idx_next.item())
                idx = torch.cat([idx, idx_next], dim=1)

                new_ids = idx[0, prompt_len:].tolist()
                decoded = self.tokenizer.decode(new_ids)
                boundary = find_turn_boundary(decoded)
                if boundary is not None:
                    decoded = trim_assistant_reply(decoded)
                    delta = decoded[len(decoded_prefix) :]
                    if delta:
                        yield delta, token_id
                    break

                delta = decoded[len(decoded_prefix) :]
                decoded_prefix = decoded
                yield delta, token_id

                if idx.size(1) >= self.model.max_seq_len:
                    break

    def config_summary(self) -> str:
        cfg = self.config
        parts = [
            f"vocab={cfg.get('vocab_size')}",
            f"d_model={cfg.get('d_model')}",
            f"layers={cfg.get('n_layers')}",
            f"heads={cfg.get('n_heads')}",
            f"d_ff={cfg.get('d_ff')}",
            f"max_seq_len={cfg.get('max_seq_len')}",
        ]
        if cfg.get("use_moe"):
            parts.append(f"moe_experts={cfg.get('moe_n_experts')}")
            parts.append(f"moe_top_k={cfg.get('moe_top_k')}")
            parts.append(f"moe_shared_d_ff={cfg.get('moe_shared_d_ff')}")
        return " | ".join(parts)


def print_banner(runner: NanoLLMInteractive, warnings: Optional[List[str]] = None) -> None:
    mode = "chat (User/Assistant)" if runner.chat_mode else "completion (raw prompt)"
    sampling = "greedy argmax" if runner.settings.greedy else (
        f"sample temp={runner.settings.temperature} top_k={runner.settings.top_k or 'off'}"
    )
    rep = runner.settings.repetition_penalty
    rep_note = f", rep_penalty={rep}" if rep > 1.0 else ""
    print("=" * 72)
    print("NanoLLM Interactive Tester")
    print("=" * 72)
    print(f"Checkpoint : {runner.checkpoint_path}")
    print(f"Tokenizer  : {runner.tokenizer_path}")
    print(f"Device     : {runner.device}")
    print(f"Config     : {runner.config_summary()}")
    print(f"Mode       : {mode}")
    print(f"Generation : max_new_tokens={runner.settings.max_new_tokens}, {sampling}{rep_note}")
    print(f"Output       : {'streaming' if runner.stream_output else 'buffered (print when complete)'}")
    if warnings:
        print("-" * 72)
        for note in warnings:
            for line in textwrap.wrap(note, width=70):
                print(f"Warning    : {line}")
    print("-" * 72)
    print("Type a prompt and press Enter. Commands:")
    print("  /help              show commands")
    print("  /clear             reset conversation history")
    print("  /temp <float>      set temperature (e.g. 0.8)")
    print("  /tokens <int>      set max new tokens")
    print("  /topk <int|off>    set top-k sampling (0 or off disables)")
    print("  /greedy            deterministic argmax decoding")
    print("  /sample            re-enable sampling")
    print("  /mode chat|raw     switch prompt formatting")
    print("  /show              print current settings")
    print("  /debug on|off      toggle token-id debug output")
    print("  /stream on|off     toggle token streaming")
    print("  /quit or /exit     leave interactive mode")
    print("-" * 72)


def handle_command(runner: NanoLLMInteractive, line: str) -> Optional[str]:
    """Process slash commands. Returns an optional status message."""
    parts = line.strip().split()
    if not parts:
        return None

    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in {"/quit", "/exit", "/q"}:
        return "__QUIT__"
    if cmd == "/help":
        return (
            "Commands: /help /clear /temp /tokens /topk /greedy /sample "
            "/mode /show /debug /stream /quit"
        )
    if cmd == "/clear":
        runner.history.clear()
        return "Conversation history cleared."
    if cmd == "/debug":
        if arg.lower() in {"on", "1", "true", "yes"}:
            runner.show_debug = True
            return "Debug output enabled."
        if arg.lower() in {"off", "0", "false", "no"}:
            runner.show_debug = False
            return "Debug output disabled."
        runner.show_debug = not runner.show_debug
        return f"Debug output {'enabled' if runner.show_debug else 'disabled'}."
    if cmd == "/stream":
        if arg.lower() in {"on", "1", "true", "yes"}:
            runner.stream_output = True
            return "Token streaming enabled."
        if arg.lower() in {"off", "0", "false", "no"}:
            runner.stream_output = False
            return "Token streaming disabled (buffered output)."
        runner.stream_output = not runner.stream_output
        return f"Token streaming {'enabled' if runner.stream_output else 'disabled'}."
    if cmd == "/show":
        s = runner.settings
        mode = "chat" if runner.chat_mode else "raw"
        sampling = "greedy" if s.greedy else f"sample (temp={s.temperature}, top_k={s.top_k})"
        return (
            f"mode={mode}, max_new_tokens={s.max_new_tokens}, {sampling}, "
            f"rep_penalty={s.repetition_penalty}, stream={runner.stream_output}, "
            f"history_turns={len(runner.history)}, debug={runner.show_debug}"
        )
    if cmd == "/temp":
        if not arg:
            return "Usage: /temp <float>"
        runner.settings.greedy = False
        runner.settings.temperature = max(1e-3, float(arg))
        return f"Temperature set to {runner.settings.temperature}"
    if cmd == "/tokens":
        if not arg:
            return "Usage: /tokens <int>"
        runner.settings.max_new_tokens = max(1, int(arg))
        return f"max_new_tokens set to {runner.settings.max_new_tokens}"
    if cmd == "/topk":
        if not arg:
            return "Usage: /topk <int|off>"
        if arg.lower() in {"off", "none", "0"}:
            runner.settings.top_k = None
            return "top_k disabled"
        runner.settings.top_k = max(1, int(arg))
        runner.settings.greedy = False
        return f"top_k set to {runner.settings.top_k}"
    if cmd == "/greedy":
        runner.settings.greedy = True
        return "Using greedy argmax decoding."
    if cmd == "/sample":
        runner.settings.greedy = False
        return "Using stochastic sampling."
    if cmd == "/mode":
        if arg not in {"chat", "raw"}:
            return "Usage: /mode chat|raw"
        runner.chat_mode = arg == "chat"
        runner.history.clear()
        return f"Mode set to {'chat' if runner.chat_mode else 'raw completion'} (history cleared)."

    return f"Unknown command: {cmd}. Type /help."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat with the latest NanoLLM checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: newest model_best.pt under checkpoints/)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Directory to search when --checkpoint is omitted",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer.json path (overrides checkpoint metadata)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto (most free GPU, else CPU), cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "chat", "raw"],
        help="Prompt format: auto (raw for foundation pretrain, else chat), chat, or raw",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Default max tokens to generate per turn",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Default sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Default top-k (0 disables)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Start in greedy argmax mode",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Down-weight recently seen tokens (default: 1.15 for foundation pretrain, 1.12 for chat)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for full generation before printing (disable token streaming)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print prompt/new token ids after each response",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint(args.checkpoints_dir.expanduser().resolve())
    else:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    tokenizer_path = resolve_tokenizer_path(
        checkpoint_path,
        checkpoint_data.get("tokenizer_path"),
        args.tokenizer,
    )

    device = resolve_device(args.device)

    foundation_pretrain = is_pretrain_foundation_checkpoint(checkpoint_path, checkpoint_data)
    if args.mode == "auto":
        chat_mode = not foundation_pretrain
    else:
        chat_mode = args.mode == "chat"

    runner = NanoLLMInteractive(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=device,
        chat_mode=chat_mode,
    )
    runner.settings.max_new_tokens = args.max_new_tokens
    runner.settings.temperature = args.temperature
    runner.settings.top_k = args.top_k if args.top_k > 0 else None
    runner.settings.greedy = args.greedy
    if args.repetition_penalty is not None:
        runner.settings.repetition_penalty = max(1.0, args.repetition_penalty)
    elif foundation_pretrain:
        runner.settings.repetition_penalty = 1.15
    elif runner.chat_mode:
        runner.settings.repetition_penalty = 1.12
    runner.show_debug = args.debug
    runner.stream_output = not args.no_stream

    quality_warnings = checkpoint_quality_warnings(checkpoint_path, checkpoint_data)
    print_banner(runner, quality_warnings)

    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    while True:
        try:
            user_line = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user_line:
            continue

        if user_line.startswith("/"):
            result = handle_command(runner, user_line)
            if result == "__QUIT__":
                print("Bye.")
                return 0
            if result:
                print(result)
            continue

        prompt = runner.build_prompt(user_line)
        try:
            if runner.stream_output:
                prompt_ids, token_stream = runner.stream_generate(prompt)
                print("\nAssistant> ", end="", flush=True)
                new_ids: List[int] = []
                for delta, token_id in token_stream:
                    new_ids.append(token_id)
                    if delta:
                        print(delta, end="", flush=True)
                print(flush=True)
                reply = trim_assistant_reply(runner.tokenizer.decode(new_ids)) if new_ids else ""
            else:
                reply, prompt_ids, new_ids = runner.generate(prompt)
                wrapped = textwrap.fill(reply, width=88) if reply else "(empty response)"
                print(f"\nAssistant> {wrapped}")
        except Exception as exc:
            print(f"[Error] {exc}")
            continue

        if runner.show_debug:
            print(f"[debug] prompt_tokens={len(prompt_ids)} new_tokens={len(new_ids)} ids={new_ids[:12]}")

        if runner.chat_mode:
            runner.history.append((user_line, reply))


if __name__ == "__main__":
    raise SystemExit(main())
