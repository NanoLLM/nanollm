"""Gradio demo for interacting with a trained NanoLLM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import torch

from chat_template import format_prompt
from chat_eval import sample_chat_reply
from model import NanoLLM
from tokenizer import BPETokenizer


def _resolve_tokenizer_path(
    checkpoint_path: Path,
    checkpoint_hint: Optional[str],
    override: Optional[str],
) -> Path:
    """Select a tokenizer path using CLI override, checkpoint metadata, or fallbacks."""

    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    if checkpoint_hint:
        candidates.append(Path(checkpoint_hint).expanduser())

    ckpt_dir = checkpoint_path.parent
    candidates.append(ckpt_dir / "tokenizer" / "tokenizer.json")
    candidates.append(ckpt_dir / "tokenizer.json")

    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Unable to locate tokenizer.json. Pass --tokenizer or ensure the checkpoint"
        " was saved with a valid 'tokenizer_path'."
    )


class NanoLLMRunner:
    """Thin wrapper that bundles a loaded NanoLLM model and tokenizer."""

    def __init__(
        self,
        checkpoint_path: Path,
        tokenizer_path: Path,
        device: torch.device,
        checkpoint_data: Optional[dict] = None,
    ) -> None:
        self.device = device
        if checkpoint_data is None:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        if "model_state_dict" not in checkpoint_data:
            raise KeyError("Checkpoint missing 'model_state_dict'")
        config = checkpoint_data.get("config")
        if not config:
            raise KeyError("Checkpoint missing 'config' metadata")

        self.model = NanoLLM(
            vocab_size=config.get("vocab_size"),
            d_model=config.get("d_model"),
            n_layers=config.get("n_layers"),
            n_heads=config.get("n_heads"),
            d_ff=config.get("d_ff"),
            max_seq_len=config.get("max_seq_len", 128),
            dropout=config.get("dropout", 0.0),
        )
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BPETokenizer.from_file(str(tokenizer_path))

        bpe_impl = getattr(self.tokenizer, "tokenizer", None)
        self._bos_id = 0
        if bpe_impl is not None and hasattr(bpe_impl, "token_to_id"):
            bos = bpe_impl.token_to_id("<BOS>")
            if bos is not None:
                self._bos_id = bos

        self.model_config = config
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        repetition_penalty: float = 1.12,
    ) -> str:
        # Extract the latest user turn from a chat prompt when possible.
        user_message = prompt
        history: List[Tuple[str, str]] = []
        if prompt.startswith("User: "):
            lines = prompt.split("\n")
            turns: List[Tuple[str, str]] = []
            current_user: Optional[str] = None
            current_assistant: Optional[str] = None
            for line in lines:
                if line.startswith("User: "):
                    if current_user and current_assistant:
                        turns.append((current_user, current_assistant))
                    current_user = line[len("User: ") :].strip()
                    current_assistant = None
                elif line.startswith("Assistant:"):
                    suffix = line[len("Assistant:") :].strip()
                    current_assistant = suffix if suffix else ""
            if current_user is not None and current_assistant is not None:
                turns.append((current_user, current_assistant))
            if turns:
                history = turns[:-1]
                user_message = turns[-1][0]

        return sample_chat_reply(
            self.model,
            self.tokenizer,
            self.device,
            user_message,
            history=history,
            max_new_tokens=max(1, int(max_new_tokens)),
            temperature=max(1e-3, float(temperature)),
            top_k=top_k if top_k and top_k > 0 else None,
            repetition_penalty=max(1.0, float(repetition_penalty)),
        )


def build_prompt(history: List[Tuple[str, str]], user_message: str) -> str:
    """Flatten chat history into a simple conversation prompt."""
    return format_prompt(user_message, history=history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Gradio demo for NanoLLM")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/model_best.pt"),
        help="Path to the trained NanoLLM checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="checkpoints/tokenizer/tokenizer.json",
        help="Optional path to tokenizer.json (overrides checkpoint metadata)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface for Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share link",
    )
    parser.add_argument(
        "--default-max-new-tokens",
        type=int,
        default=50,
        help="Initial value for the max new tokens slider",
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=0.7,
        help="Initial value for the temperature slider",
    )
    parser.add_argument(
        "--default-top-k",
        type=int,
        default=40,
        help="Initial value for the top-k slider (0 disables)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_hint = checkpoint_data.get("tokenizer_path")
    tokenizer_path = _resolve_tokenizer_path(checkpoint_path, checkpoint_hint, args.tokenizer)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    runner = NanoLLMRunner(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=device,
        checkpoint_data=checkpoint_data,
    )

    title_lines = [
        "# NanoLLM Gradio Demo",
        f"**Checkpoint:** `{checkpoint_path}`",
        f"**Tokenizer:** `{tokenizer_path}`",
        f"**Device:** `{device}`",
        (
            "**Config:** "
            f"vocab={runner.model_config.get('vocab_size')} | d_model={runner.model_config.get('d_model')} "
            f"| layers={runner.model_config.get('n_layers')} | heads={runner.model_config.get('n_heads')} "
            f"| d_ff={runner.model_config.get('d_ff')} | max_seq_len={runner.model_config.get('max_seq_len')}"
        ),
    ]
    title_md = "  \n".join(title_lines)

    with gr.Blocks(title="NanoLLM Demo") as demo:
        gr.Markdown(title_md)
        chatbot = gr.Chatbot(height=400)
        with gr.Row():
            max_new_tokens = gr.Slider(
                label="Max New Tokens",
                minimum=1,
                maximum=256,
                step=1,
                value=args.default_max_new_tokens,
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=1.5,
                step=0.05,
                value=args.default_temperature,
            )
            top_k = gr.Slider(
                label="Top-k (0 = disabled)",
                minimum=0,
                maximum=200,
                step=1,
                value=args.default_top_k,
            )
        msg = gr.Textbox(label="Your message", placeholder="Ask NanoLLM something", lines=2)
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Conversation")

        def respond(
            user_message: str,
            history: List[Tuple[str, str]],
            max_tokens: int,
            temp: float,
            topk: int,
        ):
            if not user_message or not user_message.strip():
                return history, gr.update(value="")
            prompt = build_prompt(history, user_message.strip())
            try:
                reply = runner.generate(prompt, max_tokens, temp, topk)
            except Exception as exc:
                reply = f"[Error] {exc}"
            updated_history = history + [(user_message, reply)]
            return updated_history, gr.update(value="")

        def clear_history():
            return [], gr.update(value="")

        send_btn.click(
            respond,
            inputs=[msg, chatbot, max_new_tokens, temperature, top_k],
            outputs=[chatbot, msg],
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, max_new_tokens, temperature, top_k],
            outputs=[chatbot, msg],
        )
        clear_btn.click(clear_history, outputs=[chatbot, msg])

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
