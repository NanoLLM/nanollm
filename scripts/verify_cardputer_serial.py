#!/usr/bin/env python3
"""Compare Cardputer serial inference output against local PyTorch."""

import argparse
import json
import sys
import time
from pathlib import Path

import serial
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from model import NanoLLM  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from quantized_runtime import QuantizedNanoLLM  # noqa: E402


def load_torch_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = NanoLLM(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_path = checkpoint.get("tokenizer_path")
    if not tokenizer_path or not Path(tokenizer_path).is_file():
        tokenizer_path = Path(checkpoint_path).parent / "tokenizer" / "tokenizer.json"
    if not Path(tokenizer_path).is_file():
        # Release layout keeps tokenizer.json at the release root, not under
        # <checkpoint-dir>/tokenizer/. Walk up to find it.
        for parent in Path(checkpoint_path).resolve().parents:
            cand = parent / "tokenizer" / "tokenizer.json"
            if cand.is_file():
                tokenizer_path = cand
                break
            cand2 = parent / "tokenizer.json"
            if cand2.is_file():
                tokenizer_path = cand2
                break
    tokenizer = BPETokenizer.from_file(str(tokenizer_path))
    return model, tokenizer, config



class CardputerSerial:
    def __init__(self, port, baud=115200, timeout=0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def _readline(self):
        line = self.ser.readline().decode("utf-8", errors="replace").strip()
        return line

    def reset_device(self):
        """Toggle DTR to reset the ESP32 and drain stale serial output."""
        self.ser.dtr = False
        time.sleep(0.05)
        self.ser.dtr = True
        self.ser.reset_input_buffer()

    def wait_ready(self, timeout_sec=90):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            line = self._readline()
            if not line:
                continue
            if line == "@READY":
                return True
        return False

    def command(self, cmd, expect_prefix, timeout_sec=30):
        payload, _streams = self.command_with_streams(cmd, expect_prefix, timeout_sec=timeout_sec)
        return payload

    def command_with_streams(self, cmd, expect_prefix, timeout_sec=30):
        """Send a command; capture @STREAM timestamps until expect_prefix payload.

        Returns (payload_dict, stream_events) where each stream event is
        {"t_sec": float, "step": int|None, "token": int|None, "text": str|None}
        with t_sec relative to command send (perf_counter).
        """
        self.ser.write((cmd + "\n").encode("utf-8"))
        self.ser.flush()
        t0 = time.perf_counter()
        deadline = time.time() + timeout_sec
        streams = []
        while time.time() < deadline:
            line = self._readline()
            if not line:
                continue
            if line.startswith("@ERROR|"):
                raise RuntimeError(line.split("|", 1)[1])
            if line.startswith("@ERROR"):
                raise RuntimeError(line)
            if line.startswith("@STREAM|"):
                parts = line.split("|", 3)
                step = None
                token = None
                text = None
                if len(parts) >= 3:
                    try:
                        step = int(parts[1])
                    except ValueError:
                        step = None
                    try:
                        token = int(parts[2])
                    except ValueError:
                        token = None
                if len(parts) >= 4:
                    text = parts[3]
                streams.append(
                    {
                        "t_sec": round(time.perf_counter() - t0, 6),
                        "step": step,
                        "token": token,
                        "text": text,
                    }
                )
                continue
            if line.startswith(expect_prefix + "|"):
                payload = line.split("|", 1)[1]
                return json.loads(payload), streams
        raise TimeoutError(f"Timed out waiting for {expect_prefix} response to: {cmd}")

    def ping(self):
        self.ser.write(b"PING\n")
        self.ser.flush()
        deadline = time.time() + 5
        while time.time() < deadline:
            line = self._readline()
            if line == "@PONG":
                return True
        return False


def latency_from_streams(streams):
    """Derive TTFT and steady-state inter-token latency from @STREAM events."""
    if not streams:
        return {
            "ttft_sec": None,
            "steady_state_sec_per_token": None,
            "stream_timestamps_sec": [],
            "inter_token_sec": [],
        }
    timestamps = [float(s["t_sec"]) for s in streams]
    ttft = timestamps[0]
    inter = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    steady = (sum(inter) / len(inter)) if inter else None
    return {
        "ttft_sec": round(ttft, 6),
        "steady_state_sec_per_token": round(steady, 6) if steady is not None else None,
        "stream_timestamps_sec": [round(t, 6) for t in timestamps],
        "inter_token_sec": [round(dt, 6) for dt in inter],
    }

def load_quantized_model(weights_path, config_path):
    return QuantizedNanoLLM.load(weights_path, config_path)


def quantized_next_token(model: QuantizedNanoLLM, prompt_tokens):
    return model.next_token(prompt_tokens)


def quantized_generate(model: QuantizedNanoLLM, prompt_tokens, max_new_tokens, temperature=1.0):
    return model.generate(prompt_tokens, max_new_tokens=max_new_tokens, temperature=temperature)


def compare_prompt(quant_model, tokenizer, device, prompt, max_new_tokens, encode_only=False):
    py_tokens = tokenizer.encode(prompt)
    if not py_tokens:
        return {"prompt": prompt, "ok": False, "error": "python encode produced no tokens"}

    enc = device.command(f"ENCODE|{prompt}", "@ENCODE")
    dev_tokens = enc.get("prompt_tokens", [])

    result = {
        "prompt": prompt,
        "python_prompt_tokens": py_tokens,
        "device_prompt_tokens": dev_tokens,
        "encode_match": py_tokens == dev_tokens,
    }

    if not result["encode_match"]:
        result["ok"] = False
        result["error"] = "prompt token mismatch"
        return result

    if encode_only:
        result["ok"] = True
        return result

    py_next = quantized_next_token(quant_model, py_tokens)
    # Long generations on Cardputer can exceed 30s; scale timeout with token budget.
    gen_timeout = max(120, 45 * max(1, max_new_tokens))
    generation_started = time.perf_counter()
    gen, streams = device.command_with_streams(
        f"GENERATE|{prompt}|{max_new_tokens}",
        "@GENERATE",
        timeout_sec=gen_timeout,
    )
    generation_elapsed = time.perf_counter() - generation_started
    latency = latency_from_streams(streams)
    dev_next = gen.get("next_token")
    dev_generated = gen.get("generated_tokens", [])
    generated_count = max(0, len(dev_generated) - len(dev_tokens))

    py_generated = quantized_generate(quant_model, py_tokens, max_new_tokens, temperature=1.0)
    py_next = py_generated[len(py_tokens)] if len(py_generated) > len(py_tokens) else py_next

    result.update(
        {
            "python_next_token": py_next,
            "python_next_token_fp32": None,
            "device_next_token": dev_next,
            "next_token_match": py_next == dev_next,
            "python_generated_tokens": py_generated,
            "device_generated_tokens": dev_generated,
            "generated_prefix_match": py_generated[: len(dev_generated)] == dev_generated,
            "device_generation_elapsed_sec": round(generation_elapsed, 6),
            "device_generated_token_count": generated_count,
            "device_sec_per_token": (
                round(generation_elapsed / generated_count, 6) if generated_count else None
            ),
            "ttft_sec": latency["ttft_sec"],
            "steady_state_sec_per_token": latency["steady_state_sec_per_token"],
            "stream_timestamps_sec": latency["stream_timestamps_sec"],
            "inter_token_sec": latency["inter_token_sec"],
        }
    )

    result["ok"] = result["encode_match"] and result["next_token_match"]
    if max_new_tokens > 1 and not result["generated_prefix_match"]:
        result["ok"] = False
        result["warning"] = (
            "Multi-token generation may diverge until autoregressive forward is fully implemented on ESP32."
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify Cardputer serial output against PyTorch")
    parser.add_argument("--checkpoint", default="checkpoints/moe_small_chat_20260630/model_best.pt")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to exported model.bin (default: weights/model.bin)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to model_config.json (default: alongside --weights)",
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["hello", "Hello world", "The quick brown fox"],
    )
    parser.add_argument("--output", default="weights/cardputer_serial_verify.json")
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Only verify prompt tokenization (skip next-token comparison)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the device via DTR before waiting for @READY",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights or "weights/model.bin")
    config_path = Path(args.config or weights_path.with_name("model_config.json"))
    if not weights_path.is_file():
        raise FileNotFoundError(f"Quantized weights not found: {weights_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    _, tokenizer, config = load_torch_model(args.checkpoint)
    quant_model = load_quantized_model(weights_path, config_path)
    device = CardputerSerial(args.port)

    try:
        if args.reset:
            device.reset_device()
        if not device.wait_ready():
            raise RuntimeError("Device did not announce @READY (try --reset)")
        if not device.ping():
            raise RuntimeError("PING failed")

        info = device.command("INFO", "@INFO")
        results = {
            "checkpoint": args.checkpoint,
            "weights": str(weights_path),
            "config": str(config_path),
            "reference": "python_quantized_runtime",
            "device_info": info,
            "model_config": config,
            "prompts": [],
            "all_ok": True,
        }

        for prompt in args.prompts:
            item = compare_prompt(quant_model, tokenizer, device, prompt, args.max_new_tokens, args.encode_only)
            results["prompts"].append(item)
            status = "OK" if item.get("ok") else "FAIL"
            print(f"[{status}] {prompt!r}")
            print(f"  encode: py={item.get('python_prompt_tokens')} dev={item.get('device_prompt_tokens')}")
            if "python_next_token" in item:
                print(f"  next:   py={item['python_next_token']} dev={item['device_next_token']}")
            if item.get("ttft_sec") is not None:
                steady = item.get("steady_state_sec_per_token")
                steady_s = f"{steady:.3f}s/tok" if steady is not None else "n/a"
                print(
                    f"  latency: ttft={item['ttft_sec']:.3f}s "
                    f"steady={steady_s} "
                    f"avg={item.get('device_sec_per_token')}s/tok "
                    f"total={item.get('device_generation_elapsed_sec')}s"
                )
            if item.get("warning"):
                print(f"  note:   {item['warning']}")
            if not item.get("ok"):
                results["all_ok"] = False
                if item.get("error"):
                    print(f"  error:  {item['error']}")

        ttfts = [p["ttft_sec"] for p in results["prompts"] if p.get("ttft_sec") is not None]
        steadies = [
            p["steady_state_sec_per_token"]
            for p in results["prompts"]
            if p.get("steady_state_sec_per_token") is not None
        ]
        results["latency_summary"] = {
            "max_new_tokens": args.max_new_tokens,
            "ttft_sec_mean": round(sum(ttfts) / len(ttfts), 6) if ttfts else None,
            "ttft_sec_by_prompt": {
                p["prompt"]: p.get("ttft_sec") for p in results["prompts"] if "ttft_sec" in p
            },
            "steady_state_sec_per_token_mean": (
                round(sum(steadies) / len(steadies), 6) if steadies else None
            ),
            "steady_state_sec_per_token_by_prompt": {
                p["prompt"]: p.get("steady_state_sec_per_token")
                for p in results["prompts"]
                if "steady_state_sec_per_token" in p
            },
        }
        if results["latency_summary"]["ttft_sec_mean"] is not None:
            print(
                f"Latency summary: TTFT mean={results['latency_summary']['ttft_sec_mean']:.3f}s "
                f"steady mean={results['latency_summary']['steady_state_sec_per_token_mean']}"
            )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        return 0 if results["all_ok"] else 1
    finally:
        device.close()


if __name__ == "__main__":
    raise SystemExit(main())
