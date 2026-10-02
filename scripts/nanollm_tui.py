#!/usr/bin/env python3
"""NanoLLM TUI launcher.

Terminal UI that drives the full NanoLLM lifecycle end-to-end:

    1. datasets    download / verify training data + canonical tokenizer
    2. pretrain    FineWeb pretrain (BPE tokenizer fit or reuse)
    3. sft         chat / instruction fine-tune on top of the pretrain checkpoint
    4. posttrain   int8 export (+ PROGMEM embed), greedy sanity eval
    5. firmware    (Cardputer) build + flash via PlatformIO
    6. verify      (Cardputer) serial parity vs Python int8 runtime
                    (PC-GPU) desktop C++ build + inference smoke

The user picks a deployment target (PC with GPU vs ESP32-S3 Cardputer), a
profile, and optional model/architecture overrides; the TUI computes the
resulting size (via scripts/moe_tradeoff_estimator.py), shows a live size
readout, and can run any single stage or the whole chain, with live log
tail and dry-run preview.

Stdlib only (curses). No new dependencies.

Usage:
    python3 scripts/nanollm_tui.py                 # interactive TUI
    python3 scripts/nanollm_tui.py --dry-run       # print the full plan, no TUI
    python3 scripts/nanollm_tui.py --target cardputer --profile cardputer-mqa-ctx224 --dry-run
    python3 scripts/nanollm_tui.py --run datasets --yes   # run one stage headless
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "scripts" / "train_moe_pipeline.sh"
ESTIMATOR = ROOT / "scripts" / "moe_tradeoff_estimator.py"
CANONICAL_TOKENIZER_DIR = ROOT / "data" / "tokenizers" / "cardputer_vocab2048_v1"


def _detect_serial_device() -> str:
    """Return the first USB serial device that looks like a Cardputer
    (ESP32-S3, VID:PID 303a:4002/0x1209-class CDC), else the first
    /dev/ttyACM*.  Never auto-flash an unrelated ttyUSB bridge unless it is
    the only candidate.  Returns '' when nothing suitable is found."""
    import glob
    candidates = []
    for dev in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
        try:
            import subprocess as sp
            out = sp.run(["udevadm", "info", "-q", "property", "-n", dev],
                         capture_output=True, text=True, timeout=3).stdout
            vid = dict(l.split("=", 1) for l in out.splitlines() if "=" in l).get(
                "ID_VENDOR_ID", "")
            pid = dict(l.split("=", 1) for l in out.splitlines() if "=" in l).get(
                "ID_MODEL_ID", "")
        except Exception:
            vid = pid = ""
        candidates.append((dev, vid.lower(), pid.lower()))
    # ESP32-S3 / M5Stack Cardputer signatures
    for dev, vid, pid in candidates:
        if vid in ("303a",) or (vid, pid) in (("303a", "4002"), ("0403", "6001")):
            return dev
    # fallback: a bare CDC-ACM (Cardputer without udev rules) is acceptable;
    # a plain FTDI/CP210x USB serial bridge is NOT auto-flashed.
    for dev, vid, _pid in candidates:
        if not vid or vid == "303a":
            return dev
    return ""


# --------------------------------------------------------------------------
# Target / profile catalog
# --------------------------------------------------------------------------

TARGETS = {
    "cardputer": {
        "label": "ESP32-S3 Cardputer (no-PSRAM, PROGMEM flash)",
        "recommended": [
            ("cardputer-mqa-ctx224", "Promoted deploy: L18 d80 S224 MQA, vocab 2048 (~4.66M, ~5.5 MiB int8)"),
            ("cardputer-mqa-ctx208-l20", "Depth probe: L20 d80 S208 MQA"),
            ("cardputer-mqa-wide96", "Width-heavy MQA at short context"),
            ("cardputer-mqa-bal192-88", "Balanced d192 S88"),
            ("cardputer-smoke", "Boot/OOM smoke: L4 d32, 512 ctx, vocab 8192 (minutes)"),
            ("cardputer-safe", "Conservative safe point"),
            ("quick", "Tiny 128-ctx smoke: L4 d32 vocab 512 (fastest)"),
        ],
        "firmware_env": "m5stack_cardputer_nopsram",
        "needs_embed": True,
        "needs_firmware": True,
        "verify": "serial",
    },
    "pc_gpu": {
        "label": "PC with GPU (desktop C++ inference, int8 model.bin)",
        "recommended": [
            ("desktop", "Wider MoE for GPU experimentation (L8 d64, S128)"),
            ("gpu8g-mqa-instruct", "Scaled-up: L24 d128 S32768 vocab 4096 RoPE (~15.8M, ~12 GiB batch)"),
            ("quick", "Tiny smoke: L4 d32 vocab 512 (fastest)"),
            ("cardputer-mqa-ctx224", "Deploy shape, trained on GPU (no PROGMEM embed by default)"),
        ],
        "firmware_env": "",
        "needs_embed": False,
        "needs_firmware": False,
        "verify": "desktop",
    },
}

ALL_PROFILES = [
    "cardputer-mqa-ctx224", "cardputer-mqa-ctx208-l20", "cardputer-mqa-wide96",
    "cardputer-mqa-bal192-88", "cardputer-autoresearch-d256",
    "cardputer-autoresearch-ctx224", "cardputer-chat", "cardputer-chat-wide",
    "cardputer-dense-capacity", "cardputer-dense-mac", "cardputer",
    "cardputer-legacy", "cardputer-balanced", "cardputer-smoke", "cardputer-safe",
    "gpu8g-mqa-instruct", "quick", "desktop",
]

# (key, label, default, kind) — kind: int / float / str / bool
CONFIG_FIELDS = [
    # Model / architecture (env overrides; empty = use profile values)
    ("VOCAB_SIZE", "vocab size (empty = profile)", "", "int"),
    ("BLOCK_SIZE", "context / block size (empty = profile)", "", "int"),
    ("D_MODEL", "d_model (empty = profile)", "", "int"),
    ("N_LAYERS", "n_layers (empty = profile)", "", "int"),
    ("N_HEADS", "n_heads (empty = profile)", "", "int"),
    ("N_KV_HEADS", "n_kv_heads (1 = MQA, empty = profile)", "", "int"),
    ("D_FF", "routed expert d_ff (empty = profile)", "", "int"),
    ("MOE_N_EXPERTS", "MoE experts (empty = profile)", "", "int"),
    ("MOE_TOP_K", "MoE top-k (empty = profile)", "", "int"),
    ("MOE_SHARED_D_FF", "shared expert d_ff (empty = profile)", "", "int"),
    # Training
    ("PRETRAIN_EPOCHS", "pretrain epochs (empty = profile)", "", "int"),
    ("CHAT_EPOCHS", "SFT epochs (empty = profile)", "", "int"),
    ("PRETRAIN_BATCH_SIZE", "pretrain batch size (empty = 256)", "", "int"),
    ("CHAT_BATCH_SIZE", "SFT batch size (empty = 256)", "", "int"),
    ("PRETRAIN_LR", "pretrain LR (empty = 2e-4)", "", "float"),
    ("CHAT_LR", "SFT LR (empty = 5e-5)", "", "float"),
    ("TRAIN_SEED", "train seed (default 42)", "", "int"),
    ("NUM_WORKERS", "dataloader workers (default 8)", "", "int"),
    ("AMP", "amp: auto|bf16|fp16|none (default auto)", "", "str"),
    ("CUDA_DEVICE", "CUDA device index (empty = default)", "", "int"),
    # Data
    ("PRETRAIN_DATA", "pretrain file (empty = data/fineweb/fineweb_3x.txt)", "", "path"),
    ("CHAT_DATA", "chat SFT file (empty = data/chat/teacher_distilled_qwen3_5_3k.txt)", "", "path"),
    ("REUSE_TOKENIZER", "reuse tokenizer dir or .json (empty = auto: canonical for vocab 2048)", "", "path"),
    # Toggles
    ("INSTRUCT_FINE_TUNING", "instruction fine-tune mode (0/1)", "0", "bool"),
    ("TRAIN_TOKENIZER", "fit fresh BPE on train+chat data (0/1)", "0", "bool"),
    ("EXPORT_INT8", "export int8 model.bin in post-train (0/1)", "1", "bool"),
    ("RUN_SANITY", "greedy chat sanity eval in post-train (0/1)", "1", "bool"),
    # Deployment (cardputer)
    ("UPLOAD_PORT", "Cardputer serial port to flash (empty = build only, never auto-flash)", "", "path"),
]

DEFAULTS = {k: d for k, _, d, _ in CONFIG_FIELDS}


@dataclass
class Plan:
    target: str = "cardputer"
    profile: str = "cardputer-mqa-ctx224"
    cfg: dict = field(default_factory=lambda: dict(DEFAULTS))
    stages_done: dict = field(default_factory=dict)   # stage -> "ok" | "fail"
    run_dir: str = ""

    def env_for_pipeline(self) -> dict:
        """Env vars for train_moe_pipeline.sh (only non-empty overrides)."""
        env = dict(os.environ)
        env["PROFILE"] = self.profile
        for k, _, _, kind in CONFIG_FIELDS:
            v = self.cfg.get(k, "")
            if v in ("", None):
                continue
            if kind == "bool":
                v = "1" if v in ("1", True, "true", "yes") else "0"
            env[k] = str(v)
        # target-specific defaults
        if self.target == "pc_gpu":
            env.setdefault("DO_EMBED", "0")
        return env

    def latest_run_dir(self) -> Path | None:
        link = ROOT / "checkpoints" / "latest"
        try:
            if link.exists():
                p = link.resolve()
                if p.exists():
                    return p
        except OSError:
            pass
        cands = sorted((ROOT / "checkpoints").glob("moe_run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0] if cands else None


# --------------------------------------------------------------------------
# Size estimation
# --------------------------------------------------------------------------

def estimate_size(plan: Plan) -> dict:
    """Run the estimator for the effective config; parse the single-config report."""
    # Determine effective values: profile defaults if user left fields empty.
    t = TARGETS[plan.target]
    prof = plan.profile
    eff = {
        "vocab_size": 2048, "d_model": 80, "n_layers": 18, "n_heads": 4,
        "n_kv_heads": 1, "d_ff": 320, "max_seq_len": 224,
        "moe_n_experts": 4, "moe_top_k": 1, "moe_shared_d_ff": 160,
    }
    # Profile-specific base values (mirror pipeline profile blocks).
    base = {
        "cardputer-mqa-ctx224": dict(vocab_size=2048, d_model=80, n_layers=18, n_heads=4, n_kv_heads=1, d_ff=320, max_seq_len=224, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=160),
        "cardputer-mqa-ctx208-l20": dict(vocab_size=2048, d_model=80, n_layers=20, n_heads=4, n_kv_heads=1, d_ff=320, max_seq_len=208, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=160),
        "cardputer-mqa-wide96": dict(vocab_size=2048, d_model=96, n_layers=18, n_heads=4, n_kv_heads=1, d_ff=384, max_seq_len=112, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=192),
        "cardputer-mqa-bal192-88": dict(vocab_size=2048, d_model=88, n_layers=18, n_heads=4, n_kv_heads=1, d_ff=352, max_seq_len=192, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=176),
        "cardputer-smoke": dict(vocab_size=8192, d_model=32, n_layers=4, n_heads=4, n_kv_heads=1, d_ff=128, max_seq_len=512, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=64),
        "cardputer-safe": dict(vocab_size=512, d_model=40, n_layers=12, n_heads=4, n_kv_heads=1, d_ff=80, max_seq_len=128, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=40),
        "quick": dict(vocab_size=512, d_model=32, n_layers=4, n_heads=4, n_kv_heads=1, d_ff=64, max_seq_len=128, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=16),
        "desktop": dict(vocab_size=512, d_model=64, n_layers=8, n_heads=4, n_kv_heads=1, d_ff=128, max_seq_len=128, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=32),
        "gpu8g-mqa-instruct": dict(vocab_size=4096, d_model=128, n_layers=24, n_heads=4, n_kv_heads=1, d_ff=512, max_seq_len=32768, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=256),
        "cardputer-chat": dict(vocab_size=2048, d_model=48, n_layers=44, n_heads=4, n_kv_heads=1, d_ff=192, max_seq_len=192, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=96),
        "cardputer-chat-wide": dict(vocab_size=2048, d_model=80, n_layers=18, n_heads=4, n_kv_heads=1, d_ff=320, max_seq_len=112, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=160),
        "cardputer-autoresearch-d256": dict(vocab_size=2048, d_model=256, n_layers=3, n_heads=18, n_kv_heads=4, d_ff=768, max_seq_len=112, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=256),
        "cardputer-autoresearch-ctx224": dict(vocab_size=2048, d_model=192, n_layers=3, n_heads=8, n_kv_heads=1, d_ff=640, max_seq_len=224, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=256),
        "cardputer-dense-capacity": dict(dense=True, vocab_size=2048, d_model=80, n_layers=16, n_heads=4, n_kv_heads=4, d_ff=1444, max_seq_len=112),
        "cardputer-dense-mac": dict(dense=True, vocab_size=2048, d_model=80, n_layers=16, n_heads=4, n_kv_heads=4, d_ff=480, max_seq_len=112),
        "cardputer": dict(vocab_size=8192, d_model=40, n_layers=78, n_heads=4, n_kv_heads=1, d_ff=160, max_seq_len=384, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=80),
        "cardputer-legacy": dict(vocab_size=8192, d_model=32, n_layers=124, n_heads=4, n_kv_heads=1, d_ff=128, max_seq_len=512, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=64),
        "cardputer-balanced": dict(vocab_size=8192, d_model=36, n_layers=78, n_heads=4, n_kv_heads=1, d_ff=184, max_seq_len=464, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=92),
    }
    if prof in base:
        eff = {**eff, **base[prof]}
    mapping = {
        "VOCAB_SIZE": "vocab_size", "BLOCK_SIZE": "max_seq_len", "D_MODEL": "d_model",
        "N_LAYERS": "n_layers", "N_HEADS": "n_heads", "N_KV_HEADS": "n_kv_heads",
        "D_FF": "d_ff", "MOE_N_EXPERTS": "moe_n_experts", "MOE_TOP_K": "moe_top_k",
        "MOE_SHARED_D_FF": "moe_shared_d_ff",
    }
    for field_key, eff_key in mapping.items():
        v = plan.cfg.get(field_key, "")
        if v not in ("", None):
            try:
                eff[eff_key] = int(v)
            except ValueError:
                pass

    dense = bool(eff.pop("dense", False))
    cmd = [
        sys.executable, str(ESTIMATOR),
        "--vocab-size", str(eff["vocab_size"]),
        "--d-model", str(eff["d_model"]),
        "--n-layers", str(eff["n_layers"]),
        "--n-heads", str(eff["n_heads"]),
        "--n-kv-heads", str(eff["n_kv_heads"]),
        "--d-ff", str(eff["d_ff"]),
        "--max-seq-len", str(eff["max_seq_len"]),
    ]
    if dense:
        cmd.append("--dense-model")
    else:
        cmd += [
            "--moe-experts", str(eff.get("moe_n_experts", 4)),
            "--moe-top-k", str(eff.get("moe_top_k", 1)),
            "--moe-shared-d-ff", str(eff.get("moe_shared_d_ff", 0)),
        ]
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
        txt = out.stdout
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "config": eff}

    res = {"config": eff}
    patterns = {
        "param_count": r"Estimated parameter count:\s*([\d,]+)",
        "weight_bytes": r"Estimated int8 weight file:\s*([\d,]+)",
        "working_bytes": r"Working activation memory estimate:\s*([\d,]+)",
        "firmware_bytes": r"Cardputer est\. firmware total:\s*([\d,]+)",
        "ram_ok": r"Within working budget:\s*(yes|no)",
        "flash_ok": r"Cardputer feasible \(flash-backed\):\s*(yes|no)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, txt)
        if m:
            res[key] = m.group(1)
    if res.get("flash_ok") == "no" and plan.target == "cardputer":
        res["warning"] = "EXCEEDS Cardputer constraints (flash or working RAM)"
    return res


def fmt_bytes(n: str | None) -> str:
    if n is None:
        return "?"
    try:
        b = int(str(n).replace(",", ""))
    except ValueError:
        return str(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if b < 1024 or unit == "GiB":
            return f"{b:.1f} {unit}" if unit != "B" else f"{b} B"
        b /= 1024
    return str(n)


# --------------------------------------------------------------------------
# Stage command builders
# --------------------------------------------------------------------------

def stage_commands(plan: Plan) -> list[dict]:
    """Ordered list of {key, title, desc, cmd (list), log_hint, needs(target)}."""
    stages = []
    t = TARGETS[plan.target]
    env = plan.env_for_pipeline()
    py = sys.executable

    # 1) datasets — actually download whatever the profile needs (HF-backed).
    # Profile -> pretrain data mapping mirrors train_moe_pipeline.sh case blocks:
    #   quick / cardputer-smoke / cardputer-chat  -> fineweb_tiny.txt (fast)
    #   everything else (incl. cardputer-mqa-ctx224, desktop, gpu8g-*) -> fineweb_3x.txt
    TINY_PROFILES = {"quick", "cardputer-smoke", "cardputer-chat"}
    ds_cmds = []
    ds_cmds.append({"desc": "Bootstrap canonical tokenizer (vocab 2048) from committed release copy",
                    "cmd": [py, str(ROOT / "scripts" / "ensure_canonical_tokenizer.py")]})
    pd = plan.cfg.get("PRETRAIN_DATA", "").strip()
    if pd and Path(pd).exists():
        ds_cmds.append({"desc": f"Pretrain data already present: {pd}",
                        "cmd": [py, "-c", "print('pretrain data present: " + pd + "')"]})
    elif plan.profile in TINY_PROFILES:
        ds_cmds.append({"desc": "Download FineWeb tiny (1,000 docs, HF streaming) -> data/fineweb/fineweb_tiny.txt",
                        "cmd": [py, str(ROOT / "scripts" / "ensure_finetune_data.py"), "--tiny"]})
    else:
        ds_cmds.append({"desc": "Download FineWeb 3x (~223k docs / ~660 MB, HF streaming) -> data/fineweb/fineweb_3x.txt "
                                "(reused if present; ~10-30 min depending on network)",
                        "cmd": [py, str(ROOT / "scripts" / "ensure_finetune_data.py"), "--3x"]})
    if plan.cfg.get("INSTRUCT_FINE_TUNING") in ("1", "true", "yes"):
        ds_cmds.append({"desc": "Download instruction datasets (Alpaca + OpenOrca, HF) -> data/instruct/",
                        "cmd": [py, str(ROOT / "scripts" / "download_instruction_datasets.py"),
                                "--output-dir", str(ROOT / "data" / "instruct"),
                                "--sources", "alpaca", "openorca"]})
    needs = ["tiny" if plan.profile in TINY_PROFILES else "3x"]
    if not plan.cfg.get("INSTRUCT_FINE_TUNING") in ("1", "true", "yes"):
        needs.append("teacher")
    ds_cmds.append({"desc": "Verify required inputs (soft: exit 0 when the profile's inputs exist)",
                    "cmd": [py, str(ROOT / "scripts" / "ensure_finetune_data.py"),
                            "--report", "--soft", "--needs", ",".join(needs)]})
    stages.append({"key": "datasets", "title": "Datasets",
                   "desc": "Tokenizer + training data download/verify (HuggingFace-backed)",
                   "cmds": ds_cmds, "ok_hint": "All inputs present (or bootstrapped)."})

    # 2) pretrain — always start a fresh moe_run_* dir: auto-resume would pick
    # up checkpoints/latest, whose architecture may not match the selected
    # profile (shape-mismatch crash). Resume a *specific* run by setting
    # OUTPUT_DIR in the config screen instead.
    pre = [str(PIPELINE), "--pretrain-only", "--no-resume"]
    if plan.cfg.get("TRAIN_TOKENIZER") in ("1", "true", "yes"):
        pre.append("--train-tokenizer")
    stages.append({"key": "pretrain", "title": "Pretrain", "desc": "FineWeb pretrain (BPE fit or reuse)",
                   "cmd": pre, "env": env,
                   "ok_hint": "Pretrain checkpoint written (checkpoints/latest/model_best.pt)."})

    # 3) SFT — init from the run's pretrain snapshot (model_pretrain.pt is what
    # train_moe_pipeline.sh writes after stage 1; model_best.pt only exists
    # after stage 2 has already run).
    sft_env = dict(env)
    sft = [str(PIPELINE), "--finetune-only"]
    rd = plan.latest_run_dir()
    if rd is not None:
        sft_env["OUTPUT_DIR"] = str(rd)
        init_ckpt = rd / "model_pretrain.pt"
        if not init_ckpt.exists():
            init_ckpt = rd / "model_best.pt"
        sft += ["--init-checkpoint", str(init_ckpt)]
    if plan.cfg.get("INSTRUCT_FINE_TUNING") in ("1", "true", "yes"):
        sft.append("--instruction-finetune")
    stages.append({"key": "sft", "title": "SFT (fine-tune)",
                   "desc": "Chat / instruction fine-tune on the pretrain checkpoint",
                   "cmd": sft, "env": sft_env, "needs_pretrain": True,
                   "ok_hint": "SFT best checkpoint written."})

    # 4) posttrain
    pt = []
    rd = plan.latest_run_dir()
    best = str((rd / "model_best.pt")) if rd is not None else str(ROOT / "checkpoints" / "latest" / "model_best.pt")
    if plan.cfg.get("EXPORT_INT8") in ("1", "true", "yes", "") or plan.cfg.get("EXPORT_INT8", "1") != "0":
        pt.append({"desc": "Export int8 weights (SPIFFS / desktop model.bin)",
                   "cmd": [py, str(ROOT / "python" / "export_weights.py"),
                           "--checkpoint", best, "--output", str(ROOT / "weights" / "model.bin"),
                           "--allow-moe-export"]})
    if t["needs_embed"]:
        pt.append({"desc": "Embed PROGMEM weights header (firmware flash)",
                   "cmd": [py, str(ROOT / "python" / "export_weights_header.py"),
                           "--checkpoint", best, "--output",
                           str(ROOT / "esp32_m5stack" / "src" / "model_weights.h")]})
        pt.append({"desc": "Embed PROGMEM vocab header",
                   "cmd": [py, str(ROOT / "python" / "export_vocab_header.py"),
                           "--tokenizer", str(rd / "tokenizer" / "tokenizer.json") if rd else
                           str(CANONICAL_TOKENIZER_DIR / "tokenizer.json"),
                           "--output", str(ROOT / "esp32_m5stack" / "src" / "vocab_weights.h")]})
    if plan.cfg.get("RUN_SANITY", "1") != "0":
        pt.append({"desc": "Greedy chat sanity eval (fixed prompt suite)",
                   "cmd": [py, str(ROOT / "scripts" / "eval_chat_checkpoint.py"),
                           "--checkpoint", best,
                           "--output", str(ROOT / "checkpoints" / "chat_eval_latest.json"), "--greedy"]})
    stages.append({"key": "posttrain", "title": "Post-train",
                   "desc": "int8 export" + (" + PROGMEM embed" if t["needs_embed"] else "") +
                   (" + sanity eval" if plan.cfg.get("RUN_SANITY", "1") != "0" else ""),
                   "cmds": pt, "needs_pretrain": True,
                   "ok_hint": "Exports written."})

    # 5) firmware (cardputer only) — always build; flash ONLY when the user
    # explicitly set UPLOAD_PORT in Config.  We never auto-pick a serial
    # device: on this class of workstation /dev/ttyACM* is often a different
    # Espressif board (e.g. the Waveshare USB-JTAG bridge), so flashing it
    # would brick it.
    if t["needs_firmware"]:
        fw_env = t["firmware_env"]
        port = (plan.cfg.get("UPLOAD_PORT", "") or "").strip()
        fw_cmds = [{"desc": f"Build {fw_env} firmware (PlatformIO)",
                    "cmd": ["pio", "run", "-e", fw_env]}]
        if port:
            fw_cmds.append({"desc": f"Flash to {port}",
                            "cmd": ["pio", "run", "-e", fw_env, "-t", "upload",
                                    "--upload-port", port]})
        hint = ""
        if not port:
            detected = _detect_serial_device()
            if detected:
                hint = (f"  [detected {detected} — set UPLOAD_PORT in Config "
                        f"to flash it]")
        stages.append({"key": "firmware", "title": "Firmware",
                       "desc": ((f"Build + flash {fw_env} to {port}") if port
                                else (f"Build {fw_env} (set UPLOAD_PORT to "
                                      f"flash){hint[:40]}")),
                       "cmds": fw_cmds,
                       "cwd": str(ROOT / "esp32_m5stack"), "needs_pretrain": True,
                       "ok_hint": ("Firmware flashed to Cardputer." if port
                                   else "Firmware built (no UPLOAD_PORT set — "
                                        "flash: pio run -e " + fw_env +
                                        " -t upload).")})

    # 6) verify
    if t["verify"] == "serial":
        # Read-only serial check: an explicit UPLOAD_PORT wins; otherwise a
        # single detected CDC device is used.  (Safe — no flashing happens
        # here; --reset only reboots the Cardputer's app.)
        vport = (plan.cfg.get("UPLOAD_PORT", "") or "").strip()
        if not vport:
            vport = _detect_serial_device()
        vcmd = [py, str(ROOT / "scripts" / "verify_cardputer_serial.py"),
                "--reset", "--checkpoint", best]
        if vport:
            vcmd += ["--port", vport]
        stages.append({"key": "verify", "title": "Verify (serial)",
                       "desc": (f"Serial parity vs Python int8 runtime on {vport}"
                                if vport else
                                "Serial parity vs Python int8 runtime (needs Cardputer on USB)"),
                       "cmd": vcmd,
                       "needs_pretrain": True, "optional": True,
                       "ok_hint": "Serial parity matched."})
    else:
        stages.append({"key": "verify", "title": "Verify (desktop)",
                       "desc": "Build desktop C++ + inference smoke",
                       "cmds": [
                           {"desc": "Build desktop C++ inference",
                            "cmd": ["cmake", "-S", str(ROOT / "cpp"), "-B", str(ROOT / "cpp" / "build"),
                                    "-DCMAKE_BUILD_TYPE=Release"]},
                           {"desc": "Compile",
                            "cmd": ["cmake", "--build", str(ROOT / "cpp" / "build"), "-j", "4"]},
                           {"desc": "Inference smoke (needs weights/model.bin)",
                            "cmd": [str(ROOT / "cpp" / "build" / "inference"),
                                    str(ROOT / "weights" / "model.bin"), "hello"]},
                       ], "needs_pretrain": True, "optional": True,
                       "ok_hint": "Desktop inference produced output."})
    return stages


# --------------------------------------------------------------------------
# Headless / dry-run execution
# --------------------------------------------------------------------------

def print_plan(plan: Plan) -> None:
    t = TARGETS[plan.target]
    print("=" * 74)
    print("NanoLLM plan")
    print(f"  target   : {plan.target}  ({t['label']})")
    print(f"  profile  : {plan.profile}")
    est = estimate_size(plan)
    print("  model    : " + "  ".join(f"{k}={v}" for k, v in est.get("config", {}).items()))
    if est.get("param_count"):
        print(f"  params   : {est['param_count']}   int8 weights: {fmt_bytes(est.get('weight_bytes'))}")
    if plan.target == "cardputer":
        print(f"  working RAM: {fmt_bytes(est.get('working_bytes'))} (budget 200 KiB)  "
              f"firmware est: {fmt_bytes(est.get('firmware_bytes'))}")
        if est.get("flash_ok"):
            print(f"  feasible : {est['flash_ok']}")
        if est.get("warning"):
            print(f"  WARNING  : {est['warning']}")
    print("=" * 74)
    for i, st in enumerate(stage_commands(plan), 1):
        mark = "done" if plan.stages_done.get(st["key"]) == "ok" else "    "
        print(f" [{mark}] {i}. {st['title']} — {st['desc']}")
        cmds = st.get("cmds") or [st]
        for c in cmds:
            line = "    $ " + " ".join(c["cmd"])
            if c.get("env"):
                line = "    $ " + " ".join(f"{k}={v}" for k, v in c["env"].items() if k in (
                    "PROFILE", "OUTPUT_DIR", "PRETRAIN_EPOCHS", "CHAT_EPOCHS", "PRETRAIN_DATA",
                    "CHAT_DATA", "TRAIN_SEED", "AMP", "CUDA_DEVICE", "NUM_WORKERS",
                    "PRETRAIN_BATCH_SIZE", "CHAT_BATCH_SIZE", "PRETRAIN_LR", "CHAT_LR",
                    "TRAIN_TOKENIZER", "REUSE_TOKENIZER", "INSTRUCT_FINE_TUNING",
                    "VOCAB_SIZE", "BLOCK_SIZE", "D_MODEL", "N_LAYERS", "N_HEADS", "N_KV_HEADS",
                    "D_FF", "MOE_N_EXPERTS", "MOE_TOP_K", "MOE_SHARED_D_FF")) + "  " + " ".join(c["cmd"])
            print(line[:118] + ("…" if len(line) > 118 else ""))
        print()


def run_cmd_logged(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
                   log_path: Path | None = None) -> int:
    print(f"$ {' '.join(cmd)}")
    lf = open(log_path, "a") if log_path else open(os.devnull, "w")
    with lf:
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
            lf.flush()
        proc.wait()
    return proc.returncode


def run_stage_headless(plan: Plan, stage_key: str, assume_yes: bool = False) -> int:
    stages = stage_commands(plan)
    st = next((s for s in stages if s["key"] == stage_key), None)
    if st is None:
        print(f"unknown stage: {stage_key} (available: {[s['key'] for s in stages]})")
        return 2
    if st.get("needs_pretrain") and plan.latest_run_dir() is None:
        print("No prior training run found under checkpoints/. Run the pretrain stage first.")
        return 2
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / f"tui_{stage_key}_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmds = st.get("cmds") or [st]
    env = st.get("env") or (cmds[0].get("env") if cmds else None) or plan.env_for_pipeline()
    cwd = Path(st.get("cwd", ROOT))
    for c in cmds:
        if not assume_yes:
            print(f"Will run: {' '.join(c['cmd'])}")
            ans = input("Proceed? [y/N] ")
            if ans.strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1
        rc = run_cmd_logged(c["cmd"], cwd=cwd, env=c.get("env") or env, log_path=log_path)
        if rc != 0:
            print(f"Stage failed (exit {rc}). Log: {log_path}")
            return rc
    print(f"Stage '{stage_key}' OK. Log: {log_path}")
    return 0


# --------------------------------------------------------------------------
# TUI (curses)
# --------------------------------------------------------------------------

import curses  # noqa: E402  (imported late so --dry-run needs no terminal)


class TUI:
    def __init__(self, plan: Plan):
        self.plan = plan
        self.stages = stage_commands(plan)
        self.screen_idx = "main"
        self.sel = 0
        self.cfg_idx = 0
        self.editing = False
        self.msg = ""
        self.msg_err = False
        self.run_state = None   # dict when a stage is running
        self.run_log_buf = []
        self.est = None
        self.est_dirty = True

    # ---------- helpers ----------
    def hline(self, y: int, attr=0) -> None:
        w = self.stdscr.getmaxyx()[1]
        self.stdscr.hline(y, 0, "─", w - 1, attr)

    def put(self, y: int, x: int, text: str, attr=0) -> None:
        h, w = self.stdscr.getmaxyx()
        if y < 0 or y >= h - 1 or x >= w - 1:
            return
        # truncate + strip non-printable to avoid curses exceptions
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] in ("L", "N", "P", "S", "Z") or ch in " \t")
        try:
            self.stdscr.addnstr(y, x, text, w - 1 - x, attr)
        except curses.error:
            pass

    def refresh(self) -> None:
        self.stdscr.refresh()

    def notify(self, msg: str, err: bool = False) -> None:
        self.msg = msg[:200]
        self.msg_err = err

    def recompute_estimate(self) -> None:
        self.est = estimate_size(self.plan)
        self.est_dirty = False

    # ---------- main screen ----------
    def draw_main(self) -> None:
        self.stdscr.erase()
        t = TARGETS[self.plan.target]
        attr = curses.color_pair(1)
        self.put(0, 2, "NanoLLM — train / export / deploy launcher", curses.A_BOLD | attr)
        self.put(1, 2, f"target: {self.plan.target}  ({t['label']})     profile: {self.plan.profile}", curses.A_DIM)
        if self.est is None:
            self.put(2, 2, "computing size estimate…", curses.A_DIM)
        else:
            e = self.est
            line = f"params {e.get('param_count','?')}   int8 {fmt_bytes(e.get('weight_bytes'))}   working {fmt_bytes(e.get('working_bytes'))}"
            if self.plan.target == "cardputer":
                line += f"   firmware {fmt_bytes(e.get('firmware_bytes'))}   feasible: {e.get('flash_ok','?')}"
            self.put(2, 2, line, curses.A_DIM)
            if e.get("warning"):
                self.put(3, 2, e["warning"], curses.color_pair(3) | curses.A_BOLD)
        hdr_y = 5 if (self.est and self.est.get("warning")) else 4
        self.hline(hdr_y)
        self.put(hdr_y + 1, 2, "  Step  Status  Description", curses.A_BOLD)
        self.put(5, 62, "Actions: ↑↓ move  Enter run  A run-all  D dry-run  C config  Q quit", curses.A_DIM)
        for i, st in enumerate(self.stages):
            y = 6 + i
            done = self.plan.stages_done.get(st["key"])
            status = "done ✓" if done == "ok" else "fail ✗" if done == "fail" else "      "
            sattr = curses.color_pair(2) if done == "ok" else curses.color_pair(3) if done == "fail" else curses.A_DIM
            selattr = curses.A_REVERSE if i == self.sel else 0
            self.put(y, 2, f"{i+1}. {st['title']}", selattr | (curses.A_BOLD if i == self.sel else 0))
            self.put(y, 12, status, sattr)
            self.put(y, 26, st["desc"][:56], selattr)
        y = 6 + len(self.stages) + 1
        if self.plan.latest_run_dir():
            self.put(y, 2, f"latest run: {self.plan.latest_run_dir()}", curses.A_DIM)
            y += 1
        if self.msg:
            self.put(y, 2, self.msg, curses.color_pair(3) if self.msg_err else curses.A_DIM)
        self.refresh()

    # ---------- config screen ----------
    def draw_config(self) -> None:
        self.stdscr.erase()
        self.put(0, 2, "Configure — target / profile / model / training / data", curses.A_BOLD | curses.color_pair(1))
        h, w = self.stdscr.getmaxyx()
        # target + profile block
        self.put(2, 2, "Target:", curses.A_BOLD)
        for i, (k, v) in enumerate(TARGETS.items()):
            a = curses.A_REVERSE if self.plan.target == k else 0
            self.put(2, 12 + i * 46, f"{k} — {v['label'][:40]}", a)
        self.put(3, 2, "Profile:", curses.A_BOLD)
        rec = [p for p, _ in TARGETS[self.plan.target]["recommended"]]
        profs = rec + [p for p in ALL_PROFILES if p not in rec]
        shown = profs[:12]
        for i, p in enumerate(shown):
            col = i % 4
            row = 4 + (i // 4)
            a = curses.A_REVERSE if self.plan.profile == p else 0
            self.put(row, 12 + col * 26, p[:24], a)
        self.put(4 + (len(shown) - 1) // 4, w - 30, "(←→ switch target, ↑↓ profile)", curses.A_DIM)
        # fields
        top = 9
        self.put(top - 1, 2, "Overrides (empty = profile default):", curses.A_BOLD)
        n = min(len(CONFIG_FIELDS), max(1, h - top - 2))
        for i in range(n):
            idx = (self.cfg_idx + i) % len(CONFIG_FIELDS)
            key, label, default, kind = CONFIG_FIELDS[idx]
            y = top + i
            val = self.plan.cfg.get(key, "")
            cur = idx == self.cfg_idx
            a = curses.A_REVERSE if cur else 0
            self.put(y, 2, key[:20].ljust(20), a | (curses.A_BOLD if cur else 0))
            self.put(y, 23, label[:46], a)
            self.put(y, 70, (val if val not in ("", None) else f"[{default}]")[: w - 72], a)
        self.put(h - 2, 2, "↑↓ field   Enter edit   Esc save & back   Tab cycle target/profile", curses.A_DIM)
        self.refresh()

    def config_key(self, ch) -> bool:
        """Handle config key. Returns True to leave screen."""
        n = len(CONFIG_FIELDS)
        if ch in (curses.KEY_UP, ord("k")):
            self.cfg_idx = (self.cfg_idx - 1) % n
        elif ch in (curses.KEY_DOWN, ord("j")):
            self.cfg_idx = (self.cfg_idx + 1) % n
        elif ch == curses.KEY_LEFT:
            ks = list(TARGETS)
            self.plan.target = ks[(ks.index(self.plan.target) - 1) % len(ks)]
            if self.plan.profile not in [p for p, _ in TARGETS[self.plan.target]["recommended"]] and \
               self.plan.profile not in ALL_PROFILES:
                self.plan.profile = TARGETS[self.plan.target]["recommended"][0][0]
            self.est_dirty = True
        elif ch == curses.KEY_RIGHT:
            ks = list(TARGETS)
            self.plan.target = ks[(ks.index(self.plan.target) + 1) % len(ks)]
            self.est_dirty = True
        elif ch == ord("\t"):
            ks = list(TARGETS)
            self.plan.target = ks[(ks.index(self.plan.target) + 1) % len(ks)]
            self.est_dirty = True
        elif ch in (curses.KEY_ENTER, 10, 13):
            key = CONFIG_FIELDS[self.cfg_idx][0]
            if self.editing:
                self.editing = False
            else:
                self.editing = True
        elif ch == 27:  # ESC
            self.editing = False
            self.recompute_estimate()
            return True
        if self.editing and ch != 27:
            key, _, _, kind = CONFIG_FIELDS[self.cfg_idx]
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                self.plan.cfg[key] = self.plan.cfg.get(key, "")[:-1]
            elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT, curses.KEY_UP, curses.KEY_DOWN):
                pass
            else:
                c = chr(ch)
                if c.isalnum() or c in "./_-":
                    self.plan.cfg[key] = self.plan.cfg.get(key, "") + c
            self.est_dirty = True
        return False

    # ---------- run screen ----------
    def start_stage(self, idx: int, run_all: bool = False) -> None:
        # Rebuild: downstream stage commands depend on artifacts produced by
        # earlier stages (e.g. SFT's --init-checkpoint from the new run dir).
        self.stages = stage_commands(self.plan)
        st = self.stages[idx]
        if st.get("needs_pretrain") and self.plan.latest_run_dir() is None:
            self.notify("No prior training run under checkpoints/ — run Pretrain first.", err=True)
            return
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = ROOT / "logs" / f"tui_{st['key']}_{ts}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.plan.run_dir = str(self.plan.latest_run_dir() or "")
        cmds = st.get("cmds") or [st]
        self.run_state = {
            "stage_idx": idx, "cmd_idx": 0, "run_all": run_all,
            "cmds": cmds,
            "proc": None, "log_path": log_path, "offset": 0,
            "stage_env": st.get("env"), "cwd": st.get("cwd", str(ROOT)),
        }
        self._spawn_next_cmd()

    def _spawn_next_cmd(self) -> None:
        rs = self.run_state
        if rs["cmd_idx"] >= len(rs["cmds"]):
            self._stage_finished(0)
            return
        c = rs["cmds"][rs["cmd_idx"]]
        env = c.get("env") or rs.get("stage_env") or self.plan.env_for_pipeline()
        if not env:
            env = dict(os.environ)
        try:
            rs["proc"] = subprocess.Popen(c["cmd"], cwd=rs["cwd"], env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, bufsize=1)
        except FileNotFoundError as e:
            self.notify(f"command not found: {e}", err=True)
            rs["proc"] = None
            self._stage_finished(127)
            return
        # tee to log file in a background-ish fashion: we read stdout in the loop
        rs["log_fh"] = open(rs["log_path"], "a", buffering=1)
        self.run_log_buf = []
        self.notify(f"running {self.stages[rs['stage_idx']]['title']} — q to stop", err=False)

    def _read_run_lines(self, maxlines: int = 400) -> None:
        rs = self.run_state
        proc = rs.get("proc")
        if proc is None or proc.stdout is None:
            return
        import select
        r, _, _ = select.select([proc.stdout], [], [], 0.25)
        got = 0
        while r and got < maxlines:
            line = proc.stdout.readline()
            if not line:
                break
            self.run_log_buf.append(line.rstrip("\n"))
            rs["log_fh"].write(line)
            got += 1
            r, _, _ = select.select([proc.stdout], [], [], 0)
        self.run_log_buf = self.run_log_buf[-600:]

    def _stage_finished(self, rc: int) -> None:
        rs = self.run_state
        st = self.stages[rs["stage_idx"]]
        if rs.get("log_fh"):
            rs["log_fh"].close()
        if rc == 0:
            self.plan.stages_done[st["key"]] = "ok"
            self.notify(f"{st['title']} OK — {st.get('ok_hint','')}  (log: {rs['log_path']})")
        else:
            self.plan.stages_done[st["key"]] = "fail"
            self.notify(f"{st['title']} FAILED (exit {rc}) — log: {rs['log_path']}", err=True)
        if rc == 0 and rs["run_all"]:
            # advance to next incomplete stage (rebuild: commands may depend
            # on artifacts this stage just produced)
            self.stages = stage_commands(self.plan)
            nxt = None
            for j in range(rs["stage_idx"] + 1, len(self.stages)):
                if self.plan.stages_done.get(self.stages[j]["key"]) != "ok":
                    nxt = j
                    break
            if nxt is not None:
                new_st = self.stages[nxt]
                if new_st.get("needs_pretrain") and self.plan.latest_run_dir() is None:
                    self.notify("No prior training run under checkpoints/ — run Pretrain first.",
                                err=True)
                    self.run_state = None
                    self.screen_idx = "main"
                    return
                rs["stage_idx"] = nxt
                rs["cmd_idx"] = 0
                rs["cmds"] = new_st.get("cmds") or [new_st]
                rs["stage_env"] = new_st.get("env")
                rs["cwd"] = new_st.get("cwd", str(ROOT))
                self._spawn_next_cmd()
                return
        self.run_state = None
        self.screen_idx = "main"
        self.sel = min(self.sel, len(self.stages) - 1)

    def stop_run(self) -> None:
        rs = self.run_state
        if rs and rs.get("proc") and rs["proc"].poll() is None:
            try:
                rs["proc"].send_signal(signal.SIGINT)
                time.sleep(0.5)
                if rs["proc"].poll() is None:
                    rs["proc"].terminate()
            except ProcessLookupError:
                pass
            self.notify("run stopped by user", err=True)

    def draw_run(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        rs = self.run_state
        st = self.stages[rs["stage_idx"]]
        c = rs["cmds"][rs["cmd_idx"]]
        running = rs.get("proc") and rs["proc"].poll() is None
        self.put(0, 2, f"Running: {st['title']}  (cmd {rs['cmd_idx']+1}/{len(rs['cmds'])})"
                       f"{'  · running' if running else '  · waiting'}",
                 curses.A_BOLD | (curses.color_pair(2) if running else curses.color_pair(3)))
        self.put(1, 2, "$ " + " ".join(c["cmd"])[:w - 6], curses.A_DIM)
        self.hline(2)
        lines = self.run_log_buf[-(h - 6):]
        for i, ln in enumerate(lines):
            a = curses.color_pair(3) if re.search(r"error|fail|traceback|exception", ln, re.I) else 0
            self.put(3 + i, 2, ln[:w - 4], a)
        self.put(h - 2, 2, "q stop & exit to menu    (full log: " + str(rs["log_path"]) + ")", curses.A_DIM)
        self.refresh()

    # ---------- main loop ----------
    def main(self, stdscr) -> None:
        self.stdscr = stdscr
        curses.curs_set(0)
        stdscr.nodelay(False)
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
        except curses.error:
            pass
        self.recompute_estimate()
        while True:
            if self.run_state is not None:
                self._read_run_lines()
                self.draw_run()
                ch = stdscr.getch()
                if ch in (ord("q"), 27):
                    self.stop_run()
                    # wait briefly for proc to end so we capture exit
                    for _ in range(20):
                        p = self.run_state.get("proc")
                        if p is None or p.poll() is not None:
                            break
                        time.sleep(0.1)
                    if self.run_state.get("proc") is not None:
                        self._stage_finished(self.run_state["proc"].returncode or 130)
                    else:
                        self._stage_finished(130)
                continue
            if self.screen_idx == "config":
                self.draw_config()
                ch = stdscr.getch()
                if self.config_key(ch):
                    self.screen_idx = "main"
                continue
            # main
            self.draw_main()
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q"), 27):
                break
            elif ch in (curses.KEY_UP, ord("k")):
                self.sel = (self.sel - 1) % len(self.stages)
            elif ch in (curses.KEY_DOWN, ord("j")):
                self.sel = (self.sel + 1) % len(self.stages)
            elif ch in (curses.KEY_ENTER, 10, 13):
                self.start_stage(self.sel, run_all=False)
                self.screen_idx = "run"
            elif ch in (ord("a"), ord("A")):
                self.start_stage(self.sel, run_all=True)
                self.screen_idx = "run"
            elif ch in (ord("d"), ord("D")):
                self._dry_run_dialog()
            elif ch in (ord("c"), ord("C")):
                self.screen_idx = "config"
            elif ch in (ord("r"), ord("R")):
                self.notify(f"recomputed size estimate: params={self.est.get('param_count') if self.est else '?'}")
                self.recompute_estimate()

    def _dry_run_dialog(self) -> None:
        self.recompute_estimate()
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        self.put(0, 2, "Dry-run plan (no commands executed)", curses.A_BOLD | curses.color_pair(1))
        y = 2
        for i, st in enumerate(self.stages, 1):
            if y > h - 4:
                break
            self.put(y, 2, f"{i}. {st['title']}", curses.A_BOLD)
            y += 1
            cmds = st.get("cmds") or [st]
            for c in cmds:
                if y > h - 3:
                    break
                line = "    $ " + " ".join(c["cmd"])
                if c.get("env"):
                    envline = " ".join(f"{k}={v}" for k, v in c["env"].items() if k in (
                        "PROFILE", "OUTPUT_DIR", "PRETRAIN_EPOCHS", "CHAT_EPOCHS", "TRAIN_SEED",
                        "PRETRAIN_DATA", "CHAT_DATA", "AMP", "CUDA_DEVICE")[:8])
                    line = f"    $ {envline}  {line[6:]}" if envline else line
                self.put(y, 0, line[:w - 2], curses.A_DIM)
                y += 1
            y += 1
        self.put(h - 2, 2, "any key to return", curses.A_DIM)
        self.refresh()
        self.stdscr.getch()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description="NanoLLM TUI launcher (stdlib curses).")
    p.add_argument("--target", choices=list(TARGETS), default="cardputer")
    p.add_argument("--profile", default="cardputer-mqa-ctx224")
    p.add_argument("--dry-run", action="store_true", help="print plan and exit (no TUI)")
    p.add_argument("--run", metavar="STAGE", help="headless: run one stage (datasets|pretrain|sft|posttrain|firmware|verify)")
    p.add_argument("--yes", action="store_true", help="with --run: skip per-command confirmation")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="config override (repeatable), e.g. --set PRETRAIN_EPOCHS=5 --set D_MODEL=96")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = Plan(target=args.target, profile=args.profile)
    for item in args.set:
        if "=" in item:
            k, v = item.split("=", 1)
            if k in DEFAULTS:
                plan.cfg[k] = v
            else:
                print(f"warning: unknown config key {k} (ignored)")
    if args.dry_run:
        print_plan(plan)
        return 0
    if args.run:
        return run_stage_headless(plan, args.run, assume_yes=args.yes)
    try:
        curses.wrapper(TUI(plan).main)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
