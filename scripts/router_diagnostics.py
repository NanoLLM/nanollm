#!/usr/bin/env python3
"""
Router diagnostics for NanoLLM MoE checkpoints.

Measures:
1. Expert load imbalance (fraction of tokens routed to each expert)
2. Expert entropy (how deterministic the routing is)
3. Dead experts (zero utilization)
4. Top-k vs top-1 routing comparison
5. Shared expert utilization
6. Correlation between router logits and input hidden states

Output: logs/results/router_diagnostics.json

Usage (from repository root):
  python scripts/router_diagnostics.py \\
      --checkpoint checkpoints/capacity_matrix/mqa_ctx224_20260713/model_pretrain.pt \\
      --data data/fineweb/fineweb_val.txt \\
      --num-tokens 100000
"""
from __future__ import annotations

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from model import NanoLLM, MoEFeedForward
from tokenizer import BPETokenizer


def expert_load_imbalance(
    experts_used: list[int],
    n_experts: int,
) -> dict:
    """Compute load imbalance statistics."""
    counts = np.zeros(n_experts)
    for e in experts_used:
        counts[e] += 1
    total = len(experts_used) or 1
    fractions = counts / total
    imbalance = fractions.max() - fractions.min()
    entropy = -np.sum(fractions * np.log(fractions + 1e-10))
    max_entropy = np.log(n_experts)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        "counts": counts.tolist(),
        "fractions": [float(f) for f in fractions],
        "load_imbalance": float(imbalance),
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "ideal_fraction": 1.0 / n_experts,
    }


def compute_router_diagnostics(
    model: NanoLLM,
    dataloader: DataLoader,
    config: dict,
    max_tokens: int = 100000,
    device: str = "cpu",
) -> dict:
    """Run forward passes collecting router statistics."""
    model.eval()

    expert_counts: list[int] = []
    router_entropies: list[float] = []
    shared_counts: list[int] = []
    input_norms: list[float] = []
    router_log_norms: list[float] = []
    top_k_assignments: list[dict] = []

    total_tokens = 0
    n_layers = int(config.get("n_layers", len(model.blocks)))
    use_moe = bool(config.get("use_moe", False))
    n_experts = int(config.get("moe_n_experts", 4))
    top_k = int(config.get("moe_top_k", 1))

    if not use_moe:
        return {"error": "Model is not MoE-enabled"}

    def _as_token_batch(batch):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if not torch.is_tensor(batch):
            batch = torch.as_tensor(batch)
        if batch.dtype != torch.long:
            batch = batch.long()
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        return batch.to(device)

    with torch.no_grad():
        for batch in dataloader:
            tokens = _as_token_batch(batch)
            b, t = tokens.shape
            seq_len = min(t, int(getattr(model, "max_seq_len", t)))
            tokens = tokens[:, :seq_len]
            b, t = tokens.shape

            tok_emb = model.token_embedding(tokens)
            pos = torch.arange(0, t, device=device).unsqueeze(0)
            x = tok_emb + model.pos_embedding(pos)

            for block in model.blocks:
                x = x + block.dropout(block.attention(block.norm1(x)))
                ff_in = block.norm2(x)
                ff = block.feed_forward
                if isinstance(ff, MoEFeedForward):
                    flat = ff_in.reshape(-1, ff_in.size(-1))
                    router_logits = ff.router(flat)
                    router_probs = torch.softmax(router_logits, dim=-1)
                    selected = torch.argmax(router_logits, dim=-1)
                    entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-10))).sum(dim=-1)
                    max_logits = router_logits.gather(1, selected.unsqueeze(1)).squeeze(1)

                    expert_counts.extend(int(v) for v in selected.cpu().tolist())
                    router_entropies.extend(float(v) for v in entropy.cpu().tolist())
                    input_norms.extend(float(v) for v in flat.norm(dim=-1).cpu().tolist())
                    router_log_norms.extend(float(v) for v in max_logits.cpu().tolist())
                    if ff.shared_expert is not None:
                        shared_counts.extend([1] * flat.size(0))

                    if top_k > 1:
                        values, indices = torch.topk(router_logits, top_k, dim=-1)
                        for row in range(min(flat.size(0), 32)):
                            for ki in range(top_k):
                                top_k_assignments.append(
                                    {
                                        "token": total_tokens + row,
                                        "expert": int(indices[row, ki].item()),
                                        "logit": float(values[row, ki].item()),
                                        "rank": ki,
                                    }
                                )

                x = x + block.dropout(ff(ff_in))

            total_tokens += b * t
            if total_tokens >= max_tokens:
                break

    load_stats = expert_load_imbalance(expert_counts, n_experts)
    router_entropy_stats = {
        "mean": float(np.mean(router_entropies)) if router_entropies else 0.0,
        "std": float(np.std(router_entropies)) if router_entropies else 0.0,
        "min": float(np.min(router_entropies)) if router_entropies else 0.0,
        "max": float(np.max(router_entropies)) if router_entropies else 0.0,
    }

    input_norm_corr = 0.0
    if len(input_norms) > 1 and len(router_log_norms) == len(input_norms):
        corr = np.corrcoef(input_norms, router_log_norms)[0, 1]
        input_norm_corr = float(corr) if np.isfinite(corr) else 0.0

    dead_experts = [i for i in range(n_experts) if load_stats["counts"][i] == 0]

    return {
        "checkpoint": "checkpoints/multi_seed_ctx224_l18_20260718/seed44/final4/model_best.pt",
        "total_tokens_analyzed": int(total_tokens),
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "load_imbalance": load_stats,
        "router_entropy": router_entropy_stats,
        "input_norm_router_logit_correlation": input_norm_corr,
        "dead_experts": dead_experts,
        "dead_expert_fraction": len(dead_experts) / n_experts if n_experts else 0.0,
        "shared_expert_utilization": float(np.mean(shared_counts)) if shared_counts else None,
        "top_k_assignments_sample": top_k_assignments[:10],
    }



def main():
    parser = argparse.ArgumentParser(description="Router diagnostics for MoE")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to MoE checkpoint")
    parser.add_argument("--data", type=str, default="data/fineweb/fineweb_val.txt",
                        help="Validation text data")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer JSON (defaults to checkpoint's tokenizer_path)")
    parser.add_argument("--num-tokens", type=int, default=100000,
                        help="Max tokens to analyze")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Block size for processing")
    parser.add_argument("--output", type=str, default="logs/results/router_diagnostics.json")
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", {})
    model = NanoLLM(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(args.device)

    # Load tokenizer
    tokenizer_path = args.tokenizer
    tokenizer = None
    if tokenizer_path and Path(tokenizer_path).exists():
        tokenizer = BPETokenizer.from_file(str(tokenizer_path))
        print(f"[INFO] Loaded tokenizer from: {tokenizer_path}")
    if tokenizer is None:
        # Try checkpoint's stored tokenizer_path first
        stored = ckpt.get("tokenizer_path")
        if stored and Path(stored).exists():
            tokenizer = BPETokenizer()
            tokenizer.load(stored)
            print(f"[INFO] Loaded tokenizer from stored path: {stored}")
    if tokenizer is None:
        # Search in checkpoint parent directories (e.g. clean_transfer_v1_*/final4/)
        ckpt_dir = Path(args.checkpoint).parent
        for p in ckpt_dir.rglob("tokenizer.json"):
            try:
                tokenizer = BPETokenizer()
                tokenizer.load(str(p))
                print(f"[INFO] Loaded tokenizer from: {p}")
                break
            except Exception:
                continue
    if tokenizer is None:
        # Last resort: search checkpoints/ root
        repo_root = Path(__file__).resolve().parent.parent
        for p in (repo_root / "checkpoints").rglob("tokenizer.json"):
            try:
                tokenizer = BPETokenizer()
                tokenizer.load(str(p))
                print(f"[INFO] Loaded tokenizer from: {p}")
                break
            except Exception:
                continue
    if tokenizer is None:
        print("[WARN] No tokenizer found; using simple char-level tokenizer")

    # Prepare data
    if not Path(args.data).exists():
        print(f"[WARN] Data not found at {args.data}; using sample data")
        args.data = "sample_data.txt"

    with open(args.data) as f:
        text = f.read()

    if tokenizer:
        tokens = tokenizer.encode(text)
    else:
        # Simple char-level
        tokens = torch.tensor([ord(c) for c in text[:args.num_tokens * 4]], dtype=torch.long)

    # Create dataloader
    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, block_size):
            self.tokens = tokens
            self.block_size = block_size
        def __len__(self):
            return len(self.tokens) // self.block_size
        def __getitem__(self, idx):
            start = idx * self.block_size
            end = start + self.block_size
            chunk = self.tokens[start:end]
            if not torch.is_tensor(chunk):
                chunk = torch.as_tensor(chunk, dtype=torch.long)
            return chunk.long()

    dataset = TokenDataset(tokens, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Compute diagnostics
    print(f"Analyzing router behavior on {args.num_tokens} tokens...")
    result = compute_router_diagnostics(model, dataloader, config, args.num_tokens, args.device)

    # Print summary
    print(f"\nRouter Diagnostics:")
    print(f"  Tokens analyzed: {result.get('total_tokens_analyzed', 'N/A')}")
    li = result.get("load_imbalance", {})
    print(f"  Load imbalance (max-min): {li.get('load_imbalance', 'N/A'):.4f}")
    print(f"  Normalized entropy: {li.get('normalized_entropy', 'N/A'):.4f}")
    print(f"  Dead experts: {result.get('dead_experts', [])}")
    print(f"  Router entropy mean: {result.get('router_entropy', {}).get('mean', 'N/A'):.4f}")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nRouter diagnostics written to {out_path}")


if __name__ == "__main__":
    main()
