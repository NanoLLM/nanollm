"""
Ultra-compact transformer model for ESP32 deployment.
Designed to fit in 6MB with ~200KB inference memory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    angles = torch.outer(pos, inv_freq)
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., ::2] = rot1
    out[..., 1::2] = rot2
    return out


class MultiHeadAttention(nn.Module):
    """Minimal multi-head attention for tiny models."""
    
    def __init__(self, d_model, n_heads, n_kv_heads=None, use_rope=False):
        super().__init__()
        assert d_model % n_heads == 0
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
            raise ValueError("n_kv_heads must be positive and divide n_heads")
        if use_rope and (d_model // n_heads) % 2 != 0:
            raise ValueError("use_rope requires an even head dimension (d_model // n_heads)")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.kv_dim = n_kv_heads * self.d_k
        self.use_rope = use_rope
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.kv_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.kv_dim, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        if self.use_rope:
            cos, sin = _build_rope_cache(seq_len, self.d_k, x.device, x.dtype)
            Q = _apply_rope(Q, cos, sin)
            K = _apply_rope(K, cos, sin)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            K = K.repeat_interleave(repeats, dim=1)
            V = V.repeat_interleave(repeats, dim=1)
        
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    """Minimal feed-forward network."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class MoEFeedForward(nn.Module):
    """Token-level sparse MoE FFN with optional shared expert."""

    def __init__(self, d_model, expert_d_ff, n_experts=4, top_k=1, shared_d_ff=0):
        super().__init__()
        if n_experts <= 0:
            raise ValueError("n_experts must be > 0")
        if top_k <= 0 or top_k > n_experts:
            raise ValueError("top_k must be in [1, n_experts]")

        self.n_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(d_model, expert_d_ff)
            for _ in range(n_experts)
        ])
        self.shared_expert = FeedForward(d_model, shared_d_ff) if shared_d_ff > 0 else None

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        router_logits = self.router(x_flat)
        output = torch.zeros_like(x_flat)

        if self.top_k == 1:
            router_probs = torch.softmax(router_logits, dim=-1)
            selected_experts = torch.argmax(router_logits, dim=-1)
            for expert_id, expert in enumerate(self.experts):
                mask = selected_experts == expert_id
                if torch.any(mask):
                    gate = router_probs[mask, expert_id].unsqueeze(-1)
                    output[mask] = gate * expert(x_flat[mask]).to(output.dtype)
        else:
            topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
            gates = torch.softmax(topk_logits, dim=-1)

            for expert_id, expert in enumerate(self.experts):
                for k_idx in range(self.top_k):
                    mask = topk_indices[:, k_idx] == expert_id
                    if not torch.any(mask):
                        continue
                    expert_input = x_flat[mask]
                    expert_out = expert(expert_input)
                    output[mask] += gates[mask, k_idx].unsqueeze(-1) * expert_out

        if self.shared_expert is not None:
            output += self.shared_expert(x_flat).to(output.dtype)

        return output.view(batch_size, seq_len, d_model)


class TransformerBlock(nn.Module):
    """Single transformer block with layer norm and residual connections."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, n_kv_heads=None,
                 use_moe=False, moe_n_experts=4, moe_top_k=1, moe_shared_d_ff=0,
                 use_rope=False):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, n_kv_heads, use_rope=use_rope)
        if use_moe:
            self.feed_forward = MoEFeedForward(
                d_model=d_model,
                expert_d_ff=d_ff,
                n_experts=moe_n_experts,
                top_k=moe_top_k,
                shared_d_ff=moe_shared_d_ff,
            )
        else:
            self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class NanoLLM(nn.Module):
    """
    Ultra-compact language model.
    
    Architecture:
    - vocab_size: 500 (BPE tokens)
    - d_model: 64 (embedding dimension)
    - n_layers: 1-2 (transformer layers)
    - n_heads: 2-4 (attention heads)
    - d_ff: 128-256 (feed-forward dimension)
    - max_seq_len: 128 (context length)
    
    Estimated size (int8 quantized): ~2-4MB
    """
    
    def __init__(
        self,
        vocab_size=500,
        d_model=64,
        n_layers=1,
        n_heads=2,
        n_kv_heads=None,
        d_ff=128,
        max_seq_len=128,
        dropout=0.1,
        use_moe=False,
        moe_n_experts=4,
        moe_top_k=1,
        moe_shared_d_ff=0,
        causal_attention=True,
        use_rope=False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.use_moe = use_moe
        self.causal_attention = causal_attention
        self.use_rope = use_rope
        
        # Token embeddings; learned absolute positions unless RoPE is enabled.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if use_rope:
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                d_ff,
                dropout,
                n_kv_heads=self.n_kv_heads,
                use_moe=use_moe,
                moe_n_experts=moe_n_experts,
                moe_top_k=moe_top_k,
                moe_shared_d_ff=moe_shared_d_ff,
                use_rope=use_rope,
            )
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (optional, saves memory)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        
        # Embeddings
        x = self.token_embedding(idx)
        if not self.use_rope:
            pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
            x = x + self.pos_embedding(pos)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None, eos_token_id=3):
        """Generate tokens autoregressively."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if needed
                idx_cond = idx[:, -self.max_seq_len:]
                
                # Forward pass
                logits, _ = self(idx_cond)
                
                # Get last token logits
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append
                idx = torch.cat([idx, idx_next], dim=1)
                if eos_token_id is not None and bool(torch.all(idx_next == eos_token_id)):
                    break
        
        return idx
    
    def get_model_size_mb(self, quantized=True):
        """Estimate model size in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        bytes_per_param = 1 if quantized else 4
        size_mb = (total_params * bytes_per_param) / (1024 * 1024)
        return size_mb, total_params

