"""
Training script for NanoLLM.
Optimized for RTX 3090 with mixed precision training.
Uses BPE tokenization for better text encoding.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from model import NanoLLM, MoEFeedForward
from tokenizer import BPETokenizer, train_tokenizer_from_data


class TextDataset(Dataset):
    """Text dataset backed by a memory-mapped token cache to minimize RAM usage."""

    CACHE_VERSION = 4
    IGNORE_TARGET = -1

    def __init__(
        self,
        text_file,
        tokenizer,
        block_size=128,
        cache_dir=None,
        chunk_chars=1_000_000,
        window_stride=1,
        assistant_only_loss=False,
    ):
        self.block_size = block_size
        self.window_stride = max(1, int(window_stride))
        self.tokenizer = tokenizer
        self.text_file = text_file
        self.chunk_chars = max(1024, int(chunk_chars))
        self.assistant_only_loss = bool(assistant_only_loss)
        self.meta = None

        cache_root = Path(cache_dir) if cache_dir else Path(text_file).parent
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_prefix = Path(text_file).stem
        self.cache_data_path = cache_root / f"{cache_prefix}_tokens.int32"
        self.cache_meta_path = cache_root / f"{cache_prefix}_tokens_meta.json"
        self.cache_mask_path = cache_root / f"{cache_prefix}_loss_mask.uint8"

        if self._load_cache_metadata():
            print(f"Loading token cache from {self.cache_data_path}...")
        else:
            print(f"Building token cache at {self.cache_data_path}...")
            self._build_token_cache()

        self.data = np.memmap(self.cache_data_path, mode='r', dtype=np.int32)
        self.token_count = int(self.meta['token_count'])
        self.loss_mask = None
        if self.assistant_only_loss:
            if not self.cache_mask_path.exists():
                raise ValueError(
                    f"Assistant-only loss requested but mask cache missing: {self.cache_mask_path}"
                )
            self.loss_mask = np.memmap(self.cache_mask_path, mode='r', dtype=np.uint8)
            if len(self.loss_mask) != self.token_count:
                raise ValueError("Loss mask length does not match token cache length")
        print(f"Tokenized data length: {self.token_count:,} tokens (memmapped)")
        if self.assistant_only_loss:
            trainable = int(np.sum(self.loss_mask))
            print(
                f"Assistant-only loss: {trainable:,}/{self.token_count:,} tokens "
                f"({100.0 * trainable / max(self.token_count, 1):.1f}%)"
            )

    def _expected_meta(self):
        tokenizer_fingerprint = None
        fingerprint_fn = getattr(self.tokenizer, 'fingerprint', None)
        if callable(fingerprint_fn):
            tokenizer_fingerprint = fingerprint_fn()
        return {
            "cache_version": self.CACHE_VERSION,
            "text_path": os.path.abspath(self.text_file),
            "text_size": os.path.getsize(self.text_file),
            "text_mtime": os.path.getmtime(self.text_file),
            "vocab_size": self.tokenizer.get_vocab_size(),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "assistant_only_loss": self.assistant_only_loss,
        }

    def _load_cache_metadata(self):
        if not self.cache_data_path.exists() or not self.cache_meta_path.exists():
            return False
        try:
            with open(self.cache_meta_path, 'r', encoding='utf-8') as meta_file:
                meta = json.load(meta_file)
        except (OSError, json.JSONDecodeError):
            return False

        expected = self._expected_meta()
        if meta.get('cache_version') != expected['cache_version']:
            return False
        for key in ('text_path', 'text_size', 'text_mtime', 'vocab_size', 'tokenizer_fingerprint', 'assistant_only_loss'):
            if meta.get(key) != expected[key]:
                return False
        token_count = meta.get('token_count')
        if token_count is None or token_count <= 0:
            return False
        self.meta = meta
        return True

    def _flush_buffer(self, buffer, dst_handle, mask_handle=None):
        text = ''.join(buffer)
        if not text:
            return 0
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            return 0
        arr = np.asarray(token_ids, dtype=np.int32)
        arr.tofile(dst_handle)
        if mask_handle is not None:
            from chat_template import build_assistant_token_mask

            mask = np.asarray(build_assistant_token_mask(text, self.tokenizer), dtype=np.uint8)
            if mask.size != arr.size:
                raise ValueError(
                    f"Assistant mask length ({mask.size}) does not match tokenized chunk ({arr.size})"
                )
            mask.tofile(mask_handle)
        return arr.size

    def _build_token_cache(self):
        expected_meta = self._expected_meta()
        tmp_path = self.cache_data_path.with_suffix(self.cache_data_path.suffix + '.tmp')
        tmp_mask_path = self.cache_mask_path.with_suffix(self.cache_mask_path.suffix + '.tmp')
        if tmp_path.exists():
            tmp_path.unlink()
        if self.cache_data_path.exists():
            self.cache_data_path.unlink()
        if self.assistant_only_loss:
            if tmp_mask_path.exists():
                tmp_mask_path.unlink()
            if self.cache_mask_path.exists():
                self.cache_mask_path.unlink()

        token_count = 0
        with open(self.text_file, 'r', encoding='utf-8', errors='ignore') as src, \
                open(tmp_path, 'wb') as dst, \
                tqdm(desc="Tokenizing dataset", unit='lines') as progress:
            mask_dst = open(tmp_mask_path, 'wb') if self.assistant_only_loss else None
            try:
                buffer = []
                buffer_chars = 0
                for line in src:
                    buffer.append(line)
                    buffer_chars += len(line)
                    progress.update(1)
                    # Assistant masks depend on seeing a complete User/Assistant record.
                    # Delay chat-cache flushes until the blank record separator.
                    record_boundary = not line.strip()
                    if buffer_chars >= self.chunk_chars and (
                        not self.assistant_only_loss or record_boundary
                    ):
                        token_count += self._flush_buffer(buffer, dst, mask_dst)
                        buffer.clear()
                        buffer_chars = 0
                if buffer:
                    token_count += self._flush_buffer(buffer, dst, mask_dst)
            finally:
                if mask_dst is not None:
                    mask_dst.close()

        os.replace(tmp_path, self.cache_data_path)
        if self.assistant_only_loss:
            os.replace(tmp_mask_path, self.cache_mask_path)
        expected_meta['token_count'] = int(token_count)
        with open(self.cache_meta_path, 'w', encoding='utf-8') as meta_file:
            json.dump(expected_meta, meta_file, indent=2)
        self.meta = expected_meta

        if token_count <= self.block_size:
            raise ValueError("Tokenized dataset is smaller than the configured block size.")

    def __len__(self):
        if self.token_count <= self.block_size:
            return 0
        max_start = self.token_count - self.block_size - 1
        return max_start // self.window_stride + 1
    
    def __getitem__(self, idx):
        start = idx * self.window_stride
        window = np.asarray(self.data[start:start + self.block_size + 1], dtype=np.int64)
        x = torch.tensor(window[:-1], dtype=torch.long)
        y = torch.tensor(window[1:], dtype=torch.long)
        if self.loss_mask is not None:
            target_mask = self.loss_mask[start + 1:start + self.block_size + 1]
            y = y.clone()
            y[target_mask == 0] = self.IGNORE_TARGET
        return x, y


# Removed old encoding functions - now using BPE tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_num_workers(requested=None):
    if requested is not None:
        return max(0, requested)
    if os.cpu_count():
        return min(16, max(4, os.cpu_count() // 2))
    return 2


def resolve_device(cuda_device=None):
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if cuda_device is not None:
        return torch.device(f'cuda:{cuda_device}')
    return torch.device('cuda')


def resolve_amp_mode(args):
    amp = getattr(args, 'amp', 'auto')
    if amp == 'auto':
        if args.mixed_precision:
            amp = 'fp16'
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            amp = 'bf16'
        else:
            amp = 'none'
    if amp not in ('none', 'fp16', 'bf16'):
        raise ValueError(f"Unsupported amp mode: {amp}")
    if amp == 'bf16' and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("Warning: bf16 requested but unsupported; falling back to fp16.")
        amp = 'fp16'
    return amp


def configure_cuda_runtime(tf32=False):
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision('high')


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, prefetch_factor=8, drop_last=False):
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = max(2, int(prefetch_factor))
    return DataLoader(dataset, **loader_kwargs)


def build_optimizer(model, learning_rate, weight_decay, device, fused_adamw='auto'):
    use_fused = False
    if device.type == 'cuda':
        if fused_adamw == 'on':
            use_fused = True
        elif fused_adamw == 'auto':
            use_fused = True

    if use_fused:
        try:
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                fused=True,
            ), True
        except TypeError:
            # Older PyTorch builds may not expose fused AdamW.
            pass

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    ), False


def loss_to_perplexity(loss):
    return math.exp(min(float(loss), 20.0))


def split_dataset(dataset, val_split, seed):
    """Hold out the last val_split fraction of windows with a full-context no-overlap gap."""
    if val_split <= 0.0:
        return dataset, None
    n = len(dataset)
    val_size = max(1, int(n * val_split))
    block_size = max(1, int(getattr(dataset, 'block_size', 1)))
    window_stride = max(1, int(getattr(dataset, 'window_stride', 1)))
    context_gap = max(1, math.ceil(block_size / window_stride))
    val_start = n - val_size
    train_end = val_start - context_gap
    if train_end <= 0:
        raise ValueError(
            f"Validation split ({val_split}) with no-overlap gap={context_gap} leaves no "
            f"training samples (dataset size={n}). Reduce --val_split or block size."
        )
    train_size = train_end
    train_indices = list(range(train_size))
    val_indices = list(range(val_start, n))
    generator = torch.Generator().manual_seed(seed)
    # Shuffle train windows only; validation stays on the held-out tail.
    train_indices = torch.randperm(train_size, generator=generator).tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_train_val_datasets(args, tokenizer):
    """Build train/val datasets using either a split or an independent validation file."""
    dataset = TextDataset(
        args.data,
        tokenizer,
        block_size=args.block_size,
        cache_dir=args.token_cache_dir,
        chunk_chars=args.token_cache_chunk_chars,
        window_stride=args.window_stride_tokens,
        assistant_only_loss=args.assistant_only_loss,
    )

    val_dataset = None
    validation_mode = 'split'

    if args.val_data:
        train_path = os.path.abspath(args.data)
        val_path = os.path.abspath(args.val_data)
        if train_path == val_path:
            raise ValueError('--val_data must be a different file from --data to keep validation independent')
        val_cache_dir = os.path.join(args.token_cache_dir, 'val_cache')
        os.makedirs(val_cache_dir, exist_ok=True)
        try:
            val_dataset = TextDataset(
                args.val_data,
                tokenizer,
                block_size=args.block_size,
                cache_dir=val_cache_dir,
                chunk_chars=args.token_cache_chunk_chars,
                window_stride=args.window_stride_tokens,
                assistant_only_loss=args.assistant_only_loss,
            )
        except ValueError as exc:
            # The independent val file is empty or too small for one block
            # (e.g. a smoke profile whose 5% split rounded to zero turns).
            # Degrade to an internal tail split of the train file instead of
            # crashing, so tiny/smoke runs still complete.
            print(
                f"WARNING: independent val file {args.val_data} is unusable ({exc}); "
                f"falling back to an internal {args.val_split:.0%} tail split of "
                f"{args.data} for validation."
            )
            train_dataset, val_dataset = split_dataset(dataset, args.val_split, args.seed)
            return dataset, train_dataset, val_dataset, 'split'
        train_dataset = dataset
        validation_mode = 'independent_file'
    else:
        train_dataset, val_dataset = split_dataset(dataset, args.val_split, args.seed)

    return dataset, train_dataset, val_dataset, validation_mode


def model_config_dict(args):
    return {
        'vocab_size': args.vocab_size,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'n_kv_heads': args.n_kv_heads if args.n_kv_heads is not None else args.n_heads,
        'd_ff': args.d_ff,
        'use_moe': args.use_moe,
        'moe_n_experts': args.moe_n_experts,
        'moe_top_k': args.moe_top_k,
        'moe_shared_d_ff': args.moe_shared_d_ff,
        'max_seq_len': args.block_size,
        'dropout': args.dropout,
        'causal_attention': True,
        'use_rope': args.use_rope,
    }


def _tensorboard_scalar_tag(key):
    """Map flat metric keys to TensorBoard tag namespaces."""
    if key == 'learning_rate':
        return 'train/lr'
    if key.startswith('train_'):
        return f"train/{key[len('train_'):]}"
    if key.startswith('val_'):
        return f"val/{key[len('val_'):]}"
    if key.startswith('moe_'):
        return f"moe/{key[len('moe_'):]}"
    return key


def _tensorboard_hparam_value(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


class MetricsLogger:
    """Persist epoch metrics to JSON and optionally TensorBoard."""

    def __init__(self, output_dir, use_tensorboard=True, initial_global_step=0, resume_history=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / 'training_metrics.json'
        self.tensorboard_dir = self.output_dir / 'tensorboard'
        self.history = {
            'epochs': [],
            'best_epoch': None,
            'best_val_loss': None,
            'best_train_loss': None,
        }
        if resume_history and self.metrics_path.exists():
            try:
                with open(self.metrics_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    for key in ('epochs', 'best_epoch', 'best_val_loss', 'best_train_loss'):
                        if key in loaded:
                            self.history[key] = loaded[key]
                    if 'run_config' in loaded and 'pretrain_run_config' not in loaded:
                        self.history['pretrain_run_config'] = loaded['run_config']
            except (OSError, json.JSONDecodeError):
                pass
        self.writer = None
        self.global_step = initial_global_step
        self._hparams = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(
                    log_dir=str(self.tensorboard_dir),
                    flush_secs=30,
                )
                print(f"TensorBoard logs: {self.tensorboard_dir}")
                print(f"  View: tensorboard --logdir {self.tensorboard_dir}")
            except ImportError:
                print("TensorBoard unavailable (pip install tensorboard); continuing with JSON metrics only.")

    def log_run_config(self, run_config):
        if self.history.get('run_config') and self.history.get('pretrain_run_config') is None:
            self.history['pretrain_run_config'] = self.history['run_config']
        self.history['run_config'] = run_config
        self._hparams = run_config
        self._write_json()
        if self.writer is None:
            return
        self.writer.add_text('run_config', json.dumps(run_config, indent=2, sort_keys=True), 0)

    def log_train_step(self, scalars):
        if self.writer is not None:
            for tag, value in scalars.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(tag, value, self.global_step)
        self.global_step += 1

    def log_epoch(self, epoch_record):
        self.history['epochs'].append(epoch_record)
        if self.writer is not None:
            step = epoch_record['epoch']
            for key, value in epoch_record.items():
                if key == 'epoch':
                    continue
                if key == 'moe_expert_fraction' and isinstance(value, list):
                    for expert_idx, fraction in enumerate(value):
                        self.writer.add_scalar(f'moe/expert_{expert_idx}', fraction, step)
                    continue
                if not isinstance(value, (int, float)):
                    continue
                self.writer.add_scalar(_tensorboard_scalar_tag(key), value, step)
            self.writer.flush()
        self._write_json()

    def set_best(self, epoch, train_loss, val_loss):
        self.history['best_epoch'] = epoch
        self.history['best_train_loss'] = train_loss
        if val_loss is not None:
            self.history['best_val_loss'] = val_loss
        self._write_json()

    def log_hparams(self, final_metrics):
        if self.writer is None or not self._hparams:
            return
        hparams = {
            key: _tensorboard_hparam_value(value)
            for key, value in self._hparams.items()
            if value is not None
        }
        metrics = {
            key: float(value)
            for key, value in final_metrics.items()
            if isinstance(value, (int, float))
        }
        if not metrics:
            return
        try:
            self.writer.add_hparams(hparams, metrics)
        except Exception:
            pass

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _write_json(self):
        with open(self.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)


def build_lr_scheduler(optimizer, warmup_steps, total_steps):
    if total_steps <= 0:
        return None

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _grad_norm(model):
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def maybe_save_step_checkpoint(
    global_step,
    save_every_steps,
    output_dir,
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    config,
    tokenizer_path,
):
    """Save model_step_{N}.pt every save_every_steps optimizer steps (0 disables)."""
    if save_every_steps <= 0 or global_step <= 0 or global_step % save_every_steps != 0:
        return
    checkpoint_path = os.path.join(output_dir, f'model_step_{global_step}.pt')
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch,
        train_loss,
        val_loss,
        config,
        tokenizer_path,
        global_step=global_step,
    )
    print(f"Saved step checkpoint: {checkpoint_path} (global_step={global_step})")


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    amp_mode='none',
    scaler=None,
    grad_clip=1.0,
    scheduler=None,
    log_every=50,
    metrics_logger=None,
    current_epoch=1,
    save_every_steps=0,
    output_dir=None,
    checkpoint_config=None,
    tokenizer_path=None,
    last_val_loss=None,
):
    """Train for one epoch; return aggregated metrics."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    grad_norm_sum = 0.0
    grad_norm_pre_clip_sum = 0.0
    grad_norm_count = 0
    skipped_batches = 0
    t0 = time.perf_counter()
    use_amp = amp_mode in ('fp16', 'bf16') and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_mode == 'bf16' else torch.float16

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (x, y) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=device.type == 'cuda')
        y = y.to(device, non_blocking=device.type == 'cuda')
        batch_tokens = int((y != TextDataset.IGNORE_TARGET).sum().item())
        if batch_tokens == 0:
            skipped_batches += 1
            continue
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                _, loss = model(x, targets=y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
            grad_norm_pre = _grad_norm(model)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm_post = _grad_norm(model) if grad_clip > 0 else grad_norm_pre
            scaler.step(optimizer)
            scaler.update()
        elif use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                _, loss = model(x, targets=y)
            loss.backward()
            grad_norm_pre = _grad_norm(model)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm_post = _grad_norm(model) if grad_clip > 0 else grad_norm_pre
            optimizer.step()
        else:
            _, loss = model(x, targets=y)
            loss.backward()
            grad_norm_pre = _grad_norm(model)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm_post = _grad_norm(model) if grad_clip > 0 else grad_norm_pre
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val
        total_tokens += batch_tokens
        n_batches += 1

        if batch_idx % log_every == 0:
            grad_norm_sum += grad_norm_post
            grad_norm_pre_clip_sum += grad_norm_pre
            grad_norm_count += 1
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'ppl': f'{loss_to_perplexity(loss_val):.1f}',
                'lr': f'{lr:.2e}',
                'gn': f'{grad_norm_post:.2f}',
                'gn_pre': f'{grad_norm_pre:.2f}',
            })
            if metrics_logger is not None:
                metrics_logger.log_train_step({
                    'train/loss': loss_val,
                    'train/lr': lr,
                    'train/grad_norm': grad_norm_post,
                    'train/grad_norm_pre_clip': grad_norm_pre,
                })
        else:
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})
            if metrics_logger is not None:
                metrics_logger.global_step += 1

        if (
            metrics_logger is not None
            and save_every_steps > 0
            and output_dir
            and checkpoint_config is not None
            and tokenizer_path is not None
        ):
            maybe_save_step_checkpoint(
                metrics_logger.global_step,
                save_every_steps,
                output_dir,
                model,
                optimizer,
                current_epoch,
                total_loss / max(n_batches, 1),
                last_val_loss,
                checkpoint_config,
                tokenizer_path,
            )

    elapsed = max(time.perf_counter() - t0, 1e-6)
    avg_loss = total_loss / max(n_batches, 1)
    return {
        'train_loss': avg_loss,
        'train_perplexity': loss_to_perplexity(avg_loss),
        'train_tokens_per_sec': total_tokens / elapsed,
        'train_batches': n_batches,
        'train_skipped_batches': skipped_batches,
        'train_tokens': total_tokens,
        'avg_grad_norm': grad_norm_sum / max(grad_norm_count, 1) if grad_norm_count else None,
        'avg_grad_norm_pre_clip': grad_norm_pre_clip_sum / max(grad_norm_count, 1) if grad_norm_count else None,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'elapsed_sec': elapsed,
    }


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    max_batches=None,
    collect_moe_stats=False,
):
    """Run validation; return loss, perplexity, token accuracy, and optional MoE stats."""
    if dataloader is None:
        return None

    model.eval()
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    n_batches = 0
    moe_hooks = []
    expert_counts = None

    if collect_moe_stats and getattr(model, 'use_moe', False):
        n_experts = model.blocks[0].feed_forward.n_experts
        top_k = model.blocks[0].feed_forward.top_k
        expert_counts = torch.zeros(n_experts, dtype=torch.float64)

        def make_hook(top_k_local):
            def hook(_module, inputs, output):
                x_flat = inputs[0].reshape(-1, inputs[0].size(-1))
                router_logits = output
                _, topk_indices = torch.topk(router_logits, k=top_k_local, dim=-1)
                for expert_id in range(expert_counts.numel()):
                    expert_counts[expert_id] += (topk_indices == expert_id).sum().item()
            return hook

        for block in model.blocks:
            ff = block.feed_forward
            if isinstance(ff, MoEFeedForward):
                moe_hooks.append(ff.router.register_forward_hook(make_hook(ff.top_k)))

    t0 = time.perf_counter()
    try:
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            valid = y != TextDataset.IGNORE_TARGET
            if not bool(valid.any()):
                continue
            logits, loss = model(x, targets=y)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct_tokens += (preds[valid] == y[valid]).sum().item()
            total_tokens += valid.sum().item()
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break
    finally:
        for handle in moe_hooks:
            handle.remove()

    elapsed = max(time.perf_counter() - t0, 1e-6)
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct_tokens / max(total_tokens, 1)
    metrics = {
        'val_loss': avg_loss,
        'val_perplexity': loss_to_perplexity(avg_loss),
        'val_accuracy': accuracy,
        'val_batches': n_batches,
        'val_tokens': total_tokens,
        'val_tokens_per_sec': total_tokens / elapsed,
        'val_elapsed_sec': elapsed,
    }

    if expert_counts is not None and expert_counts.sum() > 0:
        fractions = (expert_counts / expert_counts.sum()).tolist()
        metrics['moe_expert_fraction'] = fractions
        n_exp = len(fractions)
        uniform = 1.0 / n_exp
        metrics['moe_load_imbalance'] = max(abs(f - uniform) for f in fractions)

    return metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    config,
    tokenizer_path,
    global_step=None,
):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'loss': val_loss if val_loss is not None else train_loss,
        'val_loss': val_loss,
        'config': config,
        'tokenizer_path': tokenizer_path,
    }
    if global_step is not None:
        payload['global_step'] = global_step
    torch.save(payload, path)


def save_best_checkpoint(path, model, epoch, train_loss, val_loss, config, tokenizer_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'loss': val_loss if val_loss is not None else train_loss,
        'val_loss': val_loss,
        'config': config,
        'tokenizer_path': tokenizer_path,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train NanoLLM')
    parser.add_argument('--data', type=str, required=True, help='Path to training data file')
    parser.add_argument(
        '--tokenizer_data',
        type=str,
        nargs='+',
        default=None,
        help='Files used to train a new tokenizer (default: --data only)',
    )
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to independent validation data file (disables --val_split holdout)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--block_size', type=int, default=128, help='Context length')
    parser.add_argument('--vocab_size', type=int, default=500, help='Vocabulary size (BPE)')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n_kv_heads', type=int, default=None,
                        help='Number of key/value heads (default: n_heads; use 1 for MQA)')
    parser.add_argument('--d_ff', type=int, default=128, help='Feed-forward dimension')
    parser.add_argument('--use_moe', action='store_true', help='Enable MoE FFN in transformer blocks')
    parser.add_argument('--moe_n_experts', type=int, default=4, help='Number of routed experts when --use_moe is enabled')
    parser.add_argument('--moe_top_k', type=int, default=1, help='Number of active experts per token when --use_moe is enabled')
    parser.add_argument('--moe_shared_d_ff', type=int, default=0, help='Shared expert FFN hidden size (0 disables shared expert)')
    parser.add_argument('--use_rope', action='store_true', help='Rotary position embeddings (no learned pos table)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--mixed_precision', action='store_true', help='Use fp16 mixed precision (prefer --amp bf16 on modern GPUs)')
    parser.add_argument('--amp', type=str, default='auto', choices=['auto', 'none', 'fp16', 'bf16'],
                        help='Mixed precision mode (auto=bf16 on supported CUDA, else none)')
    parser.add_argument('--tf32', action='store_true', help='Enable TF32 matmul on Ampere+ GPUs')
    parser.add_argument('--cuda_device', type=int, default=None, help='CUDA device index (default: current device)')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument(
        '--save_every_steps',
        type=int,
        default=10000,
        help='Save model_step_{N}.pt snapshot every N optimizer steps (0 disables)',
    )
    parser.add_argument('--init_checkpoint', type=str, default=None, help='Path to checkpoint for weight initialization and config reuse')
    parser.add_argument(
        '--reuse_tokenizer',
        type=str,
        default=None,
        help='Path to tokenizer.json (or tokenizer dir) to reuse without loading model weights',
    )
    parser.add_argument('--token_cache_dir', type=str, default=None, help='Directory for cached token files (default: <output_dir>/token_cache)')
    parser.add_argument('--token_cache_chunk_chars', type=int, default=1_000_000, help='Approximate number of characters to tokenize per chunk when streaming data')
    parser.add_argument('--window_stride_tokens', type=int, default=1,
                        help='Token stride between adjacent training windows (1 preserves full overlap)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers (default: min(16, max(4, cpu_count//2)))')
    parser.add_argument('--prefetch_factor', type=int, default=8,
                        help='DataLoader prefetch batches per worker (when num_workers > 0)')
    parser.add_argument('--fused_adamw', type=str, default='auto', choices=['auto', 'on', 'off'],
                        help='Use fused AdamW on CUDA when available (auto/on/off)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_split', type=float, default=0.05, help='Fraction of data held out for validation (0 disables)')
    parser.add_argument('--eval_every', type=int, default=1, help='Run validation every N epochs')
    parser.add_argument('--eval_max_batches', type=int, default=None, help='Cap validation batches per epoch (default: full val set)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Max gradient norm (0 disables clipping)')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Fraction of total steps used for LR warmup')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--early_stop_patience', type=int, default=0, help='Stop if val loss does not improve for N evals (0 disables)')
    parser.add_argument('--log_every', type=int, default=50, help='Log batch metrics every N training steps')
    parser.add_argument(
        '--tensorboard',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Write TensorBoard scalars to output_dir/tensorboard (default: enabled)',
    )
    parser.add_argument(
        '--chat_eval',
        action='store_true',
        help='Run fixed chat prompt suite after each validation epoch',
    )
    parser.add_argument(
        '--chat_eval_greedy',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use greedy decoding for chat eval (default: enabled)',
    )
    parser.add_argument(
        '--chat_eval_select',
        action='store_true',
        help='Select best checkpoint by deterministic chat eval score instead of val loss',
    )
    parser.add_argument(
        '--chat_eval_require_hard',
        action='store_true',
        help='When chat_eval_select is enabled, only accept checkpoints that pass hard task checks',
    )
    parser.add_argument(
        '--assistant_only_loss',
        action='store_true',
        help='Mask loss to assistant reply tokens only (chat fine-tune)',
    )
    parser.add_argument(
        '--instruct_benchmark',
        action='store_true',
        help='Run instruction-following benchmark (category-level pass/fail) after each validation epoch',
    )
    
    args = parser.parse_args()
    
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError('--val_split must be in [0, 1)')
    if args.warmup_ratio < 0 or args.warmup_ratio > 1:
        raise ValueError('--warmup_ratio must be in [0, 1]')
    if args.prefetch_factor < 2:
        raise ValueError('--prefetch_factor must be >= 2')
    if args.save_every_steps < 0:
        raise ValueError('--save_every_steps must be >= 0')
    if args.window_stride_tokens < 1:
        raise ValueError('--window_stride_tokens must be >= 1')
    
    set_seed(args.seed)
    args.num_workers = default_num_workers(args.num_workers)
    amp_mode = resolve_amp_mode(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = resolve_device(args.cuda_device)
    configure_cuda_runtime(tf32=args.tf32)
    if device.type == 'cuda':
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
        print(
            f"AMP: {amp_mode} | TF32: {args.tf32} | "
            f"DataLoader workers: {args.num_workers} | prefetch: {args.prefetch_factor}"
        )
    else:
        print(f"Using device: {device}")
        amp_mode = 'none'

    if args.token_cache_dir is None:
        args.token_cache_dir = os.path.join(args.output_dir, "token_cache")
    os.makedirs(args.token_cache_dir, exist_ok=True)
    print(f"Token cache directory: {args.token_cache_dir}")
    
    init_checkpoint_data = None
    checkpoint_tokenizer_path = None
    start_epoch = 0

    if args.init_checkpoint:
        if not os.path.isfile(args.init_checkpoint):
            raise FileNotFoundError(f"Init checkpoint not found: {args.init_checkpoint}")
        print(f"Loading initialization checkpoint from {args.init_checkpoint}...")
        init_checkpoint_data = torch.load(args.init_checkpoint, map_location='cpu')
        if not isinstance(init_checkpoint_data, dict):
            raise ValueError("Initialization checkpoint does not contain expected metadata")
        ckpt_config = init_checkpoint_data.get('config') or {}
        args.vocab_size = ckpt_config.get('vocab_size', args.vocab_size)
        args.d_model = ckpt_config.get('d_model', args.d_model)
        args.n_layers = ckpt_config.get('n_layers', args.n_layers)
        args.n_heads = ckpt_config.get('n_heads', args.n_heads)
        args.n_kv_heads = ckpt_config.get('n_kv_heads', args.n_kv_heads)
        args.d_ff = ckpt_config.get('d_ff', args.d_ff)
        args.use_moe = ckpt_config.get('use_moe', args.use_moe)
        args.moe_n_experts = ckpt_config.get('moe_n_experts', args.moe_n_experts)
        args.moe_top_k = ckpt_config.get('moe_top_k', args.moe_top_k)
        args.moe_shared_d_ff = ckpt_config.get('moe_shared_d_ff', args.moe_shared_d_ff)
        args.use_rope = ckpt_config.get('use_rope', args.use_rope)
        args.block_size = ckpt_config.get('max_seq_len', args.block_size)
        args.dropout = ckpt_config.get('dropout', args.dropout)
        start_epoch = init_checkpoint_data.get('epoch', 0)

    # Load or train tokenizer
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")

    if init_checkpoint_data is not None:
        checkpoint_tokenizer_path = init_checkpoint_data.get('tokenizer_path')
        if checkpoint_tokenizer_path and not os.path.isfile(checkpoint_tokenizer_path):
            checkpoint_tokenizer_path = os.path.join(
                os.path.dirname(args.init_checkpoint),
                "tokenizer",
                "tokenizer.json",
            )
        if not checkpoint_tokenizer_path or not os.path.isfile(checkpoint_tokenizer_path):
            raise FileNotFoundError(
                "Initialization checkpoint tokenizer not found. "
                "Fine-tuning must reuse the pretraining tokenizer."
            )

        print(f"Loading checkpoint tokenizer from {checkpoint_tokenizer_path}...")
        tokenizer = BPETokenizer.from_file(checkpoint_tokenizer_path)
        os.makedirs(tokenizer_dir, exist_ok=True)
        if os.path.abspath(checkpoint_tokenizer_path) != os.path.abspath(tokenizer_path):
            shutil.copy2(checkpoint_tokenizer_path, tokenizer_path)
        vocab_info_src = os.path.join(os.path.dirname(checkpoint_tokenizer_path), "vocab_info.json")
        vocab_info_dst = os.path.join(tokenizer_dir, "vocab_info.json")
        if os.path.isfile(vocab_info_src) and os.path.abspath(vocab_info_src) != os.path.abspath(vocab_info_dst):
            shutil.copy2(vocab_info_src, vocab_info_dst)
        args.vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer loaded with vocab_size={args.vocab_size}")
    elif args.reuse_tokenizer:
        reuse_path = args.reuse_tokenizer
        if os.path.isdir(reuse_path):
            reuse_path = os.path.join(reuse_path, "tokenizer.json")
        if not os.path.isfile(reuse_path):
            raise FileNotFoundError(f"reuse_tokenizer not found: {args.reuse_tokenizer}")
        print(f"Reusing tokenizer from {reuse_path}...")
        tokenizer = BPETokenizer.from_file(reuse_path)
        os.makedirs(tokenizer_dir, exist_ok=True)
        if os.path.abspath(reuse_path) != os.path.abspath(tokenizer_path):
            shutil.copy2(reuse_path, tokenizer_path)
        vocab_info_src = os.path.join(os.path.dirname(reuse_path), "vocab_info.json")
        vocab_info_dst = os.path.join(tokenizer_dir, "vocab_info.json")
        if os.path.isfile(vocab_info_src) and os.path.abspath(vocab_info_src) != os.path.abspath(vocab_info_dst):
            shutil.copy2(vocab_info_src, vocab_info_dst)
        args.vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer reused with vocab_size={args.vocab_size}")
    else:
        print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
        tokenizer = train_tokenizer_from_data(
            args.tokenizer_data or args.data,
            vocab_size=args.vocab_size,
            output_dir=tokenizer_dir
        )
        args.vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer trained with vocab_size={args.vocab_size}")

    # Create dataset and dataloaders
    dataset, train_dataset, val_dataset, validation_mode = build_train_val_datasets(args, tokenizer)
    pin_memory = device.type == 'cuda'
    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = make_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            drop_last=False,
        )
        if validation_mode == 'independent_file':
            print(
                f"Train windows: {len(train_dataset):,} | Val windows: {len(val_dataset):,} "
                f"(independent file: {os.path.abspath(args.val_data)})"
            )
        else:
            print(
                f"Train windows: {len(train_dataset):,} | Val windows: {len(val_dataset):,} "
                f"({args.val_split:.0%} holdout, gap={args.block_size} windows)"
            )
    else:
        print(f"Train windows: {len(train_dataset):,} | Validation disabled")
    
    # Create model and show a structural summary when torchinfo is available
    print("Creating model...")
    model = NanoLLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        max_seq_len=args.block_size,
        dropout=args.dropout,
        use_moe=args.use_moe,
        moe_n_experts=args.moe_n_experts,
        moe_top_k=args.moe_top_k,
        moe_shared_d_ff=args.moe_shared_d_ff,
        use_rope=args.use_rope,
    ).to(device)

    if init_checkpoint_data is not None:
        state_dict = init_checkpoint_data.get('model_state_dict')
        if state_dict is None:
            raise KeyError("Initialization checkpoint missing 'model_state_dict'")
        model.load_state_dict(state_dict)
        print("Model weights initialized from checkpoint.")

    try:
        from torchinfo import summary as model_summary
    except ImportError:
        model_summary = None

    if model_summary is not None:
        try:
            summary_stats = model_summary(
                model,
                input_size=(1, args.block_size),
                dtypes=[torch.long],
                col_names=("input_size", "output_size", "num_params"),
                depth=3,
                verbose=0,
            )
            print(summary_stats)
        except Exception as err:
            print(f"Model summary unavailable: {err}")
    else:
        print("Install torchinfo to see a model summary (pip install torchinfo).")
    
    size_mb, n_params = model.get_model_size_mb(quantized=False)
    print(f"Model parameters: {n_params:,}")
    print(f"Model size (FP32): {size_mb:.2f} MB")
    print(f"Model size (int8): {size_mb/4:.2f} MB")
    
    # Optimizer and LR schedule
    optimizer, using_fused_adamw = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        fused_adamw=args.fused_adamw,
    )
    print(f"Optimizer: AdamW (fused={using_fused_adamw})")
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    # Mixed precision scaler (fp16 only; bf16 typically does not need scaling)
    scaler = torch.amp.GradScaler('cuda') if amp_mode == 'fp16' and device.type == 'cuda' else None
    
    initial_global_step = start_epoch * len(train_loader)
    if init_checkpoint_data is not None and init_checkpoint_data.get('global_step') is not None:
        initial_global_step = int(init_checkpoint_data['global_step'])
    metrics_logger = MetricsLogger(
        args.output_dir,
        use_tensorboard=args.tensorboard,
        initial_global_step=initial_global_step,
    )
    run_config = {
        **model_config_dict(args),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'val_split': args.val_split,
        'grad_clip': args.grad_clip,
        'warmup_ratio': args.warmup_ratio,
        'weight_decay': args.weight_decay,
        'amp': amp_mode,
        'tf32': args.tf32,
        'num_workers': args.num_workers,
        'prefetch_factor': args.prefetch_factor,
        'fused_adamw': args.fused_adamw,
        'fused_adamw_enabled': using_fused_adamw,
        'data': os.path.abspath(args.data),
        'val_data': os.path.abspath(args.val_data) if args.val_data else None,
        'validation_mode': validation_mode,
        'dataset_token_count': getattr(dataset, 'token_count', None),
        'val_dataset_token_count': getattr(val_dataset, 'token_count', None) if val_dataset is not None else None,
        'window_stride_tokens': args.window_stride_tokens,
        'assistant_only_loss': args.assistant_only_loss,
        'chat_eval_select': args.chat_eval_select,
        'chat_eval_require_hard': args.chat_eval_require_hard,
        'validation_gap_windows': math.ceil(args.block_size / args.window_stride_tokens) if validation_mode == 'split' and args.val_split > 0.0 else 0,
        'seed': args.seed,
        'start_epoch': start_epoch,
        'save_every_steps': args.save_every_steps,
    }
    metrics_logger.log_run_config(run_config)
    config = model_config_dict(args)

    # Training loop
    print("\nStarting training...")
    init_val_loss = None
    if init_checkpoint_data is not None:
        init_val_loss = init_checkpoint_data.get('val_loss')
        init_train_loss = init_checkpoint_data.get('train_loss', init_checkpoint_data.get('loss'))
    else:
        init_train_loss = None

    best_train_loss = init_train_loss if init_train_loss is not None else float('inf')
    best_val_loss = init_val_loss if init_val_loss is not None else float('inf')
    best_chat_eval_score = float('-inf')
    best_epoch = init_checkpoint_data.get('epoch') if init_checkpoint_data else None
    epochs_without_improvement = 0

    # Validation-loss continuation can legitimately retain the initialization
    # checkpoint for the whole stage. Materialize it locally so downstream
    # stages always receive the actual selected best checkpoint.
    initial_best_path = os.path.join(args.output_dir, 'model_best.pt')
    if init_checkpoint_data is not None and not os.path.isfile(initial_best_path):
        shutil.copy2(args.init_checkpoint, initial_best_path)
        print(f"Initial checkpoint is current best -> {initial_best_path}")
    
    for epoch_idx in range(args.epochs):
        current_epoch = start_epoch + epoch_idx + 1
        total_epochs = start_epoch + args.epochs
        print(f"\nEpoch {current_epoch}/{total_epochs}")

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp_mode=amp_mode,
            scaler=scaler,
            grad_clip=args.grad_clip,
            scheduler=scheduler,
            log_every=args.log_every,
            metrics_logger=metrics_logger,
            current_epoch=current_epoch,
            save_every_steps=args.save_every_steps,
            output_dir=args.output_dir,
            checkpoint_config=config,
            tokenizer_path=tokenizer_path,
            last_val_loss=best_val_loss if best_val_loss != float('inf') else None,
        )

        val_metrics = None
        run_eval = val_loader is not None and (
            (epoch_idx + 1) % args.eval_every == 0 or epoch_idx == args.epochs - 1
        )
        if run_eval:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                max_batches=args.eval_max_batches,
                collect_moe_stats=args.use_moe,
            )

        train_loss = train_metrics['train_loss']
        val_loss = val_metrics['val_loss'] if val_metrics else None
        selection_loss = val_loss if val_loss is not None else train_loss

        print(
            f"Train loss: {train_loss:.4f} (ppl {train_metrics['train_perplexity']:.2f}) | "
            f"tokens/s: {train_metrics['train_tokens_per_sec']:,.0f} | "
            f"lr: {train_metrics['learning_rate']:.2e}"
        )
        if val_metrics:
            moe_note = ""
            if 'moe_expert_fraction' in val_metrics:
                fracs = val_metrics['moe_expert_fraction']
                moe_note = f" | MoE load: {[f'{f:.2f}' for f in fracs]} imbalance={val_metrics['moe_load_imbalance']:.3f}"
            print(
                f"Val   loss: {val_loss:.4f} (ppl {val_metrics['val_perplexity']:.2f}) | "
                f"acc: {val_metrics['val_accuracy']:.2%}{moe_note}"
            )

        chat_eval_results = None
        chat_eval_score = None
        chat_eval_hard_pass = None
        if args.chat_eval and val_metrics is not None:
            from chat_eval import (
                aggregate_chat_eval_score,
                passes_hard_chat_eval,
                print_chat_eval,
                run_chat_eval,
                save_chat_eval,
            )

            chat_eval_results = run_chat_eval(
                model,
                tokenizer,
                device,
                greedy=args.chat_eval_greedy,
            )
            chat_eval_score = aggregate_chat_eval_score(chat_eval_results)
            chat_eval_hard_pass = passes_hard_chat_eval(chat_eval_results)
            print_chat_eval(chat_eval_results, score=chat_eval_score)
            print(f"  [chat_eval] hard_pass={chat_eval_hard_pass}")
            chat_eval_path = os.path.join(args.output_dir, f'chat_eval_epoch_{current_epoch}.json')
            save_chat_eval(chat_eval_results, chat_eval_path, score=chat_eval_score)

        # Optional instruction-following benchmark (category-level pass/fail).
        instruct_bench_path = None
        if args.instruct_benchmark and val_metrics is not None:
            try:
                bench_script = Path(__file__).parent.parent / 'scripts' / 'benchmark_instruction_following.py'
                if bench_script.is_file():
                    instruct_bench_path = os.path.join(
                        args.output_dir, f'instruct_benchmark_epoch_{current_epoch}.json'
                    )
                    result = subprocess.run(
                        [
                            'python', str(bench_script),
                            '--checkpoint', os.path.join(args.output_dir, f'model_epoch_{current_epoch}.pt'),
                            '--tokenizer', os.path.join(args.output_dir, 'tokenizer', 'tokenizer.json'),
                            '--device', str(device),
                            '--output', instruct_bench_path,
                            '--max-new-tokens', '60',
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    if result.returncode == 0:
                        print(f"  [instruct_benchmark] epoch={current_epoch} output={instruct_bench_path}")
                        for line in result.stdout.strip().splitlines():
                            print(f"    {line}")
                    else:
                        print(f"  [instruct_benchmark] epoch={current_epoch} FAILED (rc={result.returncode})")
                        if result.stderr:
                            print(f"    stderr: {result.stderr.strip()[:200]}")
                        instruct_bench_path = None
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                print(f"  [instruct_benchmark] skipped: {exc}")
                instruct_bench_path = None

        epoch_record = {'epoch': current_epoch, **train_metrics}
        if val_metrics:
            epoch_record.update(val_metrics)
        if chat_eval_results is not None:
            epoch_record['chat_eval'] = chat_eval_results
            if chat_eval_score is not None:
                epoch_record['chat_eval_score'] = chat_eval_score
            epoch_record['chat_eval_hard_pass'] = chat_eval_hard_pass
        metrics_logger.log_epoch(epoch_record)

        improved = False
        if args.chat_eval_select and chat_eval_score is not None:
            hard_ok = (not args.chat_eval_require_hard) or chat_eval_hard_pass
            improved = hard_ok and chat_eval_score > best_chat_eval_score
        elif val_metrics is not None:
            improved = val_loss < best_val_loss
        elif val_loader is None:
            improved = train_loss < best_train_loss

        if improved:
            if val_loss is not None and not args.chat_eval_select:
                best_val_loss = val_loss
            if chat_eval_score is not None and args.chat_eval_select:
                best_chat_eval_score = chat_eval_score
            best_train_loss = train_loss
            best_epoch = current_epoch
            epochs_without_improvement = 0
            metrics_logger.set_best(best_epoch, best_train_loss, val_loss)
        elif val_metrics is not None and args.early_stop_patience > 0:
            epochs_without_improvement += 1

        should_save_epoch = (current_epoch % args.save_every == 0) or improved
        if should_save_epoch:
            checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{current_epoch}.pt')
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                current_epoch,
                train_loss,
                val_loss,
                config,
                tokenizer_path,
                global_step=metrics_logger.global_step,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        if improved:
            best_path = os.path.join(args.output_dir, 'model_best.pt')
            save_best_checkpoint(
                best_path,
                model,
                current_epoch,
                train_loss,
                val_loss,
                config,
                tokenizer_path,
            )
            metric_name = 'chat_eval_score' if args.chat_eval_select and chat_eval_score is not None else (
                'val_loss' if val_loss is not None else 'train_loss'
            )
            metric_value = (
                chat_eval_score if metric_name == 'chat_eval_score'
                else selection_loss
            )
            print(f"New best {metric_name}={metric_value:.4f} -> {best_path}")

        if args.early_stop_patience > 0 and val_loss is not None and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"\nEarly stopping: val loss did not improve for {args.early_stop_patience} evaluation(s)."
            )
            break

    final_metrics = {'hparam/train_loss': best_train_loss}
    if best_val_loss < float('inf'):
        final_metrics['hparam/val_loss'] = best_val_loss
        final_metrics['hparam/val_perplexity'] = loss_to_perplexity(best_val_loss)
    metrics_logger.log_hparams(final_metrics)
    metrics_logger.close()
    print("\nTraining complete!")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'model_best.pt')}")
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch} | train_loss={best_train_loss:.4f}", end='')
        if best_val_loss < float('inf'):
            print(f" | val_loss={best_val_loss:.4f} (ppl {loss_to_perplexity(best_val_loss):.2f})")
        else:
            print()
    print(f"Metrics log: {metrics_logger.metrics_path}")
    if args.tensorboard:
        print(f"TensorBoard: tensorboard --logdir {metrics_logger.tensorboard_dir}")

if __name__ == '__main__':
    main()

