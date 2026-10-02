"""
Export PyTorch model weights to C++ binary format.
Quantizes weights to int8 for deployment.
"""
import argparse
import torch
import numpy as np
import struct
import json
import os
from model import NanoLLM, MoEFeedForward
from export_tokenizer import export_tokenizer_for_cpp
from quantize import quantize_weights

DENSE_MAGIC = b"NLMD"
DENSE_FORMAT_VERSION = 2
MOE_FORMAT_VERSION = 2


def _resolved_n_kv_heads(config):
    return int(config.get('n_kv_heads') or config['n_heads'])


def _write_versioned_header(output_file, magic, version, quantize, n_layers, n_kv_heads):
    output_file.write(struct.pack('<4sIB3xII', magic, version, int(quantize), n_layers, n_kv_heads))


def export_layer_weights(layer_name, layer, output_file, quantize=True):
    """Export a layer's weights to binary file."""
    weights_written = 0
    
    if isinstance(layer, torch.nn.Linear):
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        if quantize:
            weight_q, scale = quantize_weights(weight)
            # Write scale factor
            output_file.write(struct.pack('f', scale))
            # Write weight shape
            output_file.write(struct.pack('II', weight_q.shape[0], weight_q.shape[1]))
            # Write quantized weights
            output_file.write(weight_q.tobytes())
            weights_written += weight_q.size + 1  # +1 for scale
        else:
            # Write weight shape
            output_file.write(struct.pack('II', weight.shape[0], weight.shape[1]))
            # Write weights as float32
            output_file.write(weight.cpu().numpy().astype(np.float32).tobytes())
            weights_written += weight.numel()
        
        # Write bias if present
        if bias is not None:
            if quantize:
                bias_q, bias_scale = quantize_weights(bias)
                output_file.write(struct.pack('f', bias_scale))
                output_file.write(struct.pack('I', bias_q.size))
                output_file.write(bias_q.tobytes())
                weights_written += bias_q.size + 1
            else:
                output_file.write(struct.pack('I', bias.numel()))
                output_file.write(bias.cpu().numpy().astype(np.float32).tobytes())
                weights_written += bias.numel()
        else:
            # No bias
            output_file.write(struct.pack('I', 0))
    
    elif isinstance(layer, torch.nn.Embedding):
        weight = layer.weight.data
        
        if quantize:
            weight_q, scale = quantize_weights(weight)
            output_file.write(struct.pack('f', scale))
            output_file.write(struct.pack('II', weight_q.shape[0], weight_q.shape[1]))
            output_file.write(weight_q.tobytes())
            weights_written += weight_q.size + 1
        else:
            output_file.write(struct.pack('II', weight.shape[0], weight.shape[1]))
            output_file.write(weight.cpu().numpy().astype(np.float32).tobytes())
            weights_written += weight.numel()
    
    elif isinstance(layer, torch.nn.LayerNorm):
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        if quantize:
            weight_q, scale = quantize_weights(weight)
            output_file.write(struct.pack('f', scale))
            output_file.write(struct.pack('I', weight_q.size))
            output_file.write(weight_q.tobytes())
            weights_written += weight_q.size + 1
        else:
            output_file.write(struct.pack('I', weight.numel()))
            output_file.write(weight.cpu().numpy().astype(np.float32).tobytes())
            weights_written += weight.numel()
        
        # Write bias
        if bias is not None:
            if quantize:
                bias_q, bias_scale = quantize_weights(bias)
                output_file.write(struct.pack('f', bias_scale))
                output_file.write(struct.pack('I', bias_q.size))
                output_file.write(bias_q.tobytes())
                weights_written += bias_q.size + 1
            else:
                output_file.write(struct.pack('I', bias.numel()))
                output_file.write(bias.cpu().numpy().astype(np.float32).tobytes())
                weights_written += bias.numel()
        else:
            # No bias - write zero size
            output_file.write(struct.pack('f', 1.0))  # Dummy scale
            output_file.write(struct.pack('I', 0))
    
    return weights_written


def lm_head_is_tied(model) -> bool:
    """Return True when lm_head shares storage with token embedding."""
    return model.lm_head.weight.data.data_ptr() == model.token_embedding.weight.data.data_ptr()


def export_tied_lm_head_stub(output_file):
    """Write an lm_head record that aliases token_embedding at load time."""
    output_file.write(struct.pack('f', 1.0))
    output_file.write(struct.pack('II', 0, 0))
    output_file.write(struct.pack('I', 0))


def export_lm_head_weights(model, output_file, quantize=True):
    if lm_head_is_tied(model):
        print("  Exporting lm_head (tied to token_embedding)...")
        export_tied_lm_head_stub(output_file)
        return
    print("  Exporting lm_head...")
    export_layer_weights('lm_head', model.lm_head, output_file, quantize)


def _write_moe_linear(linear, output_file, quantize=True):
    """Write one MoE linear layer in a compact record layout."""
    if quantize:
        weight_q, w_scale = quantize_weights(linear.weight.data)
        output_file.write(struct.pack('f', w_scale))
        output_file.write(struct.pack('II', weight_q.shape[0], weight_q.shape[1]))
        output_file.write(weight_q.tobytes())

        if linear.bias is not None:
            bias_q, b_scale = quantize_weights(linear.bias.data)
            output_file.write(struct.pack('f', b_scale))
            output_file.write(struct.pack('I', bias_q.size))
            output_file.write(bias_q.tobytes())
        else:
            output_file.write(struct.pack('f', 1.0))
            output_file.write(struct.pack('I', 0))
    else:
        w = linear.weight.data.cpu().numpy().astype(np.float32)
        output_file.write(struct.pack('II', w.shape[0], w.shape[1]))
        output_file.write(w.tobytes())

        if linear.bias is not None:
            b = linear.bias.data.cpu().numpy().astype(np.float32)
            output_file.write(struct.pack('I', b.size))
            output_file.write(b.tobytes())
        else:
            output_file.write(struct.pack('I', 0))


def export_moe_model(checkpoint_path, output_path, quantize=True):
    """
    Export an experimental MoE-only binary format.

    This output is not yet consumed by current C++/ESP32 runtime code.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    use_rope = bool(config.get('use_rope', False))

    if not config.get('use_moe', False):
        raise ValueError("Checkpoint is not MoE-enabled. Use standard export for dense models.")

    model = NanoLLM(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    config_path = output_path.replace('.bin', '_config.json')
    n_kv_heads = _resolved_n_kv_heads(config)
    format_version = MOE_FORMAT_VERSION if n_kv_heads != config['n_heads'] else 1
    with open(config_path, 'w') as f:
        json.dump({
            **config,
            'n_kv_heads': n_kv_heads,
            'quantized': quantize,
            'format': f'nanollm_moe_v{format_version}',
            'format_version': format_version,
            'runtime_support': 'python_only_experimental'
        }, f, indent=2)
    print(f"Saved config: {config_path}")

    spec_path = output_path.replace('.bin', '_format.json')
    with open(spec_path, 'w') as f:
        json.dump({
            'magic': 'NLMO',
            'version': format_version,
            'quantized': quantize,
            'records': [
                'header',
                'token_embedding',
                'pos_embedding',
                'blocks(attn,norm,router,experts,shared)',
                'final_norm',
                'lm_head'
            ],
            'note': 'Experimental MoE format; not yet supported in cpp/esp32 loaders'
        }, f, indent=2)
    print(f"Saved format spec: {spec_path}")

    tokenizer_path = checkpoint.get('tokenizer_path')
    if tokenizer_path and os.path.exists(tokenizer_path):
        print("\nExporting tokenizer for C++...")
        export_tokenizer_for_cpp(tokenizer_path, os.path.dirname(output_path))

    print(f"Exporting experimental MoE weights to: {output_path}")
    with open(output_path, 'wb') as f:
        if format_version == 1:
            # Preserve byte-for-byte NLMO v1 header semantics for MHA artifacts.
            f.write(b'NLMO')
            f.write(struct.pack('I?I', 1, quantize, config['n_layers']))
        else:
            _write_versioned_header(f, b'NLMO', format_version, quantize,
                                    config['n_layers'], n_kv_heads)

        export_layer_weights('token_embedding', model.token_embedding, f, quantize)
        if not use_rope:
            export_layer_weights('pos_embedding', model.pos_embedding, f, quantize)

        for i, block in enumerate(model.blocks):
            print(f"  Exporting block {i}...")
            export_layer_weights(f'block_{i}_attn_q', block.attention.w_q, f, quantize)
            export_layer_weights(f'block_{i}_attn_k', block.attention.w_k, f, quantize)
            export_layer_weights(f'block_{i}_attn_v', block.attention.w_v, f, quantize)
            export_layer_weights(f'block_{i}_attn_o', block.attention.w_o, f, quantize)
            export_layer_weights(f'block_{i}_norm1', block.norm1, f, quantize)
            export_layer_weights(f'block_{i}_norm2', block.norm2, f, quantize)

            ff = block.feed_forward
            if not isinstance(ff, MoEFeedForward):
                raise TypeError(f"Expected MoEFeedForward in block {i}, got {type(ff).__name__}")

            # MoE metadata
            has_shared = ff.shared_expert is not None
            f.write(struct.pack('III?', ff.n_experts, ff.top_k, ff.experts[0].linear1.out_features, has_shared))

            # Router (d_model -> n_experts)
            _write_moe_linear(ff.router, f, quantize)

            # Routed experts
            for expert_idx, expert in enumerate(ff.experts):
                print(f"    expert {expert_idx}")
                _write_moe_linear(expert.linear1, f, quantize)
                _write_moe_linear(expert.linear2, f, quantize)

            # Shared expert
            if has_shared:
                _write_moe_linear(ff.shared_expert.linear1, f, quantize)
                _write_moe_linear(ff.shared_expert.linear2, f, quantize)

        export_layer_weights('norm', model.norm, f, quantize)
        export_lm_head_weights(model, f, quantize)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("\nExport complete!")
    print(f"Output file size: {file_size_mb:.2f} MB")
    print(f"Experimental MoE weights saved to: {output_path}")


def export_model(checkpoint_path, output_path, quantize=True, allow_moe_export=False):
    """Export entire model to binary format."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint.get('config', {})
    if config.get('use_moe', False):
        if not allow_moe_export:
            raise NotImplementedError(
                "MoE checkpoint export for current C++/ESP32 runtime is not implemented yet. "
                "Re-run with --allow-moe-export to produce an experimental Python-only MoE artifact."
            )
        return export_moe_model(checkpoint_path, output_path, quantize=quantize)
    use_rope = bool(config.get('use_rope', False))
    model = NanoLLM(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model config: {config}")
    size_mb, n_params = model.get_model_size_mb(quantized=quantize)
    print(f"Model size ({'int8' if quantize else 'FP32'}): {size_mb:.2f} MB")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    n_kv_heads = _resolved_n_kv_heads(config)
    format_version = DENSE_FORMAT_VERSION if n_kv_heads != config['n_heads'] else 1
    # Write config JSON
    config_path = output_path.replace('.bin', '_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            **config,
            'n_kv_heads': n_kv_heads,
            'quantized': quantize,
            'format': 'nanollm_dense_legacy' if format_version == 1 else 'nanollm_dense_v2',
            'format_version': format_version,
            'n_params': n_params,
            'lm_head_tied': lm_head_is_tied(model),
        }, f, indent=2)
    print(f"Saved config: {config_path}")
    
    # Export tokenizer if available
    tokenizer_path = checkpoint.get('tokenizer_path')
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"\nExporting tokenizer for C++...")
        from export_tokenizer import export_tokenizer_for_cpp
        export_tokenizer_for_cpp(tokenizer_path, os.path.dirname(output_path))
    
    # Export weights
    print(f"Exporting weights to: {output_path}")
    with open(output_path, 'wb') as f:
        if format_version == 1:
            # Keep existing MHA artifacts readable by older runtimes.
            f.write(struct.pack('?I', quantize, config['n_layers']))
        else:
            _write_versioned_header(f, DENSE_MAGIC, format_version, quantize,
                                    config['n_layers'], n_kv_heads)
        
        # Export token embedding
        print("  Exporting token_embedding...")
        export_layer_weights('token_embedding', model.token_embedding, f, quantize)
        
        if not use_rope:
            print("  Exporting pos_embedding...")
            export_layer_weights('pos_embedding', model.pos_embedding, f, quantize)
        
        # Export transformer blocks
        for i, block in enumerate(model.blocks):
            print(f"  Exporting block {i}...")
            
            # Attention layers
            export_layer_weights(f'block_{i}_attn_q', block.attention.w_q, f, quantize)
            export_layer_weights(f'block_{i}_attn_k', block.attention.w_k, f, quantize)
            export_layer_weights(f'block_{i}_attn_v', block.attention.w_v, f, quantize)
            export_layer_weights(f'block_{i}_attn_o', block.attention.w_o, f, quantize)
            
            # Layer norms
            export_layer_weights(f'block_{i}_norm1', block.norm1, f, quantize)
            export_layer_weights(f'block_{i}_norm2', block.norm2, f, quantize)
            
            # Feed forward
            export_layer_weights(f'block_{i}_ff1', block.feed_forward.linear1, f, quantize)
            export_layer_weights(f'block_{i}_ff2', block.feed_forward.linear2, f, quantize)
        
        # Export final norm and output
        print("  Exporting final norm...")
        export_layer_weights('norm', model.norm, f, quantize)
        export_lm_head_weights(model, f, quantize)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nExport complete!")
    print(f"Output file size: {file_size_mb:.2f} MB")
    print(f"Weights saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to C++ format')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='../weights/model.bin', help='Output binary file path')
    parser.add_argument('--no-quantize', action='store_true', help='Export as FP32 (larger file)')
    parser.add_argument('--allow-moe-export', action='store_true', help='Allow experimental MoE export format (not supported by cpp/esp32 runtime yet)')
    
    args = parser.parse_args()
    
    export_model(
        args.checkpoint,
        args.output,
        quantize=not args.no_quantize,
        allow_moe_export=args.allow_moe_export,
    )


if __name__ == '__main__':
    main()

