"""
Export GPT-2 checkpoint weights to CSV files for simulation or transfer.

Behavior:
- Loads model checkpoint from Adaptive-Sparse-Trainer format (.pt file)
- Supports both standard GPT checkpoint and sparse GPT checkpoint
- Iterates state_dict and writes each tensor to `<save_dir>/<name_with_underscores>.csv`
- Writes an index JSON with parameter shapes and dtypes for quick reference
- If `--only_feature` is set, skip weight export and only write a random token-id feature matrix to `feature.csv`.

Typical usage:
  # Export from checkpoint
  python scripts/export_gpt2_weights.py --checkpoint out_llama/gpt2_xxx/model.pt --save_dir /path/to/export
  
  # Export only specific layers (e.g., transformer blocks)
  python scripts/export_gpt2_weights.py --checkpoint out_llama/gpt2_xxx/model.pt --save_dir /path/to/export --filter "transformer.h"
  
  # Generate feature matrix only
  python scripts/export_gpt2_weights.py --checkpoint out_llama/gpt2_xxx/model.pt --only_feature --save_dir /path/to/export

Note: Exporting large models to CSV can be slow and disk-intensive.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_model_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load checkpoint and extract model state_dict.
    Handles various checkpoint formats:
    - {'model': state_dict, ...}
    - {'model_state_dict': state_dict, ...}
    - {'state_dict': state_dict, ...}
    - Direct state_dict
    """
    print(f"Loading checkpoint from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(ckpt, dict):
        # Try various keys
        for key in ['model', 'model_state_dict', 'state_dict', 'student']:
            if key in ckpt:
                state_dict = ckpt[key]
                print(f"  Found state_dict under key '{key}'")
                
                # Handle nested structure (e.g., student model in Distill_Model)
                if isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']
                    print(f"  Extracted nested 'model' state_dict")
                
                return state_dict
        
        # Check if it's directly a state_dict (keys look like parameter names)
        sample_keys = list(ckpt.keys())[:3]
        if any('weight' in k or 'bias' in k or 'transformer' in k for k in sample_keys):
            print("  Checkpoint appears to be a direct state_dict")
            return ckpt
        
        # Return as-is if nothing else matches
        print(f"  Warning: Unknown checkpoint format. Keys: {list(ckpt.keys())[:10]}")
        return ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")


def filter_state_dict(state_dict: Dict[str, torch.Tensor], 
                      include_patterns: Optional[List[str]] = None,
                      exclude_patterns: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """
    Filter state_dict by patterns.
    """
    filtered = {}
    
    for name, tensor in state_dict.items():
        # Check include patterns
        if include_patterns:
            if not any(p in name for p in include_patterns):
                continue
        
        # Check exclude patterns
        if exclude_patterns:
            if any(p in name for p in exclude_patterns):
                continue
        
        filtered[name] = tensor
    
    return filtered


def export_weights(checkpoint_path: str, 
                   save_dir: str,
                   include_filter: Optional[str] = None,
                   exclude_filter: Optional[str] = None,
                   fmt: str = '%.8f',
                   save_masks: bool = True) -> None:
    """
    Export all weights from checkpoint to CSV files.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        save_dir: Directory to save CSV files
        include_filter: Comma-separated patterns to include (e.g., "transformer.h,lm_head")
        exclude_filter: Comma-separated patterns to exclude (e.g., "mask,optimizer")
        fmt: Format string for numpy savetxt
        save_masks: Whether to save mask tensors (for sparse models)
    """
    ensure_dir(save_dir)
    
    state_dict = get_model_state_dict(checkpoint_path)
    
    # Parse filters
    include_patterns = include_filter.split(',') if include_filter else None
    exclude_patterns = exclude_filter.split(',') if exclude_filter else ['optimizer', 'scaler']
    
    if not save_masks:
        exclude_patterns = exclude_patterns or []
        exclude_patterns.append('mask')
    
    # Filter state_dict
    state_dict = filter_state_dict(state_dict, include_patterns, exclude_patterns)
    
    print(f"\nExporting {len(state_dict)} tensors to {save_dir} ...")
    
    shapes_index: Dict[str, Dict[str, Any]] = {}
    layer_stats: Dict[str, List[str]] = {}  # Group by layer for reporting
    
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            print(f"  Skipping {name} (not a tensor: {type(tensor)})")
            continue
        
        # Convert to numpy
        arr = tensor.detach().cpu().to(torch.float32).numpy()
        
        # Create filename (replace dots with underscores)
        fname = name.replace('.', '_') + '.csv'
        fpath = os.path.join(save_dir, fname)
        
        # Handle different tensor shapes
        if arr.ndim == 0:
            # Scalar
            np.savetxt(fpath, arr.reshape(1), delimiter=',', fmt=fmt)
        elif arr.ndim == 1:
            # 1D vector - save as column
            np.savetxt(fpath, arr.reshape(-1, 1), delimiter=',', fmt=fmt)
        else:
            # 2D or higher - flatten higher dims if needed
            if arr.ndim > 2:
                original_shape = arr.shape
                arr = arr.reshape(arr.shape[0], -1)
                print(f"  Note: {name} reshaped from {original_shape} to {arr.shape}")
            np.savetxt(fpath, arr, delimiter=',', fmt=fmt)
        
        # Record in index
        shapes_index[name] = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'file': fname,
            'size_mb': tensor.numel() * 4 / (1024 * 1024),  # assuming float32
        }
        
        # Group by layer
        parts = name.split('.')
        layer_key = '.'.join(parts[:3]) if len(parts) > 3 else name
        if layer_key not in layer_stats:
            layer_stats[layer_key] = []
        layer_stats[layer_key].append(name)
        
        print(f"  ✓ {name} -> {fname} shape={list(tensor.shape)}")
    
    # Write index JSON
    idx_path = os.path.join(save_dir, 'weights_index.json')
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump(shapes_index, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Wrote index: {idx_path}")
    
    # Write layer summary
    summary_path = os.path.join(save_dir, 'layer_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_params': len(shapes_index),
            'layers': {k: len(v) for k, v in layer_stats.items()},
            'layer_details': layer_stats,
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Wrote layer summary: {summary_path}")
    
    # Calculate total size
    total_size_mb = sum(info['size_mb'] for info in shapes_index.values())
    print(f"\n✓ Total exported: {len(shapes_index)} tensors, {total_size_mb:.2f} MB")


def save_feature_matrix(save_dir: str,
                        batch: int = 4,
                        seq_len: int = 32,
                        vocab_size: int = 50257) -> None:
    """
    Generate a small integer token-id matrix and save to `feature.csv` for simulator input.
    Shape: (batch, seq_len)
    """
    ensure_dir(save_dir)
    
    arr = np.random.randint(0, vocab_size, size=(batch, seq_len), dtype=np.int64)
    out = os.path.join(save_dir, 'feature.csv')
    np.savetxt(out, arr, delimiter=',', fmt='%d')
    print(f"✓ Saved feature matrix to {out} with shape {arr.shape} (vocab_size={vocab_size})")


def export_by_layer(checkpoint_path: str, save_dir: str) -> None:
    """
    Export weights organized by layer directory structure.
    Creates a directory for each transformer block.
    
    Structure:
      save_dir/
        embedding/
          wte_weight.csv
          wpe_weight.csv
        layer_00/
          ln_1_weight.csv
          attn_c_attn_weight.csv
          ...
        layer_01/
          ...
        lm_head/
          weight.csv
    """
    ensure_dir(save_dir)
    
    state_dict = get_model_state_dict(checkpoint_path)
    
    print(f"\nExporting weights by layer to {save_dir} ...")
    
    shapes_index: Dict[str, Dict[str, Any]] = {}
    
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        
        # Skip optimizer states, masks, etc.
        if 'optimizer' in name or 'scaler' in name:
            continue
        
        # Determine directory based on layer
        parts = name.split('.')
        
        if 'transformer.wte' in name or 'transformer.wpe' in name:
            layer_dir = os.path.join(save_dir, 'embedding')
            fname = name.replace('transformer.', '').replace('.', '_') + '.csv'
        elif 'transformer.ln_f' in name:
            layer_dir = os.path.join(save_dir, 'final_layernorm')
            fname = name.replace('transformer.', '').replace('.', '_') + '.csv'
        elif 'transformer.h.' in name:
            # Extract layer number
            layer_num = int(parts[2])
            layer_dir = os.path.join(save_dir, f'layer_{layer_num:02d}')
            # Remove transformer.h.N prefix
            fname = '.'.join(parts[3:]).replace('.', '_') + '.csv'
        elif 'lm_head' in name:
            layer_dir = os.path.join(save_dir, 'lm_head')
            fname = name.replace('lm_head.', '').replace('.', '_') + '.csv'
        else:
            layer_dir = os.path.join(save_dir, 'other')
            fname = name.replace('.', '_') + '.csv'
        
        ensure_dir(layer_dir)
        
        # Convert and save
        arr = tensor.detach().cpu().to(torch.float32).numpy()
        fpath = os.path.join(layer_dir, fname)
        
        if arr.ndim == 0:
            np.savetxt(fpath, arr.reshape(1), delimiter=',', fmt='%.8f')
        elif arr.ndim == 1:
            np.savetxt(fpath, arr.reshape(-1, 1), delimiter=',', fmt='%.8f')
        else:
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            np.savetxt(fpath, arr, delimiter=',', fmt='%.8f')
        
        rel_path = os.path.relpath(fpath, save_dir)
        shapes_index[name] = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'file': rel_path,
        }
        
        print(f"  ✓ {name} -> {rel_path}")
    
    # Write index
    idx_path = os.path.join(save_dir, 'weights_index.json')
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump(shapes_index, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Wrote index: {idx_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export GPT-2 checkpoint weights to CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all weights flat
  python scripts/export_gpt2_weights.py --checkpoint out_llama/gpt2_xxx/model.pt --save_dir /tmp/gpt2_weights
  
  # Export organized by layer
  python scripts/export_gpt2_weights.py --checkpoint out_llama/gpt2_xxx/model.pt --save_dir /tmp/gpt2_weights --by_layer
  
  # Export only transformer blocks
  python scripts/export_gpt2_weights.py --checkpoint out_llama/gpt2_xxx/model.pt --save_dir /tmp/gpt2_weights --filter "transformer.h"
  
  # Generate feature matrix only
  python scripts/export_gpt2_weights.py --only_feature --save_dir /tmp/gpt2_weights
"""
    )
    
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint file (.pt)'
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Directory to save CSV files'
    )
    parser.add_argument(
        '--filter', type=str, default=None, dest='include_filter',
        help='Comma-separated patterns to include (e.g., "transformer.h,lm_head")'
    )
    parser.add_argument(
        '--exclude', type=str, default=None,
        help='Comma-separated patterns to exclude (e.g., "mask,optimizer")'
    )
    parser.add_argument(
        '--by_layer', action='store_true',
        help='Organize output by layer directories'
    )
    parser.add_argument(
        '--no_masks', action='store_true',
        help='Do not export mask tensors (for sparse models)'
    )
    parser.add_argument(
        '--fmt', type=str, default='%.8f',
        help='Format string for numpy savetxt (default: %%.8f)'
    )
    
    # Feature options
    parser.add_argument(
        '--only_feature', action='store_true',
        help='Only generate feature.csv and skip weight export'
    )
    parser.add_argument(
        '--save_feature', action='store_true',
        help='Also generate a feature.csv with random token ids'
    )
    parser.add_argument(
        '--feature_batch', type=int, default=4,
        help='Feature batch size'
    )
    parser.add_argument(
        '--feature_seq_len', type=int, default=32,
        help='Feature sequence length'
    )
    parser.add_argument(
        '--vocab_size', type=int, default=50257,
        help='Vocabulary size for GPT-2 (default: 50257)'
    )
    
    args = parser.parse_args()
    
    # Only generate feature.csv and exit
    if args.only_feature:
        save_feature_matrix(
            args.save_dir,
            batch=args.feature_batch,
            seq_len=args.feature_seq_len,
            vocab_size=args.vocab_size
        )
        return
    
    # Require checkpoint for weight export
    if args.checkpoint is None:
        parser.error("--checkpoint is required for weight export")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Export weights
    if args.by_layer:
        export_by_layer(args.checkpoint, args.save_dir)
    else:
        export_weights(
            args.checkpoint,
            args.save_dir,
            include_filter=args.include_filter,
            exclude_filter=args.exclude,
            fmt=args.fmt,
            save_masks=not args.no_masks
        )
    
    # Optionally save feature matrix
    if args.save_feature:
        save_feature_matrix(
            args.save_dir,
            batch=args.feature_batch,
            seq_len=args.feature_seq_len,
            vocab_size=args.vocab_size
        )
    
    print("\n" + "=" * 60)
    print("✓ Export complete!")
    print(f"  Output directory: {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
