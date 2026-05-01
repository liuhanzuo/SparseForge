"""Model Factory: unified model creation interface for multiple HuggingFace architectures.

Supported models:
- LLaMA (LLaMA-2, LLaMA-3, CodeLLaMA, etc.)
- OPT (OPT-125m, OPT-1.3b, OPT-2.7b, OPT-6.7b, etc.)
- GPT-2 (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- Qwen (Qwen2, Qwen2.5 series)
- Mistral (Mistral-7B series)
- DeepSeek MoE (deepseek-moe-16b)
- Hunyuan (tencent/Hunyuan-1.8B-Pretrain, etc.)
"""

from typing import Optional, Type
import torch
import torch.nn as nn
import torch.distributed as dist
import io
import gc


# Lazy imports to avoid import-time errors
def _get_model_class(model_type: str) -> Type[nn.Module]:
    """Return the model class corresponding to the given model type."""
    model_type = model_type.lower()
    
    if model_type == "llama":
        from model_llama import LlamaSparse
        return LlamaSparse
    elif model_type == "opt":
        from model_opt import OPTSparse
        return OPTSparse
    elif model_type == "gpt2":
        from model import GPT
        return GPT
    elif model_type == "qwen":
        from model_qwen import QwenSparse
        return QwenSparse
    elif model_type == "mistral":
        from model_mistral import MistralSparse
        return MistralSparse
    elif model_type == "deepseek_moe":
        from model_deepseek_moe import DeepSeekMoESparse
        return DeepSeekMoESparse
    elif model_type == "hunyuan":
        from model_hunyuan import HunyuanSparse
        return HunyuanSparse
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: {', '.join(SUPPORTED_MODEL_TYPES)}")


def detect_model_type(model_name: str) -> str:
    """Auto-detect model type from model name.

    Args:
        model_name: HuggingFace model name or local path.

    Returns:
        Model type string: "llama", "opt", "gpt2", "qwen", "mistral", "deepseek_moe", "hunyuan".
    """
    model_name_lower = model_name.lower()
    
    # Hunyuan series (Tencent Hunyuan, LLaMA-based with custom wrapper)
    hunyuan_patterns = ['hunyuan']
    for pattern in hunyuan_patterns:
        if pattern in model_name_lower:
            return "hunyuan"
    
    # DeepSeek MoE detection (must come before other DeepSeek models)
    # Note: deepseek-moe uses a special MoE architecture requiring separate handling
    deepseek_moe_patterns = ['deepseek-moe', 'deepseek_moe', 'deepseekmoe']
    for pattern in deepseek_moe_patterns:
        if pattern in model_name_lower:
            return "deepseek_moe"
    
    # Qwen series (Qwen2, Qwen2.5, etc.)
    qwen_patterns = ['qwen', 'qwen2', 'qwen1.5', 'qwen-']
    for pattern in qwen_patterns:
        if pattern in model_name_lower:
            return "qwen"
    
    # Mistral series
    mistral_patterns = ['mistral', 'mixtral']
    for pattern in mistral_patterns:
        if pattern in model_name_lower:
            return "mistral"
    
    # LLaMA series
    llama_patterns = [
        'llama', 'codellama', 'vicuna', 'alpaca', 
        'yi-', 'deepseek',  # DeepSeek dense models (non-MoE) use LLaMA architecture
        'nous', 'meta-llama', 'tinyllama'
    ]
    for pattern in llama_patterns:
        if pattern in model_name_lower:
            return "llama"
    
    # OPT series
    opt_patterns = ['opt-', '/opt', 'facebook/opt']
    for pattern in opt_patterns:
        if pattern in model_name_lower:
            return "opt"
    
    # GPT-2 series
    gpt2_patterns = ['gpt2', 'gpt-2', 'distilgpt2']
    for pattern in gpt2_patterns:
        if pattern in model_name_lower:
            return "gpt2"
    
    # Default to LLaMA (most modern models use LLaMA architecture)
    print(f"[WARNING] Could not auto-detect model type for '{model_name}', defaulting to 'llama'")
    return "llama"


def broadcast_state_dict(state_dict: Optional[dict], src: int = 0) -> dict:
    """Broadcast state_dict from src rank to all other ranks.

    Strategy: per-tensor broadcast to avoid serializing the entire state_dict
    to GPU (which would cause OOM). First broadcasts the key list, then
    broadcasts each tensor individually.

    Args:
        state_dict: Model state_dict on the src rank; other ranks pass None.
        src: Source rank (default 0).

    Returns:
        Complete state_dict available on all ranks.
    """
    import pickle
    
    rank = dist.get_rank()
    
    # Step 1: Broadcast keys and metadata (serialize small data)
    if rank == src:
        keys = list(state_dict.keys())
        # Serialize key list + shape/dtype for each tensor
        meta = [(k, state_dict[k].shape, str(state_dict[k].dtype)) for k in keys]
        meta_bytes = pickle.dumps(meta)
        meta_size = torch.tensor([len(meta_bytes)], dtype=torch.long, device='cuda')
    else:
        meta_size = torch.tensor([0], dtype=torch.long, device='cuda')
    
    dist.broadcast(meta_size, src=src)
    
    if rank == src:
        meta_tensor = torch.frombuffer(bytearray(meta_bytes), dtype=torch.uint8).cuda()
    else:
        meta_tensor = torch.empty(int(meta_size.item()), dtype=torch.uint8, device='cuda')
    
    dist.broadcast(meta_tensor, src=src)
    
    if rank != src:
        meta = pickle.loads(meta_tensor.cpu().numpy().tobytes())
    
    del meta_tensor
    torch.cuda.empty_cache()
    
    # Step 2: Per-tensor broadcast
    # For efficiency, batch small tensors together; large tensors are sent individually
    CHUNK_SIZE = 256 * 1024 * 1024  # 256MB per chunk
    
    result_sd = {} if rank != src else state_dict
    
    # Group tensors: accumulate until chunk_size then transmit together
    current_chunk_keys = []
    current_chunk_size = 0
    chunks = []
    
    for key, shape, dtype_str in meta:
        tensor_bytes = 1
        for s in shape:
            tensor_bytes *= s
        # Estimate byte count based on dtype
        if '16' in dtype_str:
            tensor_bytes *= 2
        elif '32' in dtype_str:
            tensor_bytes *= 4
        elif '64' in dtype_str:
            tensor_bytes *= 8
        
        current_chunk_keys.append((key, shape, dtype_str))
        current_chunk_size += tensor_bytes
        
        if current_chunk_size >= CHUNK_SIZE:
            chunks.append(current_chunk_keys)
            current_chunk_keys = []
            current_chunk_size = 0
    
    if current_chunk_keys:
        chunks.append(current_chunk_keys)
    
    # Transmit chunk by chunk
    for chunk_keys in chunks:
        for key, shape, dtype_str in chunk_keys:
            dtype_map = {
                'torch.float16': torch.float16, 'torch.bfloat16': torch.bfloat16,
                'torch.float32': torch.float32, 'torch.float64': torch.float64,
                'torch.int32': torch.int32, 'torch.int64': torch.int64,
                'torch.int8': torch.int8, 'torch.uint8': torch.uint8,
                'torch.bool': torch.bool,
            }
            dtype = dtype_map.get(dtype_str, torch.float32)
            
            if rank == src:
                tensor = state_dict[key].cuda()
            else:
                tensor = torch.empty(shape, dtype=dtype, device='cuda')
            
            dist.broadcast(tensor, src=src)
            
            if rank != src:
                result_sd[key] = tensor.cpu()
            
            del tensor
        
        torch.cuda.empty_cache()
    
    if rank == src:
        return state_dict
    else:
        return result_sd


def get_sparse_model(
    model_name: str,
    *,
    model_type: Optional[str] = None,
    override_args: Optional[dict] = None,
    sparselinear_config=None,
    is_teacher: bool = False,
) -> nn.Module:
    """Universal model factory: create a model with SparseLinear layers.

    Args:
        model_name: HuggingFace model name or local path.
        model_type: Optional, explicitly specify model type ("llama", "opt", "gpt2",
                   "qwen", "mistral", "deepseek_moe"). Auto-detected if not specified.
        override_args: Config overrides (dropout, gradient_checkpointing, etc.).
        sparselinear_config: SparseLinear config; None means no sparse replacement.
        is_teacher: Whether this is a teacher model (teachers skip sparse layers).

    Returns:
        The constructed model instance.

    Example:
        >>> # Auto-detect model type
        >>> model = get_sparse_model("NousResearch/Llama-2-7b-hf", sparselinear_config=cfg)
        >>>
        >>> # Explicitly specify model type
        >>> model = get_sparse_model("facebook/opt-2.7b", model_type="opt", sparselinear_config=cfg)
        >>>
        >>> # Qwen model
        >>> model = get_sparse_model("Qwen/Qwen2.5-1.5B", sparselinear_config=cfg)
        >>>
        >>> # Mistral model
        >>> model = get_sparse_model("mistralai/Mistral-7B-v0.3", sparselinear_config=cfg)
        >>>
        >>> # DeepSeek MoE model
        >>> model = get_sparse_model("deepseek-ai/deepseek-moe-16b-base", sparselinear_config=cfg)
        >>>
        >>> # Hunyuan model
        >>> model = get_sparse_model("tencent/Hunyuan-1.8B-Pretrain", sparselinear_config=cfg)
    """
    # Auto-detect or use specified model type
    if model_type is None:
        model_type = detect_model_type(model_name)
    
    print(f"[model_factory] Loading model: {model_name} (type: {model_type})")
    
    # Get the corresponding model class
    model_class = _get_model_class(model_type)
    
    # Create model
    model = model_class.from_pretrained(
        model_name,
        override_args=override_args,
        sparselinear_config=sparselinear_config,
        is_teacher=is_teacher,
    )
    
    return model


def create_model_skeleton(
    model_name: str,
    *,
    model_type: Optional[str] = None,
    override_args: Optional[dict] = None,
    sparselinear_config=None,
    is_teacher: bool = False,
) -> nn.Module:
    """
    Create model skeleton (random init) without loading pretrained weights.
    Used for broadcast-loading: rank 0 loads the full model, other ranks only
    need the structure shell, then receive state_dict via broadcast.

    For HF models (LLaMA/OPT/Qwen/Mistral), builds an empty model from config.
    For GPT-2, creates an empty GPT structure (skips HF weight conversion).
    """
    if model_type is None:
        model_type = detect_model_type(model_name)
    
    override_args = override_args or {}
    model_type_lower = model_type.lower()
    
    if model_type_lower == "gpt2":
        # GPT-2: create empty GPT model shell (config only, no HF loading)
        from model import GPT, GPTConfig
        import os, json
        
        canonical = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }
        
        if model_name in canonical:
            config_args = canonical[model_name]
            config_args['vocab_size'] = 50257
            config_args['block_size'] = 1024
            config_args['bias'] = True
        else:
            config_json_path = os.path.join(model_name, 'config.json')
            if os.path.isfile(config_json_path):
                with open(config_json_path, 'r') as f:
                    hf_conf = json.load(f)
                config_args = dict(
                    n_layer=int(hf_conf['n_layer']),
                    n_head=int(hf_conf['n_head']),
                    n_embd=int(hf_conf['n_embd']),
                    vocab_size=int(hf_conf.get('vocab_size', 50257)),
                    block_size=int(hf_conf.get('n_positions', 1024)),
                    bias=True,
                )
            else:
                from transformers import AutoConfig
                hf_conf = AutoConfig.from_pretrained(model_name)
                config_args = dict(
                    n_layer=int(hf_conf.n_layer),
                    n_head=int(hf_conf.n_head),
                    n_embd=int(hf_conf.n_embd),
                    vocab_size=int(getattr(hf_conf, 'vocab_size', 50257)),
                    block_size=int(getattr(hf_conf, 'n_positions', 1024)),
                    bias=True,
                )
        
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        config_args['output_hidden_state'] = override_args.get('output_hidden_state', False)
        config_args['gradient_checkpointing'] = override_args.get('gradient_checkpointing', False)
        config_args['is_teacher'] = is_teacher
        # Only add eager_attention if GPTConfig supports it (for backward compatibility)
        if hasattr(GPTConfig, '__dataclass_fields__') and 'eager_attention' in GPTConfig.__dataclass_fields__:
            config_args['eager_attention'] = override_args.get('eager_attention', False)
        
        config = GPTConfig(**config_args)
        model = GPT(config, sparselinear_config)
        print(f"[model_factory] Created GPT-2 skeleton: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
        return model
    
    elif model_type_lower == "llama":
        from model_llama import LlamaSparse
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(model_name)
        if "dropout" in override_args:
            config.hidden_dropout = override_args["dropout"]
            config.attention_dropout = override_args["dropout"]
        config.output_hidden_states = bool(override_args.get("output_hidden_state", False))
    # Do not enable gradient_checkpointing via config (refactored in transformers 4.50+).
    # Skeleton creation skips gradient_checkpointing; enable it later after loading
    # weights via model.gradient_checkpointing_enable(use_reentrant=False).
        config.use_cache = False
    # attn_implementation follows eager_attention flag to ensure consistent attention impl across ranks
        if override_args.get("eager_attention", False):
            config._attn_implementation = "eager"
        return LlamaSparse(config, sparselinear_config=sparselinear_config, is_teacher=is_teacher)
    
    elif model_type_lower == "opt":
        from model_opt import OPTSparse
        from transformers import OPTConfig
        config = OPTConfig.from_pretrained(model_name)
        if "dropout" in override_args:
            config.dropout = override_args["dropout"]
            config.attention_dropout = override_args["dropout"]
        config.output_hidden_states = bool(override_args.get("output_hidden_state", False))
        config.use_cache = False
    # attn_implementation follows eager_attention flag to ensure consistent attention impl across ranks
        if override_args.get("eager_attention", False):
            config._attn_implementation = "eager"
        return OPTSparse(config, sparselinear_config=sparselinear_config, is_teacher=is_teacher)
    
    elif model_type_lower == "qwen":
        from model_qwen import QwenSparse
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if "dropout" in override_args and hasattr(config, 'attention_dropout'):
            config.attention_dropout = override_args["dropout"]
        config.output_hidden_states = bool(override_args.get("output_hidden_state", False))
        config.use_cache = False
    # attn_implementation follows eager_attention flag to ensure consistent attention impl across ranks
        if override_args.get("eager_attention", False):
            config._attn_implementation = "eager"
        return QwenSparse(config, sparselinear_config=sparselinear_config, is_teacher=is_teacher)
    
    elif model_type_lower == "mistral":
        from model_mistral import MistralSparse
        from transformers import MistralConfig
        config = MistralConfig.from_pretrained(model_name)
        if "dropout" in override_args:
            config.attention_dropout = override_args["dropout"]
        config.output_hidden_states = bool(override_args.get("output_hidden_state", False))
        config.use_cache = False
    # attn_implementation follows eager_attention flag to ensure consistent attention impl across ranks
        if override_args.get("eager_attention", False):
            config._attn_implementation = "eager"
        return MistralSparse(config, sparselinear_config=sparselinear_config, is_teacher=is_teacher)
    
    elif model_type_lower == "deepseek_moe":
    # DeepSeek MoE does not support skeleton creation (model=None errors); fall back to full load
        print(f"[model_factory] WARNING: DeepSeek MoE does not support skeleton creation, falling back to full load")
        return get_sparse_model(
            model_name, model_type=model_type, override_args=override_args,
            sparselinear_config=sparselinear_config, is_teacher=is_teacher,
        )
    
    elif model_type_lower == "hunyuan":
        from model_hunyuan import HunyuanSparse
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if "dropout" in override_args:
            if hasattr(config, 'hidden_dropout'):
                config.hidden_dropout = override_args["dropout"]
            if hasattr(config, 'attention_dropout'):
                config.attention_dropout = override_args["dropout"]
        config.output_hidden_states = bool(override_args.get("output_hidden_state", False))
        config.use_cache = False
    # attn_implementation follows eager_attention flag to ensure consistent attention impl across ranks
        if override_args.get("eager_attention", False):
            config._attn_implementation = "eager"
        return HunyuanSparse(config, sparselinear_config=sparselinear_config, is_teacher=is_teacher)
    
    else:
        raise ValueError(f"Unsupported model type for skeleton: {model_type}")


def get_model_info(model_name: str, model_type: Optional[str] = None) -> dict:
    """Get basic model information without loading weights.

    Returns:
        Dictionary containing model metadata.
    """
    if model_type is None:
        model_type = detect_model_type(model_name)
    
    model_type_lower = model_type.lower()
    
    if model_type_lower == "llama":
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(model_name)
    elif model_type_lower == "opt":
        from transformers import OPTConfig
        config = OPTConfig.from_pretrained(model_name)
    elif model_type_lower == "gpt2":
        from transformers import GPT2Config
        config = GPT2Config.from_pretrained(model_name)
    elif model_type_lower == "qwen":
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    elif model_type_lower == "mistral":
        from transformers import MistralConfig
        config = MistralConfig.from_pretrained(model_name)
    elif model_type_lower == "deepseek_moe":
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    elif model_type_lower == "hunyuan":
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Build return info, handling different config attribute names across models
    info = {
        "model_type": model_type,
        "model_name": model_name,
        "hidden_size": getattr(config, 'hidden_size', getattr(config, 'd_model', None)),
        "num_hidden_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None)),
        "num_attention_heads": getattr(config, 'num_attention_heads', getattr(config, 'n_head', None)),
        "vocab_size": config.vocab_size,
        "max_position_embeddings": getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', None)),
    }
    
    # For MoE models, add extra information
    if model_type_lower == "deepseek_moe":
        info["num_experts"] = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', None))
        info["num_experts_per_tok"] = getattr(config, 'num_experts_per_tok', getattr(config, 'top_k', None))
        info["is_moe"] = True
    else:
        info["is_moe"] = False
    
    return info


# Supported model types
SUPPORTED_MODEL_TYPES = ["llama", "opt", "gpt2", "qwen", "mistral", "deepseek_moe", "hunyuan"]
