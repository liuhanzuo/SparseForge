"""
Model Factory: 统一的模型创建接口，支持多种 HuggingFace 模型架构。
支持的模型：
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


# 延迟导入，避免导入时错误
def _get_model_class(model_type: str) -> Type[nn.Module]:
    """根据模型类型返回对应的模型类。"""
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
    """
    根据模型名称自动检测模型类型。
    
    Args:
        model_name: HuggingFace 模型名称或本地路径
        
    Returns:
        模型类型字符串: "llama", "opt", "gpt2", "qwen", "mistral", "deepseek_moe", "hunyuan"
    """
    model_name_lower = model_name.lower()
    
    # Hunyuan 系列检测（腾讯混元，基于 LLaMA 架构但有独立 wrapper）
    hunyuan_patterns = ['hunyuan']
    for pattern in hunyuan_patterns:
        if pattern in model_name_lower:
            return "hunyuan"
    
    # DeepSeek MoE 检测（需要在其他 DeepSeek 模型之前检测）
    # 注意：deepseek-moe 使用特殊的 MoE 架构，需要单独处理
    deepseek_moe_patterns = ['deepseek-moe', 'deepseek_moe', 'deepseekmoe']
    for pattern in deepseek_moe_patterns:
        if pattern in model_name_lower:
            return "deepseek_moe"
    
    # Qwen 系列检测（Qwen2, Qwen2.5 等）
    qwen_patterns = ['qwen', 'qwen2', 'qwen1.5', 'qwen-']
    for pattern in qwen_patterns:
        if pattern in model_name_lower:
            return "qwen"
    
    # Mistral 系列检测
    mistral_patterns = ['mistral', 'mixtral']
    for pattern in mistral_patterns:
        if pattern in model_name_lower:
            return "mistral"
    
    # LLaMA 系列检测
    llama_patterns = [
        'llama', 'codellama', 'vicuna', 'alpaca', 
        'yi-', 'deepseek',  # DeepSeek dense models (non-MoE) use LLaMA architecture
        'nous', 'meta-llama', 'tinyllama'
    ]
    for pattern in llama_patterns:
        if pattern in model_name_lower:
            return "llama"
    
    # OPT 系列检测
    opt_patterns = ['opt-', '/opt', 'facebook/opt']
    for pattern in opt_patterns:
        if pattern in model_name_lower:
            return "opt"
    
    # GPT-2 系列检测
    gpt2_patterns = ['gpt2', 'gpt-2', 'distilgpt2']
    for pattern in gpt2_patterns:
        if pattern in model_name_lower:
            return "gpt2"
    
    # 默认尝试 LLaMA（大多数现代模型是 LLaMA 架构）
    print(f"[WARNING] Could not auto-detect model type for '{model_name}', defaulting to 'llama'")
    return "llama"


def broadcast_state_dict(state_dict: Optional[dict], src: int = 0) -> dict:
    """
    将 state_dict 从 src rank broadcast 到所有其他 rank。
    
    策略：逐 tensor broadcast，避免序列化整个 state_dict 到 GPU 导致 OOM。
    先 broadcast keys 列表，然后逐个 tensor broadcast。
    
    Args:
        state_dict: src rank 上的模型 state_dict，其他 rank 传 None
        src: 源 rank（默认 0）
    
    Returns:
        所有 rank 上都有的完整 state_dict
    """
    import pickle
    
    rank = dist.get_rank()
    
    # Step 1: Broadcast keys 和 metadata（使用序列化传输小数据）
    if rank == src:
        keys = list(state_dict.keys())
        # 序列化 key 列表 + 每个 tensor 的 shape/dtype
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
    
    # Step 2: 逐个 tensor broadcast
    # 为了效率，将小 tensor 打包一起传输，大 tensor 单独传输
    CHUNK_SIZE = 256 * 1024 * 1024  # 256MB per chunk
    
    result_sd = {} if rank != src else state_dict
    
    # 将 tensors 分组：累积到 chunk_size 后一起传输
    current_chunk_keys = []
    current_chunk_size = 0
    chunks = []
    
    for key, shape, dtype_str in meta:
        tensor_bytes = 1
        for s in shape:
            tensor_bytes *= s
        # 根据 dtype 估算字节数
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
    
    # 逐 chunk 传输
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
    """
    通用模型工厂函数：根据模型名称或类型创建带 SparseLinear 的模型。
    
    Args:
        model_name: HuggingFace 模型名称或本地路径
        model_type: 可选，显式指定模型类型 ("llama", "opt", "gpt2", "qwen", "mistral", "deepseek_moe")
                   如果不指定，会自动检测
        override_args: 配置覆盖参数 (dropout, gradient_checkpointing, etc.)
        sparselinear_config: SparseLinear 配置，为 None 时不替换为稀疏层
        is_teacher: 是否为 teacher 模型（teacher 不使用稀疏层）
        
    Returns:
        包装好的模型实例
        
    Example:
        >>> # 自动检测模型类型
        >>> model = get_sparse_model("NousResearch/Llama-2-7b-hf", sparselinear_config=cfg)
        >>> 
        >>> # 显式指定模型类型
        >>> model = get_sparse_model("facebook/opt-2.7b", model_type="opt", sparselinear_config=cfg)
        >>>
        >>> # Qwen 模型
        >>> model = get_sparse_model("Qwen/Qwen2.5-1.5B", sparselinear_config=cfg)
        >>>
        >>> # Mistral 模型
        >>> model = get_sparse_model("mistralai/Mistral-7B-v0.3", sparselinear_config=cfg)
        >>>
        >>> # DeepSeek MoE 模型
        >>> model = get_sparse_model("deepseek-ai/deepseek-moe-16b-base", sparselinear_config=cfg)
        >>>
        >>> # Hunyuan 模型
        >>> model = get_sparse_model("tencent/Hunyuan-1.8B-Pretrain", sparselinear_config=cfg)
    """
    # 自动检测或使用指定的模型类型
    if model_type is None:
        model_type = detect_model_type(model_name)
    
    print(f"[model_factory] Loading model: {model_name} (type: {model_type})")
    
    # 获取对应的模型类
    model_class = _get_model_class(model_type)
    
    # 创建模型
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
    只创建模型结构（随机初始化），不从磁盘加载预训练权重。
    用于 broadcast 加载场景：rank 0 加载完整模型，其他 rank 只需要结构壳，
    然后通过 broadcast 接收 state_dict 恢复权重。
    
    对于 HF 模型（LLaMA/OPT/Qwen/Mistral），从 config 直接构建空模型。
    对于 GPT-2，创建空 GPT 结构（跳过 HF 权重转换）。
    """
    if model_type is None:
        model_type = detect_model_type(model_name)
    
    override_args = override_args or {}
    model_type_lower = model_type.lower()
    
    if model_type_lower == "gpt2":
        # GPT-2: 创建空壳 GPT 模型（只需要 config，不从 HF 加载）
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
        # 不再通过 config.gradient_checkpointing 启用（新版 transformers 4.50+ 已重构）。
        # skeleton 创建时不设置 gradient_checkpointing，等后续加载完权重后
        # 通过 model.gradient_checkpointing_enable(use_reentrant=False) 启用。
        config.use_cache = False
        # attn_implementation 跟随 eager_attention 参数，保证所有 rank 的 attention 实现一致
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
        # attn_implementation 跟随 eager_attention 参数，保证所有 rank 的 attention 实现一致
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
        # attn_implementation 跟随 eager_attention 参数，保证所有 rank 的 attention 实现一致
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
        # attn_implementation 跟随 eager_attention 参数，保证所有 rank 的 attention 实现一致
        if override_args.get("eager_attention", False):
            config._attn_implementation = "eager"
        return MistralSparse(config, sparselinear_config=sparselinear_config, is_teacher=is_teacher)
    
    elif model_type_lower == "deepseek_moe":
        # DeepSeek MoE 不支持空壳创建（model=None 会出错），回退到完整加载
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
        # attn_implementation 跟随 eager_attention 参数，保证所有 rank 的 attention 实现一致
        if override_args.get("eager_attention", False):
            config._attn_implementation = "eager"
        return HunyuanSparse(config, sparselinear_config=sparselinear_config, is_teacher=is_teacher)
    
    else:
        raise ValueError(f"Unsupported model type for skeleton: {model_type}")


def get_model_info(model_name: str, model_type: Optional[str] = None) -> dict:
    """
    获取模型的基本信息（不加载权重）。
    
    Returns:
        包含模型信息的字典
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
    
    # 构建返回信息，处理不同模型配置属性名差异
    info = {
        "model_type": model_type,
        "model_name": model_name,
        "hidden_size": getattr(config, 'hidden_size', getattr(config, 'd_model', None)),
        "num_hidden_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None)),
        "num_attention_heads": getattr(config, 'num_attention_heads', getattr(config, 'n_head', None)),
        "vocab_size": config.vocab_size,
        "max_position_embeddings": getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', None)),
    }
    
    # 对于 MoE 模型，添加额外信息
    if model_type_lower == "deepseek_moe":
        info["num_experts"] = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', None))
        info["num_experts_per_tok"] = getattr(config, 'num_experts_per_tok', getattr(config, 'top_k', None))
        info["is_moe"] = True
    else:
        info["is_moe"] = False
    
    return info


# 支持的模型类型列表
SUPPORTED_MODEL_TYPES = ["llama", "opt", "gpt2", "qwen", "mistral", "deepseek_moe", "hunyuan"]
