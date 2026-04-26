# sparse_modeling.py
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional
import wandb


class STE(torch.autograd.Function):
    """Forward: weight * mask; Backward: pass grad to dense weight, no grad to mask."""
    @staticmethod
    def forward(ctx, weight, mask):
        ctx.save_for_backward(weight)
        ctx.mask = mask
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output):
        # AST-style: always pass grad to dense weights (NOT multiplied by mask)
        weight, = ctx.saved_tensors
        return grad_output, None


class SRSTE(torch.autograd.Function):
    """SR-STE: adds decay term for dense weights in backward."""
    @staticmethod
    def forward(ctx, weight, mask, decay):
        ctx.save_for_backward(weight)
        ctx.mask = mask
        ctx.decay = decay
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        # ctx.mask is a tensor of same shape as weight; apply decay term only where mask < 0.5
        mask_lt = (ctx.mask < 0.5).to(dtype=weight.dtype)
        extra = ctx.decay * weight * mask_lt
        return grad_output + extra, None, None


class FusedMaskedLinearSTE(torch.autograd.Function):
    """Fused masked linear with STE-style weight gradient.

    Key property: does NOT materialize and save `masked_weight` for backward.

    Semantics:
    - Forward uses `w = weight * mask`.
    - Backward uses STE for weight: grad_weight = grad_out^T @ x (no mask).
    - Optionally adds SRSTE decay term: + decay * weight * (mask < 0.5).
    - No grad for mask.
    
    FSDP Note: mask is now a frozen Parameter (requires_grad=False) that gets
    sharded together with weight, so shape should always match.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, mask, decay: float, scale_factor: float = 1.0):
        # Cast mask to weight dtype to avoid fp16/bf16 -> fp32 promotion.
        mask_cast = mask.to(dtype=weight.dtype)
        
        # Compute output using masked weight, but do not save masked_weight.
        # Weight Scaling: 当 scale_factor > 1.0 时，对被 mask 保留的权重做缩放补偿，
        # 弥补因稀疏化丢失的信息（CAST 论文核心思想之一）。
        w = weight * mask_cast
        if scale_factor != 1.0:
            w = w * scale_factor
        
        x_2d = x.reshape(-1, x.shape[-1])
        out_2d = x_2d.matmul(w.t())
        if bias is not None:
            out_2d = out_2d + bias
        out = out_2d.view(*x.shape[:-1], w.shape[0])

        ctx.save_for_backward(x, weight, mask_cast)
        ctx.has_bias = bias is not None
        ctx.decay = float(decay)
        
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, mask_cast = ctx.saved_tensors
        decay = float(getattr(ctx, "decay", 0.0))

        # CRITICAL: Ensure all tensors have matching dtype for matmul operations.
        # grad_out may be bfloat16/float16 from autocast, weight may be bfloat16,
        # but mask_cast was saved from forward and may have different dtype.
        compute_dtype = grad_out.dtype
        
        grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        
        # Cast weight and mask to the same dtype as grad_out for consistent matmul.
        weight_cast = weight.to(dtype=compute_dtype) if weight.dtype != compute_dtype else weight
        mask_for_grad = mask_cast.to(dtype=compute_dtype) if mask_cast.dtype != compute_dtype else mask_cast

        # Recompute masked weight for grad_x (do not save it).
        w = weight_cast * mask_for_grad
        grad_x_2d = grad_out_2d.matmul(w)
        grad_x = grad_x_2d.view_as(x)

        # STE: weight gradient does NOT apply mask.
        # Cast x_2d to match grad_out dtype if needed.
        x_2d_cast = x_2d.to(dtype=compute_dtype) if x_2d.dtype != compute_dtype else x_2d
        grad_weight = grad_out_2d.t().matmul(x_2d_cast)

        # Optional SRSTE decay term (applies where mask < 0.5).
        if decay != 0.0:
            mask_lt = (mask_for_grad < 0.5).to(dtype=compute_dtype)
            grad_weight = grad_weight + (decay * weight_cast * mask_lt)

        grad_bias = None
        if getattr(ctx, "has_bias", False):
            grad_bias = grad_out_2d.sum(dim=0)

        # No grad for mask; no grad for decay; no grad for scale_factor.
        return grad_x, grad_weight, grad_bias, None, None, None



@dataclass
class SparseLinearConfig:
    # behavior toggles
    change_mask: bool = True
    mode: str = "sparse_forward"   # ["sparse_forward", "dense_forward"]
    mask_type: str = "structured"  # ["structured", "unstructured", "none"]

    # importance metric used to build score
    # ["magnitude", "movement", "hessian_obd", "hessian_ratio", "wanda"]
    mask_metric: str = "hessian_obd"

    # pruning target
    sparsity_ratio: float = 0.5
    N: int = 2
    M: int = 4

    # EMA for grad/hessian and score
    beta: float = 0.99
    score_ema_beta: float = 0.99

    # mask update dynamics (no-grad, continuous)
    mask_update_period: int = 10     # update mask every K steps
    # optional piecewise schedule for mask_update_period
    # if mask_update_switch_step>0 and both before/after are provided, use:
    #   step < switch: period_before
    #   step >= switch: period_after
    mask_update_switch_step: int = 0
    mask_update_period_before: Optional[int] = None
    mask_update_period_after: Optional[int] = None
    mask_lr: float = 0.10            # EMA step: mask <- (1-a)mask + a*G
    # mask penalty learning rate: scale applied to (1-2m) gradient term when using mid_penalty
    mask_penalty_lr: float = None
    # how to apply mid-penalty inside no-grad mask updates
    # - "mid": elementwise push away from 0.5 using (1-2m)
    # - "structured_topn": within each M-group, push top-N members toward 1 and others toward 0
    mask_penalty_mode: str = "mid"
    
    # mask initialization: by default initialize to all ones
    # (previously supported Gaussian-centered sampling; removed)

    # annealed soft gating temperature (lower => harder 0/1)
    temp_init: float = 1.0
    temp_min: float = 0.05
    temp_decay: float = 0.97         # applied per mask update

    # multiplicative binarization decay: elements <0.5 multiply by (1 - mask_binarize_decay),
    # elements >0.5 multiply by (1 + mask_binarize_decay) each mask update to push masks toward 0/1
    mask_binarize_decay: float = 0
    # soft->hard hardening schedule: start step and duration (steps to anneal x from 1->0)
    mask_hardening_start: int = 1000
    mask_hardening_duration: int = 10000

    # warmup before enabling sparsity (mask updates)
    sparsity_warmup_steps: int = 100

    # quantile threshold estimation for unstructured
    tau_sample_size: int = 262144

    # movement scaling
    temperature: float = 1.0

    # optional hard-freeze near extremes
    # 若你要“始终连续”，保持 freeze_low=0.0, freeze_high=1.0 即可（不会触发任何二值化）
    freeze_low: float = 0.0
    freeze_high: float = 1.0

    # SLoRB
    SLoRB: bool = False
    SLoRB_k: int = 64
    SLoRB_init_type: str = "mean"    # ["mean","sum","xavier"]
    trainable_projection: bool = False

    # If True, enforce exact top-N per M-group (useful for 2:4 pruning)
    structured_exact: bool = False

    # Hard-mask projection used during forward hardening/finalization.
    # "match" follows mask_type; "structured" enforces exact top-N per M-group;
    # "unstructured" uses threshold at 0.5.
    hard_mask_type: str = "match"

    # β structural mixing schedule: gradually mix group-level (N:M) structure into
    # the global soft gate G.  G_final = (1-β)*G_global + β*target_nm.
    # β rises from 0 to 1 between [beta_structural_start, beta_structural_end].
    # Uses smooth-step (cubic hermite) to avoid gradient discontinuities.
    beta_structural_start: int = 0      # step where β starts rising from 0
    beta_structural_end: int = 0        # step where β reaches 1.0  (0 = disabled)

    gradient_checkpointing: bool = False
    srste_decay: float = 0.0
    
    # GLU Joint Mask: For SwiGLU architectures (LLaMA, Qwen, Mistral), gate_proj and up_proj
    # are element-wise multiplied in the forward pass. Independent 2:4 masks on each can cause
    # misalignment: if gate[j] is pruned but up[j] is kept, the computation is wasted.
    # When glu_joint_mask=True, we compute a JOINT mask based on combined importance scores
    # and apply the SAME mask to both gate_proj and up_proj, ensuring aligned pruning.
    # This significantly improves lm_eval performance on GLU models (e.g., -8% -> -2% drop).
    glu_joint_mask: bool = False

    # Weight Scaling (inspired by CAST paper):
    # 当 mask 趋近 binary 时，被保留的权重需要补偿被剪枝权重的能量损失。
    # 对于 N:M 稀疏，scaling factor = M/N（例如 2:4 → scale=2.0）。
    # 对于 unstructured sparsity，scaling factor = 1/(1-sparsity_ratio)。
    # 这在训练中逐步引入（跟随 hardening_x 退火），让模型平滑适应 scaled 输出。
    weight_scaling: bool = False


class SparseLinear(nn.Linear):
    """
    Continuous-mask pruning with no-grad mask dynamics:
    - mask is a buffer in [0,1] (NOT sharded by FSDP to avoid shape mismatch)
    - importance_ema tracks long-term importance (bigger => keep)
    - periodic soft gate update + temperature annealing pushes mask toward 0/1
    
    FSDP Optimization:
    - mask is a REGISTERED BUFFER (not Parameter), so it's NOT sharded by FSDP.
      This keeps mask full-sized across all ranks (14GB total), which is acceptable
      for 8x80GB A100. After each mask update, we sync mask across ranks with all_reduce.
    - Other weight-aligned tensors (grad_ema, hessian_diag, frozen_mask_flags) are
      registered as frozen nn.Parameter (requires_grad=False) so they ARE sharded
      by FSDP for memory efficiency.
    - This hybrid approach avoids shape mismatch (sharded weight vs sharded other tensors)
      while keeping memory usage reasonable.
    """
    def __init__(self, in_features, out_features, sparselinear_config=None, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        assert sparselinear_config is not None
        cfg = sparselinear_config
        self.cfg = cfg

        self.mode = cfg.mode
        self.change_mask = cfg.change_mask
        self.mask_type = cfg.mask_type
        self.mask_metric = cfg.mask_metric

        self.sparsity_ratio = cfg.sparsity_ratio
        self.N = cfg.N
        self.M = cfg.M
        
        # mask: Register as FROZEN PARAMETER (requires_grad=False) so it's sharded by FSDP.
        # CRITICAL CHANGE: Previous version used buffer which is NOT sharded, causing OOM
        # on smaller GPUs (e.g., L20A 48GB with 8B model). Now mask is a Parameter (frozen)
        # which FSDP will shard across ranks, reducing per-GPU memory from 14GB to ~0.5GB.
        # 
        # IMPORTANT: mask is used in FusedMaskedLinearSTE.forward() and update_mask().
        # Being a Parameter (not buffer) means:
        # 1. FSDP will shard it -> saves memory
        # 2. Must use FSDP.summon_full_params() when reading full mask for eval/checkpoint
        # 3. .data access still works for in-place updates
        dtype = self.weight.dtype
        self.mask = nn.Parameter(
            torch.ones(out_features, in_features, device=self.weight.device, dtype=dtype),
            requires_grad=False  # Frozen: no gradients computed for mask
        )
        
        # init_mask: DELETED to save 14GB/rank (only used for init_flip logging, not critical)
        # If you really need it, compute init_flip once and discard, or use uint8 frozen param
        
        # Pruning state tracking - USE FROZEN PARAMETERS (requires_grad=False) for FSDP sharding
        # KEY: FSDP doesn't shard buffers but DOES shard parameters. By using frozen params,
        # these large tensors get sharded across ranks instead of replicated.
        
        # grad_ema: ONLY if mask_metric == "movement" (NOT for hessian metrics)
        # FSDP-CRITICAL: Always use frozen nn.Parameter (not buffer) for weight-aligned tensors.
        # When not needed, use a small placeholder Parameter to avoid FSDP sharding mismatch.
        need_grad_ema = cfg.change_mask and (cfg.mask_metric == "movement")
        if need_grad_ema:
            self.grad_ema = nn.Parameter(
                torch.zeros(out_features, in_features, device=self.weight.device, dtype=dtype),
                requires_grad=False
            )
        else:
            # Use a scalar placeholder Parameter instead of buffer to maintain consistent module structure
            # across all SparseLinear instances. FSDP handles Parameters uniformly.
            self.grad_ema = nn.Parameter(
                torch.zeros(1, device=self.weight.device, dtype=dtype),
                requires_grad=False
            )
            self._grad_ema_placeholder = True
        
        # hessian_diag: ONLY if hessian-based metric OR Hutchinson enabled
        # FSDP-CRITICAL: Keep as frozen nn.Parameter (NOT buffer) so it's sharded by FSDP.
        # Only mask is a buffer (not sharded) to avoid shape mismatch. Other weight-aligned
        # tensors (grad_ema, hessian_diag) remain frozen parameters for memory efficiency.
        need_hessian = cfg.change_mask and (
            cfg.mask_metric in ["hessian_obd", "hessian_ratio", "hessian"] 
            or getattr(cfg, "enable_hutchinson", False)
        )
        if need_hessian:
            self.hessian_diag = nn.Parameter(
                torch.zeros(out_features, in_features, device=self.weight.device, dtype=dtype),
                requires_grad=False
            )
        else:
            # Use a scalar placeholder Parameter to maintain consistent module structure.
            self.hessian_diag = nn.Parameter(
                torch.zeros(1, device=self.weight.device, dtype=dtype),
                requires_grad=False
            )
            self._hessian_diag_placeholder = True
        
        # frozen_mask_flags: ONLY if there's actual freeze region
        # FSDP-CRITICAL: Always use frozen nn.Parameter for consistent sharding behavior.
        # NOTE: Must use same dtype as weight (float) because FSDP requires uniform dtype for flattening.
        need_frozen = cfg.change_mask and ((cfg.freeze_low > 0.0) or (cfg.freeze_high < 1.0))
        if need_frozen:
            self.frozen_mask_flags = nn.Parameter(
                torch.zeros(out_features, in_features, device=self.weight.device, dtype=dtype),
                requires_grad=False
            )
        else:
            # Use a scalar placeholder Parameter.
            self.frozen_mask_flags = nn.Parameter(
                torch.zeros(1, device=self.weight.device, dtype=dtype),
                requires_grad=False
            )
            self._frozen_mask_flags_placeholder = True

        # WANDA calibration scaler
        # FSDP-CRITICAL: Use frozen Parameter instead of buffer.
        # Note: scaler_row is (in_features,) not (out_features, in_features), so it's 1D.
        # Still use Parameter for consistency, but mark as placeholder when not needed.
        if cfg.mask_metric == "wanda":
            self.scaler_row = nn.Parameter(
                torch.zeros(in_features, device=self.weight.device, dtype=dtype),
                requires_grad=False
            )
        else:
            self.scaler_row = nn.Parameter(
                torch.zeros(1, device=self.weight.device, dtype=dtype),
                requires_grad=False
            )
            self._scaler_row_placeholder = True
        
        self.nsamples = 0

        # compatibility for existing code that reads p.mask
        self.weight.mask = self.mask
        # init_mask deleted to save memory; init_flipped_mask set to 0
        self.weight.flipped_mask = 0
        self.weight.init_flipped_mask = 0
        self.weight.param_count = self.weight.numel()
        self.srste_decay = cfg.srste_decay

        # penalty learning rate: use mask_lr if not explicitly provided
        self.mask_penalty_lr = float(cfg.mask_penalty_lr) if getattr(cfg, 'mask_penalty_lr', None) is not None else float(cfg.mask_lr)

        # how we compute the hard view used in forward hardening/finalization
        self.hard_mask_type = str(getattr(cfg, 'hard_mask_type', 'match') or 'match')

        # hardening state: x in [1,0] where 1 means fully soft, 0 means fully hard
        self.hardening_x = 1.0

        # SLoRB
        self.SLoRB = cfg.SLoRB
        if self.SLoRB:
            self.SLoRB_k = cfg.SLoRB_k
            self.SLoRB_init_type = cfg.SLoRB_init_type
            self.trainable_projection = cfg.trainable_projection
            # IMPORTANT: Do NOT lazily create Parameters under FSDP.
            # Register parameters here (before wrapping) and only initialize data later.
            k = int(self.SLoRB_k)
            assert k > 0, f"SLoRB_k must be > 0 (got {k})"
            assert in_features % k == 0, f"in_features ({in_features}) must be divisible by SLoRB_k ({k})."
            rows = in_features // k
            cols = in_features
            self.x_proj = nn.Parameter(
                torch.zeros(rows, cols, device=self.weight.device, dtype=self.weight.dtype),
                requires_grad=bool(self.trainable_projection)
            )
            self.SLoRB_Weight = nn.Parameter(
                torch.zeros((out_features, rows), device=self.weight.device, dtype=self.weight.dtype),
                requires_grad=True
            )

    @torch.no_grad()
    def initialize(self):
        # mask is now a buffer (not Parameter), but fill_() still works
        self.mask.fill_(1.0)
        # init_mask deleted
        # Check placeholder flag instead of numel() since placeholder parameters have numel()=1
        if hasattr(self, 'frozen_mask_flags') and isinstance(self.frozen_mask_flags, nn.Parameter):
            if not getattr(self, '_frozen_mask_flags_placeholder', False):
                self.frozen_mask_flags.zero_()
    
        if hasattr(self, 'grad_ema') and isinstance(self.grad_ema, nn.Parameter):
            if not getattr(self, '_grad_ema_placeholder', False):
                self.grad_ema.zero_()
    
        # ----Initialize hessian with warm start if using hessian-based metric----
        # (importance_ema removed - scores computed on-the-fly from weight/grad_ema/hessian_diag)
        if self.mask_metric == "hessian_obd":
            # Warm start: H≈1 to avoid score collapse from zero Hessian
            if hasattr(self, 'hessian_diag') and isinstance(self.hessian_diag, nn.Parameter):
                if not getattr(self, '_hessian_diag_placeholder', False):
                    self.hessian_diag.fill_(1.0)
        else:
            if hasattr(self, 'hessian_diag') and isinstance(self.hessian_diag, nn.Parameter):
                if not getattr(self, '_hessian_diag_placeholder', False):
                    self.hessian_diag.zero_()

        # compatibility pointers (excluding init_mask which was deleted)
        self.weight.mask = self.mask
        self.weight.flipped_mask = 0
        self.weight.init_flipped_mask = 0

    @torch.no_grad()
    def sync_weight(self):
        # Ensure pointers remain valid after .to(device)
        self.weight.mask = self.mask
        # init_mask deleted
        # restore auxiliary logging attributes that may be lost when params move devices
        try:
            self.weight.flipped_mask = int(getattr(self.weight, 'flipped_mask', 0))
        except Exception:
            self.weight.flipped_mask = 0
        try:
            self.weight.init_flipped_mask = int(getattr(self.weight, 'init_flipped_mask', 0))
        except Exception:
            self.weight.init_flipped_mask = 0
        try:
            self.weight.param_count = int(getattr(self.weight, 'param_count', self.weight.numel()))
        except Exception:
            self.weight.param_count = self.weight.numel()

    @torch.no_grad()
    def _blockify_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """将 2D tensor 按 block 取均值，使同一 block 内所有元素相同。
        
        仅在 hard_mask_type 为 block_sparse16 或 block_sparse32 时生效。
        对于 1D tensor 或非 block_sparse 模式，直接返回原始 tensor。
        
        这确保了 soft mask 在训练过程中始终维护 block 粒度的一致性，
        避免 block 内部元素的 mask 值分化导致 hardening 时的剧烈跳变。
        """
        hard_type = str(getattr(self, 'hard_mask_type', 'match') or 'match')
        if hard_type == 'block_sparse16':
            bs = 16
        elif hard_type == 'block_sparse32':
            bs = 32
        else:
            return tensor
        
        if tensor.dim() != 2:
            return tensor
        
        out_dim, in_dim = tensor.shape
        out_full = (out_dim // bs) * bs
        in_full = (in_dim // bs) * bs
        
        if out_full == 0 or in_full == 0:
            return tensor
        
        # 提取 core 区域（能被 block size 整除的部分）
        core = tensor[:out_full, :in_full].float()
        ob = out_full // bs
        ib = in_full // bs
        
        # reshape to (ob, bs, ib, bs) -> 计算每个 block 的均值
        tiles = core.view(ob, bs, ib, bs)
        # block_mean: (ob, ib)
        block_mean = tiles.mean(dim=(1, 3))
        
        # 将 block 均值广播回原始形状
        # (ob, ib) -> (ob, 1, ib, 1) -> (ob, bs, ib, bs)
        expanded = block_mean.unsqueeze(1).unsqueeze(3).expand(ob, bs, ib, bs)
        # reshape back to (out_full, in_full)
        core_blockified = expanded.contiguous().view(out_full, in_full)
        
        # 就地更新（避免创建新 tensor，兼容 nn.Parameter）
        result = tensor.clone()
        result[:out_full, :in_full] = core_blockified.to(dtype=tensor.dtype)
        return result

    @torch.no_grad()
    def _hard_mask_from_soft(self, soft_mask: torch.Tensor) -> torch.Tensor:
        """Project a soft mask in [0,1] to a hard mask.

        - unstructured: threshold at 0.5
        - structured (N:M): per group of size M, keep exactly top-N entries

        This makes the hard mask consistent with the *ordering* of the soft mask,
        which reduces soft/hard mismatch during annealing.
        """
        hard_type = str(getattr(self, 'hard_mask_type', 'match') or 'match')
        mask_type = self.mask_type if hard_type == 'match' else hard_type

        if mask_type == "none":
            return torch.ones_like(soft_mask)

        if mask_type == "unstructured":
            return (soft_mask > 0.5).to(dtype=soft_mask.dtype)

        # 如果 mask 是 1D 的（例如 bias），按 unstructured 处理
        if soft_mask.dim() == 1:
            return (soft_mask > 0.5).to(dtype=soft_mask.dtype)

        # Hopper-style logical 16x16 tile sparsity:
        # Within EACH 16x16 tile, keep a fixed fraction of elements (top-k by soft_mask),
        # e.g. with sparsity_ratio=0.5 we keep 128/256 elements per tile.
        # This is a *logical* projection only (training may still use dense matmul with an expanded mask).
        if mask_type == "block16":
            bs = 16
            if soft_mask.dim() != 2:
                # 非 2D mask 按 unstructured 处理
                return (soft_mask > 0.5).to(dtype=soft_mask.dtype)
            out_dim, in_dim = soft_mask.shape
            out_full = (out_dim // bs) * bs
            in_full = (in_dim // bs) * bs

            # Default: keep everything; only apply tile pruning to the fully covered core.
            hard = torch.ones_like(soft_mask, dtype=soft_mask.dtype)
            if out_full == 0 or in_full == 0:
                return hard

            core = soft_mask.detach().float()[:out_full, :in_full]
            ob = out_full // bs
            ib = in_full // bs
            # (ob, bs, ib, bs) -> (ob, ib, bs, bs)
            tiles = core.view(ob, bs, ib, bs).permute(0, 2, 1, 3).contiguous()
            flat_tiles = tiles.reshape(ob, ib, bs * bs)  # (ob, ib, 256)

            # Keep top-k elements within each tile.
            sr = float(getattr(self, 'sparsity_ratio', 0.0))
            keep_k = int(round((1.0 - sr) * (bs * bs)))
            keep_k = max(0, min(bs * bs, keep_k))

            if keep_k == 0:
                tile_mask = torch.zeros_like(flat_tiles, dtype=soft_mask.dtype)
            elif keep_k == bs * bs:
                tile_mask = torch.ones_like(flat_tiles, dtype=soft_mask.dtype)
            else:
                topi = torch.topk(flat_tiles, k=keep_k, dim=-1, largest=True).indices
                tile_mask = torch.zeros_like(flat_tiles, dtype=soft_mask.dtype)
                tile_mask.scatter_(-1, topi, 1.0)

            expanded = tile_mask.view(ob, ib, bs, bs).permute(0, 2, 1, 3).contiguous().view(out_full, in_full)
            hard[:out_full, :in_full] = expanded
            return hard

        # 2:4 Structured Sparsity: 在每4个元素的group中保留top-2
        # 和 block16 操作方式类似，但 group size = 4, keep = 2
        if mask_type == "nm_2_4":
            N, M = 2, 4  # 每4个元素保留2个
            if soft_mask.dim() != 2:
                # 非 2D mask 按 unstructured 处理
                return (soft_mask > 0.5).to(dtype=soft_mask.dtype)
            out_dim, in_dim = soft_mask.shape
            in_full = (in_dim // M) * M

            # Default: keep everything; only apply N:M pruning to the fully covered columns.
            hard = torch.ones_like(soft_mask, dtype=soft_mask.dtype)
            if in_full == 0:
                return hard

            core = soft_mask.detach().float()[:, :in_full]
            groups = in_full // M
            # reshape to (out_dim, groups, M)
            grouped = core.view(out_dim, groups, M)

            # 在每个group内选择top-N个元素
            topi = torch.topk(grouped, k=N, dim=-1, largest=True).indices
            group_mask = torch.zeros_like(grouped, dtype=soft_mask.dtype)
            group_mask.scatter_(-1, topi, 1.0)

            # reshape back to (out_dim, in_full)
            hard[:, :in_full] = group_mask.view(out_dim, in_full)
            return hard

        # block_sparse32: 将一半的32x32 block设为全0（pruned），另一半为全dense
        # 通过计算每个block内soft mask的均值来决定哪些block保留
        if mask_type == "block_sparse32":
            bs = 32
            if soft_mask.dim() != 2:
                # 非 2D mask 按 unstructured 处理
                return (soft_mask > 0.5).to(dtype=soft_mask.dtype)
            out_dim, in_dim = soft_mask.shape
            out_full = (out_dim // bs) * bs
            in_full = (in_dim // bs) * bs

            # Default: keep everything; only apply block pruning to the fully covered core.
            hard = torch.ones_like(soft_mask, dtype=soft_mask.dtype)
            if out_full == 0 or in_full == 0:
                return hard

            core = soft_mask.detach().float()[:out_full, :in_full]
            ob = out_full // bs  # number of blocks in output dimension
            ib = in_full // bs   # number of blocks in input dimension
            # reshape to (ob, bs, ib, bs) -> (ob, ib, bs, bs)
            tiles = core.view(ob, bs, ib, bs).permute(0, 2, 1, 3).contiguous()
            flat_tiles = tiles.reshape(ob, ib, bs * bs)  # (ob, ib, 1024)

            # 计算每个block的均值（使用平方平均 RMS 来更好地区分重要性）
            # block_scores shape: (ob, ib)
            block_scores = (flat_tiles ** 2).mean(dim=-1).sqrt()  # RMS

            # 目标稀疏度：50% 的 blocks 被 pruned（设为0）
            sr = float(getattr(self, 'sparsity_ratio', 0.5))
            total_blocks = ob * ib
            keep_blocks = int(round((1.0 - sr) * total_blocks))
            keep_blocks = max(0, min(total_blocks, keep_blocks))

            # 展平 block_scores 选择 top-k 个 blocks 保留
            flat_scores = block_scores.view(-1)  # (ob * ib,)
            
            if keep_blocks == 0:
                # 所有 blocks 都 pruned
                block_keep_mask = torch.zeros(ob, ib, dtype=torch.bool, device=soft_mask.device)
            elif keep_blocks == total_blocks:
                # 所有 blocks 都保留
                block_keep_mask = torch.ones(ob, ib, dtype=torch.bool, device=soft_mask.device)
            else:
                # 选择 top-k 个 blocks
                _, top_indices = torch.topk(flat_scores, k=keep_blocks, largest=True)
                block_keep_mask = torch.zeros(total_blocks, dtype=torch.bool, device=soft_mask.device)
                block_keep_mask[top_indices] = True
                block_keep_mask = block_keep_mask.view(ob, ib)

            # 将 block-level mask 扩展回原始大小
            # block_keep_mask: (ob, ib) -> (ob, 1, ib, 1) -> broadcast to (ob, bs, ib, bs)
            tile_mask = block_keep_mask.unsqueeze(1).unsqueeze(3).expand(ob, bs, ib, bs)
            # reshape back: (ob, bs, ib, bs) -> (ob * bs, ib * bs) = (out_full, in_full)
            expanded = tile_mask.permute(0, 1, 2, 3).contiguous().view(out_full, in_full)
            hard[:out_full, :in_full] = expanded.to(dtype=soft_mask.dtype)
            return hard

        # block_sparse16: 将部分的16x16 block设为全0（pruned），其余为全dense
        # 和 block_sparse32 逻辑完全一致，只是 block size 为 16
        # 这是真正的 block sparsity，每个 block 整体保留或丢弃，可以获得硬件加速
        if mask_type == "block_sparse16":
            bs = 16
            if soft_mask.dim() != 2:
                # 非 2D mask 按 unstructured 处理
                return (soft_mask > 0.5).to(dtype=soft_mask.dtype)
            out_dim, in_dim = soft_mask.shape
            out_full = (out_dim // bs) * bs
            in_full = (in_dim // bs) * bs

            # Default: keep everything; only apply block pruning to the fully covered core.
            hard = torch.ones_like(soft_mask, dtype=soft_mask.dtype)
            if out_full == 0 or in_full == 0:
                return hard

            core = soft_mask.detach().float()[:out_full, :in_full]
            ob = out_full // bs  # number of blocks in output dimension
            ib = in_full // bs   # number of blocks in input dimension
            # reshape to (ob, bs, ib, bs) -> (ob, ib, bs, bs)
            tiles = core.view(ob, bs, ib, bs).permute(0, 2, 1, 3).contiguous()
            flat_tiles = tiles.reshape(ob, ib, bs * bs)  # (ob, ib, 256)

            # 计算每个block的均值（使用平方平均 RMS 来更好地区分重要性）
            # block_scores shape: (ob, ib)
            block_scores = (flat_tiles ** 2).mean(dim=-1).sqrt()  # RMS

            # 目标稀疏度：sr% 的 blocks 被 pruned（设为0）
            sr = float(getattr(self, 'sparsity_ratio', 0.5))
            total_blocks = ob * ib
            keep_blocks = int(round((1.0 - sr) * total_blocks))
            keep_blocks = max(0, min(total_blocks, keep_blocks))

            # 展平 block_scores 选择 top-k 个 blocks 保留
            flat_scores = block_scores.view(-1)  # (ob * ib,)
            
            if keep_blocks == 0:
                # 所有 blocks 都 pruned
                block_keep_mask = torch.zeros(ob, ib, dtype=torch.bool, device=soft_mask.device)
            elif keep_blocks == total_blocks:
                # 所有 blocks 都保留
                block_keep_mask = torch.ones(ob, ib, dtype=torch.bool, device=soft_mask.device)
            else:
                # 选择 top-k 个 blocks
                _, top_indices = torch.topk(flat_scores, k=keep_blocks, largest=True)
                block_keep_mask = torch.zeros(total_blocks, dtype=torch.bool, device=soft_mask.device)
                block_keep_mask[top_indices] = True
                block_keep_mask = block_keep_mask.view(ob, ib)

            # 将 block-level mask 扩展回原始大小
            # block_keep_mask: (ob, ib) -> (ob, 1, ib, 1) -> broadcast to (ob, bs, ib, bs)
            tile_mask = block_keep_mask.unsqueeze(1).unsqueeze(3).expand(ob, bs, ib, bs)
            # reshape back: (ob, bs, ib, bs) -> (ob * bs, ib * bs) = (out_full, in_full)
            expanded = tile_mask.permute(0, 1, 2, 3).contiguous().view(out_full, in_full)
            hard[:out_full, :in_full] = expanded.to(dtype=soft_mask.dtype)
            return hard

        if mask_type != "structured":
            raise ValueError(f"Invalid hard mask_type: {mask_type}")

        N, M = int(self.N), int(self.M)
        # 非 2D mask 按 unstructured 处理
        if soft_mask.dim() != 2:
            return (soft_mask > 0.5).to(dtype=soft_mask.dtype)
        out_dim, in_dim = soft_mask.shape
        if in_dim % M != 0:
            raise ValueError(f"in_features ({in_dim}) must be divisible by M ({M}).")
        groups = in_dim // M
        sm = soft_mask.detach().float().view(out_dim, groups, M)
        topi = torch.topk(sm, k=N, dim=-1, largest=True).indices
        hard_g = torch.zeros_like(sm, dtype=soft_mask.dtype)
        hard_g.scatter_(-1, topi, 1.0)
        return hard_g.view(out_dim, in_dim)

    def forward(self, x):
        if self.mode == "dense_forward" or self.mask_type == "none":
            out = F.linear(x, self.weight, self.bias)
        elif self.mode == "sparse_forward":
            # masked forward, dense backward (STE/SRSTE)
            # Note: torch.autograd.Function.apply does not accept keyword args — pass decay positionally
            # Build effective mask: convex combination of soft mask and hard mask
            # hard_mask is binary view of current soft mask
            try:
                hx = float(getattr(self, 'hardening_x', 1.0))
            except Exception:
                hx = 1.0
            
            # Check if mask is already finalized (hard binary mask from checkpoint)
            # In this case, skip _hard_mask_from_soft to avoid:
            # 1. FSDP sharding issues (topk on sharded tensors)
            # 2. Recomputing hard mask which may differ slightly due to tie-breaking
            is_finalized = bool(getattr(self, '_hardening_finalized', False))
            
            # FSDP SAFETY: When hardening_x == 1.0, skip _hard_mask_from_soft entirely.
            # This avoids calling topk() on sharded mask tensors which can cause rank divergence.
            # _hard_mask_from_soft uses topk for block16/structured modes, which assumes full mask shape.
            if hx >= 1.0 - 1e-6:
                # Pure soft mask mode - no hard mask needed
                effective_mask = self.mask
            elif is_finalized:
                # Mask is already hardened (from finalized checkpoint or harden_masks call).
                # Use it directly without recomputing hard mask.
                # This is critical for retrain: the mask is already binary and weights
                # have been multiplied by mask during finalization.
                effective_mask = self.mask
            else:
                # Hardening active: compute convex combination of soft and hard masks
                hard_mask = self._hard_mask_from_soft(self.mask)
                effective_mask = (hx * self.mask) + ((1.0 - hx) * hard_mask)
            
            # Fused masked linear: avoids saving a per-layer `masked_weight` tensor for backward.
            # NOTE: backward 路径的硬阈值衰减(mask<0.5)与 soft mask 不兼容，已禁用。
            # 衰减统一由 AdamW 路径处理，使用连续的 (1-mask) 权重，见 adamw.py。
            
            # Weight Scaling（CAST 论文）：计算缩放因子，补偿因稀疏化丢失的能量。
            # 缩放因子跟随 hardening_x 退火，从 1.0 渐进到目标值。
            scale_factor = 1.0
            if getattr(self.cfg, 'weight_scaling', False) and hx < 1.0 - 1e-6:
                # 计算目标缩放因子
                target_scale = self._compute_weight_scale_factor()
                # 跟随 hardening_x 退火：hx=1.0 时 scale=1.0，hx=0.0 时 scale=target
                scale_factor = 1.0 + (1.0 - hx) * (target_scale - 1.0)
            
            out = FusedMaskedLinearSTE.apply(x, self.weight, self.bias, effective_mask, 0.0, scale_factor)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # SLoRB add-on
        if self.SLoRB and hasattr(self, "x_proj"):
            lora_out = F.linear(x, self.x_proj, bias=None)
            lora_out = F.linear(lora_out, self.SLoRB_Weight, bias=None)
            out = out + lora_out

        return out

    # ---- Weight Scaling ----
    def _compute_weight_scale_factor(self) -> float:
        """计算 weight scaling 因子，补偿因稀疏化丢失的能量（CAST 论文核心思想之一）。
        
        对于 N:M 结构化稀疏：scale = M / N（例如 2:4 → 2.0）
        对于 unstructured 稀疏：scale = 1 / (1 - sparsity_ratio)
        对于 block 稀疏（block_sparse16/32）：scale = 1 / (1 - sparsity_ratio)
        
        返回值 >= 1.0，表示保留权重需要放大多少倍来补偿被剪枝的部分。
        """
        hard_type = str(getattr(self, 'hard_mask_type', 'match') or 'match')
        mask_type = self.mask_type if hard_type == 'match' else hard_type
        
        if mask_type == "none":
            return 1.0
        
        if mask_type == "structured":
            N, M = int(self.N), int(self.M)
            if N <= 0 or M <= 0 or N >= M:
                return 1.0
            return float(M) / float(N)
        
        if mask_type == "nm_2_4":
            return 2.0  # 4/2
        
        # unstructured, block16, block_sparse16, block_sparse32
        sr = float(self.sparsity_ratio)
        if sr <= 0.0 or sr >= 1.0:
            return 1.0
        return 1.0 / (1.0 - sr)

    # ---- SLoRB init ----
    @torch.no_grad()
    def init_SLoRB(self):
        """
        Initialize SLoRB projection and weights.
        - x_proj: (in_features//k, in_features), block-sum projection by default
        - SLoRB_Weight: (out_features, in_features//k)
        """
        N, d = self.weight.shape  # out, in
        k = int(self.SLoRB_k)
        assert d % k == 0, f"in_features ({d}) must be divisible by SLoRB_k ({k})."
        rows = d // k
        cols = d

        # Parameters are created in __init__ (FSDP-safe). Here we only (re)initialize values.
        if not hasattr(self, "x_proj") or not isinstance(self.x_proj, nn.Parameter):
            self.x_proj = nn.Parameter(
                torch.zeros(rows, cols, device=self.weight.device, dtype=self.weight.dtype),
                requires_grad=bool(self.trainable_projection)
            )
        if not hasattr(self, "SLoRB_Weight") or not isinstance(self.SLoRB_Weight, nn.Parameter):
            self.SLoRB_Weight = nn.Parameter(
                torch.zeros((N, rows), device=self.weight.device, dtype=self.weight.dtype),
                requires_grad=True
            )

        x_proj = self.x_proj.data
        x_proj.zero_()
        indices = torch.arange(rows, device=self.weight.device) * k
        x_proj[torch.arange(rows, device=self.weight.device)[:, None], indices[:, None] + torch.arange(k, device=self.weight.device)] = 1

        if self.SLoRB_init_type == "xavier":
            nn.init.xavier_uniform_(self.x_proj)
        else:
            # Use "currently pruned part" as initialization signal:
            # pruned_weight = W * (1 - mask)
            pruned = (self.weight.detach() * (1.0 - self.mask.detach())).view(N, rows, k)
            if self.SLoRB_init_type == "mean":
                init = pruned.mean(dim=2)
            elif self.SLoRB_init_type == "sum":
                init = pruned.sum(dim=2)
            else:
                raise ValueError(f"Invalid SLoRB_init_type: {self.SLoRB_init_type}")
            self.SLoRB_Weight.data.copy_(init)

    # ---- WANDA calibration ----
    def _is_placeholder(self, name: str) -> bool:
        """Check if a parameter is a placeholder (not the full weight-aligned tensor).
        
        FSDP shards all Parameters (including requires_grad=False), but we use
        placeholder Parameters (size=1) to avoid registering large buffers when
        they're not needed. This method checks if a parameter is such a placeholder.
        """
        return getattr(self, f'_{name}_placeholder', False)
    
    @torch.no_grad()
    def add_batch(self, inp, out):
        # Skip if scaler_row is placeholder (when not using WANDA metric)
        if self._is_placeholder('scaler_row'):
            return
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.scaler_row.mul_(self.nsamples / (self.nsamples + tmp + 1e-8))
        self.nsamples += tmp
        inp = inp.float()
        self.scaler_row.add_((torch.norm(inp, p=2, dim=1) ** 2) / max(self.nsamples, 1))

    # ---- grad / hessian EMA ----
    @torch.no_grad()
    def update_grad_hessian_ema(self, update_hessian_with_grad2: bool = True):
        # Skip if grad_ema is placeholder (when change_mask=False or not using movement metric)
        if self._is_placeholder('grad_ema') or self.weight.grad is None:
            return
        g = self.weight.grad.detach()
        # FSDP may materialize grads as flattened shards/views.
        # Best-effort reshape back to weight shape when numel matches.
        if g.shape != self.grad_ema.shape:
            if g.numel() == self.grad_ema.numel():
                g = g.view_as(self.grad_ema)
            else:
                return
        b = float(self.cfg.beta)
        self.grad_ema.mul_(b).add_(g, alpha=1 - b)
        if update_hessian_with_grad2 and not self._is_placeholder('hessian_diag'):
            # FSDP-safe: only update if shapes match
            if self.hessian_diag.shape == g.shape:
                self.hessian_diag.mul_(b).add_(g * g, alpha=1 - b)
            elif hasattr(self, '_hessian_diag_local') and self._hessian_diag_local.shape == g.shape:
                # Use local buffer if available (created by Hutchinson in FSDP mode)
                self._hessian_diag_local.mul_(b).add_(g * g, alpha=1 - b)

    # ---- importance EMA ----
    @torch.no_grad()
    def update_importance_ema(self):
        """DEPRECATED: importance_ema buffer removed. Scores computed on-the-fly in compute_gate_target()."""
        # This is now a no-op for backward compatibility
        pass

    # ---- schedules ----
    def _temperature(self, step: int) -> float:
        K = max(1, int(self._mask_update_period(step)))
        updates = step // K
        init_temp = float(self.cfg.temp_init)
        # block_sparse 模式下提高初始温度，让初期 sigmoid 更平缓，减少激进二值化
        hard_type = str(getattr(self, 'hard_mask_type', 'match') or 'match')
        if hard_type in ('block_sparse16', 'block_sparse32'):
            block_temp_mult = float(getattr(self.cfg, 'block_sparse_temp_mult', 2.0))
            init_temp = init_temp * block_temp_mult
        T = init_temp * (float(self.cfg.temp_decay) ** updates)
        return float(max(float(self.cfg.temp_min), T))

    def _mask_update_period(self, step: int) -> int:
        """Return effective mask update period K for the given step."""
        base = int(getattr(self.cfg, 'mask_update_period', 10))
        sw = int(getattr(self.cfg, 'mask_update_switch_step', 0) or 0)
        before = getattr(self.cfg, 'mask_update_period_before', None)
        after = getattr(self.cfg, 'mask_update_period_after', None)
        if sw > 0 and before is not None and after is not None:
            return int(before) if step < sw else int(after)
        return base

    def _effective_sparsity(self, step: int) -> float:
        warm = int(self.cfg.sparsity_warmup_steps)
        if warm <= 0:
            return float(self.sparsity_ratio)
        r = min(1.0, max(0.0, step / max(1, warm)))
        # block_sparse 模式使用 cubic warmup：前期稀疏度增长更缓慢，
        # 避免大粒度 block（16x16=256参数）被过早整体清零导致 PPL 飙升
        hard_type = str(getattr(self, 'hard_mask_type', 'match') or 'match')
        if hard_type in ('block_sparse16', 'block_sparse32'):
            r = r * r * r  # cubic: 50% warmup 时只有 12.5% 稀疏度（vs 线性的 50%）
        return float(self.sparsity_ratio * r)

    def _beta_structural(self, step: int) -> float:
        """Compute β(step) for structural mixing.

        β controls how much group-level N:M structure is blended into the
        global soft gate G:  G_final = (1-β)*G_global + β*target_nm.

        Schedule: smooth-step (cubic Hermite) rise from 0→1 over
        [beta_structural_start, beta_structural_end], then clamp at 1.
        """
        start = int(getattr(self.cfg, 'beta_structural_start', 0))
        end = int(getattr(self.cfg, 'beta_structural_end', 0))
        if end <= start:
            return 0.0  # disabled
        if step < start:
            return 0.0
        if step >= end:
            return 1.0
        t = float(step - start) / float(end - start)  # linear 0→1
        # smooth-step: 3t² - 2t³  (C¹ continuous, 0-derivative at endpoints)
        return float(3.0 * t * t - 2.0 * t * t * t)

    @torch.no_grad()
    def _tau_unstructured(self, scores: torch.Tensor, sparsity: float) -> torch.Tensor:
        flat = scores.flatten()
        n = flat.numel()
        m = int(self.cfg.tau_sample_size) if self.cfg.tau_sample_size is not None else n
        if m > 0 and n > m:
            idx = torch.randint(0, n, (m,), device=flat.device)
            sample = flat[idx]
        else:
            sample = flat
        # local quantile estimate
        tau = torch.quantile(sample.float(), q=float(sparsity))
        # If running under DDP, average the scalar tau across ranks so all ranks use the
        # same threshold (prevents masks diverging due to independent sampling).
        # CRITICAL: Skip all_reduce when inside summon_full_params context because:
        # 1. All ranks see identical full parameters, so tau is already consistent.
        # 2. Some ranks may skip update_mask (due to shape mismatch in partial unshard),
        #    causing NCCL collective mismatch if all_reduce is attempted.
        if getattr(self, '_in_summon_context', False):
            return tau
        try:
            if dist.is_available() and dist.is_initialized():
                t = tau.clone()
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t.div_(dist.get_world_size())
                tau = t
        except Exception:
            pass
        return tau

    # ---- compute target gate G ----
    @torch.no_grad()
    def compute_gate_target(self, step: int) -> torch.Tensor:
        """Compute soft mask target [0,1] from importance scores (directly from weight/grad/hessian, no importance_ema)."""
        # If change_mask disabled or mask_type is none, keep all 1s
        if (not self.change_mask) or (self.mask_type == "none"):
            return torch.ones_like(self.mask)
        
        # Compute scores directly from weight/grad_ema/hessian_diag (no importance_ema buffer!)
        eps = 1e-8
        metric = self.mask_metric
        W = self.weight.detach()
        
        # Helper to get hessian_diag that matches weight shape (FSDP-safe)
        def get_hessian():
            # First try _hessian_diag_local (created by Hutchinson in FSDP mode)
            if hasattr(self, '_hessian_diag_local') and self._hessian_diag_local.shape == W.shape:
                return self._hessian_diag_local.detach()
            # Then try hessian_diag if it matches weight shape
            if not self._is_placeholder('hessian_diag') and self.hessian_diag.shape == W.shape:
                return self.hessian_diag.detach()
            # Shapes don't match (FSDP sharding) - return None
            return None
        
        if metric == "magnitude":
            scores = W.abs()
        elif metric == "movement":
            # score = |W * grad_ema|
            if not self._is_placeholder('grad_ema') and self.grad_ema.shape == W.shape:
                g = self.grad_ema.detach()
                scores = (W * g).abs() / max(float(self.cfg.temperature), eps)
            else:
                scores = W.abs()  # fallback if no grad_ema or shape mismatch
        elif metric == "hessian_obd":
            # score = (H + eps) * W^2
            H = get_hessian()
            if H is not None:
                scores = (H + eps) * (W * W)
            else:
                scores = W * W  # fallback
        elif metric == "hessian_ratio":
            # score = |W| / sqrt(H + eps)
            H = get_hessian()
            if H is not None:
                scores = W.abs() / torch.sqrt(H + eps)
            else:
                scores = W.abs()  # fallback
        elif metric == "hessian":
            # score = H (hessian diagonal)
            H = get_hessian()
            if H is not None:
                scores = H
            else:
                scores = W.abs()  # fallback
        elif metric == "wanda":
            # score = |W| * scaler_row
            if not self._is_placeholder('scaler_row'):
                scaler = self.scaler_row.detach()
                scores = W.abs() * torch.sqrt(scaler.reshape(1, -1) + eps)
            else:
                scores = W.abs()  # fallback
        else:
            scores = W.abs()  # default to magnitude
        
        if scores.numel() == 0:
            return self.mask
        
        # Normalize scores in float32 for numerical stability, then cast back to native dtype
        scores_fp32 = scores.float()
        mu = scores_fp32.mean()
        sigma = scores_fp32.std(unbiased=False) + 1e-6
        scores = ((scores_fp32 - mu) / sigma).to(scores.dtype)
        
        T = self._temperature(step)
        Tt = torch.tensor(T, device=scores.device, dtype=torch.float32)
        
        # Extract frozen mask flags (explicit .bool() conversion to avoid implicit conversion in indexing)
        frozen = (
            self.frozen_mask_flags.bool()
            if not self._is_placeholder('frozen_mask_flags') else torch.zeros_like(self.mask, dtype=torch.bool)
        )

        if self.mask_type == "unstructured":
            sparsity = self._effective_sparsity(step)
            hard_type = str(getattr(self, 'hard_mask_type', 'match') or 'match')
            # ── Block-level soft mask: 在 block 级别计算 G ──────────────
            # 当 hard_mask_type 为 block_sparse16/32 时，先将逐元素 scores pool 到
            # block 级别（RMS），在 block 级别做 threshold + sigmoid，再 expand 回去。
            # 这保证 G 从一开始就是 block 粒度一致的，不存在 block 内部分化问题。
            # ── CRITICAL FIX for NCCL collective mismatch ──
            # 在 FSDP summon_full_params 上下文中，不同 rank 的 scores 可能有
            # 不同的 dim（某些 rank 正确恢复为 2D，某些仍为 1D flat tensor）。
            # 如果 block_sparse16 路径有独立的 all_reduce（同步 tau_block），
            # 而 else 路径调用 _tau_unstructured（也有 all_reduce），
            # 不同 rank 走不同分支就会导致 NCCL collective 不匹配 → 死锁/超时。
            #
            # 修复方案：所有 rank 统一先调用 _tau_unstructured 计算 tau（保证
            # all_reduce 一致），然后在 tau 基础上做 block-level 或 element-level
            # sigmoid。block_sparse 路径不再做独立的 all_reduce。
            # ──────────────────────────────────────────────────────────────

            # 第一步：统一计算逐元素 tau（所有 rank 走同一条 all_reduce 路径）
            tau = self._tau_unstructured(scores, sparsity)

            # 第二步：根据 hard_type 决定如何用 tau 生成 G
            if hard_type in ('block_sparse16', 'block_sparse32') and scores.dim() == 2:
                bs = 16 if hard_type == 'block_sparse16' else 32
                out_dim, in_dim = scores.shape
                out_full = (out_dim // bs) * bs
                in_full = (in_dim // bs) * bs
                if out_full > 0 and in_full > 0:
                    # 1. 将 scores pool 到 block 级别（RMS）
                    core = scores.detach().float()[:out_full, :in_full]
                    ob = out_full // bs
                    ib = in_full // bs
                    tiles = core.view(ob, bs, ib, bs)  # (ob, bs, ib, bs)
                    # block RMS: sqrt(mean(scores^2)) per block
                    block_scores = (tiles ** 2).mean(dim=(1, 3)).sqrt()  # (ob, ib)
                    
                    # 2. 用 block_scores 自己的 quantile 作为 block-level tau
                    #    ── 关键修复 ──
                    #    之前错误地使用逐元素 tau，但 block_scores 是 RMS 聚合值，
                    #    量级远大于逐元素 scores，导致 sigmoid 输出全部接近 1。
                    #    正确做法：用 block_scores 的 sparsity-quantile 作为 tau_block，
                    #    这样大的一半 block → sigmoid > 0.5 → 推向 1，
                    #    小的一半 block → sigmoid < 0.5 → 推向 0。
                    #    注意：不需要 all_reduce，因为在 summon_full_params 上下文中
                    #    所有 rank 看到的参数完全一致，tau_block 本就相同。
                    block_flat = block_scores.flatten()
                    tau_block = torch.quantile(block_flat.float(), q=float(sparsity))
                    
                    # 3. block 级别 sigmoid
                    G_block = torch.sigmoid((block_scores.float() - tau_block) / (Tt + 1e-8))  # (ob, ib)
                    
                    # 4. expand 回元素级别
                    G_expanded = G_block.unsqueeze(1).unsqueeze(3).expand(ob, bs, ib, bs)
                    G_core = G_expanded.contiguous().view(out_full, in_full)
                    
                    # 5. 构造完整 G（边缘部分保持 1.0）
                    G = torch.ones_like(scores, dtype=scores.dtype)
                    G[:out_full, :in_full] = G_core.to(dtype=scores.dtype)
                else:
                    # 无法分 block，退化为逐元素
                    G = torch.sigmoid((scores.float() - tau) / (Tt + 1e-8)).to(scores.dtype)
            else:
                G = torch.sigmoid((scores.float() - tau) / (Tt + 1e-8)).to(scores.dtype)

        elif self.mask_type == "structured":
            N, M = int(self.N), int(self.M)
            out_dim, in_dim = scores.shape
            assert in_dim % M == 0, f"in_features ({in_dim}) must be divisible by M ({M})."
            groups = in_dim // M
            s = scores.float().view(out_dim, groups, M)
            # group-wise threshold: top-N in each M-sized group
            # obtain top-N values and indices per group
            topk = torch.topk(s, k=N, dim=-1, largest=True)
            topv = topk.values
            topi = topk.indices
            tau_g = topv[..., -1].unsqueeze(-1)
            if getattr(self.cfg, 'structured_exact', False):
                # enforce exact top-N per group using indices (avoid >= threshold tie issues)
                Gg = torch.zeros_like(s, dtype=scores.dtype)
                # scatter ones at topk indices
                # topi shape: (out_dim, groups, N)
                Gg.scatter_(-1, topi, 1.0)
                G = Gg.view(out_dim, in_dim)
            else:
                # soft sigmoid gating around group threshold
                G = torch.sigmoid((s - tau_g) / (Tt + 1e-8)).view(out_dim, in_dim).to(scores.dtype)

        else:
            raise ValueError(f"Invalid mask_type: {self.mask_type}")

        # ── β structural mixing ──────────────────────────────────────────
        # Blend group-level N:M hard target into G so that soft mask
        # gradually acquires the 2:4 (or N:M) structure *before* hardening
        # kicks in, reducing the soft↔hard gap that causes PPL spikes.
        #
        # G_final = (1 - β) * G_global  +  β * target_nm
        #
        # target_nm is computed from the raw *scores* (not G) to keep
        # ordering stable regardless of temperature / sigmoid saturation.
        beta_s = self._beta_structural(step)
        # 注意：beta_structural 是 N:M 结构化混合，仅对非 block_sparse 模式有效。
        # block_sparse 模式下跳过，因为 N:M 分组（每 4 个元素）会破坏 block 级一致性。
        hard_type_for_beta = str(getattr(self, 'hard_mask_type', 'match') or 'match')
        is_block_sparse = hard_type_for_beta in ('block_sparse16', 'block_sparse32')
        if beta_s > 0.0 and scores.dim() == 2 and not is_block_sparse:
            N_s, M_s = int(self.N), int(self.M)
            out_dim_s, in_dim_s = scores.shape
            in_full_s = (in_dim_s // M_s) * M_s
            if in_full_s > 0:
                # 基于原始 scores 在每个 group 内做 top-N 选择 → 二值 target
                sc_core = scores.detach().float()[:, :in_full_s]
                groups_s = in_full_s // M_s
                sc_grouped = sc_core.view(out_dim_s, groups_s, M_s)
                topi_s = torch.topk(sc_grouped, k=N_s, dim=-1, largest=True).indices
                target_nm = torch.zeros_like(sc_grouped, dtype=G.dtype)
                target_nm.scatter_(-1, topi_s, 1.0)
                target_nm = target_nm.view(out_dim_s, in_full_s)

                # 构造 full-size target（对于 in_dim 不能被 M 整除的尾部保持 G 原值）
                if in_full_s < in_dim_s:
                    target_full = G.clone()
                    target_full[:, :in_full_s] = target_nm
                else:
                    target_full = target_nm

                G = ((1.0 - beta_s) * G.float() + beta_s * target_full.float()).to(G.dtype)

        if frozen.any():
            G = G.clone()
            G[frozen] = self.mask[frozen]

        # ── Block-level soft mask 一致性 ──────────────────────────────────
        # 当 hard_mask_type 为 block_sparse16/32 时，将 G 按 block 取均值。
        # 虽然 block 路径本身产出的 G 已经是 block 一致的，但经过
        # beta_structural mixing 和 frozen 处理后可能引入微小的不一致，
        # 此处做最终的一致性保障。
        # 对于 nm_2_4 等非 block_sparse 模式，_blockify_tensor 直接返回原值，零开销。
        G = self._blockify_tensor(G)

        return G

    # ---- periodic mask update ----
    @torch.no_grad()
    def update_mask(self, step: int, lambda_mid: float = 0.0):
        # FSDP 兼容性说明：
        # calculate_model_mask() 现在在 summon_full_params() 上下文中调用本方法，
        # 所以 self.mask 和 self.weight 都应该恢复为完整的 2D shape (out_features, in_features)。
        # 如果仍然不是 2D，说明 FSDP 配置有问题或未正确调用，跳过并警告。
        
        # 跳过空 tensor
        if self.mask.numel() == 0:
            return
        
        # 在 summon_full_params 上下文中，mask 应该是 2D
        if self.mask.dim() != 2:
            import warnings
            if not getattr(self, '_warned_mask_dim_at_entry', False):
                warnings.warn(
                    f"[update_mask] mask.dim()={self.mask.dim()}, shape={self.mask.shape}. "
                    f"Expected 2D (inside summon_full_params). "
                    f"weight.shape={self.weight.shape}. Skipping this layer."
                )
                self._warned_mask_dim_at_entry = True
            return
        
        # weight 和 mask shape 应该匹配
        if self.weight.shape != self.mask.shape:
            import warnings
            if not getattr(self, '_warned_fsdp_shape_mismatch', False):
                warnings.warn(
                    f"[update_mask] Shape mismatch: weight.shape={self.weight.shape}, "
                    f"mask.shape={self.mask.shape}. Skipping this layer's mask update."
                )
                self._warned_fsdp_shape_mismatch = True
            return
        
        if (not self.change_mask) or (self.mask_type == "none"):
            self.weight.flipped_mask = 0
            self.weight.init_flipped_mask = 0
            return

        if step < int(self.cfg.sparsity_warmup_steps):
            return

        K = max(1, int(self._mask_update_period(step)))
        if step % K != 0:
            return

        prev = self.mask.detach().clone()
        G = self.compute_gate_target(step)

        a = float(self.cfg.mask_lr)
        # Apply mid_penalty correction to the target gate G so penalty affects mask dynamics.
        if lambda_mid is None:
            lambda_mid = 0.0
        if float(lambda_mid) != 0.0:
            pen_scale = float(self.mask_penalty_lr)
            mode = str(getattr(self.cfg, 'mask_penalty_mode', 'mid') or 'mid')
            if mode == 'block16':
                # Tile-level top-k penalty for 16x16 logical tile sparsity.
                # Build a per-tile hard target (top-k -> 1, rest -> 0) and pull G toward it.
                # User request: hard target is based on CURRENT soft mask ordering within each tile.
                bs = 16
                # 跳过 1D mask 或 1D G（FSDP 分片可能导致临时展平，或者 weight 是 sharded 的）
                if self.mask.dim() != 2 or G.dim() != 2:
                    # DEBUG: log unexpected shape (only once to avoid spam)
                    if not getattr(self, '_warned_1d_mask', False):
                        import warnings
                        warnings.warn(
                            f"[block16] mask is {self.mask.dim()}D (shape {self.mask.shape}), "
                            f"G is {G.dim()}D (shape {G.shape}), expected both to be 2D. "
                            f"Falling back to 'mid' mode. "
                            f"weight shape: {self.weight.shape if hasattr(self, 'weight') else 'N/A'}"
                        )
                        self._warned_1d_mask = True
                    mode = 'mid'
                else:
                    out_dim, in_dim = self.mask.shape
                    out_full = (out_dim // bs) * bs
                    in_full = (in_dim // bs) * bs
                    if out_full == 0 or in_full == 0:
                        mode = 'mid'
                    else:
                        sr = float(getattr(self, 'sparsity_ratio', 0.0))
                        keep_k = int(round((1.0 - sr) * (bs * bs)))
                        keep_k = max(0, min(bs * bs, keep_k))

                        # target based on current gate ordering (use G to determine importance)
                        # (use G rather than self.mask so penalty follows the same scoring used to form G)
                        core = G.detach().float()[:out_full, :in_full]
                        ob = out_full // bs
                        ib = in_full // bs
                        tiles = core.view(ob, bs, ib, bs).permute(0, 2, 1, 3).contiguous().view(ob, ib, bs * bs)

                        if keep_k == 0:
                            target_tiles = torch.zeros_like(tiles, dtype=G.dtype)
                        elif keep_k == bs * bs:
                            target_tiles = torch.ones_like(tiles, dtype=G.dtype)
                        else:
                            topi = torch.topk(tiles, k=keep_k, dim=-1, largest=True).indices
                            target_tiles = torch.zeros_like(tiles, dtype=G.dtype)
                            target_tiles.scatter_(-1, topi, 1.0)

                        target_core = target_tiles.view(ob, ib, bs, bs).permute(0, 2, 1, 3).contiguous().view(out_full, in_full)
                        target = torch.ones_like(G, dtype=G.dtype)
                        target[:out_full, :in_full] = target_core

                        delta = (self.mask.to(dtype=G.dtype) - target)
                        G = G - (pen_scale * float(lambda_mid)) * delta

            if mode == 'nm_2_4':
                # 2:4 Structured Sparsity penalty mode
                # 在每4个元素的group中，拉高top-2的mask，降低bottom-2的mask
                # 和 block16 操作方式类似
                N, M = 2, 4
                # 跳过 1D mask 或 1D G
                if self.mask.dim() != 2 or G.dim() != 2:
                    if not getattr(self, '_warned_1d_mask_nm_2_4', False):
                        import warnings
                        warnings.warn(
                            f"[nm_2_4] mask is {self.mask.dim()}D (shape {self.mask.shape}), "
                            f"G is {G.dim()}D (shape {G.shape}), expected both to be 2D. "
                            f"Falling back to 'mid' mode. "
                            f"weight shape: {self.weight.shape if hasattr(self, 'weight') else 'N/A'}"
                        )
                        self._warned_1d_mask_nm_2_4 = True
                    mode = 'mid'
                else:
                    out_dim, in_dim = self.mask.shape
                    in_full = (in_dim // M) * M
                    if in_full == 0:
                        mode = 'mid'
                    else:
                        # 使用 G 的值来决定 group 内元素的重要性
                        core = G.detach().float()[:, :in_full]
                        groups = in_full // M
                        # (out_dim, groups, M)
                        grouped = core.view(out_dim, groups, M)

                        # 在每个 group 内选择 top-N 作为 target=1，其余为 target=0
                        topi = torch.topk(grouped, k=N, dim=-1, largest=True).indices
                        target_grouped = torch.zeros_like(grouped, dtype=G.dtype)
                        target_grouped.scatter_(-1, topi, 1.0)

                        # reshape back to (out_dim, in_full)
                        target_core = target_grouped.view(out_dim, in_full)

                        # Full target: 默认保持1，只在 in_full 列应用 2:4 target
                        target = torch.ones_like(G, dtype=G.dtype)
                        target[:, :in_full] = target_core

                        # Pull G toward target
                        delta = (self.mask.to(dtype=G.dtype) - target)
                        G = G - (pen_scale * float(lambda_mid)) * delta

            if mode == 'block_sparse32':
                # Block-level penalty for 32x32 block sparsity.
                # 目标：一半的blocks被pruned（全0），另一半为dense（全1）
                # 计算每个block的均值（RMS），较大的50%保留，较小的50%pruned
                # 然后 pull G 向这个 target 靠近
                bs = 32
                # 跳过 1D mask 或 1D G
                if self.mask.dim() != 2 or G.dim() != 2:
                    if not getattr(self, '_warned_1d_mask_block_sparse32', False):
                        import warnings
                        warnings.warn(
                            f"[block_sparse32] mask is {self.mask.dim()}D (shape {self.mask.shape}), "
                            f"G is {G.dim()}D (shape {G.shape}), expected both to be 2D. "
                            f"Falling back to 'mid' mode. "
                            f"weight shape: {self.weight.shape if hasattr(self, 'weight') else 'N/A'}"
                        )
                        self._warned_1d_mask_block_sparse32 = True
                    mode = 'mid'
                else:
                    out_dim, in_dim = self.mask.shape
                    out_full = (out_dim // bs) * bs
                    in_full = (in_dim // bs) * bs
                    if out_full == 0 or in_full == 0:
                        mode = 'mid'
                    else:
                        sr = float(getattr(self, 'sparsity_ratio', 0.5))
                        ob = out_full // bs
                        ib = in_full // bs
                        total_blocks = ob * ib
                        keep_blocks = int(round((1.0 - sr) * total_blocks))
                        keep_blocks = max(0, min(total_blocks, keep_blocks))

                        # 用 mask 当前值来决定 block 排序（大的推1，小的推0）
                        # ── 关键修复 ──
                        # 之前用 G.detach() 排序，但 penalty 的目的是"根据 mask 当前状态推动二值化"。
                        # 正确做法：用 self.mask 的 block-level 均值排序，
                        # 和 mid mode 用 mask 值分 >0.5/<0.5 的逻辑一致。
                        core_mask = self.mask.detach().float()[:out_full, :in_full]
                        tiles_mask = core_mask.view(ob, bs, ib, bs).permute(0, 2, 1, 3).contiguous()
                        flat_tiles_mask = tiles_mask.reshape(ob, ib, bs * bs)  # (ob, ib, 1024)

                        # 计算每个 block 的均值作为排序依据
                        block_mean = flat_tiles_mask.mean(dim=-1)  # (ob, ib)
                        flat_scores = block_mean.view(-1)  # (total_blocks,)

                        # 生成 block-level 的 target mask
                        if keep_blocks == 0:
                            block_target = torch.zeros(ob, ib, dtype=G.dtype, device=G.device)
                        elif keep_blocks == total_blocks:
                            block_target = torch.ones(ob, ib, dtype=G.dtype, device=G.device)
                        else:
                            _, top_indices = torch.topk(flat_scores, k=keep_blocks, largest=True)
                            block_target = torch.zeros(total_blocks, dtype=G.dtype, device=G.device)
                            block_target[top_indices] = 1.0
                            block_target = block_target.view(ob, ib)

                        # 将 block-level target 扩展到 element-level
                        # block_target: (ob, ib) -> (ob, 1, ib, 1) -> (ob, bs, ib, bs) -> (out_full, in_full)
                        target_expanded = block_target.unsqueeze(1).unsqueeze(3).expand(ob, bs, ib, bs)
                        target_core = target_expanded.permute(0, 1, 2, 3).contiguous().view(out_full, in_full)

                        # Full target: 默认保持1，只在 core 区域应用 block target
                        target = torch.ones_like(G, dtype=G.dtype)
                        target[:out_full, :in_full] = target_core

                        # Pull G toward target
                        delta = (self.mask.to(dtype=G.dtype) - target)
                        G = G - (pen_scale * float(lambda_mid)) * delta

            if mode == 'block_sparse16':
                # Block-level penalty for 16x16 block sparsity.
                # 和 block_sparse32 逻辑完全一致，只是 block size 为 16
                bs = 16
                # 跳过 1D mask 或 1D G
                if self.mask.dim() != 2 or G.dim() != 2:
                    if not getattr(self, '_warned_1d_mask_block_sparse16', False):
                        import warnings
                        warnings.warn(
                            f"[block_sparse16] mask is {self.mask.dim()}D (shape {self.mask.shape}), "
                            f"G is {G.dim()}D (shape {G.shape}), expected both to be 2D. "
                            f"Falling back to 'mid' mode. "
                            f"weight shape: {self.weight.shape if hasattr(self, 'weight') else 'N/A'}"
                        )
                        self._warned_1d_mask_block_sparse16 = True
                    mode = 'mid'
                else:
                    out_dim, in_dim = self.mask.shape
                    out_full = (out_dim // bs) * bs
                    in_full = (in_dim // bs) * bs
                    if out_full == 0 or in_full == 0:
                        mode = 'mid'
                    else:
                        sr = float(getattr(self, 'sparsity_ratio', 0.5))
                        ob = out_full // bs
                        ib = in_full // bs
                        total_blocks = ob * ib
                        keep_blocks = int(round((1.0 - sr) * total_blocks))
                        keep_blocks = max(0, min(total_blocks, keep_blocks))

                        # 用 mask 当前值来决定 block 排序（大的推1，小的推0）
                        # ── 关键修复 ──
                        # 之前用 G.detach() 排序，但 G 是 gate target（基于 scores 的 sigmoid），
                        # 而 penalty 的目的是"根据 mask 当前状态来推动二值化"。
                        # 正确做法：用 self.mask 的 block-level 均值排序，
                        # 和 mid mode 用 mask 值分 >0.5/<0.5 的逻辑一致。
                        core_mask = self.mask.detach().float()[:out_full, :in_full]
                        tiles_mask = core_mask.view(ob, bs, ib, bs).permute(0, 2, 1, 3).contiguous()
                        flat_tiles_mask = tiles_mask.reshape(ob, ib, bs * bs)  # (ob, ib, bs*bs)

                        # 计算每个 block 的均值作为排序依据
                        block_mean = flat_tiles_mask.mean(dim=-1)  # (ob, ib)
                        flat_scores = block_mean.view(-1)  # (total_blocks,)

                        # 生成 block-level 的 target mask
                        if keep_blocks == 0:
                            block_target = torch.zeros(ob, ib, dtype=G.dtype, device=G.device)
                        elif keep_blocks == total_blocks:
                            block_target = torch.ones(ob, ib, dtype=G.dtype, device=G.device)
                        else:
                            _, top_indices = torch.topk(flat_scores, k=keep_blocks, largest=True)
                            block_target = torch.zeros(total_blocks, dtype=G.dtype, device=G.device)
                            block_target[top_indices] = 1.0
                            block_target = block_target.view(ob, ib)

                        # 将 block-level target 扩展到 element-level
                        # block_target: (ob, ib) -> (ob, 1, ib, 1) -> (ob, bs, ib, bs) -> (out_full, in_full)
                        target_expanded = block_target.unsqueeze(1).unsqueeze(3).expand(ob, bs, ib, bs)
                        target_core = target_expanded.permute(0, 1, 2, 3).contiguous().view(out_full, in_full)

                        # Full target: 默认保持1，只在 core 区域应用 block target
                        target = torch.ones_like(G, dtype=G.dtype)
                        target[:out_full, :in_full] = target_core

                        # Pull G toward target
                        delta = (self.mask.to(dtype=G.dtype) - target)
                        G = G - (pen_scale * float(lambda_mid)) * delta

            if mode == 'structured_topn':
                # Within each M-group, compute a hard target based on the CURRENT soft mask ordering,
                # then push top-N toward 1 and the rest toward 0 by pulling G toward that target.
                # This avoids the pathological case where 3/4 entries all drift upward together.
                N, M = int(self.N), int(self.M)
                # 跳过 1D mask 或 1D G（FSDP 分片可能导致临时展平，或者 weight 是 sharded 的）
                if self.mask.dim() != 2 or G.dim() != 2:
                    if not getattr(self, '_warned_1d_mask_topn', False):
                        import warnings
                        warnings.warn(
                            f"[structured_topn] mask is {self.mask.dim()}D (shape {self.mask.shape}), "
                            f"G is {G.dim()}D (shape {G.shape}), expected both to be 2D. "
                            f"Falling back to 'mid' mode."
                        )
                        self._warned_1d_mask_topn = True
                    mode = 'mid'
                else:
                    out_dim, in_dim = self.mask.shape
                    if in_dim % M != 0:
                        # fallback to elementwise mode if shape doesn't fit grouping
                        mode = 'mid'
                    else:
                        groups = in_dim // M
                        # IMPORTANT: use the same ordering source as the gate target (G),
                        # not the current mask ordering. Otherwise the penalty can fight
                        # the importance-based target and create oscillations.
                        sm = G.detach().float().view(out_dim, groups, M)
                        topi = torch.topk(sm, k=N, dim=-1, largest=True).indices
                        target = torch.zeros_like(sm, dtype=G.dtype)
                        target.scatter_(-1, topi, 1.0)
                        target = target.view(out_dim, in_dim)
                        # Pull G toward target: for top-N entries (target=1) increase, others decrease.
                        # Using (mask - target) makes the push stronger when far from target.
                        delta = (self.mask.to(dtype=G.dtype) - target)
                        G = G - (pen_scale * float(lambda_mid)) * delta
            if mode == 'mid':
                # d/dm [m*(1-m)] = (1 - 2m): push away from 0.5 elementwise
                pen_grad = (1.0 - 2.0 * self.mask).to(dtype=G.dtype)
                G = G - (pen_scale * float(lambda_mid)) * pen_grad

            # keep target gate in valid range
            G = G.clamp(0.0, 1.0)

        # multiplicative binarization push: move values <0.5 downward and >0.5 upward
        try:
            d = float(self.cfg.mask_binarize_decay) if getattr(self.cfg, 'mask_binarize_decay', None) is not None else 0.0
        except Exception:
            d = 0.0
        if d != 0.0:
            # apply scaling to target gate G (before EMA) so EMA will move mask toward pushed target
            lt = G < 0.5
            gt = G > 0.5
            if lt.any():
                G = G.clone()
                G[lt] = G[lt] * (1.0 - d)
            if gt.any():
                if 'G' not in locals():
                    G = G.clone()
                G[gt] = G[gt] * (1.0 + d)

        # EMA update toward corrected (and binarization-pushed) gate
        self.mask.mul_(1 - a).add_(G, alpha=a).clamp_(0.0, 1.0)

        # ── Block-level soft mask 一致性（EMA 后） ────────────────────────
        # EMA 更新可能因 G 中的微小浮点差异导致 block 内元素轻微偏移。
        # 再次 blockify 确保 mask 始终保持 block 粒度一致。
        hard_type = str(getattr(self, 'hard_mask_type', 'match') or 'match')
        if hard_type in ('block_sparse16', 'block_sparse32') and self.mask.dim() == 2:
            blockified = self._blockify_tensor(self.mask.data)
            self.mask.data.copy_(blockified)

        # update per-module hardening parameter x (1->0 over schedule)
        try:
            start = int(getattr(self.cfg, 'mask_hardening_start', 0))
            duration = int(getattr(self.cfg, 'mask_hardening_duration', 0))
        except Exception:
            start = 0
            duration = 0
        if duration <= 0:
            self.hardening_x = 0.0 if step >= start and start > 0 else 1.0
        else:
            if step < start:
                self.hardening_x = 1.0
            elif step >= start + duration:
                self.hardening_x = 0.0
            else:
                progress = float(step - start) / float(max(1, duration))
                self.hardening_x = max(0.0, min(1.0, 1.0 - progress))

        # IMPORTANT:
        # 只有当 freeze_low>0 或 freeze_high<1 才会发生"真正二值化/冻结"
        low = float(self.cfg.freeze_low)
        high = float(self.cfg.freeze_high)
        # Only apply freezing if user configured non-trivial thresholds
        if (low > 0.0) or (high < 1.0):
            freeze0 = self.mask <= low
            freeze1 = self.mask >= high
            # always apply low-side freezing immediately (snap to 0)
            if freeze0.any():
                self.mask[freeze0] = 0.0
            # HIGH-side freeze: only snap to 1 and freeze after step >= 2000
            applied_freeze = freeze0.clone()
            if step >= 2000:
                if freeze1.any():
                    self.mask[freeze1] = 1.0
                applied_freeze = applied_freeze | freeze1
            # update frozen flags only for actually applied freezes (use max for float dtype)
            self.frozen_mask_flags.data = torch.maximum(
                self.frozen_mask_flags.data, 
                applied_freeze.to(dtype=self.frozen_mask_flags.dtype)
            )

        # SYNC: Since mask is now a frozen Parameter (sharded by FSDP), each rank only has
        # a portion of the full mask. FSDP handles synchronization internally during
        # all_gather in forward and reduce_scatter in backward.
        # 
        # NOTE: We no longer need explicit all_reduce sync for mask because:
        # 1. All ranks compute the same update logic (same score, same threshold)
        # 2. FSDP ensures the sharded portions are consistent across ranks
        # 3. During forward, FSDP all_gathers the full params for computation
        #
        # The previous all_reduce sync was needed when mask was a buffer (not sharded)
        # to ensure all ranks had identical copies. Now with sharded Parameter, this
        # is handled by FSDP's internal collective operations.
        #
        # REMOVED: explicit all_reduce sync (no longer needed with sharded mask Parameter)
        # flip logging uses hard view ONLY for统计，不会影响mask连续值
        hard_prev = (self._hard_mask_from_soft(prev) > 0.5)
        hard_now = (self._hard_mask_from_soft(self.mask) > 0.5)
        self.weight.flipped_mask = int((hard_prev ^ hard_now).sum().item())
        # init_flipped_mask deleted (no init_mask to compare against)
        self.weight.init_flipped_mask = 0

        # keep pointers
        self.weight.mask = self.mask


class Distill_Model(torch.nn.Module):
    def __init__(self, model, teacher=None, output_hidden_state=False):
        super().__init__()
        self.student = model
        self.teacher = teacher
        self.student.config.output_hidden_state = output_hidden_state
        self.teacher.config.output_hidden_state = output_hidden_state
        self.output_hidden_state = output_hidden_state
        self.teacher.eval()

    def forward(self, idx, targets=None):
        # If teacher is None (released after finalization), skip distillation
        if self.teacher is None:
            student_logits, task_loss, _ = self.student(idx, targets)
            return student_logits, task_loss, None, torch.tensor(0.0, device=idx.device)
        
        if self.output_hidden_state:
            student_logits, task_loss, student_hidden_states = self.student(idx, targets)
            with torch.no_grad():
                # Teacher may be on CPU or GPU (including FSDP). Follow teacher's parameter device.
                try:
                    tdev = next(self.teacher.parameters()).device
                except Exception:
                    tdev = idx.device
                idx_t = idx.to(device=tdev, non_blocking=False) if idx.device != tdev else idx
                targets_t = None
                if targets is not None:
                    targets_t = targets.to(device=tdev, non_blocking=False) if targets.device != tdev else targets
                teacher_logits, _, teacher_hidden_states = self.teacher(idx_t, targets_t)

            # Move teacher outputs to student's device so we can compute distill losses
            # without moving the student's activations off GPU (keeps student gradients valid).
            if teacher_logits.device != student_logits.device:
                teacher_logits = teacher_logits.to(device=student_logits.device, dtype=student_logits.dtype, non_blocking=False)
            if teacher_hidden_states is not None and student_hidden_states is not None:
                if len(teacher_hidden_states) == len(student_hidden_states):
                    teacher_hidden_states = [t.to(device=student_logits.device, dtype=student_logits.dtype, non_blocking=False) for t in teacher_hidden_states]
            if student_hidden_states is not None and teacher_hidden_states is not None:
                layerwise_loss = self.layerwise_loss(student_hidden_states, teacher_hidden_states)
            else:
                layerwise_loss = 0.0
            kl_loss = self.kl_loss(student_logits, teacher_logits)
            return student_logits, task_loss, layerwise_loss, kl_loss
        else:
            # GPT.forward returns (logits, loss, hidden_states_or_None) even when
            # output_hidden_state=False. Keep this robust across student/teacher.
            student_logits, task_loss, _ = self.student(idx, targets)
            
            # DEBUG: Check student output for NaN/Inf
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(f"[Distill_Model][rank{rank}] STUDENT logits has NaN={torch.isnan(student_logits).any().item()} Inf={torch.isinf(student_logits).any().item()}")
                print(f"[Distill_Model][rank{rank}] student_logits stats: min={student_logits.min().item():.2f} max={student_logits.max().item():.2f}")
            
            with torch.no_grad():
                # Teacher may be on CPU or GPU (including FSDP). Follow teacher's parameter device.
                try:
                    tdev = next(self.teacher.parameters()).device
                except Exception:
                    tdev = idx.device
                idx_t = idx.to(device=tdev, non_blocking=False) if idx.device != tdev else idx
                targets_t = None
                if targets is not None:
                    targets_t = targets.to(device=tdev, non_blocking=False) if targets.device != tdev else targets
                teacher_logits, _, _ = self.teacher(idx_t, targets_t)
                
                # DEBUG: Check teacher output for NaN/Inf
                if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
                    import torch.distributed as dist
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    print(f"[Distill_Model][rank{rank}] TEACHER logits has NaN={torch.isnan(teacher_logits).any().item()} Inf={torch.isinf(teacher_logits).any().item()}")
                    print(f"[Distill_Model][rank{rank}] teacher_logits stats: min={teacher_logits.min().item():.2f} max={teacher_logits.max().item():.2f}")

            # Move teacher logits to student's device for KL computation.
            if teacher_logits.device != student_logits.device:
                teacher_logits = teacher_logits.to(device=student_logits.device, dtype=student_logits.dtype, non_blocking=False)
            kl_loss = self.kl_loss(student_logits, teacher_logits)
            return student_logits, task_loss, None, kl_loss

    def kl_loss(self, student_logits, teacher_logits, temperature=2):
        """
        计算 KL 散度损失，使用分块计算来减少内存峰值。
        针对大 vocab_size 模型（如 Qwen3-14B vocab=151936）进行了内存优化：
        1. 手动分步计算 log_softmax + kl_div，避免多个 [chunk, vocab] 中间张量同时驻留
        2. 动态调整 chunk_size，vocab 越大分块越小
        3. 每个 chunk 计算完毕后立即释放中间变量
        """
        # Flatten to (total_tokens, vocab_size) for chunked processing
        original_shape = student_logits.shape
        device = student_logits.device
        dtype = student_logits.dtype
        student_flat = student_logits.view(-1, original_shape[-1])
        teacher_flat = teacher_logits.view(-1, original_shape[-1])
        # 尽早释放原始引用，减少峰值内存
        del student_logits, teacher_logits
        
        num_tokens = student_flat.size(0)
        vocab_size = student_flat.size(-1)
        
        # 分块大小：根据 vocab_size 动态调整
        # Qwen3-14B (151936): 每个 [chunk, vocab] bf16 ≈ chunk * 151936 * 2 bytes
        # chunk=128 → ~37MB per tensor, 峰值约 ~150MB（4个中间变量）
        if vocab_size > 100000:
            chunk_size = 128  # 超大vocab，极保守分块
        elif vocab_size > 50000:
            chunk_size = 512
        else:
            chunk_size = 2048
        
        max_logit = 100.0
        inv_temp = 1.0 / temperature
        total_kl = torch.tensor(0.0, device=device, dtype=torch.float32)  # float32 累加避免精度丢失
        
        # 分块计算 KL 散度
        # KL(P||Q) = sum(P * (log P - log Q))，其中 P=teacher softmax, Q=student softmax
        for i in range(0, num_tokens, chunk_size):
            end_idx = min(i + chunk_size, num_tokens)
            
            # --- Student log_softmax ---
            s_chunk = student_flat[i:end_idx].float().clamp_(-max_logit, max_logit)
            s_chunk.mul_(inv_temp)
            s_log_prob = F.log_softmax(s_chunk, dim=-1)
            del s_chunk  # 立即释放
            
            # --- Teacher log_softmax ---
            t_chunk = teacher_flat[i:end_idx].float().clamp_(-max_logit, max_logit)
            t_chunk.mul_(inv_temp)
            t_log_prob = F.log_softmax(t_chunk, dim=-1)
            del t_chunk  # 立即释放
            
            # Skip if NaN/Inf
            if torch.isnan(s_log_prob).any() or torch.isnan(t_log_prob).any():
                del s_log_prob, t_log_prob
                continue
            
            # KL = sum(exp(t_log_prob) * (t_log_prob - s_log_prob))
            # 分步计算避免同时存在多个 [chunk, vocab] 张量
            diff = t_log_prob - s_log_prob  # [chunk, vocab]
            del s_log_prob  # 释放 student log_prob
            diff.mul_(t_log_prob.exp())     # inplace: diff = P * (log P - log Q)
            del t_log_prob  # 释放 teacher log_prob
            
            chunk_kl = diff.sum()
            del diff
            
            if torch.isfinite(chunk_kl):
                total_kl = total_kl + chunk_kl
            del chunk_kl
        
        del student_flat, teacher_flat
        
        kl = total_kl * (temperature ** 2) / num_tokens
        kl = kl.to(dtype)  # 转回原始 dtype
        
        # Final safety check
        if not torch.isfinite(kl):
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        return kl

    def layerwise_loss(self, student_hidden_states, teacher_hidden_states):
        length = len(student_hidden_states)
        loss = 0.0
        eps = torch.finfo(torch.bfloat16).eps
        for i in range(length):
            loss += (student_hidden_states[i] - teacher_hidden_states[i]).pow(2).mean() / (
                teacher_hidden_states[i].pow(2).mean() + eps
            )
        return loss
