"""
Channel-level Structured Pruning Trainer.

This is the main training script for channel-level FFN pruning.
It implements:
1. Model loading with optional distillation
2. Channel importance scoring using Hessian-based metrics
3. Soft-to-hard mask annealing following HASAST mechanism
4. Training with masked forward pass
5. Export of pruned model with reduced dimensions

Usage:
    python -m channel_pruning.train_channel_pruning --model_name Qwen/Qwen3-1.7B --ffn_keep_ratio 0.75
"""

import os
import sys
import time
import math
import json
import argparse
from dataclasses import asdict
from contextlib import nullcontext
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import ChannelPruningConfig
from .channel_groups import (
    get_mlp_projections, 
    MLPChannelGroupManager,
    get_intermediate_size,
    get_num_layers
)
from .channel_score import ChannelScoreComputer
from .channel_mask import (
    ChannelMaskState, 
    ChannelMaskApplier,
    patch_model_mlp_forward,
    restore_model_mlp_forward
)
from .export_pruned import export_pruned_model, compute_param_reduction


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Channel-level Structured Pruning")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model_type", type=str, default="qwen",
                        choices=["qwen", "llama", "opt", "mistral"])
    
    # Pruning
    parser.add_argument("--ffn_keep_ratio", type=float, default=0.75)
    parser.add_argument("--importance_metric", type=str, default="hessian_obd",
                        choices=["hessian_obd", "magnitude", "taylor", "wanda"])
    
    # Mask dynamics
    parser.add_argument("--mask_update_period", type=int, default=50)
    parser.add_argument("--mask_lr", type=float, default=0.1)
    parser.add_argument("--sparsity_warmup_steps", type=int, default=500)
    parser.add_argument("--hardening_start_step", type=int, default=2000)
    parser.add_argument("--hardening_duration", type=int, default=5000)
    
    # Training
    parser.add_argument("--dataset", type=str, default="c4_qwen")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=500)
    
    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Distillation
    parser.add_argument("--use_distillation", action="store_true", default=True)
    parser.add_argument("--distill_alpha", type=float, default=0.5)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=100)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--wandb_log", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="channel-pruning")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="outputs/channel_pruning")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--export_pruned", action="store_true", default=True)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--compile", action="store_true", default=False)
    
    # Distributed
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("--backend", type=str, default="nccl")
    
    return parser.parse_args()


def args_to_config(args) -> ChannelPruningConfig:
    """Convert argparse args to ChannelPruningConfig."""
    return ChannelPruningConfig(
        model_name=args.model_name,
        model_type=args.model_type,
        ffn_keep_ratio=args.ffn_keep_ratio,
        importance_metric=args.importance_metric,
        mask_update_period=args.mask_update_period,
        mask_lr=args.mask_lr,
        sparsity_warmup_steps=args.sparsity_warmup_steps,
        hardening_start_step=args.hardening_start_step,
        hardening_duration=args.hardening_duration,
        dataset=args.dataset,
        block_size=args.block_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_distillation=args.use_distillation,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        wandb_log=args.wandb_log,
        wandb_project=args.wandb_project,
        out_dir=args.out_dir,
        save_interval=args.save_interval,
        export_pruned=args.export_pruned,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        ddp=args.ddp,
        backend=args.backend,
    )


class ChannelPruningTrainer:
    """Main trainer for channel-level structured pruning."""
    
    def __init__(self, config: ChannelPruningConfig):
        self.config = config
        self.setup_distributed()
        self.setup_device()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_pruning()
        self.setup_logging()
    
    def setup_distributed(self):
        """Setup distributed training."""
        self.ddp = self.config.ddp
        if self.ddp:
            dist.init_process_group(backend=self.config.backend)
            self.ddp_rank = dist.get_rank()
            self.ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.ddp_world_size = dist.get_world_size()
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.device = self.config.device
            self.master_process = True
        
        if self.master_process:
            print(f"[ChannelPruningTrainer] DDP={self.ddp}, rank={self.ddp_rank}, world_size={self.ddp_world_size}")
    
    def setup_device(self):
        """Setup compute device and dtype."""
        self.device = torch.device(self.device)
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map[self.config.dtype]
        
        # Setup autocast context
        if self.device.type == "cuda":
            self.ctx = torch.amp.autocast(device_type="cuda", dtype=self.dtype)
        else:
            self.ctx = nullcontext()
        
        if self.master_process:
            print(f"[ChannelPruningTrainer] device={self.device}, dtype={self.dtype}")
    
    def setup_data(self):
        """Setup training and validation data."""
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            self.config.dataset
        )
        
        if self.master_process:
            print(f"[ChannelPruningTrainer] Loading data from {data_dir}")
        
        # Load data
        train_path = os.path.join(data_dir, "train.bin")
        val_path = os.path.join(data_dir, "val.bin")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        # Memory map for large files
        self.train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        
        if os.path.exists(val_path):
            self.val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
        else:
            # Use last 10% of train for validation
            split_idx = int(len(self.train_data) * 0.9)
            self.val_data = self.train_data[split_idx:]
            self.train_data = self.train_data[:split_idx]
        
        if self.master_process:
            print(f"[ChannelPruningTrainer] train_tokens={len(self.train_data):,}, val_tokens={len(self.val_data):,}")
    
    def get_batch(self, split: str):
        """Get a batch of data."""
        data = self.train_data if split == "train" else self.val_data
        block_size = self.config.block_size
        batch_size = self.config.batch_size
        
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        return x, y
    
    def setup_model(self):
        """Setup student and teacher models."""
        config = self.config
        
        if self.master_process:
            print(f"[ChannelPruningTrainer] Loading model: {config.model_name}")
        
        # Import model wrapper based on model type
        if config.model_type == "qwen":
            from model_qwen import QwenSparse
            
            # Load student model
            self.student = QwenSparse.from_pretrained(
                config.model_name,
                override_args={"dropout": 0.0},
                sparselinear_config=None,  # No sparse layers for channel pruning
                is_teacher=False,
            )
            
            # Load teacher model for distillation
            if config.use_distillation:
                self.teacher = QwenSparse.from_pretrained(
                    config.teacher_model or config.model_name,
                    override_args={"dropout": 0.0},
                    sparselinear_config=None,
                    is_teacher=True,
                )
                self.teacher.eval()
                for p in self.teacher.parameters():
                    p.requires_grad = False
            else:
                self.teacher = None
        
        elif config.model_type == "llama":
            from model_llama import LLaMASparse
            
            self.student = LLaMASparse.from_pretrained(
                config.model_name,
                override_args={"dropout": 0.0},
                sparselinear_config=None,
                is_teacher=False,
            )
            
            if config.use_distillation:
                self.teacher = LLaMASparse.from_pretrained(
                    config.teacher_model or config.model_name,
                    override_args={"dropout": 0.0},
                    sparselinear_config=None,
                    is_teacher=True,
                )
                self.teacher.eval()
                for p in self.teacher.parameters():
                    p.requires_grad = False
            else:
                self.teacher = None
        else:
            raise ValueError(f"Unsupported model_type: {config.model_type}")
        
        # Move to device
        self.student = self.student.to(self.device)
        if self.teacher is not None:
            self.teacher = self.teacher.to(self.device)
        
        # DDP wrapping
        if self.ddp:
            self.student = DDP(self.student, device_ids=[self.ddp_local_rank])
        
        # Compile if requested
        if config.compile and hasattr(torch, "compile"):
            if self.master_process:
                print("[ChannelPruningTrainer] Compiling model with torch.compile")
            self.student = torch.compile(self.student)
        
        # Get raw model for pruning operations
        self.raw_student = self.student.module if self.ddp else self.student
        
        if self.master_process:
            num_params = sum(p.numel() for p in self.student.parameters())
            print(f"[ChannelPruningTrainer] Model loaded: {num_params/1e9:.2f}B parameters")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        config = self.config
        
        # Separate parameters for weight decay
        param_dict = {pn: p for pn, p in self.student.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=config.betas,
            fused=True if self.device.type == "cuda" else False
        )
        
        # GradScaler for mixed precision
        self.scaler = torch.amp.GradScaler(enabled=(self.dtype == torch.float16))
        
        if self.master_process:
            print(f"[ChannelPruningTrainer] Optimizer: AdamW, lr={config.learning_rate}")
    
    def get_lr(self, iter_num: int) -> float:
        """Get learning rate with warmup and cosine decay."""
        config = self.config
        
        # Warmup
        if iter_num < config.warmup_iters:
            return config.learning_rate * iter_num / config.warmup_iters
        
        # Cosine decay
        if iter_num > config.lr_decay_iters:
            return config.min_lr
        
        decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    
    def setup_pruning(self):
        """Setup channel pruning components."""
        config = self.config
        
        if self.master_process:
            print("[ChannelPruningTrainer] Setting up channel pruning...")
        
        # Channel mask state
        self.mask_state = ChannelMaskState(
            self.raw_student,
            config,
            device=self.device
        )
        
        # Channel score computer
        self.score_computer = ChannelScoreComputer(
            self.raw_student,
            config,
            device=self.device
        )
        
        # Patch model forward to apply masks
        self.original_forwards, self._lora_bypass_modules = patch_model_mlp_forward(
            self.raw_student,
            self.mask_state,
            config
        )
        
        if self.master_process:
            intermediate_size = get_intermediate_size(self.raw_student, config.model_type)
            num_layers = get_num_layers(self.raw_student, config.model_type)
            keep_k = self.mask_state.get_keep_k(0)
            print(f"  Intermediate size: {intermediate_size}")
            print(f"  Num layers: {num_layers}")
            print(f"  Target keep channels: {keep_k} ({config.ffn_keep_ratio*100:.1f}%)")
    
    def setup_logging(self):
        """Setup logging and wandb."""
        config = self.config
        
        if self.master_process:
            os.makedirs(config.out_dir, exist_ok=True)
            
            # Save config
            config_path = os.path.join(config.out_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(asdict(config), f, indent=2, default=str)
            
            # Setup wandb
            if config.wandb_log:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=asdict(config),
                )
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss with optional distillation.
        
        Args:
            x: Input token IDs [batch, seq_len]
            y: Target token IDs [batch, seq_len]
            
        Returns:
            Dict with loss, task_loss, distill_loss
        """
        config = self.config
        
        # Student forward
        student_logits, task_loss, _ = self.raw_student(x, y)
        
        if config.use_distillation and self.teacher is not None:
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits, _, _ = self.teacher(x, y)
            
            # KL divergence loss
            T = config.distill_temperature
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
            
            distill_loss = F.kl_div(
                student_log_probs,
                teacher_log_probs,
                reduction="batchmean",
                log_target=True
            ) * (T * T)
            
            # Combined loss
            alpha = config.distill_alpha
            loss = alpha * distill_loss + (1 - alpha) * task_loss
            
            return {
                "loss": loss,
                "task_loss": task_loss,
                "distill_loss": distill_loss,
            }
        else:
            return {
                "loss": task_loss,
                "task_loss": task_loss,
                "distill_loss": torch.tensor(0.0, device=self.device),
            }
    
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate loss on train and val splits."""
        self.student.eval()
        out = {}
        
        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.get_batch(split)
                with self.ctx:
                    loss_dict = self.compute_loss(x, y)
                losses[k] = loss_dict["loss"].item()
            out[split] = losses.mean().item()
        
        self.student.train()
        return out
    
    def train(self):
        """Main training loop."""
        config = self.config
        
        if self.master_process:
            print("\n" + "="*60)
            print("Starting Channel Pruning Training")
            print("="*60)
        
        self.student.train()
        
        iter_num = 0
        t0 = time.time()
        
        # Get first batch
        x, y = self.get_batch("train")
        
        while iter_num < config.max_iters:
            # Update learning rate
            lr = self.get_lr(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            
            # Forward pass with gradient accumulation
            for micro_step in range(config.gradient_accumulation_steps):
                with self.ctx:
                    loss_dict = self.compute_loss(x, y)
                    loss = loss_dict["loss"] / config.gradient_accumulation_steps
                
                # Backward
                self.scaler.scale(loss).backward()
                
                # Get next batch
                x, y = self.get_batch("train")
            
            # Gradient clipping
            if config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), config.grad_clip)
            
            # Update Hessian estimates (before optimizer step clears grads)
            if config.importance_metric in ["hessian_obd", "taylor"]:
                self.score_computer.update_all_hessian_ema()
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Update channel masks
            if iter_num % config.mask_update_period == 0 and iter_num >= config.sparsity_warmup_steps:
                all_scores = self.score_computer.compute_all_scores()
                score_tensors = [ls.scores for ls in all_scores]
                self.mask_state.update_all_masks(score_tensors, iter_num)
                
                # Update score EMA
                self.score_computer.update_score_ema(all_scores)
            
            iter_num += 1
            
            # Logging
            if self.master_process and iter_num % config.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                
                lossf = loss_dict["loss"].item()
                task_lossf = loss_dict["task_loss"].item()
                distill_lossf = loss_dict["distill_loss"].item()
                
                sparsity_stats = self.mask_state.get_sparsity_stats()
                
                print(f"iter {iter_num}/{config.max_iters} | "
                      f"loss {lossf:.4f} | task {task_lossf:.4f} | distill {distill_lossf:.4f} | "
                      f"sparsity {sparsity_stats['hard_sparsity']:.3f} | "
                      f"lr {lr:.2e} | dt {dt*1000:.1f}ms")
                
                if config.wandb_log:
                    import wandb
                    wandb.log({
                        "iter": iter_num,
                        "loss": lossf,
                        "task_loss": task_lossf,
                        "distill_loss": distill_lossf,
                        "lr": lr,
                        **sparsity_stats
                    })
            
            # Evaluation
            if iter_num % config.eval_interval == 0:
                losses = self.estimate_loss()
                
                if self.master_process:
                    print(f"[Eval] iter {iter_num}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")
                    
                    if config.wandb_log:
                        import wandb
                        wandb.log({
                            "iter": iter_num,
                            "eval/train_loss": losses["train"],
                            "eval/val_loss": losses["val"],
                        })
            
            # Checkpointing
            if self.master_process and iter_num % config.save_interval == 0:
                self.save_checkpoint(iter_num)
        
        # Final export
        if self.master_process and config.export_pruned:
            self.export_model()
        
        if self.master_process:
            print("\n" + "="*60)
            print("Training Complete!")
            print("="*60)
    
    def save_checkpoint(self, iter_num: int):
        """Save training checkpoint."""
        config = self.config
        ckpt_path = os.path.join(config.out_dir, f"ckpt_{iter_num}.pt")
        
        checkpoint = {
            "iter_num": iter_num,
            "model_state_dict": self.raw_student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(config),
            "masks": {
                i: {
                    "mask": self.mask_state.masks[i].mask.cpu(),
                    "temperature": self.mask_state.masks[i].temperature,
                    "hardening_x": self.mask_state.masks[i].hardening_x,
                }
                for i in range(self.mask_state.num_layers)
            },
            "score_ema": {
                i: self.score_computer.score_ema[i].cpu() 
                for i in self.score_computer.score_ema.keys()
            } if self.score_computer.score_ema else None,
        }
        
        torch.save(checkpoint, ckpt_path)
        print(f"[Checkpoint] Saved to {ckpt_path}")
    
    def export_model(self):
        """Export the pruned model."""
        config = self.config
        
        print("\n[Export] Exporting pruned model...")
        
        # Restore original forwards before export
        restore_model_mlp_forward(self.raw_student, self.original_forwards, config)
        
        # Compute reduction stats
        reduction_stats = compute_param_reduction(
            self.raw_student,
            self.mask_state,
            config
        )
        print(f"[Export] FFN param reduction: {reduction_stats['ffn_param_reduction']*100:.1f}%")
        
        # Export
        export_path = config.export_path or os.path.join(config.out_dir, "pruned_model")
        export_pruned_model(
            self.raw_student,
            self.mask_state,
            config,
            export_path
        )
        
        print(f"[Export] Pruned model saved to {export_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.ddp:
            dist.destroy_process_group()


def main():
    """Main entry point."""
    args = get_args()
    config = args_to_config(args)
    
    trainer = ChannelPruningTrainer(config)
    
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
