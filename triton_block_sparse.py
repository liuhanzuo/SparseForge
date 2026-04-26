#!/usr/bin/env python3
"""
Triton Block-16 Sparse Matrix Multiplication

这是一个针对 block-16 稀疏模式优化的 Triton kernel，可以直接集成到你的 AST 框架中。

关键优化：
1. Block-level mask 检查，跳过全零 block（真正的计算节省）
2. 使用 CSR-like 格式存储非零 block 索引
3. Tensor Core 友好的 tile 大小

Usage:
    from triton_block_sparse import BlockSparseLinear, convert_to_block_sparse
    
    # 转换稀疏权重
    sparse_weight, block_indices = convert_to_block_sparse(weight, mask, block_size=16)
    
    # 创建稀疏线性层
    layer = BlockSparseLinear(sparse_weight, block_indices, bias)
    
    # Forward（有加速）
    output = layer(input)

Author: AST Framework
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

# 尝试导入 Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[WARN] Triton not installed. Block-sparse acceleration disabled.")
    print("       Install with: pip install triton")


# =============================================================================
# Block-Sparse Data Structure
# =============================================================================
def convert_to_block_sparse(
    weight: torch.Tensor,
    mask: torch.Tensor,
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 dense weight + block mask 转换为压缩的 block-sparse 格式
    
    Args:
        weight: (out_features, in_features) dense 权重
        mask: (out_features, in_features) binary mask (block-16 aligned)
        block_size: block 大小，默认 16
    
    Returns:
        sparse_blocks: (num_nonzero_blocks, block_size, block_size) 非零 block 数据
        block_row_indices: (num_block_rows + 1,) CSR 格式的 row pointer
        block_col_indices: (num_nonzero_blocks,) 每个非零 block 的列索引
    """
    out_features, in_features = weight.shape
    assert out_features % block_size == 0 and in_features % block_size == 0
    
    num_block_rows = out_features // block_size
    num_block_cols = in_features // block_size
    
    # Apply mask
    sparse_weight = weight * mask
    
    # 重塑为 block 形式
    blocked = sparse_weight.view(
        num_block_rows, block_size, num_block_cols, block_size
    ).permute(0, 2, 1, 3)  # (num_block_rows, num_block_cols, block_size, block_size)
    
    # 找出非零 block（block 中有任意非零元素即为非零 block）
    block_norms = blocked.abs().sum(dim=(-1, -2))  # (num_block_rows, num_block_cols)
    
    # 构建 CSR 格式
    sparse_blocks = []
    block_row_indices = [0]
    block_col_indices = []
    
    for row_idx in range(num_block_rows):
        nonzero_cols = torch.where(block_norms[row_idx] > 0)[0]
        for col_idx in nonzero_cols:
            sparse_blocks.append(blocked[row_idx, col_idx])
            block_col_indices.append(col_idx.item())
        block_row_indices.append(len(sparse_blocks))
    
    if len(sparse_blocks) == 0:
        # 全零矩阵
        sparse_blocks = torch.zeros(1, block_size, block_size, 
                                     device=weight.device, dtype=weight.dtype)
    else:
        sparse_blocks = torch.stack(sparse_blocks)
    
    block_row_indices = torch.tensor(block_row_indices, device=weight.device, dtype=torch.int32)
    block_col_indices = torch.tensor(block_col_indices, device=weight.device, dtype=torch.int32)
    
    return sparse_blocks, block_row_indices, block_col_indices


def get_sparsity_info(block_row_indices: torch.Tensor, num_block_cols: int) -> dict:
    """获取稀疏度统计信息"""
    num_block_rows = len(block_row_indices) - 1
    num_nonzero_blocks = block_row_indices[-1].item()
    total_blocks = num_block_rows * num_block_cols
    block_sparsity = 1 - num_nonzero_blocks / total_blocks
    
    return {
        "num_block_rows": num_block_rows,
        "num_block_cols": num_block_cols,
        "num_nonzero_blocks": num_nonzero_blocks,
        "total_blocks": total_blocks,
        "block_sparsity": block_sparsity,
        "theoretical_speedup": 1 / (1 - block_sparsity) if block_sparsity < 1 else float('inf'),
    }


# =============================================================================
# Triton Kernels
# =============================================================================
if HAS_TRITON:
    
    @triton.jit
    def block_sparse_matmul_kernel(
        # Input tensor
        x_ptr,
        # Sparse weight in CSR-like block format
        sparse_blocks_ptr,
        block_row_indices_ptr,
        block_col_indices_ptr,
        # Output tensor
        output_ptr,
        # Dimensions
        batch_size,
        in_features,
        out_features,
        num_nonzero_blocks,
        # Block configuration
        BLOCK_SIZE: tl.constexpr,
        BATCH_BLOCK: tl.constexpr,
    ):
        """
        Block-sparse matrix multiplication: output = x @ W^T
        
        Where W is stored in CSR-like block format:
        - sparse_blocks: (num_nonzero_blocks, BLOCK_SIZE, BLOCK_SIZE)
        - block_row_indices: CSR row pointers
        - block_col_indices: column index for each block
        
        Each program handles one output block row and one batch block.
        """
        # Program IDs
        pid_batch = tl.program_id(0)  # Which batch block
        pid_out_row = tl.program_id(1)  # Which output block row (corresponds to weight row)
        
        # Batch offsets
        batch_start = pid_batch * BATCH_BLOCK
        batch_offs = batch_start + tl.arange(0, BATCH_BLOCK)
        batch_mask = batch_offs < batch_size
        
        # Output row offsets (BLOCK_SIZE rows per block)
        out_row_start = pid_out_row * BLOCK_SIZE
        out_row_offs = out_row_start + tl.arange(0, BLOCK_SIZE)
        out_row_mask = out_row_offs < out_features
        
        # Load CSR row pointers for this output block row
        row_start = tl.load(block_row_indices_ptr + pid_out_row)
        row_end = tl.load(block_row_indices_ptr + pid_out_row + 1)
        
        # Initialize accumulator: (BATCH_BLOCK, BLOCK_SIZE)
        acc = tl.zeros((BATCH_BLOCK, BLOCK_SIZE), dtype=tl.float32)
        
        # Iterate over non-zero blocks in this row
        for block_idx in range(row_start, row_end):
            # Get column index for this block
            col_idx = tl.load(block_col_indices_ptr + block_idx)
            
            # Input column offsets
            in_col_start = col_idx * BLOCK_SIZE
            in_col_offs = in_col_start + tl.arange(0, BLOCK_SIZE)
            
            # Load input block: (BATCH_BLOCK, BLOCK_SIZE)
            # x_ptr shape: (batch_size, in_features)
            x_ptrs = x_ptr + batch_offs[:, None] * in_features + in_col_offs[None, :]
            x_block = tl.load(x_ptrs, mask=batch_mask[:, None], other=0.0)
            
            # Load weight block: (BLOCK_SIZE, BLOCK_SIZE)
            # Note: weight is stored as (out_features, in_features), so block is (BLOCK_SIZE, BLOCK_SIZE)
            # For x @ W^T, we need W^T which is (in_features, out_features)
            # So we load W block and transpose implicitly
            block_ptr_start = block_idx * BLOCK_SIZE * BLOCK_SIZE
            w_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
            for i in range(BLOCK_SIZE):
                w_row_ptrs = sparse_blocks_ptr + block_ptr_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                w_row = tl.load(w_row_ptrs)
                # w_block[i, :] = w_row, but we need to transpose for x @ W^T
                # So we accumulate: acc += x_block @ w_block^T
                # Which is: acc[:, i] += x_block @ w_row (dot product)
                w_row_expanded = w_row[None, :]  # (1, BLOCK_SIZE)
                x_w_prod = tl.sum(x_block * w_row_expanded, axis=1)  # (BATCH_BLOCK,)
                # This gives us contribution to output column i within this block
                # We need to scatter this properly
            
            # Simpler approach: load full block and do block matmul
            # Load W block as contiguous (BLOCK_SIZE, BLOCK_SIZE)
            w_ptrs = sparse_blocks_ptr + block_ptr_start + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
            w_block = tl.load(w_ptrs).to(tl.float32)
            
            # acc += x_block @ W^T = x_block @ W.T
            # x_block: (BATCH_BLOCK, BLOCK_SIZE), W: (BLOCK_SIZE, BLOCK_SIZE)
            # Result: (BATCH_BLOCK, BLOCK_SIZE)
            acc += tl.dot(x_block.to(tl.float32), tl.trans(w_block))
        
        # Store output
        out_ptrs = output_ptr + batch_offs[:, None] * out_features + out_row_offs[None, :]
        tl.store(out_ptrs, acc.to(tl.float16), mask=batch_mask[:, None] & out_row_mask[None, :])
    
    
    @triton.jit  
    def block_sparse_matmul_v2_kernel(
        # Inputs
        x_ptr,
        sparse_blocks_ptr,
        block_row_indices_ptr,
        block_col_indices_ptr,
        output_ptr,
        # Dimensions
        batch_size,
        in_features,
        out_features,
        # Block config
        BLOCK_SIZE: tl.constexpr,
        BATCH_TILE: tl.constexpr,
    ):
        """
        优化版本：使用更高效的 tiling 策略
        """
        pid_batch = tl.program_id(0)
        pid_out = tl.program_id(1)
        
        # Batch range
        batch_start = pid_batch * BATCH_TILE
        batch_range = batch_start + tl.arange(0, BATCH_TILE)
        batch_valid = batch_range < batch_size
        
        # Output range
        out_start = pid_out * BLOCK_SIZE
        out_range = out_start + tl.arange(0, BLOCK_SIZE)
        out_valid = out_range < out_features
        
        # Get non-zero block range for this output block
        nnz_start = tl.load(block_row_indices_ptr + pid_out)
        nnz_end = tl.load(block_row_indices_ptr + pid_out + 1)
        
        # Accumulator
        acc = tl.zeros((BATCH_TILE, BLOCK_SIZE), dtype=tl.float32)
        
        # Process each non-zero block
        for nnz_idx in range(nnz_start, nnz_end):
            # Column index
            col_blk = tl.load(block_col_indices_ptr + nnz_idx)
            col_start = col_blk * BLOCK_SIZE
            col_range = col_start + tl.arange(0, BLOCK_SIZE)
            
            # Load X tile: (BATCH_TILE, BLOCK_SIZE)
            x_tile_ptrs = x_ptr + batch_range[:, None] * in_features + col_range[None, :]
            x_tile = tl.load(x_tile_ptrs, mask=batch_valid[:, None], other=0.0).to(tl.float32)
            
            # Load W block: (BLOCK_SIZE, BLOCK_SIZE)
            w_base = nnz_idx * BLOCK_SIZE * BLOCK_SIZE
            w_ptrs = sparse_blocks_ptr + w_base + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
            w_tile = tl.load(w_ptrs).to(tl.float32)
            
            # Y += X @ W^T
            acc += tl.dot(x_tile, tl.trans(w_tile))
        
        # Store result
        out_ptrs = output_ptr + batch_range[:, None] * out_features + out_range[None, :]
        tl.store(out_ptrs, acc.to(tl.float16), mask=batch_valid[:, None] & out_valid[None, :])


# =============================================================================
# PyTorch Wrapper
# =============================================================================
class BlockSparseLinear(nn.Module):
    """
    Block-sparse linear layer with Triton acceleration
    
    Usage:
        # 从 dense 权重和 mask 创建
        layer = BlockSparseLinear.from_dense(weight, mask, bias, block_size=16)
        
        # 或直接从预处理的稀疏数据创建
        layer = BlockSparseLinear(sparse_blocks, block_row_indices, block_col_indices, 
                                   in_features, out_features, bias)
        
        # Forward
        output = layer(input)
    """
    
    def __init__(
        self,
        sparse_blocks: torch.Tensor,
        block_row_indices: torch.Tensor,
        block_col_indices: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor] = None,
        block_size: int = 16,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # 注册为 buffer（不参与梯度计算）
        self.register_buffer('sparse_blocks', sparse_blocks.contiguous())
        self.register_buffer('block_row_indices', block_row_indices.contiguous())
        self.register_buffer('block_col_indices', block_col_indices.contiguous())
        
        if bias is not None:
            self.register_buffer('bias', bias.contiguous())
        else:
            self.bias = None
        
        # 统计信息
        num_block_cols = in_features // block_size
        self.sparsity_info = get_sparsity_info(block_row_indices, num_block_cols)
    
    @classmethod
    def from_dense(
        cls,
        weight: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        block_size: int = 16,
    ) -> 'BlockSparseLinear':
        """从 dense 权重和 mask 创建 BlockSparseLinear"""
        out_features, in_features = weight.shape
        sparse_blocks, block_row_indices, block_col_indices = convert_to_block_sparse(
            weight, mask, block_size
        )
        return cls(
            sparse_blocks.to(torch.float16),
            block_row_indices,
            block_col_indices,
            in_features,
            out_features,
            bias.to(torch.float16) if bias is not None else None,
            block_size,
        )
    
    @classmethod
    def from_sparse_linear(cls, sparse_linear_module, block_size: int = 16) -> 'BlockSparseLinear':
        """从你的 SparseLinear 模块转换"""
        weight = sparse_linear_module.weight.data
        mask = sparse_linear_module.hard_mask if hasattr(sparse_linear_module, 'hard_mask') else \
               sparse_linear_module.mask.data
        bias = sparse_linear_module.bias.data if sparse_linear_module.bias is not None else None
        return cls.from_dense(weight, mask, bias, block_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with block-sparse acceleration
        
        Args:
            x: (batch, seq_len, in_features) or (batch, in_features)
        
        Returns:
            output: same shape as input but last dim is out_features
        """
        original_shape = x.shape
        
        # Flatten to 2D
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x = x.view(batch * seq_len, self.in_features)
        
        batch_size = x.shape[0]
        
        # 选择实现
        if HAS_TRITON and x.is_cuda:
            output = self._triton_forward(x)
        else:
            output = self._torch_forward(x)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        # Restore shape
        if len(original_shape) == 3:
            output = output.view(batch, seq_len, self.out_features)
        
        return output
    
    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Triton accelerated forward"""
        batch_size = x.shape[0]
        
        # Allocate output
        output = torch.zeros(batch_size, self.out_features, device=x.device, dtype=torch.float16)
        
        # Grid configuration
        num_out_blocks = self.out_features // self.block_size
        BATCH_TILE = 32  # 可调参数
        
        grid = (
            triton.cdiv(batch_size, BATCH_TILE),
            num_out_blocks,
        )
        
        # Launch kernel
        block_sparse_matmul_v2_kernel[grid](
            x,
            self.sparse_blocks,
            self.block_row_indices,
            self.block_col_indices,
            output,
            batch_size,
            self.in_features,
            self.out_features,
            BLOCK_SIZE=self.block_size,
            BATCH_TILE=BATCH_TILE,
        )
        
        return output
    
    def _torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: 使用 PyTorch 实现（无加速，用于 CPU 或验证）"""
        # 重建 dense weight（仅用于验证，生产环境应避免）
        num_block_rows = self.out_features // self.block_size
        num_block_cols = self.in_features // self.block_size
        
        dense_weight = torch.zeros(
            self.out_features, self.in_features,
            device=x.device, dtype=self.sparse_blocks.dtype
        )
        
        block_idx = 0
        for row_idx in range(num_block_rows):
            row_start = self.block_row_indices[row_idx].item()
            row_end = self.block_row_indices[row_idx + 1].item()
            
            for nnz_idx in range(row_start, row_end):
                col_idx = self.block_col_indices[nnz_idx].item()
                block_data = self.sparse_blocks[nnz_idx]
                
                out_start = row_idx * self.block_size
                out_end = out_start + self.block_size
                in_start = col_idx * self.block_size
                in_end = in_start + self.block_size
                
                dense_weight[out_start:out_end, in_start:in_end] = block_data
        
        return torch.mm(x.to(dense_weight.dtype), dense_weight.t())
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'block_size={self.block_size}, '
            f'sparsity={self.sparsity_info["block_sparsity"]:.1%}, '
            f'nnz_blocks={self.sparsity_info["num_nonzero_blocks"]}'
        )


# =============================================================================
# Integration with AST Framework
# =============================================================================
def convert_model_to_block_sparse(
    model: nn.Module,
    block_size: int = 16,
    target_modules: Optional[list] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    将模型中的 SparseLinear 层转换为 BlockSparseLinear
    
    Args:
        model: 包含 SparseLinear 层的模型
        block_size: block 大小
        target_modules: 要转换的模块名（None 表示全部）
        verbose: 是否打印转换信息
    
    Returns:
        转换后的模型（in-place 修改）
    """
    converted = 0
    
    for name, module in model.named_modules():
        # 检查是否是你的 SparseLinear
        if hasattr(module, 'hard_mask') or (hasattr(module, 'mask') and hasattr(module, 'weight')):
            if target_modules is not None and not any(t in name for t in target_modules):
                continue
            
            try:
                # 获取父模块
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                # 转换
                sparse_linear = BlockSparseLinear.from_sparse_linear(module, block_size)
                setattr(parent, attr_name, sparse_linear)
                
                if verbose:
                    info = sparse_linear.sparsity_info
                    print(f"[convert] {name}: sparsity={info['block_sparsity']:.1%}, "
                          f"nnz_blocks={info['num_nonzero_blocks']}/{info['total_blocks']}")
                
                converted += 1
            except Exception as e:
                if verbose:
                    print(f"[convert] {name}: FAILED - {e}")
    
    if verbose:
        print(f"[convert] Total converted: {converted} layers")
    
    return model


# =============================================================================
# Benchmark
# =============================================================================
def benchmark_block_sparse_linear(
    in_features: int = 4096,
    out_features: int = 4096,
    batch_size: int = 32,
    sparsity: float = 0.5,
    block_size: int = 16,
    warmup: int = 10,
    iterations: int = 100,
):
    """Benchmark BlockSparseLinear vs dense Linear"""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    print(f"\n{'='*60}")
    print(f"Benchmark: BlockSparseLinear vs Dense")
    print(f"{'='*60}")
    print(f"  Shape: ({batch_size}, {in_features}) x ({out_features}, {in_features})^T")
    print(f"  Sparsity: {sparsity*100:.0f}%")
    print(f"  Block size: {block_size}")
    print(f"  Device: {device}")
    
    # Create test data
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
    
    # Create block mask
    num_block_rows = out_features // block_size
    num_block_cols = in_features // block_size
    block_mask = (torch.rand(num_block_rows, num_block_cols, device=device) > sparsity)
    mask = block_mask.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1).float()
    
    # Dense linear
    dense_linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
    dense_linear.weight.data = weight * mask
    
    # Block sparse linear
    sparse_linear = BlockSparseLinear.from_dense(weight, mask, block_size=block_size)
    sparse_linear = sparse_linear.to(device)
    
    print(f"\n  Sparse info: {sparse_linear.sparsity_info}")
    
    # Warmup
    print("\n  Warming up...")
    for _ in range(warmup):
        _ = dense_linear(x)
        _ = sparse_linear(x)
    torch.cuda.synchronize()
    
    # Benchmark dense
    print("  Benchmarking dense...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = dense_linear(x)
    torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) / iterations * 1000
    
    # Benchmark sparse
    print("  Benchmarking sparse...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = sparse_linear(x)
    torch.cuda.synchronize()
    sparse_time = (time.perf_counter() - start) / iterations * 1000
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Dense:  {dense_time:.3f} ms")
    print(f"  Sparse: {sparse_time:.3f} ms")
    print(f"  Speedup: {dense_time/sparse_time:.2f}x")
    print(f"  Theoretical speedup (from sparsity): {sparse_linear.sparsity_info['theoretical_speedup']:.2f}x")
    
    # Verify correctness
    with torch.no_grad():
        dense_out = dense_linear(x)
        sparse_out = sparse_linear(x)
        diff = (dense_out - sparse_out).abs().max().item()
        print(f"\n  Max difference: {diff:.6f}")
        print(f"  Outputs match: {'✓' if diff < 1e-2 else '✗'}")
    
    return dense_time, sparse_time


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        exit(1)
    
    # 运行 benchmark
    benchmark_block_sparse_linear(
        in_features=4096,
        out_features=4096,
        batch_size=64,
        sparsity=0.5,
        block_size=16,
    )
