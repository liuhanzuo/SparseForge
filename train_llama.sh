#!/usr/bin/env bash
set -euo pipefail

# ========================================
# 网络代理配置（可选）
# 如果需要通过代理访问外网，取消以下注释并修改为你的代理地址
# ========================================
# export http_proxy=http://your-proxy-host:port
# export https_proxy=http://your-proxy-host:port
# export HTTP_PROXY=http://your-proxy-host:port
# export HTTPS_PROXY=http://your-proxy-host:port
# export no_proxy=localhost,127.0.0.1,.local,.internal

# ========================================
# HuggingFace 镜像配置（可选）
# 如果在中国网络环境，使用 HF 镜像加速模型下载
# ========================================
# export HF_ENDPOINT=https://hf-mirror.com
# ========================================

export CLUSTER_ENV_SETUP='source ~/.bashrc && conda activate minillm &&'
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/cluster_launcher.sh"
SCRIPT_PATH=$(realpath "$0")

# Ensure the launcher uses the correct repo directory on all nodes (shared FS assumed).
export WORKDIR=${WORKDIR:-"${SCRIPT_DIR}"}

# Best-effort per-run trace directory for capturing per-rank Python tracebacks.
# Controller mode (cluster_launcher) will set RUN_ID/AST_TRACE_DIR explicitly.
if [[ -z "${RUN_ID:-}" ]]; then
  export RUN_ID="$(date +%Y%m%d_%H%M%S).$$"
fi
if [[ -z "${AST_TRACE_DIR:-}" ]]; then
  export AST_TRACE_DIR="${WORKDIR}/outputs/ast_trace/${RUN_ID}"
fi
mkdir -p "${AST_TRACE_DIR}" >/dev/null 2>&1 || true

# New mode: controller launch by selecting a subset of nodes.
# Usage: bash train_llama.sh <NNODES> <IDX1> ... <IDXN> [-- extra main_llama.py args]
if [[ "${INTERNAL_LAUNCH:-0}" != "1" ]]; then
  if [[ $# -ge 1 ]] && cluster_is_integer "$1"; then
    nnodes="$1"
    shift
    if [[ $# -lt "$nnodes" ]]; then
      echo "Usage: bash $(basename "$0") <NNODES> <IDX1> ... <IDXN> [-- extra main_llama.py args]" >&2
      exit 2
    fi
    idxs=()
    for ((i=0; i<nnodes; i++)); do
      idxs+=("$1")
      shift
    done
    cluster_launch_selected_nodes "$SCRIPT_PATH" "$nnodes" "${idxs[@]}" "$@"
    exit $?
  fi
fi

echo "Training launched (LLaMA-2-7B, continuous mask, Hutchinson Hessian)."

# 默认 master 地址（rank0）和备节点地址（rank1），仅在多机模式下使用
DEFAULT_MASTER_ADDR=<MASTER_IP>
DEFAULT_WORKER_ADDR=<WORKER_IP>
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_DISABLE=1
# Force NCCL to use the bonded interface that is reachable between nodes
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}

# CRITICAL: Disable NVLS to prevent "Cuda failure 802 'system not yet initialized'" errors
# on H800/H100 GPUs with NCCL 2.21.x. NVLS (NVLink SHARP) has compatibility issues
# with CUDA context initialization in multi-node environments.
export NCCL_NVLS_ENABLE=0

echo "[trace] RUN_ID=${RUN_ID}"
echo "[trace] AST_TRACE_DIR=${AST_TRACE_DIR}"

# Optional: enable launching with PyTorch FSDP in fully-sharded mode.
# When set to 1 the launcher will prefer `torchrun` and append FSDP flags
# into the training args so `main_llama.py` can wrap the model in FSDP (fully shard).
export USE_FSDP_FULLY_SHARDED=${USE_FSDP_FULLY_SHARDED:-0}

# Hostfile for DeepSpeed multi-node (used in LAUNCH_MODE=single)
export DEEPSPEED_HOSTFILE=${DEEPSPEED_HOSTFILE:-configs/hosts.txt}
if [ -f "${DEEPSPEED_HOSTFILE}" ]; then
  export DEEPSPEED_HOSTFILE=$(realpath "${DEEPSPEED_HOSTFILE}")
fi

# Multi-node launch mode:
# - single (default): run ONLY on master node; DeepSpeed will ssh to workers using hostfile.
# - multi: run on EACH node; specify NODE_RANK explicitly; DeepSpeed will NOT ssh.
LAUNCH_MODE=${LAUNCH_MODE:-single}

# 启动参数： ./train_llama.sh [MASTER_ADDR] [NODE_RANK] [NNODES]
# 注意：只有当第一个参数看起来像 IP 地址时才解析为 MASTER_ADDR
# 否则认为是直接传递给 main_llama.py 的参数
MASTER_ADDR_ARG=""
NODE_RANK_ARG=""
NNODES_ARG=""
if [[ $# -ge 1 ]] && cluster_is_ip_like "${1:-}"; then
  MASTER_ADDR_ARG=${1:-}
  NODE_RANK_ARG=${2:-}
  NNODES_ARG=${3:-}
fi

NNODES=${NNODES_ARG:-${NNODES:-1}}
NODE_RANK=${NODE_RANK_ARG:-${NODE_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29600}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Remote worker mode (invoked by controller): rely on env vars for launcher settings.
if [[ "${INTERNAL_LAUNCH:-0}" == "1" ]]; then
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  export MASTER_PORT
fi

if [ "${NNODES}" -eq 1 ]; then
  export MASTER_ADDR=${MASTER_ADDR_ARG:-${MASTER_ADDR:-127.0.0.1}}
  # single-node: by default use DeepSpeed runner. Set USE_TORCHRUN_SINGLE_NODE=1 to use torchrun instead.
  if [[ "${USE_TORCHRUN_SINGLE_NODE:-0}" == "1" ]]; then
    launch_cmd=(torchrun --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --node_rank=0 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
  else
    if [[ "${USE_FSDP_FULLY_SHARDED}" == "1" ]]; then
      # Single-node FSDP: use torchrun so main_llama.py can enable FSDP wrapping.
      launch_cmd=(torchrun --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --node_rank=0 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
    else
      launch_cmd=(deepspeed --num_gpus ${NPROC_PER_NODE})
    fi
  fi
else
  export MASTER_ADDR=${MASTER_ADDR_ARG:-${MASTER_ADDR:-$DEFAULT_MASTER_ADDR}}
  if [ "${LAUNCH_MODE}" = "single" ]; then
    if [ "${NODE_RANK}" -ne 0 ]; then
      echo "[LAUNCH_MODE=single] 请只在 master 节点运行本脚本 (NODE_RANK=0)。"
      echo "  当前 NODE_RANK=${NODE_RANK}，将直接退出以避免重复/冲突拉起。"
      exit 0
    fi
    if [ ! -f "${DEEPSPEED_HOSTFILE}" ]; then
      echo "ERROR: DEEPSPEED_HOSTFILE not found: ${DEEPSPEED_HOSTFILE}" >&2
      exit 2
    fi
    if [[ "${USE_FSDP_FULLY_SHARDED}" == "1" ]]; then
      # Controller-driven single-launch but using FSDP: prefer torchrun so the
      # training process can initialize PyTorch FSDP (fully sharded).
      echo "[INFO] LAUNCH_MODE=single with USE_FSDP_FULLY_SHARDED=1: using torchrun across ${NNODES} nodes"
      launch_cmd=(torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
    else
      launch_cmd=(deepspeed --hostfile "${DEEPSPEED_HOSTFILE}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" --num_gpus ${NPROC_PER_NODE})
    fi
  elif [ "${LAUNCH_MODE}" = "multi" ]; then
    # In multi mode, we launch via torchrun (each node is started by the controller SSH).
    launch_cmd=(torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port "${MASTER_PORT}")
  else
    echo "ERROR: Unknown LAUNCH_MODE='${LAUNCH_MODE}'. Use 'single' or 'multi'." >&2
    exit 2
  fi
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

  # build argument array
  # === Core Training ===
  args=(
    --student_model NousResearch/Llama-2-7b-hf
    --teacher_model NousResearch/Llama-2-7b-hf
    --learning_rate 1e-4
    --min_lr 1e-5
    --warmup_iters 1000
    --lr_decay_iters 20000
    --max_iters 20000
    --batch_size 1
    --global_batch_size 64
    --dtype auto
  )
  # === Mask & Sparsity ===
  args+=(
    --mode sparse_forward
    --mask_type unstructured
    --hard_mask_type block16
    --mask_metric hessian_ratio
    --sparsity_ratio 0.5
    --change_mask True
    --beta 0.98
  )
  # === Hutchinson Hessian ===
  args+=(
    --enable_hutchinson True
    --mask_update_period_before 30
    --mask_update_period_after 10
    --mask_update_switch_step 2000
    --mask_lr 0.1
    --score_ema_beta 0.99
  )
  # === Temperature Scheduling ===
  args+=(
    --temp_init 1.0
    --temp_decay 0.97
    --temp_min 0.05
  )
  # === Structured Pruning ===
  args+=(
    --structured_exact False
    --structured_n 2
    --structured_m 4
  )
  # === Sparsity Penalties ===
  args+=(
    --sparsity_warmup_steps 500
    --tau_sample_size 262144
    --mask_penalty_lr 0.1
    --mask_penalty_mode block16
    --sparsity_alpha 0.2
    --lambda_mid_max 0.1
  )
  # === Hardening ===
  args+=(
    --mask_hardening_start 2000
    --mask_hardening_duration 15000
    --hardening_period 0
    --hardening_fraction 0
    --freeze_low 0.0
    --freeze_high 1.0
    --final_finetune_iters 13000
  )
  # === SLoRB ===
  args+=(
    --SLoRB True
    --SLoRB_init_type sum
    --SLoRB_k 64
    --trainable_projection True
  )
  # === Distillation ===
  args+=(
    --distill_model True
    --hardness_task 1.0
    --hardness_kldiv 1.0
    --hardness_squarehead 0.0
  )
  # === Other ===
  args+=(
    --gradient_checkpointing True
    --eval_interval 200
    --srste_decay 0
    --wandb_logging True
    --wandb_run_name "llama-2-7b-AST-block16-hutchinson"
    --wandb_project "AST-Pruning"
    --dataset c4_llama
    --out_dir out_llama
    --use_fsdp True
    --fsdp_mode hybrid_shard
    --fsdp_cpu_offload True
  )

# Debug: Print the actual command that will be executed
if [[ "${DEBUG_LAUNCH:-0}" == "1" ]]; then
  echo "[DEBUG] Command: ${launch_cmd[@]}"
  echo "[DEBUG] Main script: main_llama.py"
  echo "[DEBUG] Total args: ${#args[@]}"
  echo "[DEBUG] Args: ${args[@]}"
fi

# append any user-supplied args
# - INTERNAL_LAUNCH=1: treat all positionals as extra args
# - legacy mode: if first positional is IP-like, consume up to 3 launcher positionals
if [[ "${INTERNAL_LAUNCH:-0}" != "1" ]]; then
  consume=0
  if [[ $# -ge 1 ]] && cluster_is_ip_like "$1"; then
    consume=$#
    if [ $consume -gt 3 ]; then
      consume=3
    fi
  fi
  if [ $consume -gt 0 ]; then
    shift "$consume"
  fi
fi
# 跳过 -- 分隔符（如果存在），将后续参数传递给 main_llama.py
if [[ $# -ge 1 ]] && [[ "$1" == "--" ]]; then
  shift
fi
for var in "$@"; do
  args+=("$var")
done

# Filter out empty arguments to prevent "unrecognized arguments" errors
clean_args=()
for arg in "${args[@]}"; do
  if [ -n "$arg" ]; then
    clean_args+=("$arg")
  fi
done
args=("${clean_args[@]}")

if [ "${NNODES}" -gt 1 ]; then
  echo "Multi-node: NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} LAUNCH_MODE=${LAUNCH_MODE}"
  if [ "${LAUNCH_MODE}" = "single" ]; then
    echo "Hostfile: ${DEEPSPEED_HOSTFILE}"
  fi
fi

# detect launcher name for clearer logging
launcher_name="${launch_cmd[0]}"
echo "Launching with ${launcher_name}"
"${launch_cmd[@]}" main_llama.py "${args[@]}"
