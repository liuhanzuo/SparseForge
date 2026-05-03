#!/usr/bin/env bash
###############################################################################
# train_universal.sh — Universal Sparse Trainer (OPT / LLaMA / GPT-2 / Qwen)
###############################################################################
# This script launches sparse training for various model architectures using
# SparseForge. It relies on `scripts/cluster_launcher.sh` for multi-node
# orchestration.
#
# ─── Launch Modes ───────────────────────────────────────────────────────────
#
# 1) Controller mode (recommended for multi-node):
#    Automatically selects nodes from the pool and launches via SSH.
#      bash train_universal.sh <NNODES> <IDX1> ... <IDXN> [-- extra args]
#    Examples:
#      bash train_universal.sh 2 1 3       # 2 nodes: pool indices 1 and 3
#      bash train_universal.sh 1 2         # single node: pool index 2
#      bash train_universal.sh 2 1 3 -- --model_type llama
#
# 2) Legacy manual mode (run on each node separately):
#      bash train_universal.sh <MASTER_IP> <NODE_RANK> <NNODES>
#    Example (on node 0):
#      bash train_universal.sh 10.0.0.1 0 2
#
# 3) Single-node (no arguments):
#      bash train_universal.sh
#
# ─── Key Environment Variables ──────────────────────────────────────────────
#
#   NNODES              Number of nodes (default: 1)
#   NPROC_PER_NODE      GPUs per node (default: 8)
#   MASTER_ADDR         Master node IP (auto-detected in controller mode)
#   MASTER_PORT         Master port (default: 29500)
#   LAUNCH_MODE         "single" (DeepSpeed SSH) or "multi" (torchrun per node)
#   USE_FSDP_FULLY_SHARDED  Set to 1 to use PyTorch FSDP fully-sharded mode
#   WORKDIR             Shared filesystem path (default: script directory)
#   CLUSTER_NODE_IPS    Override node pool IPs (space-separated)
#
# ─── Prerequisites ──────────────────────────────────────────────────────────
#
#   - Passwordless SSH between all nodes (for controller mode)
#   - Shared filesystem accessible from all nodes at the same path
#   - CUDA-capable GPUs with PyTorch + DeepSpeed/FSDP installed
#
###############################################################################
set -euo pipefail
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

# HuggingFace cache directories (using shared storage)
export HF_HOME=${HF_HOME:-"${WORKDIR}/models"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"${WORKDIR}/models"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${WORKDIR}/data/hf_datasets"}
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" >/dev/null 2>&1 || true

# New mode: controller launch by selecting a subset of nodes.
# Usage: bash train_universal.sh <NNODES> <IDX1> ... <IDXN> [-- extra main_universal.py args]
if [[ "${INTERNAL_LAUNCH:-0}" != "1" ]]; then
  if [[ $# -ge 1 ]] && cluster_is_integer "$1"; then
    nnodes="$1"
    shift
    if [[ $# -lt "$nnodes" ]]; then
      echo "Usage: bash $(basename "$0") <NNODES> <IDX1> ... <IDXN> [-- extra main_universal.py args]" >&2
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

echo "Training launched (Universal Sparse Trainer - OPT/LLaMA/GPT-2)."

# Default master (rank0) and worker (rank1) addresses — only used in multi-node mode.
# >>> Replace with your actual node IPs <<<
DEFAULT_MASTER_ADDR=<MASTER_IP>
DEFAULT_WORKER_ADDR=<WORKER_IP>
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
# Force NCCL to use the bonded interface that is reachable between nodes
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}

# CRITICAL: Disable NVLS to prevent "Cuda failure 802 'system not yet initialized'" errors
# on H800/H100 GPUs with NCCL 2.21.x. NVLS (NVLink SHARP) has compatibility issues
# with CUDA context initialization in multi-node environments.
export NCCL_NVLS_ENABLE=0

# Skip config consistency check in multi-node (all_gather_object can be unstable at scale)
export AST_SKIP_CONFIG_CHECK=${AST_SKIP_CONFIG_CHECK:-1}

echo "[trace] RUN_ID=${RUN_ID}"
echo "[trace] AST_TRACE_DIR=${AST_TRACE_DIR}"

# Optional: enable launching with PyTorch FSDP in fully-sharded mode.
# When set to 1 the launcher will prefer `torchrun` and append FSDP flags
# into the training args so `main_universal.py` can wrap the model in FSDP (fully shard).
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

# Legacy positional args: ./train_universal.sh [MASTER_ADDR] [NODE_RANK] [NNODES]
MASTER_ADDR_ARG=${1:-}
NODE_RANK_ARG=${2:-}
NNODES_ARG=${3:-}

NNODES=${NNODES_ARG:-${NNODES:-1}}
NODE_RANK=${NODE_RANK_ARG:-${NODE_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29500}
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
      # Single-node FSDP: use torchrun so main_universal.py can enable FSDP wrapping.
      launch_cmd=(torchrun --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --node_rank=0 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
    else
      launch_cmd=(deepspeed --num_gpus ${NPROC_PER_NODE})
    fi
  fi
else
  export MASTER_ADDR=${MASTER_ADDR_ARG:-${MASTER_ADDR:-$DEFAULT_MASTER_ADDR}}
  if [ "${LAUNCH_MODE}" = "single" ]; then
    if [ "${NODE_RANK}" -ne 0 ]; then
echo "[LAUNCH_MODE=single] This script should only run on the master node (NODE_RANK=0)."
      echo "  Current NODE_RANK=${NODE_RANK}; exiting to avoid duplicate launches."
      exit 0
    fi
    if [ ! -f "${DEEPSPEED_HOSTFILE}" ]; then
      echo "ERROR: DEEPSPEED_HOSTFILE not found: ${DEEPSPEED_HOSTFILE}" >&2
      exit 2
    fi
    if [[ "${USE_FSDP_FULLY_SHARDED}" == "1" ]]; then
      # Controller-driven single-launch but using FSDP: prefer torchrun so the
      # training process can initialize PyTorch FSDP (fully sharded). Note:
      # torchrun will not SSH to workers like DeepSpeed; ensure remote nodes
      # are reachable/started or use cluster launcher to start torchrun on
      # each node when required.
      echo "[INFO] LAUNCH_MODE=single with USE_FSDP_FULLY_SHARDED=1: using torchrun across ${NNODES} nodes"
      launch_cmd=(torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
    else
      launch_cmd=(deepspeed --hostfile "${DEEPSPEED_HOSTFILE}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" --num_gpus ${NPROC_PER_NODE})
    fi
  elif [ "${LAUNCH_MODE}" = "multi" ]; then
    # In multi mode, we launch via torchrun (each node is started by the controller SSH).
    # DeepSpeed runner still expects a hostfile for >1 nodes; torchrun avoids that.
    launch_cmd=(torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port "${MASTER_PORT}")
  else
    echo "ERROR: Unknown LAUNCH_MODE='${LAUNCH_MODE}'. Use 'single' or 'multi'." >&2
    exit 2
  fi
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# =============================================================================
# Model Selection - Modify these to switch between models
# =============================================================================
# Option 1: OPT-2.7B (default)
# STUDENT_MODEL="models/facebook--opt-2.7b"
# TEACHER_MODEL="models/facebook--opt-2.7b"
# MODEL_TYPE="opt"

# Network proxy (optional) — uncomment and modify if needed for model downloads.
# export http_proxy=http://your-proxy-host:port
# export https_proxy=http://your-proxy-host:port
# export all_proxy=http://your-proxy-host:port
# export no_proxy=localhost,127.0.0.1,.local
 
# Option 2: LLaMA-2-7B (uncomment to use)
# STUDENT_MODEL="models/Llama--Llama2-7b"
# TEACHER_MODEL="models/Llama--Llama2-7b"
# MODEL_TYPE="llama"

# Option 3: GPT-2-XL (uncomment to use)
# STUDENT_MODEL="models/openai-community--gpt2-medium"
# TEACHER_MODEL="models/openai-community--gpt2-medium"
# MODEL_TYPE="gpt2"
STUDENT_MODEL="models/Qwen--Qwen3-1.7b"
TEACHER_MODEL="models/Qwen--Qwen3-1.7b"
MODEL_TYPE="qwen"
# STUDENT_MODEL="models/deepseek-ai--deepseek-moe-16b-base"
# TEACHER_MODEL="models/deepseek-ai--deepseek-moe-16b-base"
# MODEL_TYPE="deepseek_moe"
MASK_TYPE="nm_2_4"
# STUDENT_MODEL="models/openai-community--gpt2-xl"
# TEACHER_MODEL="models/openai-community--gpt2-xl"
# MODEL_TYPE="gpt2"
  args=(
    --dataset c4_${MODEL_TYPE} \
    --student_model ${STUDENT_MODEL} \
    --teacher_model ${TEACHER_MODEL} \
    --model_type ${MODEL_TYPE} \
    \
    --srste_decay 1e-4 \
    --learning_rate 1e-4 \
    --min_lr 1e-5 \
    --warmup_iters 1000 \
    --lr_decay_iters 16000 \
    --max_iters 17000 \
    --dtype auto \
    --batch_size 2 \
    --global_batch_size 64 \
    --eval_interval 200 \
    --mode sparse_forward \
    --mask_type unstructured \
    --hard_mask_type ${MASK_TYPE} \
    --structured_exact False \
    --mask_metric hessian_obd \
    --structured_n 2 \
    --structured_m 4 \
    --change_mask True \
    --sparsity_ratio 0.5 \
    --beta 0.98 \
    --enable_hutchinson True \
    --mask_update_period_before 30 \
    --mask_update_period_after 10 \
    --mask_update_switch_step 2000 \
    --mask_lr 0.1 \
    --score_ema_beta 0.99 \
    --temp_init 1.0 \
    --temp_decay 0.98 \
    --temp_min 0.05 \
    --sparsity_warmup_steps 500 \
    --tau_sample_size 262144 \
    \
    --wandb_logging True \
    --wandb_run_name "${MODEL_TYPE}-AST-${MASK_TYPE}-hutchinson" \
    --wandb_project "AST-Pruning" \
    \
    --SLoRB True \
    --SLoRB_init_type sum \
    --SLoRB_k 16 \
    --trainable_projection True \
    \
    --distill_model True \
    --hardness_task 1.0 \
    --hardness_kldiv 1.0 \
    --hardness_squarehead 0.0 \
    \
    --gradient_checkpointing False \
    \
    --hardening_period 0 \
    --hardening_fraction 0 \
    --lambda_mid_max 0.3 \
    --mask_penalty_lr 1.0 \
    --mask_penalty_lr_min 0.2 \
    --mask_penalty_lr_schedule linear \
    --mask_penalty_mode ${MASK_TYPE} \
    --sparsity_alpha 0.2 \
    --freeze_low 0 \
    --freeze_high 1 \
    --mask_hardening_start 2000 \
    --mask_hardening_duration 15000 \
    --final_finetune_iters 3000 \
    --use_fsdp True \
    --fsdp_mode hybrid_sharded \
    --fsdp_cpu_offload False \
    --skip_wiki_ppl False \
    --beta_structural_start 2000 \
    --beta_structural_end 17000 \
    --glu_joint_mask False \
    --eager_attention False \
    # --finalize_lm_eval True \
    # --lm_eval_tasks "boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
    # --lm_eval_batch_size 64 \
  )
  # Uncomment below to enable DeepSpeed:
  # args+=(--use_deepspeed --deepspeed_config configs/deepspeed_xl.json)
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
  # Only add non-empty arguments to avoid "unrecognized arguments" error
  for var in "$@"; do
    if [[ -n "$var" ]]; then
      args+=("$var")
    fi
  done


if [ "${NNODES}" -gt 1 ]; then
  echo "Multi-node: NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} LAUNCH_MODE=${LAUNCH_MODE}"
  if [ "${LAUNCH_MODE}" = "single" ]; then
    echo "Hostfile: ${DEEPSPEED_HOSTFILE}"
  fi
fi

  # detect launcher name for clearer logging
  launcher_name="${launch_cmd[0]}"
  echo "Launching with ${launcher_name}"
  echo "Model: ${STUDENT_MODEL} (type: ${MODEL_TYPE})"
"${launch_cmd[@]}" legacy/main_universal.py "${args[@]}"
