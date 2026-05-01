#!/usr/bin/env bash
###############################################################################
# eval_wiki_ppl.sh
#
# Standalone evaluation script for WikiText-2 perplexity + lm_eval benchmarks.
# Supports running on any cluster node (single-node, 8 GPUs) via
# cluster_launcher.sh remote node selection.
#
# Usage:
#   # Run on the current node (default)
#   bash eval_wiki_ppl.sh
#
#   # Run on node 3 (using cluster_launcher node selection)
#   bash eval_wiki_ppl.sh 1 3
#
#   # Run on node 1
#   bash eval_wiki_ppl.sh 1 1
#
#   # Override parameters via environment variables
#   CKPT_PATH=outputs/.../model.pt MODEL_PATH=models/xxx BLOCK_SIZE=2048 bash eval_wiki_ppl.sh 1 2
#
#   # Append extra arguments (everything after -- is forwarded to eval_wiki_ppl.py)
#   bash eval_wiki_ppl.sh 1 3 -- --ckpt_path2 retrain_best.pt --output_json results.json
#
# Node pool (shared with training scripts, defined in cluster_launcher.sh):
#   1: <NODE_IP_1>
#   2: <NODE_IP_2>
#   3: <NODE_IP_3>
#   4: <NODE_IP_4>
# Override via CLUSTER_NODE_IPS environment variable.
###############################################################################

set -euo pipefail
export CLUSTER_ENV_SETUP='source ~/.bashrc && conda activate minillm &&'
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/cluster_launcher.sh"
SCRIPT_PATH=$(realpath "$0")

# Workspace directory (shared FS)
export WORKDIR=${WORKDIR:-"${SCRIPT_DIR}"}

# HuggingFace cache directories
export HF_HOME=${HF_HOME:-"${WORKDIR}/models"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"${WORKDIR}/models"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${WORKDIR}/data/hf_datasets"}
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" >/dev/null 2>&1 || true

# =============================================================================
# Controller mode: launch on selected node
# =============================================================================
if [[ "${INTERNAL_LAUNCH:-0}" != "1" ]]; then
  if [[ $# -ge 1 ]] && cluster_is_integer "$1"; then
    nnodes="$1"
    shift
    if [[ $# -lt "$nnodes" ]]; then
      echo "Usage: bash $(basename "$0") <NNODES> <IDX1> ... <IDXN> [-- extra args]" >&2
      echo "  Example: bash $(basename "$0") 1 3        # run on node 3" >&2
      echo "  Example: bash $(basename "$0") 1 1        # run on node 1" >&2
      exit 2
    fi
    idxs=()
    for ((i=0; i<nnodes; i++)); do
      idxs+=("$1")
      shift
    done
    # Eval only needs a single node; force NNODES=1 to avoid multi-node distributed launch.
    if [[ "$nnodes" -gt 1 ]]; then
      echo "[WARN] eval_wiki_ppl only needs 1 node, using first selected node (idx=${idxs[0]})"
      nnodes=1
      idxs=("${idxs[0]}")
    fi
    cluster_launch_selected_nodes "$SCRIPT_PATH" "$nnodes" "${idxs[@]}" "$@"
    exit $?
  fi
fi

# =============================================================================
# Proxy Settings (optional) — uncomment and modify if needed for model downloads.
# =============================================================================
# export http_proxy=http://your-proxy-host:port
# export https_proxy=http://your-proxy-host:port
# export all_proxy=http://your-proxy-host:port
# export no_proxy=localhost,127.0.0.1,.local

# =============================================================================
# Evaluation Parameters (override via environment variables)
# =============================================================================
CKPT_PATH=${CKPT_PATH:-"outputs/channel_pruning/qwen/models_Qwen--Qwen3-1.7B_mask-unstructured_s0.0_m-hessian_obd_20260212_162623/model.pt"}
MODEL_PATH=${MODEL_PATH:-"models/Qwen--Qwen3-1.7B"}
BLOCK_SIZE=${BLOCK_SIZE:-4096}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}

# Optional flags (set to "1" or "true" to enable)
EVAL_BASE_MODEL=${EVAL_BASE_MODEL:-"true"}
RUN_LM_EVAL=${RUN_LM_EVAL:-"true"}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-32}
HF_TOKEN=${HF_TOKEN:-""}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# =============================================================================
# Build launch command
# =============================================================================
launch_cmd=(torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}")

# Build eval arguments
args=()

if [[ -n "${CKPT_PATH}" ]]; then
  args+=(--ckpt_path "${CKPT_PATH}")
fi

args+=(--model_path "${MODEL_PATH}")
args+=(--block_size "${BLOCK_SIZE}")

if [[ "${EVAL_BASE_MODEL}" == "true" || "${EVAL_BASE_MODEL}" == "1" ]]; then
  args+=(--eval_base_model)
fi

if [[ "${RUN_LM_EVAL}" == "true" || "${RUN_LM_EVAL}" == "1" ]]; then
  args+=(--run_lm_eval)
  args+=(--lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}")
fi

if [[ -n "${HF_TOKEN}" ]]; then
  args+=(--hf_token "${HF_TOKEN}")
fi

# =============================================================================
# Append extra arguments (passed after -- in command line)
# =============================================================================
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

for var in "$@"; do
  if [[ -n "$var" ]]; then
    args+=("$var")
  fi
done

# =============================================================================
# Launch
# =============================================================================
echo "=============================================================="
echo "Wiki PPL + lm_eval Evaluation"
echo "=============================================================="
echo "[CONFIG] Checkpoint: ${CKPT_PATH:-'(none)'}"
echo "[CONFIG] Model path: ${MODEL_PATH}"
echo "[CONFIG] Block size: ${BLOCK_SIZE}"
echo "[CONFIG] GPUs: ${NPROC_PER_NODE}"
echo "[CONFIG] Eval base model: ${EVAL_BASE_MODEL}"
echo "[CONFIG] Run lm_eval: ${RUN_LM_EVAL}"
echo "[CONFIG] Node: $(hostname) ($(hostname -I | awk '{print $1}'))"
echo "=============================================================="
echo ""

"${launch_cmd[@]}" legacy/eval_wiki_ppl.py "${args[@]}"
