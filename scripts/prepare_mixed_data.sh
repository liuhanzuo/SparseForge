#!/usr/bin/env bash
###############################################################################
# prepare_mixed_data.sh
#
# Prepare mixed domain dataset (C4 + ArXiv + GitHub + StackExchange) for LLaMA
# training. This script tokenizes data and creates train.bin + val.bin.
#
# Features:
#   - Samples data according to specified domain ratios
#   - Supports target token count specification
#   - Handles insufficient data with repeat or truncation
#
# Usage:
#   # Default: use all available data, 60% C4 + 15% ArXiv + 15% GitHub + 10% SE
#   bash scripts/prepare_mixed_data.sh
#
#   # Specify target tokens (e.g., 10 billion)
#   TARGET_TOKENS=10B bash scripts/prepare_mixed_data.sh
#
#   # Custom ratios
#   C4_RATIO=0.5 ARXIV_RATIO=0.2 bash scripts/prepare_mixed_data.sh
#
#   # Allow data repetition for insufficient domains
#   ALLOW_REPEAT=1 TARGET_TOKENS=20B bash scripts/prepare_mixed_data.sh
#
#   # Force regenerate
#   bash scripts/prepare_mixed_data.sh --force
###############################################################################

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

# Tokenizer path
TOKENIZER="${TOKENIZER:-models/NousResearch--Llama-2-7b-hf}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-data/mixed_llama}"

# Number of parallel workers
NUM_WORKERS="${NUM_WORKERS:-48}"

# =============================================================================
# Domain Ratios (must sum to ~1.0, will be normalized)
# =============================================================================

C4_RATIO="${C4_RATIO:-0.60}"         # 60% - General web text
ARXIV_RATIO="${ARXIV_RATIO:-0.15}"   # 15% - Academic/scientific
GITHUB_RATIO="${GITHUB_RATIO:-0.15}" # 15% - Code
SE_RATIO="${SE_RATIO:-0.10}"         # 10% - Q&A

# =============================================================================
# Target Tokens & Sampling Options
# =============================================================================

# Target total tokens (e.g., "10B", "500M", "1G")
# If not set, uses all available data
TARGET_TOKENS="${TARGET_TOKENS:-}"

# Allow repeating data if a domain has insufficient samples
ALLOW_REPEAT="${ALLOW_REPEAT:-0}"

# Random seed
SEED="${SEED:-42}"

# =============================================================================
# Parse Arguments
# =============================================================================

EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --force)
            EXTRA_ARGS+=("--force")
            ;;
        --allow_repeat|--allow-repeat)
            ALLOW_REPEAT=1
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# =============================================================================
# Build Command
# =============================================================================

echo "=============================================================="
echo "Mixed Domain Dataset Preparation (with Ratio Sampling)"
echo "=============================================================="
echo "Project directory: ${PROJECT_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo ""
echo "Domain Ratios:"
echo "  C4:           ${C4_RATIO} (general web text)"
echo "  ArXiv:        ${ARXIV_RATIO} (academic/scientific)"
echo "  GitHub:       ${GITHUB_RATIO} (code)"
echo "  StackExchange: ${SE_RATIO} (Q&A)"
echo ""

if [[ -n "${TARGET_TOKENS}" ]]; then
    echo "Target Tokens: ${TARGET_TOKENS}"
else
    echo "Target Tokens: (all available data)"
fi

if [[ "${ALLOW_REPEAT}" == "1" ]]; then
    echo "Allow Repeat: YES (will repeat data if domain is insufficient)"
else
    echo "Allow Repeat: NO (will use all available if insufficient)"
fi
echo "=============================================================="
echo ""

# Build Python command
CMD=(
    python "${SCRIPT_DIR}/prepare_mixed_llama.py"
    --tokenizer "${TOKENIZER}"
    --output_dir "${OUTPUT_DIR}"
    --num_workers "${NUM_WORKERS}"
    --c4_ratio "${C4_RATIO}"
    --arxiv_ratio "${ARXIV_RATIO}"
    --github_ratio "${GITHUB_RATIO}"
    --stackexchange_ratio "${SE_RATIO}"
    --seed "${SEED}"
)

# Add target tokens if specified
if [[ -n "${TARGET_TOKENS}" ]]; then
    CMD+=(--target_tokens "${TARGET_TOKENS}")
fi

# Add allow repeat if enabled
if [[ "${ALLOW_REPEAT}" == "1" ]]; then
    CMD+=(--allow_repeat)
fi

# Add extra arguments
CMD+=("${EXTRA_ARGS[@]}")

# =============================================================================
# Execute
# =============================================================================

cd "${PROJECT_DIR}"
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "=============================================================="
echo "Done!"
echo "Output: ${OUTPUT_DIR}/train.bin, ${OUTPUT_DIR}/val.bin"
echo "=============================================================="
