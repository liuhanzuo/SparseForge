#!/usr/bin/env bash
#===============================================================================
# quick_setup_nodes.sh
# 快速为所有节点配置 minillm conda 环境（简化版）
#
# 用法:
#   bash scripts/quick_setup_nodes.sh
#===============================================================================
set -uo pipefail  # 移除 -e 以防止提前退出

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOSTS_FILE="${SCRIPT_DIR}/hosts.txt"
REQUIREMENTS_FILE="${SCRIPT_DIR}/../requirements.txt"

# 如果 requirements.txt 不在上级目录，尝试其他位置
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    REQUIREMENTS_FILE="/apdcephfs_wzc1/share_304376610/pighzliu_code/requirements.txt"
fi

echo "=========================================="
echo " Quick MiniLLM Environment Setup"
echo "=========================================="
echo "Hosts file: $HOSTS_FILE"
echo "Requirements: $REQUIREMENTS_FILE"
echo ""

# 检查 hosts 文件是否存在
if [[ ! -f "$HOSTS_FILE" ]]; then
    echo "ERROR: Hosts file not found: $HOSTS_FILE"
    exit 1
fi

# 读取所有节点到数组
mapfile -t hosts < <(grep -v '^#' "$HOSTS_FILE" | grep -v '^$' | tr -d '\r')
total_nodes=${#hosts[@]}

echo "Total nodes to setup: $total_nodes"
echo "Nodes list:"
for i in "${!hosts[@]}"; do
    echo "  [$((i+1))] ${hosts[$i]}"
done
echo ""

if [[ $total_nodes -eq 0 ]]; then
    echo "ERROR: No valid hosts found in $HOSTS_FILE"
    exit 1
fi

SETUP_COMMANDS=$(cat << 'EOF'
set -e

# 日志函数，带时间戳和主机名
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$(hostname)] $1"
}

log "=========================================="
log "Starting environment setup..."
log "=========================================="
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
export all_proxy=http://your-proxy:port
export no_proxy=localhost,127.0.0.1,.local

# Step 1: 初始化 conda
log "[Step 1/6] Initializing conda..."
eval "$(conda shell.bash hook)" 2>/dev/null || source ~/.bashrc
log "[Step 1/6] ✓ Conda initialized"

# Step 2: 创建或激活环境
log "[Step 2/6] Setting up conda environment 'minillm'..."
if conda env list | grep -q "^minillm "; then
    log "[Step 2/6] Environment 'minillm' already exists, activating..."
    conda activate minillm
else
    log "[Step 2/6] Creating new environment 'minillm' with Python 3.9..."
    conda create -n minillm python=3.9 -y
    conda activate minillm
fi
log "[Step 2/6] ✓ Conda environment ready"

# Step 3: 安装 pyarrow
log "[Step 3/6] Installing pyarrow..."
conda install -c conda-forge pyarrow -y 2>/dev/null || pip install pyarrow
log "[Step 3/6] ✓ pyarrow installed"

# Step 4: 安装 PyTorch
log "[Step 4/6] Installing PyTorch (CUDA 12.1)... (this may take a while)"
pip install --upgrade torch torchvision torchaudio
log "[Step 4/6] ✓ PyTorch installed"

# Step 5: 安装其他依赖
log "[Step 5/6] Installing requirements from requirements.txt..."
REQUIREMENTS_FILE_REMOTE="${REQUIREMENTS_FILE:-/apdcephfs/pig_data/MiniLLM/requirements.txt}"
if [[ -f "$REQUIREMENTS_FILE_REMOTE" ]]; then
    pip install -r "$REQUIREMENTS_FILE_REMOTE"
else
    log "[Step 5/6] WARNING: requirements.txt not found, skipping..."
fi
log "[Step 5/6] ✓ Requirements installed"

wandb login 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

# Step 6: 验证安装
log "[Step 6/6] Verifying installation..."
PYTORCH_VER=$(python -c "import torch; print(torch.__version__)")
CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())")
CUDA_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
log "[Step 6/6] PyTorch version: $PYTORCH_VER"
log "[Step 6/6] CUDA available: $CUDA_AVAIL"
log "[Step 6/6] CUDA device count: $CUDA_COUNT"
log "[Step 6/6] ✓ Verification complete"
# pip install transformers==4.57.6


log "=========================================="
log "✓ Environment setup completed successfully!"
log "=========================================="
EOF
)

# ========== 并行执行 ==========
LOG_DIR="/tmp/quick_setup_logs_$$"
mkdir -p "$LOG_DIR"

echo "🚀 并行启动所有 $total_nodes 个节点的环境配置..."
echo "   日志目录: $LOG_DIR"
echo ""

# 存储后台进程 PID 和对应的 host
declare -A pid_to_host
declare -A pid_to_idx

current_node=0
for host in "${hosts[@]}"; do
    [[ -z "$host" ]] && continue
    ((current_node++))

    log_file="${LOG_DIR}/node_${current_node}_${host}.log"
    export_cmd="export REQUIREMENTS_FILE='${REQUIREMENTS_FILE}'"

    # 后台启动 SSH，输出重定向到独立日志文件
    (
        echo "##########################################################"
        echo "# Node $current_node/$total_nodes: $host"
        echo "# Started at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "##########################################################"
        echo "[DEBUG] Connecting to root@${host}..."
        ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -o BatchMode=yes \
            "root@${host}" "${export_cmd}; ${SETUP_COMMANDS}" 2>&1
        exit_code=$?
        echo ""
        echo "# Finished at: $(date '+%Y-%m-%d %H:%M:%S') | exit_code=$exit_code"
        exit $exit_code
    ) > "$log_file" 2>&1 &

    pid=$!
    pid_to_host[$pid]="$host"
    pid_to_idx[$pid]="$current_node"
    echo "  [Node $current_node] $host -> PID $pid (log: $log_file)"
done

echo ""
echo "所有节点已启动，等待完成..."
echo ""

# 等待所有后台进程完成，收集结果
failed_nodes=()
success_nodes=()

for pid in "${!pid_to_host[@]}"; do
    host="${pid_to_host[$pid]}"
    idx="${pid_to_idx[$pid]}"
    if wait "$pid"; then
        echo "✅ [Node $idx] $host 完成"
        success_nodes+=("$host")
    else
        echo "❌ [Node $idx] $host 失败 (exit code: $?)"
        failed_nodes+=("$host")
    fi
done

echo ""
echo "##########################################################"
echo "# SUMMARY"
echo "# Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "##########################################################"
echo ""
echo "Total nodes: $total_nodes"
echo "Successful: ${#success_nodes[@]}"
echo "Failed: ${#failed_nodes[@]}"
echo "Log directory: $LOG_DIR"
echo ""

# 打印每个节点日志的最后几行作为摘要
echo "========== 各节点日志摘要（最后5行）=========="
for host in "${hosts[@]}"; do
    [[ -z "$host" ]] && continue
    log_file=$(ls "${LOG_DIR}"/node_*_"${host}".log 2>/dev/null | head -1)
    if [[ -n "$log_file" && -f "$log_file" ]]; then
        echo ""
        echo "--- $host ---"
        tail -5 "$log_file"
    fi
done
echo ""

if [[ ${#failed_nodes[@]} -eq 0 ]]; then
    echo "✅ All nodes configured successfully!"
else
    echo "❌ Failed nodes: ${failed_nodes[*]}"
    echo ""
    echo "查看失败节点详细日志："
    for fhost in "${failed_nodes[@]}"; do
        log_file=$(ls "${LOG_DIR}"/node_*_"${fhost}".log 2>/dev/null | head -1)
        echo "  cat $log_file"
    done
    exit 1
fi
