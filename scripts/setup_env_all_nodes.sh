#!/bin/bash
# =============================================================================
# 批量在所有节点上安装 MiniLLM 环境
# 用法: bash scripts/setup_env_all_nodes.sh [hosts_file]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTS_FILE="${1:-$SCRIPT_DIR/hosts.txt}"

# 检查 hosts.txt 是否存在
if [[ ! -f "$HOSTS_FILE" ]]; then
    echo "[ERROR] hosts file not found: $HOSTS_FILE"
    exit 1
fi

# 读取节点列表（跳过空行和注释）
NODES=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line=$(echo "$line" | xargs)  # 去除首尾空格
    [[ -z "$line" || "$line" == \#* ]] && continue
    NODES+=("$line")
done < "$HOSTS_FILE"

if [[ ${#NODES[@]} -eq 0 ]]; then
    echo "[ERROR] No nodes found in $HOSTS_FILE"
    exit 1
fi

echo "========================================"
echo "Found ${#NODES[@]} node(s) to setup:"
printf '  - %s\n' "${NODES[@]}"
echo "========================================"

# 定义要在远程节点执行的安装命令
INSTALL_SCRIPT='
#!/bin/bash
set -e

echo ">>> [$(hostname)] Starting environment setup..."

# 初始化 conda（尝试多个常见路径）
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
{ echo "[ERROR] conda not found"; exit 1; }

# 检查环境是否已存在
if conda env list | grep -q "^minillm "; then
    echo ">>> Environment minillm already exists, skipping create..."
else
    echo ">>> Creating conda environment: minillm"
    conda create -n minillm python=3.9 -y
fi

# 激活环境
conda activate minillm

# 安装 pyarrow
echo ">>> Installing pyarrow..."
conda install -c conda-forge pyarrow -y

# 安装 PyTorch (CUDA 12.1)
echo ">>> Installing PyTorch (cu121)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖（如果 requirements.txt 存在）
REQ_FILE="/apdcephfs/pig_data/Adaptive-Sparse-Trainer/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo ">>> Installing requirements.txt..."
    pip install -r "$REQ_FILE"
else
    echo ">>> No requirements.txt found at $REQ_FILE, skipping..."
fi

echo ">>> [$(hostname)] Environment setup completed!"
'

# 日志目录
LOG_DIR="$SCRIPT_DIR/setup_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 并行执行函数
setup_node() {
    local node=$1
    local log_file="$LOG_DIR/setup_${node}_${TIMESTAMP}.log"
    local show_output=${2:-false}
    
    echo "[$(date +%H:%M:%S)] Starting setup on $node ..."
    
    if [[ "$show_output" == "true" ]]; then
        # 实时显示输出，同时保存到日志
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$node" "$INSTALL_SCRIPT" 2>&1 | tee "$log_file"
        local exit_code=${PIPESTATUS[0]}
    else
        # 只保存到日志，但定期显示状态
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$node" "$INSTALL_SCRIPT" \
            > "$log_file" 2>&1 &
        local ssh_pid=$!
        
        # 等待并定期显示状态
        while kill -0 $ssh_pid 2>/dev/null; do
            if [[ -f "$log_file" ]]; then
                local last_line=$(tail -1 "$log_file" 2>/dev/null | head -c 80)
                if [[ -n "$last_line" ]]; then
                    echo "  [$node] $last_line"
                fi
            fi
            sleep 10
        done
        wait $ssh_pid
        local exit_code=$?
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] ✓ $node: SUCCESS (log: $log_file)"
        return 0
    else
        echo "[$(date +%H:%M:%S)] ✗ $node: FAILED (log: $log_file)"
        echo "  Last 5 lines of log:"
        tail -5 "$log_file" 2>/dev/null | sed 's/^/    /'
        return 1
    fi
}

# 询问是否并行执行
echo ""
read -p "Run in parallel? [Y/n]: " PARALLEL
PARALLEL=${PARALLEL:-Y}

FAILED_NODES=()

if [[ "$PARALLEL" =~ ^[Yy]$ ]]; then
    echo ""
    echo ">>> Running setup on all nodes in parallel..."
    echo ">>> Progress will be shown every 10 seconds. Full logs in: $LOG_DIR"
    echo ""
    
    # 并行执行 - 在后台启动所有节点
    declare -A NODE_PIDS
    declare -A NODE_LOGS
    
    for node in "${NODES[@]}"; do
        log_file="$LOG_DIR/setup_${node}_${TIMESTAMP}.log"
        NODE_LOGS[$node]="$log_file"
        (
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$node" "$INSTALL_SCRIPT" \
                > "$log_file" 2>&1
        ) &
        NODE_PIDS[$node]=$!
        echo "[$(date +%H:%M:%S)] Started: $node (pid: ${NODE_PIDS[$node]})"
    done
    
    echo ""
    echo ">>> All jobs launched. Monitoring progress..."
    echo ""
    
    # 监控进度
    completed=0
    while [[ $completed -lt ${#NODES[@]} ]]; do
        sleep 10
        completed=0
        echo "--- [$(date +%H:%M:%S)] Status Update ---"
        for node in "${NODES[@]}"; do
            pid=${NODE_PIDS[$node]}
            log_file=${NODE_LOGS[$node]}
            if kill -0 $pid 2>/dev/null; then
                # 进程仍在运行
                if [[ -f "$log_file" ]]; then
                    last_line=$(tail -1 "$log_file" 2>/dev/null | head -c 100)
                    echo "  [RUNNING] $node: $last_line"
                else
                    echo "  [RUNNING] $node: (waiting for output...)"
                fi
            else
                # 进程已结束
                ((completed++))
                wait $pid 2>/dev/null
                if [[ $? -eq 0 ]]; then
                    echo "  [DONE ✓] $node"
                else
                    echo "  [FAIL ✗] $node"
                    FAILED_NODES+=("$node")
                fi
            fi
        done
        echo ""
    done
else
    echo ""
    echo ">>> Running setup on all nodes sequentially..."
    echo ""
    
    # 顺序执行
    for node in "${NODES[@]}"; do
        setup_node "$node" || FAILED_NODES+=("$node")
    done
fi

# 输出结果汇总
echo ""
echo "========================================"
echo "Setup completed!"
echo "  Total nodes: ${#NODES[@]}"
echo "  Success: $((${#NODES[@]} - ${#FAILED_NODES[@]}))"
echo "  Failed: ${#FAILED_NODES[@]}"

if [[ ${#FAILED_NODES[@]} -gt 0 ]]; then
    echo ""
    echo "Failed nodes:"
    printf '  - %s\n' "${FAILED_NODES[@]}"
    echo ""
    echo "Check logs in: $LOG_DIR"
    exit 1
fi

echo "========================================"
