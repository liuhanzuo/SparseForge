#!/usr/bin/env bash
#===============================================================================
# setup_all_nodes.sh
# 使用 pssh 为 hosts.txt 中所有节点批量创建 conda 环境
#
# 用法:
#   bash scripts/setup_all_nodes.sh [hosts_file] [requirements_file]
#
# 默认:
#   hosts_file = scripts/hosts.txt
#   requirements_file = /apdcephfs/pig_data/MiniLLM/requirements.txt
#
# 前提条件:
#   1. 安装 pssh: sudo apt-get install pssh 或 pip install parallel-ssh
#   2. 配置好 SSH 免密登录到所有节点
#   3. 所有节点都能访问共享文件系统 /apdcephfs
#===============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 默认参数
HOSTS_FILE="${1:-${SCRIPT_DIR}/hosts.txt}"
REQUIREMENTS_FILE="${2:-/apdcephfs/pig_data/MiniLLM/requirements.txt}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

#===============================================================================
# 检查前提条件
#===============================================================================
check_prerequisites() {
    # 检查 hosts 文件
    if [[ ! -f "$HOSTS_FILE" ]]; then
        log_error "Hosts file not found: $HOSTS_FILE"
        exit 1
    fi
    
    # 检查 requirements 文件
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        log_error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
    
    # 检查 pssh 是否安装
    if ! command -v pssh &> /dev/null && ! command -v parallel-ssh &> /dev/null; then
        log_warn "pssh not found. Installing..."
        pip install parallel-ssh 2>/dev/null || {
            log_error "Failed to install pssh. Please install manually:"
            log_error "  Ubuntu/Debian: sudo apt-get install pssh"
            log_error "  CentOS/RHEL:   sudo yum install pssh"
            log_error "  pip:           pip install parallel-ssh"
            exit 1
        }
    fi
    
    # 确定 pssh 命令名（不同系统可能不同）
    if command -v pssh &> /dev/null; then
        PSSH_CMD="pssh"
    elif command -v parallel-ssh &> /dev/null; then
        PSSH_CMD="parallel-ssh"
    fi
    
    log_info "Using pssh command: $PSSH_CMD"
    log_info "Hosts file: $HOSTS_FILE"
    log_info "Requirements file: $REQUIREMENTS_FILE"
}

#===============================================================================
# 创建临时的 setup 脚本（会被复制到每个节点执行）
#===============================================================================
create_setup_script() {
    local setup_script="/tmp/minillm_node_setup_$$.sh"
    
    cat > "$setup_script" << 'SETUP_SCRIPT'
#!/usr/bin/env bash
set -e

REQUIREMENTS_FILE="__REQUIREMENTS_FILE__"
CONDA_ENV_NAME="minillm"

echo "[$(hostname)] Starting environment setup..."

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "[$(hostname)] ERROR: conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# 初始化 conda（如果需要）
eval "$(conda shell.bash hook)" 2>/dev/null || source ~/.bashrc

# 检查环境是否已存在
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "[$(hostname)] Conda env '${CONDA_ENV_NAME}' already exists. Updating..."
    conda activate ${CONDA_ENV_NAME}
else
    echo "[$(hostname)] Creating conda env '${CONDA_ENV_NAME}'..."
    conda create -n ${CONDA_ENV_NAME} python=3.9 -y
    conda activate ${CONDA_ENV_NAME}
fi

# 安装 pyarrow (用于数据处理)
echo "[$(hostname)] Installing pyarrow..."
conda install -c conda-forge pyarrow -y || pip install pyarrow

# 安装 PyTorch (CUDA 12.1)
echo "[$(hostname)] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
if [[ -f "$REQUIREMENTS_FILE" ]]; then
    echo "[$(hostname)] Installing requirements from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "[$(hostname)] WARNING: Requirements file not found: $REQUIREMENTS_FILE"
fi

# 验证安装
echo "[$(hostname)] Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')" || true

echo "[$(hostname)] Environment setup completed successfully!"
SETUP_SCRIPT

    # 替换占位符
    sed -i "s|__REQUIREMENTS_FILE__|${REQUIREMENTS_FILE}|g" "$setup_script"
    
    chmod +x "$setup_script"
    echo "$setup_script"
}

#===============================================================================
# 清理无效的 hosts（移除空行和注释）
#===============================================================================
create_clean_hosts() {
    local clean_hosts="/tmp/minillm_hosts_$$.txt"
    grep -v '^#' "$HOSTS_FILE" | grep -v '^$' | sed 's/[[:space:]]*$//' > "$clean_hosts"
    
    # 显示将要配置的节点
    log_info "Nodes to configure:"
    cat -n "$clean_hosts"
    echo ""
    
    echo "$clean_hosts"
}

#===============================================================================
# 方法1: 使用 pssh 并行执行
#===============================================================================
run_with_pssh() {
    local setup_script="$1"
    local clean_hosts="$2"
    
    log_info "Running setup on all nodes using pssh..."
    
    # pssh 参数说明:
    # -h: hosts 文件
    # -l: 用户名（默认 root）
    # -t: 超时时间（秒）
    # -p: 并行度
    # -i: 显示输出
    
    $PSSH_CMD \
        -h "$clean_hosts" \
        -l root \
        -t 3600 \
        -p 4 \
        -i \
        "bash -s" < "$setup_script"
}

#===============================================================================
# 方法2: 使用 SSH 循环（备用方案，更可靠但较慢）
#===============================================================================
run_with_ssh_loop() {
    local setup_script="$1"
    local clean_hosts="$2"
    
    log_info "Running setup on all nodes using SSH loop..."
    
    local failed_hosts=()
    
    while IFS= read -r host || [[ -n "$host" ]]; do
        # 跳过空行
        [[ -z "$host" ]] && continue
        
        log_info "Setting up node: $host"
        
        if ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "root@${host}" "bash -s" < "$setup_script"; then
            log_info "✓ Node $host completed successfully"
        else
            log_error "✗ Node $host failed"
            failed_hosts+=("$host")
        fi
        
        echo ""
    done < "$clean_hosts"
    
    # 汇总结果
    echo ""
    echo "==============================================================================="
    if [[ ${#failed_hosts[@]} -eq 0 ]]; then
        log_info "All nodes configured successfully!"
    else
        log_error "Failed nodes: ${failed_hosts[*]}"
        exit 1
    fi
}

#===============================================================================
# 方法3: 使用 GNU Parallel（如果可用）
#===============================================================================
run_with_parallel() {
    local setup_script="$1"
    local clean_hosts="$2"
    
    if ! command -v parallel &> /dev/null; then
        log_warn "GNU parallel not found, falling back to SSH loop"
        run_with_ssh_loop "$setup_script" "$clean_hosts"
        return
    fi
    
    log_info "Running setup on all nodes using GNU parallel..."
    
    cat "$clean_hosts" | parallel -j 4 --tag \
        "ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no root@{} 'bash -s' < $setup_script"
}

#===============================================================================
# 主函数
#===============================================================================
main() {
    log_info "=========================================="
    log_info " MiniLLM Multi-Node Environment Setup"
    log_info "=========================================="
    
    check_prerequisites
    
    local setup_script=$(create_setup_script)
    local clean_hosts=$(create_clean_hosts)
    
    # 确认执行
    echo ""
    read -p "Do you want to proceed with setup? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user."
        rm -f "$setup_script" "$clean_hosts"
        exit 0
    fi
    
    # 选择执行方法
    local method="${SETUP_METHOD:-ssh}"
    
    case "$method" in
        pssh)
            run_with_pssh "$setup_script" "$clean_hosts"
            ;;
        parallel)
            run_with_parallel "$setup_script" "$clean_hosts"
            ;;
        ssh|*)
            run_with_ssh_loop "$setup_script" "$clean_hosts"
            ;;
    esac
    
    # 清理临时文件
    rm -f "$setup_script" "$clean_hosts"
    
    log_info "Setup completed!"
}

# 运行
main "$@"
