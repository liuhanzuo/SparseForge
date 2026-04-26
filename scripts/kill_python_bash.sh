#!/usr/bin/env bash
# 多节点 Python 进程批量终止脚本
# 复用 cluster_launcher.sh 的节点池，支持与 train_universal.sh 相同的节点选择语法。
#
# 用法:
#   1) Kill 所有节点池中的节点:
#        ./scripts/kill_python_bash.sh all
#
#   2) 选择 N 个节点 kill（与 train_universal.sh 相同语法）:
#        ./scripts/kill_python_bash.sh <NNODES> <IDX1> <IDX2> ... <IDXN>
#      例如 kill 节点 0 和 2:
#        ./scripts/kill_python_bash.sh 2 0 2
#
#   3) 直接指定主机名或 hosts.txt 中的索引（旧用法）:
#        ./scripts/kill_python_bash.sh host1.example.com host2.example.com
#        ./scripts/kill_python_bash.sh 0 1 3
#
# 环境变量:
#   SSH_USER   - SSH 用户名（默认 root）
#   SSH_PORT   - SSH 端口（默认不指定）
#   KILL_CMD   - 远程执行的 kill 命令（默认 "sudo pkill -9 -f python"）

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOSTS_FILE="$SCRIPT_DIR/hosts.txt"
SSH_USER=${SSH_USER:-root}
SSH_PORT=${SSH_PORT:-}
KILL_CMD=${KILL_CMD:-"sudo pkill -9 -f python"}

# 引入 cluster_launcher.sh 以复用节点池和选择函数
source "${SCRIPT_DIR}/cluster_launcher.sh"

# ---------------------------------------------------------------------------
# 辅助函数：SSH 到指定 host 执行 kill 命令
# ---------------------------------------------------------------------------
do_kill() {
  local host="$1"
  local target="$host"
  if [ -n "$SSH_USER" ]; then
    target="${SSH_USER}@${host}"
  fi

  local -a ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=no)
  if [ -n "$SSH_PORT" ]; then
    ssh_opts+=("-p" "$SSH_PORT")
  fi

  echo "[kill] ${target} -> ${KILL_CMD}"
  ssh "${ssh_opts[@]}" "${target}" "${KILL_CMD} || true" 2>/dev/null || \
    echo "[warn] 无法连接到 ${target}，跳过"
}

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
  cat <<'EOF'
用法:
  kill_python_bash.sh all                          # kill 所有节点
  kill_python_bash.sh <NNODES> <IDX1> ... <IDXN>  # kill 指定节点（同 train_universal.sh 语法）
  kill_python_bash.sh <host|index> [host|index ...]  # 直接指定主机名或 hosts.txt 索引
EOF
  exit 1
fi

# ---------------------------------------------------------------------------
# 模式 1: all - kill 节点池中的所有节点
# ---------------------------------------------------------------------------
if [[ "$1" == "all" ]]; then
  cluster_load_nodes
  echo "[kill] 将终止所有 ${#CLUSTER_NODE_IPS_ARR[@]} 个节点上的 Python 进程..."
  for ip in "${CLUSTER_NODE_IPS_ARR[@]}"; do
    do_kill "$ip"
  done
  echo "[kill] 完成。共处理 ${#CLUSTER_NODE_IPS_ARR[@]} 个节点。"
  exit 0
fi

# ---------------------------------------------------------------------------
# 模式 2: <NNODES> <IDX1> ... <IDXN> - 与 train_universal.sh 相同的节点选择语法
#   检测条件: 第一个参数是数字，且后续有足够多的数字参数
# ---------------------------------------------------------------------------
if cluster_is_integer "$1"; then
  nnodes="$1"
  # 检查是否有足够的后续参数且全是数字 -> 进入 controller 模式
  all_integer=true
  if (( $# >= nnodes + 1 )); then
    for ((i=2; i<=nnodes+1; i++)); do
      arg="${!i:-}"
      if ! cluster_is_integer "$arg"; then
        all_integer=false
        break
      fi
    done
  else
    all_integer=false
  fi

  if $all_integer && (( nnodes > 0 )) && (( $# >= nnodes + 1 )); then
    shift
    idxs=()
    for ((i=0; i<nnodes; i++)); do
      idxs+=("$1")
      shift
    done

    cluster_select_ips "$nnodes" "${idxs[@]}"
    echo "[kill] 将终止 ${nnodes} 个选定节点上的 Python 进程: ${CLUSTER_SELECTED_IPS[*]}"
    for ip in "${CLUSTER_SELECTED_IPS[@]}"; do
      do_kill "$ip"
    done
    echo "[kill] 完成。共处理 ${nnodes} 个节点。"
    exit 0
  fi
fi

# ---------------------------------------------------------------------------
# 模式 3: 兼容旧用法 - 直接传入主机名或 hosts.txt 中的索引
# ---------------------------------------------------------------------------
hosts_to_kill=()
for t in "$@"; do
  if [[ "$t" =~ ^[0-9]+$ ]] && [ -f "$HOSTS_FILE" ]; then
    host_line=$(sed -n "$((t+1))p" "$HOSTS_FILE" || true)
    host=$(echo "$host_line" | awk '{print $1}')
    if [ -z "$host" ]; then
      echo "[warn] 索引 $t 超出范围（$HOSTS_FILE 中无对应主机），跳过。"
      continue
    fi
  else
    host="$t"
  fi
  hosts_to_kill+=("$host")
done

if [ ${#hosts_to_kill[@]} -eq 0 ]; then
  echo "[error] 未找到有效的目标主机。"
  exit 1
fi

echo "[kill] 将终止 ${#hosts_to_kill[@]} 个节点上的 Python 进程..."
for host in "${hosts_to_kill[@]}"; do
  do_kill "$host"
done
echo "[kill] 完成。共处理 ${#hosts_to_kill[@]} 个节点。"
