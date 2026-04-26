#!/usr/bin/env bash
#===============================================================================
# upgrade_transformers.sh
# 在所有节点上升级 transformers 以支持 Qwen3
#
# 用法:
#   bash scripts/upgrade_transformers.sh
#===============================================================================
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOSTS_FILE="${SCRIPT_DIR}/hosts.txt"

echo "============================================"
echo " Upgrading transformers on all nodes"
echo "============================================"

# 检查 hosts 文件
if [[ ! -f "$HOSTS_FILE" ]]; then
    echo "ERROR: Hosts file not found: $HOSTS_FILE"
    exit 1
fi

# 读取节点
mapfile -t hosts < <(grep -v '^#' "$HOSTS_FILE" | grep -v '^$' | tr -d '\r')
total_nodes=${#hosts[@]}

echo "Total nodes: $total_nodes"
echo ""

# 升级命令 - transformers >= 4.51 支持 Qwen3
UPGRADE_CMD='
source /opt/conda/etc/profile.d/conda.sh
conda activate minillm
echo "[$(hostname)] Current transformers version:"
pip show transformers | grep Version
echo "[$(hostname)] Upgrading transformers..."
pip install --upgrade transformers>=4.51.0
echo "[$(hostname)] New transformers version:"
pip show transformers | grep Version
echo "[$(hostname)] ✓ Done"
'

# 并行升级所有节点
echo "Starting parallel upgrade on all nodes..."
echo ""

pids=()
for host in "${hosts[@]}"; do
    [[ -z "$host" ]] && continue
    echo "→ Starting upgrade on $host (background)..."
    ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "root@${host}" "$UPGRADE_CMD" 2>&1 | sed "s/^/[$host] /" &
    pids+=($!)
done

# 等待所有任务完成
echo ""
echo "Waiting for all nodes to complete..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        ((failed++))
    fi
done

echo ""
echo "============================================"
if [[ $failed -eq 0 ]]; then
    echo "✓ All $total_nodes nodes upgraded successfully!"
else
    echo "✗ $failed/$total_nodes nodes failed"
    exit 1
fi
