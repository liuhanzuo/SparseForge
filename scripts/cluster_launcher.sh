#!/usr/bin/env bash
set -euo pipefail

# Cluster launcher helper.
# Supports launching a multi-node job from a single controller node via SSH.
#
# New UX (selection mode):
#   bash train_gpt_medium.sh <NNODES> <IDX1> <IDX2> ... <IDXN> [-- extra main.py args]
# Example (use 2 nodes: node 3 and node 5):
#   bash train_gpt_medium.sh 2 3 5
#
# Notes:
# - IDX can be 1-based (1..N) or 0-based (0..N-1); we auto-detect.
# - Rank0 will be the FIRST selected index (IDX1). MASTER_ADDR will be that node's IP.
# - Requires passwordless SSH to all selected nodes.

cluster_default_nodes() {
  echo "<NODE_IP_1> <NODE_IP_2> <NODE_IP_3> <NODE_IP_4>"
}

cluster_load_nodes() {
  # Allow override via env CLUSTER_NODE_IPS="ip1 ip2 ...".
  local nodes
  nodes=${CLUSTER_NODE_IPS:-""}
  if [[ -z "$nodes" ]]; then
    nodes=$(cluster_default_nodes)
  fi
  read -r -a CLUSTER_NODE_IPS_ARR <<< "$nodes"
  export CLUSTER_NODE_IPS_ARR
}

cluster_is_integer() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

cluster_is_ip_like() {
  [[ "${1:-}" == *.* ]]
}

cluster_detect_index_base() {
  # Decide whether indices are 0-based or 1-based.
  # If any idx is 0 => 0-based. Else if all in [1..len] => 1-based.
  local len=$1
  shift
  local base=1
  local idx
  for idx in "$@"; do
    if [[ "$idx" == "0" ]]; then
      echo 0
      return
    fi
  done
  # otherwise check 1-based validity
  local min=999999
  local max=-1
  for idx in "$@"; do
    (( idx < min )) && min=$idx
    (( idx > max )) && max=$idx
  done
  if (( min >= 1 && max <= len )); then
    echo 1
    return
  fi
  # fallback: if all in [0..len-1] treat as 0-based
  if (( min >= 0 && max < len )); then
    echo 0
    return
  fi
  echo -1
}

cluster_select_ips() {
  local nnodes=$1
  shift
  local -a idxs=("$@")

  cluster_load_nodes
  local len=${#CLUSTER_NODE_IPS_ARR[@]}
  if (( nnodes <= 0 )); then
    echo "ERROR: NNODES must be > 0" >&2
    return 2
  fi
  if (( nnodes > len )); then
    echo "ERROR: NNODES=${nnodes} exceeds cluster pool size=${len}" >&2
    return 2
  fi

  local base
  base=$(cluster_detect_index_base "$len" "${idxs[@]}")
  if [[ "$base" == "-1" ]]; then
    echo "ERROR: invalid node indices: ${idxs[*]} (pool size=${len}); use 0..$((len-1)) or 1..${len}" >&2
    return 2
  fi

  CLUSTER_SELECTED_IPS=()
  local i raw_idx idx
  for raw_idx in "${idxs[@]}"; do
    idx=$raw_idx
    if (( base == 1 )); then
      idx=$((raw_idx - 1))
    fi
    if (( idx < 0 || idx >= len )); then
      echo "ERROR: node index out of range: ${raw_idx} (resolved=${idx})" >&2
      return 2
    fi
    CLUSTER_SELECTED_IPS+=("${CLUSTER_NODE_IPS_ARR[$idx]}")
  done
  export CLUSTER_SELECTED_IPS
}

cluster_local_ip_match() {
  # Return 0 if the given IP appears on this host.
  local ip=$1
  if [[ -z "${ip:-}" ]]; then
    return 1
  fi

  # Fast-path loopback
  if [[ "$ip" == "127.0.0.1" ]]; then
    return 0
  fi

  # Prefer `ip addr` which is more reliable than `hostname -I` on some clusters.
  local -a local_ips=()

  if command -v ip >/dev/null 2>&1; then
# `ip -o -4 addr show` prints CIDR like 10.0.0.1/24; strip prefix.
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      local_ips+=("${line%%/*}")
    done < <(ip -o -4 addr show 2>/dev/null | awk '{print $4}' || true)
  fi

  # Fallback to hostname -I
  local ips
  ips=$(hostname -I 2>/dev/null || true)
  if [[ -n "$ips" ]]; then
    for x in $ips; do
      local_ips+=("$x")
    done
  fi

  # Optional override hook if the environment has unusual networking.
  if [[ -n "${CLUSTER_LOCAL_IPS:-}" ]]; then
    for x in ${CLUSTER_LOCAL_IPS}; do
      local_ips+=("$x")
    done
  fi

  local x
  for x in "${local_ips[@]}"; do
    if [[ "$x" == "$ip" ]]; then
      return 0
    fi
  done
  return 1
}

cluster_launch_selected_nodes() {
  # Usage: cluster_launch_selected_nodes <script_path> <nnodes> <idx1..idxN> [-- extra_args...]
  local script_path=$1
  shift
  local nnodes=$1
  shift

  local -a idxs=()
  local i
  for ((i=0; i<nnodes; i++)); do
    idxs+=("${1:-}")
    shift || true
  done

  # Remaining args are forwarded to the script (and eventually main.py)
  local -a extra_args=("$@")

  cluster_select_ips "$nnodes" "${idxs[@]}"
  local -a ips=("${CLUSTER_SELECTED_IPS[@]}")

  local master_addr=${ips[0]}
  local master_port=${MASTER_PORT:-29500}
  local nproc=${NPROC_PER_NODE:-8}
  # Prefer the directory of the launched script as default WORKDIR.
  # This avoids accidentally using a per-user/per-node $PWD (e.g. /apdcephfs/private_*)
  # which may not be accessible from other nodes.
  local script_dir
  script_dir=$(cd "$(dirname "${script_path}")" && pwd)
  local workdir=${WORKDIR:-"${script_dir}"}
  local ssh_user=${SSH_USER:-""}
  local env_setup=${CLUSTER_ENV_SETUP:-""}
  local dry_run=${DRY_RUN:-0}
  local debug_launch=${DEBUG_LAUNCH:-0}
  # Forward select feature flags to remote nodes so all ranks run the same code paths.
  local use_fsdp_fully_sharded=${USE_FSDP_FULLY_SHARDED:-0}
  local use_torchrun_single_node=${USE_TORCHRUN_SINGLE_NODE:-0}

  # Forward NCCL/network knobs so all nodes/ranks share identical transport settings.
  local force_tcp=${FORCE_TCP:-0}
  local disable_p2p=${DISABLE_P2P:-0}
  local nccl_socket_ifname=${NCCL_SOCKET_IFNAME:-}
  # CRITICAL: Disable NVLS to prevent "Cuda failure 802 'system not yet initialized'" errors
  # on H800/H100 GPUs with NCCL 2.21.x
  local nccl_nvls_enable=${NCCL_NVLS_ENABLE:-0}
  # Forward ALL NCCL env vars to ensure consistent protocol across nodes
  local nccl_ib_disable=${NCCL_IB_DISABLE:-}
  local nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL:-}
  local nccl_ib_gid_index=${NCCL_IB_GID_INDEX:-}
  local nccl_ib_timeout=${NCCL_IB_TIMEOUT:-}
  local nccl_ib_retry_cnt=${NCCL_IB_RETRY_CNT:-}
  local nccl_ib_qps=${NCCL_IB_QPS_PER_CONNECTION:-}
  local nccl_ib_cuda_support=${NCCL_IB_CUDA_SUPPORT:-}
  local nccl_shm_disable=${NCCL_SHM_DISABLE:-}
  local nccl_p2p_disable=${NCCL_P2P_DISABLE:-}
  local nccl_proto=${NCCL_PROTO:-}
  local nccl_debug=${NCCL_DEBUG:-}

  # Shared run id / trace dir so every node writes per-rank tracebacks to the same place.
  local run_id=${RUN_ID:-""}
  if [[ -z "$run_id" ]]; then
    run_id="$(date +%Y%m%d_%H%M%S).$$"
  fi
  local trace_dir=${AST_TRACE_DIR:-""}
  if [[ -z "$trace_dir" ]]; then
    trace_dir="${workdir}/outputs/ast_trace/${run_id}"
  fi

  mkdir -p "$trace_dir" >/dev/null 2>&1 || true

  # If the controller was invoked from a non-shared path (e.g., a per-node home dir),
  # remote nodes will fail `cd`. Fail fast with a clear message.
  if [[ ! -d "$workdir" ]]; then
    echo "ERROR: WORKDIR does not exist locally: $workdir" >&2
    echo "Set WORKDIR to a shared filesystem path visible on all nodes." >&2
    return 2
  fi

  # Prefer running the script from within WORKDIR on remote nodes. This avoids depending on
  # the controller's absolute path being identical on every node.
  local remote_script_path="$script_path"
  if [[ "$script_path" != "$workdir"/* ]]; then
    remote_script_path="$workdir/$(basename "$script_path")"
  fi

  # Optional: generate a DeepSpeed hostfile for the selected nodes (shared FS assumed).
  # This is only needed if your training script uses DeepSpeed LAUNCH_MODE=single with --hostfile.
  local generated_hostfile=""
  if [[ "${CLUSTER_GENERATE_HOSTFILE:-0}" == "1" ]]; then
    mkdir -p "${workdir}/configs" >/dev/null 2>&1 || true
    generated_hostfile="${workdir}/configs/hosts.selected.${nnodes}.$(date +%Y%m%d_%H%M%S).$$.txt"
    : >"${generated_hostfile}"
    local slots=${CLUSTER_HOSTFILE_SLOTS:-${nproc}}
    local ip
    for ip in "${ips[@]}"; do
      echo "${ip} slots=${slots}" >>"${generated_hostfile}"
    done
    export DEEPSPEED_HOSTFILE="${generated_hostfile}"
    echo "[cluster] generated hostfile: ${generated_hostfile}"
  fi

  local extra_args_quoted=""
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    extra_args_quoted=$(printf "%q " "${extra_args[@]}")
  fi

  local ssh_target
  local -a bg_pids=()

  echo "[cluster] selected nodes (${nnodes}): ${ips[*]}"
  echo "[cluster] rank0/master_addr: ${master_addr}  master_port: ${master_port}  nproc_per_node: ${nproc}"
  echo "[cluster] RUN_ID: ${run_id}"
  echo "[cluster] AST_TRACE_DIR: ${trace_dir}"
  echo "[cluster] ssh logs: ${trace_dir}/ssh_node_rank*.log"

  # Pre-flight check: verify WORKDIR and script exist on each remote node before launching.
  for ((i=0; i<nnodes; i++)); do
    local ip=${ips[$i]}
    if cluster_local_ip_match "$ip"; then
      # local checks
      if [[ ! -f "$script_path" ]] && [[ ! -f "$remote_script_path" ]]; then
        echo "ERROR: training script not found locally: $script_path (or $remote_script_path)" >&2
        return 2
      fi
      continue
    fi
    if [[ -n "$ssh_user" ]]; then
      ssh_target="${ssh_user}@${ip}"
    else
      ssh_target="${ip}"
    fi
    if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ssh_target" \
      "bash -lc 'test -d "$workdir" && test -f "$remote_script_path"'"; then
      echo "ERROR: remote node ${ip} cannot access WORKDIR/script." >&2
      echo "  WORKDIR: $workdir" >&2
      echo "  script : $remote_script_path" >&2
      echo "Fix: run from a shared FS path, or export WORKDIR to a shared path that contains this script." >&2
      return 2
    fi
  done

  # Launch non-rank0 nodes first in background.
  for ((i=1; i<nnodes; i++)); do
    local ip=${ips[$i]}
    if [[ -n "$ssh_user" ]]; then
      ssh_target="${ssh_user}@${ip}"
    else
      ssh_target="${ip}"
    fi

    # If the IP matches this host, launch locally in background instead of ssh-ing to self.
    # You can force SSH for all nodes by setting CLUSTER_FORCE_SSH=1 in the environment.
    if cluster_local_ip_match "$ip" && [[ "${CLUSTER_FORCE_SSH:-0}" != "1" ]]; then
      echo "[cluster] launching rank ${i} locally (background, matches ${ip})"
      local node_log="${trace_dir}/ssh_node_rank${i}.${ip}.log"
      bash -lc "cd \"$workdir\" && ${env_setup} \
        INTERNAL_LAUNCH=1 LAUNCH_MODE=multi \
        DRY_RUN=${dry_run} DEBUG_LAUNCH=${debug_launch} \
        FORCE_TCP=${force_tcp} DISABLE_P2P=${disable_p2p} \
        ${nccl_socket_ifname:+NCCL_SOCKET_IFNAME=\"${nccl_socket_ifname}\"} \
        NCCL_NVLS_ENABLE=${nccl_nvls_enable} \
        ${nccl_ib_disable:+NCCL_IB_DISABLE=${nccl_ib_disable}} \
        ${nccl_net_gdr_level:+NCCL_NET_GDR_LEVEL=${nccl_net_gdr_level}} \
        ${nccl_ib_gid_index:+NCCL_IB_GID_INDEX=${nccl_ib_gid_index}} \
        ${nccl_ib_timeout:+NCCL_IB_TIMEOUT=${nccl_ib_timeout}} \
        ${nccl_ib_retry_cnt:+NCCL_IB_RETRY_CNT=${nccl_ib_retry_cnt}} \
        ${nccl_ib_qps:+NCCL_IB_QPS_PER_CONNECTION=${nccl_ib_qps}} \
        ${nccl_ib_cuda_support:+NCCL_IB_CUDA_SUPPORT=${nccl_ib_cuda_support}} \
        ${nccl_shm_disable:+NCCL_SHM_DISABLE=${nccl_shm_disable}} \
        ${nccl_p2p_disable:+NCCL_P2P_DISABLE=${nccl_p2p_disable}} \
        ${nccl_proto:+NCCL_PROTO=${nccl_proto}} \
        ${nccl_debug:+NCCL_DEBUG=${nccl_debug}} \
        USE_FSDP_FULLY_SHARDED=${use_fsdp_fully_sharded} USE_TORCHRUN_SINGLE_NODE=${use_torchrun_single_node} \
        RUN_ID=\"${run_id}\" AST_TRACE_DIR=\"${trace_dir}\" \
        RETRAIN_DEBUG_STAGES=${RETRAIN_DEBUG_STAGES:-0} DEBUG_NCCL=${DEBUG_NCCL:-0} \
        NNODES=${nnodes} NODE_RANK=${i} \
        MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} \
        NPROC_PER_NODE=${nproc} \
        ${generated_hostfile:+DEEPSPEED_HOSTFILE=\"$generated_hostfile\"} \
        bash \"$script_path\" ${extra_args_quoted}" >"${node_log}" 2>&1 &
      bg_pids+=("$!")
      continue
    fi

    echo "[cluster] launching rank ${i} on ${ip} (background)"
    local node_log="${trace_dir}/ssh_node_rank${i}.${ip}.log"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ssh_target" \
      "bash -lc 'cd \"$workdir\" && ${env_setup} \
       mkdir -p \"${trace_dir}\" >/dev/null 2>&1 || true; \
       INTERNAL_LAUNCH=1 LAUNCH_MODE=multi \
       DRY_RUN=${dry_run} DEBUG_LAUNCH=${debug_launch} \
       FORCE_TCP=${force_tcp} DISABLE_P2P=${disable_p2p} \
       ${nccl_socket_ifname:+NCCL_SOCKET_IFNAME=\"${nccl_socket_ifname}\"} \
       NCCL_NVLS_ENABLE=${nccl_nvls_enable} \
       ${nccl_ib_disable:+NCCL_IB_DISABLE=${nccl_ib_disable}} \
       ${nccl_net_gdr_level:+NCCL_NET_GDR_LEVEL=${nccl_net_gdr_level}} \
       ${nccl_ib_gid_index:+NCCL_IB_GID_INDEX=${nccl_ib_gid_index}} \
       ${nccl_ib_timeout:+NCCL_IB_TIMEOUT=${nccl_ib_timeout}} \
       ${nccl_ib_retry_cnt:+NCCL_IB_RETRY_CNT=${nccl_ib_retry_cnt}} \
       ${nccl_ib_qps:+NCCL_IB_QPS_PER_CONNECTION=${nccl_ib_qps}} \
       ${nccl_ib_cuda_support:+NCCL_IB_CUDA_SUPPORT=${nccl_ib_cuda_support}} \
       ${nccl_shm_disable:+NCCL_SHM_DISABLE=${nccl_shm_disable}} \
       ${nccl_p2p_disable:+NCCL_P2P_DISABLE=${nccl_p2p_disable}} \
       ${nccl_proto:+NCCL_PROTO=${nccl_proto}} \
       ${nccl_debug:+NCCL_DEBUG=${nccl_debug}} \
       USE_FSDP_FULLY_SHARDED=${use_fsdp_fully_sharded} USE_TORCHRUN_SINGLE_NODE=${use_torchrun_single_node} \
       RUN_ID=\"${run_id}\" AST_TRACE_DIR=\"${trace_dir}\" \
       RETRAIN_DEBUG_STAGES=${RETRAIN_DEBUG_STAGES:-0} DEBUG_NCCL=${DEBUG_NCCL:-0} \
       NNODES=${nnodes} NODE_RANK=${i} \
       MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} \
       NPROC_PER_NODE=${nproc} \
       ${generated_hostfile:+DEEPSPEED_HOSTFILE=\"$generated_hostfile\"} \
       bash \"$remote_script_path\" ${extra_args_quoted}'" \
      >"${node_log}" 2>&1 &
    bg_pids+=("$!")
  done

  # Launch rank0 (prefer local execution if rank0 IP is local).
  if cluster_local_ip_match "$master_addr"; then
    echo "[cluster] launching rank 0 locally (matches ${master_addr})"
    bash -lc "cd \"$workdir\" && ${env_setup} \
      INTERNAL_LAUNCH=1 LAUNCH_MODE=multi \
      DRY_RUN=${dry_run} DEBUG_LAUNCH=${debug_launch} \
      FORCE_TCP=${force_tcp} DISABLE_P2P=${disable_p2p} \
      ${nccl_socket_ifname:+NCCL_SOCKET_IFNAME=\"${nccl_socket_ifname}\"} \
      NCCL_NVLS_ENABLE=${nccl_nvls_enable} \
      ${nccl_ib_disable:+NCCL_IB_DISABLE=${nccl_ib_disable}} \
      ${nccl_net_gdr_level:+NCCL_NET_GDR_LEVEL=${nccl_net_gdr_level}} \
      ${nccl_ib_gid_index:+NCCL_IB_GID_INDEX=${nccl_ib_gid_index}} \
      ${nccl_ib_timeout:+NCCL_IB_TIMEOUT=${nccl_ib_timeout}} \
      ${nccl_ib_retry_cnt:+NCCL_IB_RETRY_CNT=${nccl_ib_retry_cnt}} \
      ${nccl_ib_qps:+NCCL_IB_QPS_PER_CONNECTION=${nccl_ib_qps}} \
      ${nccl_ib_cuda_support:+NCCL_IB_CUDA_SUPPORT=${nccl_ib_cuda_support}} \
      ${nccl_shm_disable:+NCCL_SHM_DISABLE=${nccl_shm_disable}} \
      ${nccl_p2p_disable:+NCCL_P2P_DISABLE=${nccl_p2p_disable}} \
      ${nccl_proto:+NCCL_PROTO=${nccl_proto}} \
      ${nccl_debug:+NCCL_DEBUG=${nccl_debug}} \
      USE_FSDP_FULLY_SHARDED=${use_fsdp_fully_sharded} USE_TORCHRUN_SINGLE_NODE=${use_torchrun_single_node} \
      RUN_ID=\"${run_id}\" AST_TRACE_DIR=\"${trace_dir}\" \
      RETRAIN_DEBUG_STAGES=${RETRAIN_DEBUG_STAGES:-0} DEBUG_NCCL=${DEBUG_NCCL:-0} \
      NNODES=${nnodes} NODE_RANK=0 \
      MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} \
      NPROC_PER_NODE=${nproc} \
      ${generated_hostfile:+DEEPSPEED_HOSTFILE=\"$generated_hostfile\"} \
      bash \"$script_path\" ${extra_args_quoted}"
    rc=$?
  else
    if [[ -n "$ssh_user" ]]; then
      ssh_target="${ssh_user}@${master_addr}"
    else
      ssh_target="${master_addr}"
    fi
    echo "[cluster] launching rank 0 on ${master_addr} (foreground)"
     local node_log="${trace_dir}/ssh_node_rank0.${master_addr}.log"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ssh_target" \
      "bash -lc 'cd \"$workdir\" && ${env_setup} \
       mkdir -p \"${trace_dir}\" >/dev/null 2>&1 || true; \
       INTERNAL_LAUNCH=1 LAUNCH_MODE=multi \
       DRY_RUN=${dry_run} DEBUG_LAUNCH=${debug_launch} \
       FORCE_TCP=${force_tcp} DISABLE_P2P=${disable_p2p} \
       ${nccl_socket_ifname:+NCCL_SOCKET_IFNAME=\"${nccl_socket_ifname}\"} \
       NCCL_NVLS_ENABLE=${nccl_nvls_enable} \
       ${nccl_ib_disable:+NCCL_IB_DISABLE=${nccl_ib_disable}} \
       ${nccl_net_gdr_level:+NCCL_NET_GDR_LEVEL=${nccl_net_gdr_level}} \
       ${nccl_ib_gid_index:+NCCL_IB_GID_INDEX=${nccl_ib_gid_index}} \
       ${nccl_ib_timeout:+NCCL_IB_TIMEOUT=${nccl_ib_timeout}} \
       ${nccl_ib_retry_cnt:+NCCL_IB_RETRY_CNT=${nccl_ib_retry_cnt}} \
       ${nccl_ib_qps:+NCCL_IB_QPS_PER_CONNECTION=${nccl_ib_qps}} \
       ${nccl_ib_cuda_support:+NCCL_IB_CUDA_SUPPORT=${nccl_ib_cuda_support}} \
       ${nccl_shm_disable:+NCCL_SHM_DISABLE=${nccl_shm_disable}} \
       ${nccl_p2p_disable:+NCCL_P2P_DISABLE=${nccl_p2p_disable}} \
       ${nccl_proto:+NCCL_PROTO=${nccl_proto}} \
       ${nccl_debug:+NCCL_DEBUG=${nccl_debug}} \
       USE_FSDP_FULLY_SHARDED=${use_fsdp_fully_sharded} USE_TORCHRUN_SINGLE_NODE=${use_torchrun_single_node} \
       RUN_ID=\"${run_id}\" AST_TRACE_DIR=\"${trace_dir}\" \
       RETRAIN_DEBUG_STAGES=${RETRAIN_DEBUG_STAGES:-0} DEBUG_NCCL=${DEBUG_NCCL:-0} \
       NNODES=${nnodes} NODE_RANK=0 \
       MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} \
       NPROC_PER_NODE=${nproc} \
       ${generated_hostfile:+DEEPSPEED_HOSTFILE=\"$generated_hostfile\"} \
       bash \"$remote_script_path\" ${extra_args_quoted}'" \
      >"${node_log}" 2>&1
    rc=$?
    # If rank0 runs remotely, its output is captured by the redirection above.
  fi

  # Wait for background ssh sessions.
  local pid
  for pid in "${bg_pids[@]}"; do
    wait "$pid" || true
  done

  return $rc
}
