#!/usr/bin/env bash
set -euo pipefail

HOSTS_FILE=${1:-$(dirname "$0")/hosts.txt}
# Query memory total (MiB), memory used (MiB), gpu util (%), power draw (W)
CMD='nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits'

# Parser: take CSV lines (memory.total, memory.used, util.gpu, power.draw)
# and compute per-node averages: memory-util-percent, gpu-util-percent, power(W)
parse_and_print_metrics() {
  local node="$1"
  awk -F", *" '
  {
    mt = ($1+0);
    mu = ($2+0);
    gu = ($3+0);
    # power.draw may be N/A; treat non-numeric as missing
    pd_raw = $4;
    if (pd_raw ~ /^[0-9]+(\.[0-9]+)?$/) { pd = pd_raw + 0; sum_power += pd; power_count++ }
    sum_mem_total += mt;
    sum_mem_used += mu;
    sum_gpu += gu;
    count++;
  }
  END {
    if (count == 0) { print "=== " node " ===\n  No GPUs detected or nvidia-smi returned no data"; exit }
    if (sum_mem_total > 0) avg_mem_util = sum_mem_used / sum_mem_total * 100; else avg_mem_util = 0;
    avg_gpu = sum_gpu / count;
    if (power_count > 0) avg_power = sum_power / power_count; else avg_power = -1;
    if (avg_power >= 0) {
      printf "=== %s ===\n  AvgMemUtil(%%): %.1f\n  AvgGPUUtil(%%): %.1f\n  AvgPower(W): %.1f\n", node, avg_mem_util, avg_gpu, avg_power
    } else {
      printf "=== %s ===\n  AvgMemUtil(%%): %.1f\n  AvgGPUUtil(%%): %.1f\n  AvgPower(W): N/A\n", node, avg_mem_util, avg_gpu
    }
  }'
}

# collect local IPs for self-detection
_local_ips=()
if command -v ip >/dev/null 2>&1; then
  while read -r _line; do
    _local_ips+=("${_line%%/*}")
  done < <(ip -o -4 addr show | awk '{print $4}')
else
  # fallback to hostname -I
  read -r -a _local_ips <<< "$(hostname -I 2>/dev/null || echo "127.0.0.1")"
fi

if [ ! -f "$HOSTS_FILE" ]; then
  echo "hosts file not found: $HOSTS_FILE" >&2
  exit 2
fi

# If pdsh exists, use it for parallelism
if command -v pdsh >/dev/null 2>&1; then
  hosts_csv=$(paste -sd, "$HOSTS_FILE")
  echo "Using pdsh to query: $hosts_csv"
  pdsh -w "$hosts_csv" "$CMD"
  exit 0
fi

# Otherwise iterate hosts; run locally when host matches local IP
while read -r host; do
  host=$(echo "$host" | tr -d '\r\n' | sed -e 's/#.*$//' -e 's/^\s*//;s/\s*$//')
  [ -z "$host" ] && continue
  is_local=0
  for ip in "${_local_ips[@]}"; do
    if [ "$host" = "$ip" ] || [ "$host" = "localhost" ] || [ "$host" = "127.0.0.1" ]; then
      is_local=1
      break
    fi
  done
  if [ "$is_local" -eq 1 ]; then
    echo "(local) ${host}"
    if output=$($CMD 2>/dev/null); then
      echo "$output" | parse_and_print_metrics "$host"
    else
      echo "=== $host ===\n  nvidia-smi failed locally"
    fi
  else
    echo "(remote) ${host}"
    if output=$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$host" "$CMD" 2>/dev/null); then
      echo "$output" | parse_and_print_metrics "$host"
    else
      echo "=== $host ===\n  SSH or nvidia-smi failed: $host"
    fi
  fi
done < "$HOSTS_FILE"
