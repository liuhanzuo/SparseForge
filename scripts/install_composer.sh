#!/usr/bin/env bash
set -euo pipefail

HOSTS_FILE=${1:-scripts/hosts.txt}
SSH_USER=${SSH_USER:-}
PYPI_INDEX=${PYPI_INDEX:-https://pypi.org/simple}
CONDA_ENV=${CONDA_ENV:-minillm}
RETRIES=${RETRIES:-2}

if [ ! -f "$HOSTS_FILE" ]; then
  echo "Hosts file not found: $HOSTS_FILE" >&2
  exit 2
fi

run_on_host() {
  host="$1"
  prefix=""
  if [ -n "$SSH_USER" ]; then
    prefix="$SSH_USER@"
  fi
  if [[ "$host" == "$(hostname -I | awk '{print $1}')" || "$host" == "$(hostname)" ]]; then
    # local
    echo "==> [LOCAL] installing on $host"
    cmd="set -euo pipefail; source ~/.bashrc >/dev/null 2>&1 || true; conda activate $CONDA_ENV >/dev/null 2>&1 || true; python -m pip install --upgrade pip setuptools wheel; python -m pip install composer --no-cache-dir -i $PYPI_INDEX"
    bash -lc "$cmd"
    return $?
  else
    echo "==> [SSH] installing on $host"
    cmd="set -euo pipefail; source ~/.bashrc >/dev/null 2>&1 || true; conda activate $CONDA_ENV >/dev/null 2>&1 || true; python -m pip install --upgrade pip setuptools wheel; python -m pip install composer --no-cache-dir -i $PYPI_INDEX"
    ssh -o BatchMode=yes -o ConnectTimeout=10 ${prefix}${host} "$cmd"
    return $?
  fi
}

echo "Installing composer on hosts from $HOSTS_FILE (PYPI_INDEX=$PYPI_INDEX)"

pids=()
failures=()
while read -r host || [ -n "$host" ]; do
  host=$(echo "$host" | tr -d '\r' | sed -n '1p')
  if [ -z "$host" ]; then
    continue
  fi
  (
    set -x
    attempt=0
    until [ $attempt -gt $RETRIES ]; do
      if run_on_host "$host"; then
        echo "SUCCESS: $host"
        exit 0
      else
        echo "WARN: install failed on $host (attempt $attempt), retrying..."
        attempt=$((attempt+1))
        sleep 2
      fi
    done
    echo "FAIL: $host" >&2
    exit 1
  ) &
  pids+=($!)
done < "$HOSTS_FILE"

exit_code=0
for pid in "${pids[@]}"; do
  wait "$pid" || exit_code=$?
done

if [ $exit_code -ne 0 ]; then
  echo "One or more hosts failed to install composer." >&2
  exit $exit_code
fi

echo "composer installed on all hosts."
#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/install_composer.sh [hosts_file]
# Default hosts_file: scripts/hosts.txt

HOSTS_FILE=${1:-$(dirname "$0")/hosts.txt}
SSH_USER=${SSH_USER:-}
TIMEOUT=${SSH_TIMEOUT:-10}

if [ ! -f "$HOSTS_FILE" ]; then
  echo "hosts file not found: $HOSTS_FILE" >&2
  exit 2
fi

CMD="source ~/.bashrc >/dev/null 2>&1 || true; conda activate minillm >/dev/null 2>&1 && \
python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 && \
python -m pip install composer --no-cache-dir"

pids=()
results=()

while read -r host; do
  host=$(echo "$host" | tr -d '\r\n' | sed -e 's/#.*$//' -e 's/^\s*//;s/\s*$//')
  [ -z "$host" ] && continue

  if [ -n "$SSH_USER" ]; then
    ssh_target="${SSH_USER}@${host}"
  else
    ssh_target="$host"
  fi

  echo "=== $host ==="
  # run in background so all hosts install in parallel
  (
    if [ "$host" = "localhost" ] || [ "$host" = "127.0.0.1" ]; then
      bash -lc "$CMD" && echo "[OK] $host" || echo "[FAIL] $host"
    else
      ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=$TIMEOUT "$ssh_target" \
        "bash -lc '$CMD'" && echo "[OK] $host" || echo "[FAIL] $host"
    fi
  ) &
  pids+=("$!")

done < "$HOSTS_FILE"

# wait for all background jobs
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo "All installs attempted. Use scripts/check_gpus.sh or manual SSH to verify." 
