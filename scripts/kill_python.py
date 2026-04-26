#!/usr/bin/env python3
"""
kill_python.py

Usage:
  bash scripts/kill_python.py 0 1 3

This script reads `scripts/hosts.txt` for host list and runs a safe pkill on each
selected host index over SSH. If a host string (not an index) is provided, it will
be used directly as a hostname. If no hosts file found, arguments are treated as
hostnames. The SSH command attempts a graceful `pkill -f python`, and falls back
to `pkill -9` if `--force` is provided.

Requires passwordless SSH (keys) to the target hosts for non-interactive runs.
"""
import sys
import subprocess
from pathlib import Path


def load_hosts(hosts_path: Path):
    if not hosts_path.exists():
        return []
    with hosts_path.open('r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith('#')]
    return lines


def run_ssh(host: str, cmd: str, dry_run: bool = False, user: str = None, port: int = None, no_batch: bool = False):
    ssh_cmd = ['ssh']
    if port is not None:
        ssh_cmd += ['-p', str(port)]
    if not no_batch:
        ssh_cmd += ['-o', 'BatchMode=yes']
    target = f"{user}@{host}" if user else host
    ssh_cmd += [target, cmd]
    print(f"[{target}] -> {cmd}")
    if dry_run:
        return 0
    try:
        res = subprocess.run(ssh_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            out = res.stdout.strip()
            if out:
                print(f"[{target}] stdout: {out}")
            return 0
        else:
            print(f"[{target}] ssh failed rc={res.returncode} stderr: {res.stderr.strip()}")
            return res.returncode
    except Exception as e:
        print(f"[{target}] ssh exception: {e}")
        return 2


def run_local(cmd: str, dry_run: bool = False):
    print(f"[local] -> {cmd}")
    if dry_run:
        return 0
    try:
        res = subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            out = res.stdout.strip()
            if out:
                print(f"[local] stdout: {out}")
            return 0
        else:
            print(f"[local] failed rc={res.returncode} stderr: {res.stderr.strip()}")
            return res.returncode
    except Exception as e:
        print(f"[local] exception: {e}")
        return 2


def main(argv):
    if len(argv) <= 1:
        print("Usage: kill_python.py <NODE_IDX|HOST> [<NODE_IDX|HOST> ...] [--force] [--dry-run]")
        print("Nodes are indices into scripts/hosts.txt (0-based)." )
        return 1

    args = argv[1:]
    force = False
    dry = False
    ssh_user = None
    ssh_port = None
    no_batch = False
    hosts_file = Path(__file__).parent / 'hosts.txt'
    hosts_list = load_hosts(hosts_file)

    # simple flag parsing
    remaining = []
    for a in args:
        if a == '--force':
            force = True
        elif a == '--dry-run':
            dry = True
        elif a.startswith('--ssh-user='):
            ssh_user = a.split('=', 1)[1]
        elif a.startswith('--ssh-port='):
            try:
                ssh_port = int(a.split('=', 1)[1])
            except Exception:
                ssh_port = None
        elif a == '--no-batch':
            no_batch = True
        else:
            remaining.append(a)
    args = remaining

    targets = []
    for a in args:
        # numeric -> index into hosts_list
        if a.isdigit():
            idx = int(a)
            if idx < 0 or idx >= len(hosts_list):
                print(f"Index {idx} out of range (hosts file len={len(hosts_list)}). Skipping.")
                continue
            targets.append(hosts_list[idx])
        else:
            targets.append(a)

    if not targets:
        print("No valid targets resolved.")
        return 1

    # command to run on remote/local
    if force:
        cmd = "sudo pkill -9 -f python || true"
    else:
        # try graceful first, then list remaining
        cmd = "pkill -f python || true; sleep 1; pgrep -a -f python || echo 'no-python'"

    # execute
    exit_codes = []
    for h in targets:
        # if the host looks like localhost or loopback, run locally
        if h in ("localhost", "127.0.0.1", "::1") or h == subprocess.getoutput('hostname'):
            rc = run_local(cmd, dry_run=dry)
        else:
            rc = run_ssh(h, cmd, dry_run=dry, user=ssh_user, port=ssh_port, no_batch=no_batch)
        exit_codes.append(rc)

    failed = [c for c in exit_codes if c != 0]
    if failed:
        print(f"Completed with failures: {failed}")
        return 2
    print("Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
