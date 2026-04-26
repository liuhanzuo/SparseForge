#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-c4_en_train_00000_00031}"
mkdir -p "$OUTDIR"

BASE="https://huggingface.co/datasets/allenai/c4/resolve/main/en"

# 如果你遇到 401/限流，可在环境里设置 HF_TOKEN，然后解开下面两行之一：
# AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
AUTH_HEADER=()

for i in $(seq 0 0); do
  printf -v shard "%05d" "$i"
#   file="c4-train.${shard}-of-01024.json.gz"
  file="c4-validation.${shard}-of-00008.json.gz"
  url="${BASE}/${file}"

  echo "[+] ${file}"
  # -L 跟随重定向；-C - 断点续传；--fail 非 2xx 直接失败；--retry 自动重试
  curl -L --fail --retry 5 --retry-delay 2 -C - \
    "${AUTH_HEADER[@]}" \
    -o "${OUTDIR}/${file}.part" \
    "$url"
  mv "${OUTDIR}/${file}.part" "${OUTDIR}/${file}"
done

echo "[OK] Done: ${OUTDIR}"
