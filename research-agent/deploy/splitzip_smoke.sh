#!/usr/bin/env bash
# ============================================================================
# splitzip_smoke.sh — validate the SplitZip lossless KV codec on YOUR GPU
# before designing any integration around it.
#
# What it proves:
#   1) Triton kernels compile & run on your GPU (Ada sm_89 OK; Triton needs
#      Linux or WSL2 — native Windows is NOT supported)
#   2) Bitwise-lossless round-trip on synthetic BF16 KV-shaped tensors
#      (bench script asserts torch.equal before reporting ratios)
#   3) Encode/decode throughput on your card vs your PCIe link
#      (laptop PCIe4 x8 ~ 16 GB/s — codec must beat it to be "free" on transfer)
#
# Usage:
#   ./splitzip_smoke.sh                 # clone + venv + synthetic smoke test
#   TENSOR_PT=/path/kv_tensor.pt ./splitzip_smoke.sh   # test on real KV tensors
# ============================================================================
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/splitzip-test}"
REPO="https://github.com/Intelligent-Microsystems-Lab/SplitZip"
PY="${PY:-python3}"

echo "== SplitZip smoke test =="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || \
  { echo "[FATAL] nvidia-smi not found"; exit 1; }
uname -r | grep -qi microsoft && echo "[info] WSL2 detected — good (Triton supported)."
uname -s | grep -qi linux || { echo "[FATAL] Run under Linux or WSL2 (Triton requirement)."; exit 1; }

# ---- 1. fetch repo ----------------------------------------------------------
if [[ ! -d "$WORKDIR/SplitZip" ]]; then
  echo "[1/4] cloning $REPO ..."
  mkdir -p "$WORKDIR"
  git clone --depth 1 "$REPO" "$WORKDIR/SplitZip"
else
  echo "[1/4] repo present at $WORKDIR/SplitZip"
fi
cd "$WORKDIR/SplitZip"

# ---- 2. environment ---------------------------------------------------------
if [[ ! -d "$WORKDIR/venv" ]]; then
  echo "[2/4] creating venv + installing requirements (torch/triton per requirements.txt) ..."
  "$PY" -m venv "$WORKDIR/venv"
  "$WORKDIR/venv/bin/pip" install -q --upgrade pip
  "$WORKDIR/venv/bin/pip" install -q -r requirements.txt
else
  echo "[2/4] venv present"
fi
PYBIN="$WORKDIR/venv/bin/python"

# ---- 3. synthetic round-trip + throughput -----------------------------------
echo "[3/4] running codec smoke test (bitwise round-trip asserted internally) ..."
"$PYBIN" bench_codec_throughput.py \
    --device cuda:0 \
    --synthetic --calibrate-on-input \
    --rows 4096 --hidden-dim 4096 --chunk-size 1024 \
    --warmup 3 --iters 20 --repeats 3 \
    --output "$WORKDIR/splitzip_smoke_synthetic.json"
echo "     -> results: $WORKDIR/splitzip_smoke_synthetic.json"

# optional: real KV tensors saved from your own runs (bf16 .pt)
if [[ -n "${TENSOR_PT:-}" && -f "${TENSOR_PT}" ]]; then
  echo "[3b] real-tensor test on $TENSOR_PT ..."
  "$PYBIN" bench_codec_throughput.py \
      --device cuda:0 --input "$TENSOR_PT" --chunk-size 1024 \
      --output "$WORKDIR/splitzip_smoke_real.json"
fi

# ---- 4. verdict --------------------------------------------------------------
echo "[4/4] verdict guidance:"
cat <<'EOF'
  - JSON contains ratio ~= 1.31x (BF16) and PASS on bitwise verification.
  - Compare decode GB/s against your PCIe link:
      lspci | grep -i nvidia ; nvidia-smi -q | grep -i "link width\|gen"
    Laptop RTX 4060 (PCIe4 x8 ~= 16 GB/s): any codec speed >> 16 GB/s means
    compression is free on the transfer path (its intended home).
  - REMEMBER: SplitZip does NOT shrink what attention kernels read — it is a
    movement/storage codec (KV save-restore, PD-disaggregation, VeriCache-style
    swap legs). Decode-time VRAM residency is unaffected.
EOF
echo "RESULT SPLITZIP_SMOKE DONE"
