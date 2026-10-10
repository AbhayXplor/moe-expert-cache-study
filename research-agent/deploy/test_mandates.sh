#!/usr/bin/env bash
# ============================================================================
# test_mandates.sh — A/B harness for the three memory-isolation mandates
#   T1 speculative-stack arms (Mandate 2: verified layer bypass)
#   T2 KV-tier fidelity + divergence metric (Mandate 1: lossless pruning test)
#   T3 long-context scaling via llama-bench (SWA linearity)
#   T4 synthetic-acceptance calibration of the throughput model
#   T5 isolation monitor wrapper (Mandate 3) — Linux/WSL; on Windows use
#      isolate_and_verify.ps1 around this harness
# Usage: TESTS="T1 T2 T3" ./test_mandates.sh     (default: all)
# ============================================================================
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma-4-12b-qat}"
MODEL="$MODEL_DIR/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
MTP="$MODEL_DIR/MTP/mtp-gemma-4-12B-it-Q4_0.gguf"
LLAMA_CLI="${LLAMA_CLI:-./llama.cpp/build/bin/llama-cli}"
LLAMA_BENCH="${LLAMA_BENCH:-./llama.cpp/build/bin/llama-bench}"
CTX="${CTX:-32768}"
N_PREDICT="${N_PREDICT:-256}"
TESTS="${TESTS:-T1 T2 T3 T4 T5}"
OUTDIR="${OUTDIR:-./mandate_results}"; mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_NO_PINNED=1

BASE_FLAGS=(-m "$MODEL" -ngl 99 -fa on -c "$CTX" --no-mmap --cache-ram 0
            --temp 0 --seed 42 -no-cnv --no-display-prompt -n "$N_PREDICT"
            --n-batch 512 --ubatch-size 512)

# Long technical "conversation history" with exact syntax markers to protect:
read -r -d '' PROMPT <<'EOF' || true
CONTEXT HISTORY:
service: log-rotator v2.3.1 ; host: node-17.internal ; env: LOG_RETENTION=5
function rotate_logs(dir="/var/log/app", keep=5) must preserve gzip -9 archives
flags in use: --dry-run --force --utc ; cron: "17 3 * * 0" ; owner uid 1042
The parser rejects any deviation from POSIX bash; identifiers are byte-exact.
TASK: rewrite rotate_logs for /var/log/app keeping exactly 5 newest *.log and
gzipping the rest with gzip -9. Reference --dry-run, LOG_RETENTION, uid 1042.
Output only the bash code block.
EOF

ts() { date +%s.%N; }
tps_from_log() { grep -oE '[0-9.]+ tokens per second' "$1" | tail -n1 | grep -oE '^[0-9.]+' || echo "0"; }
acc_from_log() { grep -iE 'accept' "$1" | tail -n1 || echo "n/a"; }
divergence() { # $1=ref $2=candidate -> char-level similarity + first-diff position
  python3 - "$1" "$2" <<'PY'
import sys, difflib
a=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
b=open(sys.argv[2],encoding='utf-8',errors='ignore').read()
r=difflib.SequenceMatcher(None,a,b).ratio()
fd=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)) if a!=b else -1)
print(f"similarity={r:.4f} first_divergence_char={fd} identical={'YES' if a==b else 'NO'}")
PY
}

# ------------------------------- T5 monitor --------------------------------
MON_PID=""
t5_start() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    ( while true; do
        echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" \
          >> "$OUTDIR/t5_vram.log"; sleep 1; done ) & MON_PID=$!
    echo "[T5] VRAM monitor started (peak logged to $OUTDIR/t5_vram.log)"
  else echo "[T5][skip] nvidia-smi not found"; fi
}
t5_stop() {
  [[ -n "$MON_PID" ]] && kill "$MON_PID" 2>/dev/null
  if [[ -s "$OUTDIR/t5_vram.log" ]]; then
    PEAK=$(awk '{print $2}' "$OUTDIR/t5_vram.log" | sort -n | tail -n1)
    echo "[T5] peak dedicated VRAM during suite: ${PEAK} MiB (budget 8192; PASS if <=7900)"
    (( PEAK <= 7900 )) && echo "RESULT T5 PASS peak=${PEAK}MiB" || echo "RESULT T5 FAIL peak=${PEAK}MiB"
  fi
}

# ---------------------------------- T1 -------------------------------------
run_arm() { # $1=name  $2...=extra flags
  local name="$1"; shift
  local log="$OUTDIR/t1_${name}.log" out="$OUTDIR/t1_${name}.out"
  local t0 t1
  t0=$(ts)
  "$LLAMA_CLI" "${BASE_FLAGS[@]}" "$@" -p "$PROMPT" > "$out" 2> "$log"
  t1=$(ts)
  local wall tps acc
  wall=$(python3 -c "print(f'{$t1-$t0:.2f}')")
  tps=$(tps_from_log "$log"); acc=$(acc_from_log "$log")
  echo "RESULT T1 arm=${name} wall=${wall}s eval_tps=${tps} acceptance='${acc}'"
}
t1() {
  echo "== T1 speculative-stack arms (greedy => lossless verification test) =="
  run_arm none        --spec-type none
  run_arm ngrammod    --spec-type ngram-mod --spec-ngram-mod-n-match 24 \
                      --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
  run_arm mtp         -md "$MTP" --spec-type draft-mtp --spec-draft-n-max 2 \
                      --spec-draft-ngl all -ctkd q8_0 -ctvd q8_0
  run_arm mtp_ngram   -md "$MTP" --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 \
                      --spec-draft-ngl all -ctkd q8_0 -ctvd q8_0
  echo "-- T1 lossless check (greedy outputs must be identical across arms):"
  for a in ngrammod mtp mtp_ngram; do
    echo "   none vs $a: $(divergence "$OUTDIR/t1_none.out" "$OUTDIR/t1_${a}.out")"
  done
}

# ---------------------------------- T2 -------------------------------------
t2() {
  echo "== T2 KV-tier fidelity (Mandate 1 divergence metric) =="
  for kv in f16 q8_0 q4_0; do
    local log="$OUTDIR/t2_${kv}.log" out="$OUTDIR/t2_${kv}.out"
    "$LLAMA_CLI" "${BASE_FLAGS[@]}" --cache-type-k "$kv" --cache-type-v "$kv" \
        -p "$PROMPT" > "$out" 2> "$log"
    echo "RESULT T2 kv=${kv} eval_tps=$(tps_from_log "$log")"
  done
  echo "-- divergence vs f16 ground truth (lower is safer for syntax markers):"
  for kv in q8_0 q4_0; do
    echo "   f16 vs $kv: $(divergence "$OUTDIR/t2_f16.out" "$OUTDIR/t2_${kv}.out")"
  done
  echo "-- syntax-marker integrity spot check (grep exact identifiers in q8_0 output):"
  for marker in rotate_logs /var/log/app gzip LOG_RETENTION 1042; do
    grep -q -- "$marker" "$OUTDIR/t2_q8_0.out" \
      && echo "   marker '$marker' preserved" || echo "   marker '$marker' MISSING"
  done
}

# ---------------------------------- T3 -------------------------------------
t3() {
  echo "== T3 long-context decode scaling (SWA linearity) =="
  "$LLAMA_BENCH" -m "$MODEL" -ngl 99 -fa 1 -c $((CTX + 2048)) \
      -pg 8192,128 -pg 16384,128 -pg 32768,128 -r 2 2>&1 | tee "$OUTDIR/t3.log"
  echo "RESULT T3 see table above; PASS if tg(32K) >= 0.9 * tg(8K)"
}

# ---------------------------------- T4 -------------------------------------
t4() {
  echo "== T4 synthetic-acceptance calibration (throughput model) =="
  for L in 1.5 2.0 2.5 3.0; do
    local log="$OUTDIR/t4_L${L}.log"
    "$LLAMA_CLI" "${BASE_FLAGS[@]}" -md "$MTP" --spec-type draft-mtp \
        --spec-draft-n-max 3 --spec-draft-ngl all --spec-synth-len "$L" \
        -p "$PROMPT" > /dev/null 2> "$log"
    echo "RESULT T4 synth_len=${L} eval_tps=$(tps_from_log "$log")"
  done
  echo "(Compare slope against PLAYBOOK §2.4: tps ~ baseline * f(L).)"
}

# --------------------------------- main ------------------------------------
trap 't5_stop' EXIT
for t in $TESTS; do
  case "$t" in
    T5) t5_start ;;
    T1) t1 ;; T2) t2 ;; T3) t3 ;; T4) t4 ;;
    *) echo "unknown test $t" ;;
  esac
done
[[ "$TESTS" != *T5* ]] && t5_stop
echo; echo "Artifacts in $OUTDIR — grep '^RESULT' for machine-readable summary."
