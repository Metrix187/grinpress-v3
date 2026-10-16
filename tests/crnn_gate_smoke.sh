#!/usr/bin/env bash
# Dry-run of run_v3.sh's CRNN resume + gate logic (no GPU, no training): fake Drive with the real CRNN_DONE/CRNN_FT_DONE
# contents from the 2026-10-16 Colab run, dummy weights, stubbed train_crnn.py (must never be called) and a stubbed
# train_teacher_v3.py (pregen reached = success; train fails -> run exits with MAX_RETRIES=1, T_MICRO=1).
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); BIN=$T/bin; mkdir -p $BIN
command -v python >/dev/null || ln -s "$(command -v python3)" $BIN/python; export PATH=$BIN:$PATH
setup() {  # $1 = case dir
  W=$T/$1; mkdir -p $W/src/artifacts $W/drive/assets $W/drive/crnn $W/local
  cp "$REPO/run_v3.sh" $W/src/
  printf 'import sys; print("[stub] train_crnn.py CALLED (must not happen)"); sys.exit(9)\n' > $W/src/train_crnn.py
  printf 'import sys; print("[stub] train_teacher_v3.py", sys.argv[1], "-> crnn arg:", sys.argv[sys.argv.index("--crnn")+1] if "--crnn" in sys.argv else "-"); sys.exit(0 if sys.argv[1] == "pregen" else 3)\n' > $W/src/train_teacher_v3.py
  echo '{"source": "stage0", "ref48": {"n": 48, "exact": 0.9583333333333334}, "gate": false}' > $W/drive/CRNN_DONE
  echo '{"source": "stage0b_finetune", "ref48": {"n": 48, "exact": 0.9791666666666666, "exact_ci": 0.9791666666666666, "char_acc": 0.9983974358974358}, "misses": [["newspaper boy", "newspapee boy"]], "gate": false}' > $W/drive/CRNN_FT_DONE
  echo STAGE0 > $W/drive/assets/crnn.safetensors; echo STAGE0 > $W/drive/assets/crnn_stage0.safetensors; echo FINETUNED > $W/drive/assets/crnn_ft.safetensors
  echo '{"phase": "error", "note": "crnn_gate_failed"}' > $W/drive/status.json; touch $W/drive/crnn_gate_failed $W/local/v1_teacher.safetensors
}
run() {  # $1 = case, rest = env
  c=$1; shift; echo "================ case $c: $*"
  (cd $T/$c/src && env DRIVE=$T/$c/drive LOCAL=$T/$c/local MAX_RETRIES=1 T_MICRO=1 WORKERS=1 "$@" bash run_v3.sh 2>&1 | grep -v "^$"; echo "exit=${PIPESTATUS[0]}")
  echo "artifacts/crnn.safetensors = $(cat $T/$c/src/artifacts/crnn.safetensors 2>/dev/null || echo none); drive assets/crnn.safetensors = $(cat $T/$c/drive/assets/crnn.safetensors)"
  echo "markers: $(cd $T/$c/drive && ls -1 | tr '\n' ' ')"; echo "status.json: $(cat $T/$c/drive/status.json)"
  [ -f $T/$c/drive/CRNN_GATE.json ] && echo "CRNN_GATE.json: $(python3 -c "import json;d=json.load(open('$T/$c/drive/CRNN_GATE.json'));print({k:d[k] for k in ('override','ref48_correct','threshold','accept_ft','weights','reason') if k in d})")"
}
setup A; run A CRNN_GATE=0.979
setup B; run B
setup C; run C CRNN_ACCEPT_FT=1
setup D; rm $T/D/drive/assets/crnn_ft.safetensors; run D CRNN_GATE=0.979
setup E; run E CRNN_GATE=0.979; echo "--- E resumed a second time (idempotent)"; run E CRNN_GATE=0.979
rm -rf "$T"
