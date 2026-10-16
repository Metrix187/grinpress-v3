#!/usr/bin/env bash
# grinpress v3: idempotent, unattended end-to-end run (Colab A100). Safe to start again at any time: every stage
# is skipped when its marker exists, and resumes from its newest checkpoint / chunk otherwise. DONE short-circuits.
#   stages: v1 teacher download -> CRNN train (stage 0, GPU) -> pregen teacher data -> teacher fine-tune (Phase 3)
#           -> reflow pairs (Phase 4a) -> student distill (Phase 4b) -> export + eval + bench (Phase 5) -> DONE
# Every file written to Drive is < 100 MB (sharded ckpts, 1000-pair shards, fp16 weights, split fp32 ONNX).
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1   # every python stage logs to run.log immediately (no block-buffered stdout)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
DRIVE=${DRIVE:-/content/drive/MyDrive/grinpress_v3}; export DRIVE
LOCAL=${LOCAL:-/content/grinpress_v3_local}
# ---- Phase 2 stage 0 (CRNN on the GPU)
C_NTRAIN=${C_NTRAIN:-120000}; C_EPOCHS=${C_EPOCHS:-30}; C_BS=${C_BS:-256}; C_LR=${C_LR:-2e-3}; C_HARD=${C_HARD:-0.4}
C_WIDTH=${C_WIDTH:-"32 64 128 192 256"}; C_SEQ=${C_SEQ:-4}; C_SEED=${C_SEED:-31}
# stage 0b: one fine-tune pass if the stage-0 CRNN missed the gate (init from it, new hard-case data, lower LR)
C_FT=${C_FT:-1}; CF_NTRAIN=${CF_NTRAIN:-60000}; CF_EPOCHS=${CF_EPOCHS:-24}; CF_LR=${CF_LR:-6e-4}; CF_HARD=${CF_HARD:-0.6}; CF_SEED=${CF_SEED:-47}
CRNN_FROM_REPO=${CRNN_FROM_REPO:-0}; REQUIRE_CRNN_GATE=${REQUIRE_CRNN_GATE:-1}
# gate override (user-approved): CRNN_GATE = ref48-exact threshold (default 0.98 = the spec gate); CRNN_ACCEPT_FT=1 accepts the
# stage-0b fine-tuned CRNN whatever its score. Either way the run uses assets/crnn_ft.safetensors, never the stage-0 weights,
# logs the override and records it in CRNN_GATE.json, TRAIN_DONE, eval_ocr_v3.json and DONE.
CRNN_GATE=${CRNN_GATE:-0.98}; CRNN_ACCEPT_FT=${CRNN_ACCEPT_FT:-0}; CRNN_GATE_NOTE=${CRNN_GATE_NOTE:-user-approved}
# ---- Phase 3 (teacher)
PREGEN_N=${PREGEN_N:-100000}
T_BATCH=${T_BATCH:-128}; T_MICRO=${T_MICRO:-64}; T_STEPS=${T_STEPS:-40000}; T_LR=${T_LR:-1e-4}
T_VAL_EVERY=${T_VAL_EVERY:-1000}; T_STOP=${T_STOP:-0.95}; T_VAL_STEPS=${T_VAL_STEPS:-50}; VAL_LIMIT=${VAL_LIMIT:-0}
REQUIRE_TEACHER_GATE=${REQUIRE_TEACHER_GATE:-0}   # 1 = stop the run if the teacher never beats T_STOP on the 48-grid
# ---- Phase 4 (pairs + student)
NUM_PAIRS=${NUM_PAIRS:-50000}; SHARD=${SHARD:-1000}; GEN_BATCH=${GEN_BATCH:-128}; TEACHER_STEPS=${TEACHER_STEPS:-50}
S_BATCH=${S_BATCH:-128}; S_MICRO=${S_MICRO:-32}; S_STEPS=${S_STEPS:-20000}; S_LR=${S_LR:-5e-5}; S_VAL_EVERY=${S_VAL_EVERY:-2000}
EVAL_TEACHER_STEPS=${EVAL_TEACHER_STEPS:-50}; BENCH_REPS=${BENCH_REPS:-5}
MAX_RETRIES=${MAX_RETRIES:-5}
WORKERS=${WORKERS:-$(( $(nproc) - 1 ))}
mkdir -p "$DRIVE"/{teacher,teacher_ckpt,pairs_v3,student,student_ckpt,onnx,assets} "$LOCAL"/{teacher_data,teacher_ckpt,pairs,student_ckpt,export}
log() { echo "[run_v3 $(date '+%F %T')] $*"; }
beat() { python - "$@" <<PY || true
import json, sys, time, os
p = "$DRIVE/status.json"; d = {"phase": sys.argv[1], "time": time.strftime("%F %T"), "unix": int(time.time())}
if len(sys.argv) > 2: d["note"] = " ".join(sys.argv[2:])
open(p + ".tmp", "w").write(json.dumps(d)); os.replace(p + ".tmp", p)
PY
}
wait_drive() { for i in $(seq 1 60); do ls "$DRIVE" >/dev/null 2>&1 && return 0; log "Drive not reachable, waiting ($i/60)"; sleep 15; done; return 1; }
copy_to_drive() { cp "$1" "$2.tmp" && mv "$2.tmp" "$2"; }
if [ -f "$DRIVE/DONE" ]; then log "DONE marker present, nothing to do"; exit 0; fi

# ---------------------------------------------------------------- 0) inputs
if [ ! -f "$LOCAL/v1_teacher.safetensors" ]; then
  beat download_v1_teacher
  python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('skyuu72/grinpress','model.safetensors'), '$LOCAL/v1_teacher.safetensors')" || { beat error teacher_download; exit 1; }
fi

# ---------------------------------------------------------------- 0b) Phase 2 CRNN (stage 0, GPU; resumable via Drive ckpt)
# Trains the improved recognizer (scale channel, wider model, hard-class data) unless CRNN_DONE exists on Drive, or a
# CRNN that already passed the >=98% ref48 gate was committed (CRNN_FROM_REPO=1 skips training and uses artifacts/).
mkdir -p "$DRIVE/crnn" "$LOCAL/crnn_cache"
if [ "$CRNN_FROM_REPO" = "1" ]; then
  [ -f artifacts/crnn.safetensors ] || { log "artifacts/crnn.safetensors missing"; beat error no_crnn; exit 1; }
  [ -f "$DRIVE/CRNN_DONE" ] || { copy_to_drive artifacts/crnn.safetensors "$DRIVE/assets/crnn.safetensors"; echo '{"source":"repo"}' > "$DRIVE/CRNN_DONE"; }
fi
stuck=0
while [ ! -f "$DRIVE/CRNN_DONE" ]; do
  beat crnn
  python -u train_crnn.py --n_train $C_NTRAIN --n_val 2000 --epochs $C_EPOCHS --bs $C_BS --lr $C_LR --in_ch 2 --hard_frac $C_HARD \
    --width $C_WIDTH --seq_layers $C_SEQ --seed $C_SEED --workers $WORKERS --cache_dir "$LOCAL/crnn_cache" \
    --resume_ckpt "$DRIVE/crnn/crnn_ckpt.pt" --out "$LOCAL/crnn.safetensors" --report "$LOCAL/crnn_train.json" --heartbeat "$DRIVE/status.json" && {
      cp "$LOCAL/crnn.safetensors" artifacts/crnn.safetensors
      copy_to_drive "$LOCAL/crnn.safetensors" "$DRIVE/assets/crnn.safetensors"
      copy_to_drive "$LOCAL/crnn_train.json" "$DRIVE/assets/crnn_train.json"
      python -c "import json; r=json.load(open('$LOCAL/crnn_train.json')); json.dump(dict(source='stage0', ref48=r['final_ref48'], ref48_gtbox=r['final_ref48_with_gt_boxes'], val=r['final_val'], gate=r['gate_ref48_exact_ge_0_98']), open('$DRIVE/CRNN_DONE.tmp','w'))" \
        && mv "$DRIVE/CRNN_DONE.tmp" "$DRIVE/CRNN_DONE"; break; }
  wait_drive || { log "Drive gone 15 min; exiting for remount"; exit 1; }
  stuck=$((stuck + 1)); log "crnn training failed ($stuck/$MAX_RETRIES); resuming from Drive ckpt"
  [ "$stuck" -ge "$MAX_RETRIES" ] && { beat error crnn_stuck; exit 1; }
  sleep 20
done
[ -f "$DRIVE/assets/crnn.safetensors" ] && cp "$DRIVE/assets/crnn.safetensors" artifacts/crnn.safetensors
[ -f artifacts/crnn.safetensors ] || { log "no CRNN"; beat error no_crnn; exit 1; }
CGATE=$(python -c "import json; print(json.load(open('$DRIVE/CRNN_DONE')).get('gate', 'n/a'))")
log "CRNN stage 0 done; ref48 >= 0.98 gate: $CGATE"
# ---------------------------------------------------------------- 0c) CRNN fine-tune (only when stage 0 missed the gate)
# resume: a finished fine-tune is never re-rendered or retrained; its weights must be on Drive
if [ -f "$DRIVE/CRNN_FT_DONE" ]; then
  [ -f "$DRIVE/assets/crnn_ft.safetensors" ] || { log "CRNN_FT_DONE present but assets/crnn_ft.safetensors missing; refusing to retrain"; beat error no_crnn_ft; exit 1; }
  log "CRNN_FT_DONE + assets/crnn_ft.safetensors on Drive: skipping CRNN render/train"
fi
if [ "$CGATE" != "True" ] && [ "$C_FT" = "1" ] && [ "$CRNN_FROM_REPO" != "1" ]; then
  [ -f "$DRIVE/assets/crnn_stage0.safetensors" ] || copy_to_drive "$DRIVE/assets/crnn.safetensors" "$DRIVE/assets/crnn_stage0.safetensors"
  stuck=0
  while [ ! -f "$DRIVE/CRNN_FT_DONE" ]; do
    beat crnn_ft
    GP_CRNN_HARD_FT=1 python -u train_crnn.py --init "$DRIVE/assets/crnn_stage0.safetensors" --n_train $CF_NTRAIN --n_val 2000 \
      --epochs $CF_EPOCHS --bs $C_BS --lr $CF_LR --hard_frac $CF_HARD --seed $CF_SEED --workers $WORKERS --cache_dir "$LOCAL/crnn_cache" \
      --resume_ckpt "$DRIVE/crnn/crnn_ft_ckpt.pt" --out "$LOCAL/crnn_ft.safetensors" --report "$LOCAL/crnn_ft_train.json" --heartbeat "$DRIVE/status.json" && {
        copy_to_drive "$LOCAL/crnn_ft.safetensors" "$DRIVE/assets/crnn_ft.safetensors"
        copy_to_drive "$LOCAL/crnn_ft_train.json" "$DRIVE/assets/crnn_ft_train.json"
        python -c "import json; r=json.load(open('$LOCAL/crnn_ft_train.json')); json.dump(dict(source='stage0b_finetune', ref48=r['final_ref48'], ref48_gtbox=r['final_ref48_with_gt_boxes'], val=r['final_val'], misses=[(m['gt'], m['pred']) for m in r['ref48_misses']], gate=r['gate_ref48_exact_ge_0_98']), open('$DRIVE/CRNN_FT_DONE.tmp','w'))" \
          && mv "$DRIVE/CRNN_FT_DONE.tmp" "$DRIVE/CRNN_FT_DONE"; break; }
    wait_drive || { log "Drive gone 15 min; exiting for remount"; exit 1; }
    stuck=$((stuck + 1)); log "crnn fine-tune failed ($stuck/$MAX_RETRIES); resuming from Drive ckpt"
    [ "$stuck" -ge "$MAX_RETRIES" ] && { beat error crnn_ft_stuck; exit 1; }
    sleep 20
  done
  CGATE=$(python -c "import json; print(json.load(open('$DRIVE/CRNN_FT_DONE')).get('gate', 'n/a'))")
  log "CRNN fine-tune done; ref48 >= 0.98 gate: $CGATE"
  if [ "$CGATE" = "True" ]; then
    copy_to_drive "$DRIVE/assets/crnn_ft.safetensors" "$DRIVE/assets/crnn.safetensors"; cp "$DRIVE/assets/crnn_ft.safetensors" artifacts/crnn.safetensors
  fi
fi
# ---------------------------------------------------------------- 0d) CRNN gate decision (+ explicit, logged override)
rm -f "$DRIVE/CRNN_GATE_FAILED" "$DRIVE/crnn_gate_failed"   # stale failure markers (none are written any more); never touches weights/markers
GDEC=$(DRIVE="$DRIVE" CGATE="$CGATE" CRNN_GATE="$CRNN_GATE" CRNN_ACCEPT_FT="$CRNN_ACCEPT_FT" CRNN_GATE_NOTE="$CRNN_GATE_NOTE" CRNN_FROM_REPO="$CRNN_FROM_REPO" python - <<'PY'
import json, os, time
D = os.environ["DRIVE"]; ft = os.path.join(D, "CRNN_FT_DONE"); repo = os.environ["CRNN_FROM_REPO"] == "1"
src = ft if os.path.exists(ft) and not repo else os.path.join(D, "CRNN_DONE")
m = json.load(open(src)); r = m.get("ref48") or {}; ex = float(r.get("exact", 0) if isinstance(r, dict) else r or 0); n = int(r.get("n", 48) if isinstance(r, dict) else 48)
thr = float(os.environ["CRNN_GATE"]); accept_ft = os.environ["CRNN_ACCEPT_FT"] == "1"; passed = str(m.get("gate")) == "True" or os.environ["CGATE"] == "True"
rec = dict(time=time.strftime("%F %T"), marker=os.path.basename(src), source=m.get("source"), ref48_exact=ex, ref48_correct=round(ex * n), ref48_n=n,
           spec_gate=0.98, spec_gate_passed=passed, threshold=thr, accept_ft=accept_ft, misses=m.get("misses"))
if repo or passed:
    dec = "PASS"; rec.update(override=False, weights="assets/crnn.safetensors")
elif os.path.basename(src) == "CRNN_FT_DONE" and (ex >= thr - 1e-9 or accept_ft):
    dec = "OVERRIDE"; rec.update(override=True, weights="assets/crnn_ft.safetensors", note=os.environ["CRNN_GATE_NOTE"],
                                 reason=f"ref48 {ex:.4f} ({round(ex*n)}/{n}) below spec 0.98, accepted via " + ("CRNN_ACCEPT_FT=1" if accept_ft else f"CRNN_GATE={thr}"))
else:
    dec = "FAIL"; rec.update(override=False, weights=None)
if dec != "FAIL":
    json.dump(rec, open(os.path.join(D, "CRNN_GATE.json.tmp"), "w"), indent=1); os.replace(os.path.join(D, "CRNN_GATE.json.tmp"), os.path.join(D, "CRNN_GATE.json"))
print(dec, f"{ex:.4f}", f"{round(ex*n)}/{n}", thr)
PY
)
read -r GDEC_KIND GDEC_EX GDEC_FRAC GDEC_THR <<<"$GDEC"
case "$GDEC_KIND" in
  PASS) log "CRNN gate passed (ref48 $GDEC_EX, $GDEC_FRAC)";;
  OVERRIDE)
    # use the fine-tuned weights everywhere (artifacts/ for teacher/student/eval, Drive assets/crnn.safetensors for the HF push);
    # the stage-0 weights stay on Drive as assets/crnn_stage0.safetensors
    [ -f "$DRIVE/assets/crnn_stage0.safetensors" ] || copy_to_drive "$DRIVE/assets/crnn.safetensors" "$DRIVE/assets/crnn_stage0.safetensors"
    cmp -s "$DRIVE/assets/crnn_ft.safetensors" "$DRIVE/assets/crnn.safetensors" || copy_to_drive "$DRIVE/assets/crnn_ft.safetensors" "$DRIVE/assets/crnn.safetensors"
    cp "$DRIVE/assets/crnn_ft.safetensors" artifacts/crnn.safetensors
    cmp -s "$DRIVE/assets/crnn_ft.safetensors" artifacts/crnn.safetensors || { log "CRNN override: copying fine-tuned weights failed"; beat error crnn_ft_copy; exit 1; }
    log "CRNN gate override: accepting ref48 $GDEC_EX ($CRNN_GATE_NOTE $GDEC_FRAC; threshold $GDEC_THR, accept_ft=$CRNN_ACCEPT_FT); using fine-tuned weights assets/crnn_ft.safetensors sha256 $(sha256sum artifacts/crnn.safetensors | cut -c1-12)"
    beat crnn_gate_override "ref48 $GDEC_EX ($GDEC_FRAC) accepted";;
  *) log "CRNN gate failed: ref48 ${GDEC_EX:-?} (${GDEC_FRAC:-?}) < ${GDEC_THR:-$CRNN_GATE}"
     if [ "$REQUIRE_CRNN_GATE" = "1" ]; then beat error crnn_gate_failed; exit 1; fi;;
esac

# ---------------------------------------------------------------- 1) Phase 3 teacher fine-tune
if [ ! -f "$DRIVE/TEACHER_DONE" ]; then
  beat pregen
  stuck=0
  until python -u train_teacher_v3.py pregen --data_dir "$LOCAL/teacher_data" --num_samples $PREGEN_N --workers $WORKERS --heartbeat "$DRIVE/status.json"; do
    stuck=$((stuck + 1)); log "pregen failed ($stuck/$MAX_RETRIES)"; [ $stuck -ge $MAX_RETRIES ] && { beat error pregen; exit 1; }; sleep 10
  done
  stuck=0
  while [ ! -f "$DRIVE/TEACHER_DONE" ]; do
    before=$(ls "$LOCAL"/teacher_ckpt/ckpt_*.pt "$DRIVE"/teacher_ckpt/*.manifest.json 2>/dev/null | tail -2 | tr '\n' ' ')
    python -u train_teacher_v3.py train --init "$LOCAL/v1_teacher.safetensors" --crnn artifacts/crnn.safetensors \
      --data_dir "$LOCAL/teacher_data" --checkpoint_dir "$LOCAL/teacher_ckpt" --checkpoint_mirror "$DRIVE/teacher_ckpt" \
      --out_dir "$DRIVE/teacher" --batch_size $T_BATCH --micro_batch $T_MICRO --lr $T_LR --max_steps $T_STEPS \
      --val_every $T_VAL_EVERY --val_steps $T_VAL_STEPS --val_limit $VAL_LIMIT --stop_exact $T_STOP --heartbeat "$DRIVE/status.json" --done_marker "$DRIVE/TEACHER_DONE" && break
    wait_drive || { log "Drive gone 15 min; exiting so the notebook monitor can remount + relaunch"; exit 1; }
    after=$(ls "$LOCAL"/teacher_ckpt/ckpt_*.pt "$DRIVE"/teacher_ckpt/*.manifest.json 2>/dev/null | tail -2 | tr '\n' ' ')
    if [ "$after" != "$before" ]; then stuck=0; log "teacher died after progress; resuming";
    elif [ "$T_MICRO" -gt 1 ]; then T_MICRO=$((T_MICRO / 2)); stuck=0; log "teacher failed, no progress -> T_MICRO=$T_MICRO";
    else stuck=$((stuck + 1)); log "teacher failed at micro 1 ($stuck/$MAX_RETRIES)"; fi
    [ "$stuck" -ge "$MAX_RETRIES" ] && { beat error teacher_stuck; exit 1; }
    beat teacher_retry "T_MICRO=$T_MICRO"; sleep 20
  done
fi
TEACHER="$DRIVE/teacher/teacher_v3_fp16.safetensors"
[ -f "$TEACHER" ] || { log "missing $TEACHER"; beat error no_teacher; exit 1; }
GATE=$(python -c "import json; print(json.load(open('$DRIVE/TEACHER_DONE')).get('best_grid48_exact', 0))")
log "teacher best 48-grid CRNN exact = $GATE (target > $T_STOP)"
if [ "$REQUIRE_TEACHER_GATE" = "1" ] && python -c "import sys; sys.exit(0 if float('$GATE') > float('$T_STOP') else 1)"; then :;
elif [ "$REQUIRE_TEACHER_GATE" = "1" ]; then beat error teacher_gate_failed "best=$GATE"; exit 1; fi

# ---------------------------------------------------------------- 2) Phase 4a reflow pairs (Drive shards, resumable)
stuck=0
while true; do
  before=$(ls "$DRIVE"/pairs_v3/pairs_*.pt "$DRIVE"/pairs_v3/partial_*/chunk_*.pt 2>/dev/null | wc -l)
  python -u distill_v3.py gen_pairs --teacher "$TEACHER" --out_dir "$DRIVE/pairs_v3" --num_pairs $NUM_PAIRS --shard_size $SHARD \
    --steps $TEACHER_STEPS --cfg 5.0 --batch_size $GEN_BATCH --heartbeat "$DRIVE/status.json" && break
  wait_drive || { log "Drive gone 15 min; exiting for remount"; exit 1; }
  after=$(ls "$DRIVE"/pairs_v3/pairs_*.pt "$DRIVE"/pairs_v3/partial_*/chunk_*.pt 2>/dev/null | wc -l)
  if [ "$after" != "$before" ]; then stuck=0; log "gen_pairs died after progress; resuming";
  elif [ "$GEN_BATCH" -gt 8 ]; then GEN_BATCH=$((GEN_BATCH / 2)); stuck=0; log "gen_pairs failed, no progress -> GEN_BATCH=$GEN_BATCH";
  else stuck=$((stuck + 1)); log "gen_pairs failed at min batch ($stuck/$MAX_RETRIES)"; fi
  [ "$stuck" -ge "$MAX_RETRIES" ] && { beat error gen_pairs_stuck; exit 1; }
  beat gen_pairs_retry "GEN_BATCH=$GEN_BATCH"; sleep 20
done
NS=$(ls "$DRIVE"/pairs_v3/pairs_*.pt 2>/dev/null | wc -l)
[ "$NS" -ge $(( (NUM_PAIRS + SHARD - 1) / SHARD )) ] || { log "pairs incomplete ($NS)"; beat error pairs_incomplete; exit 1; }

# ---------------------------------------------------------------- 3) Phase 4b student distill
if [ ! -f "$DRIVE/STUDENT_DONE" ]; then
  beat copy_pairs
  for f in "$DRIVE"/pairs_v3/pairs_*.pt; do
    b=$(basename "$f")
    if [ ! -f "$LOCAL/pairs/$b" ] || [ "$(stat -c%s "$f")" != "$(stat -c%s "$LOCAL/pairs/$b")" ]; then cp "$f" "$LOCAL/pairs/$b.tmp" && mv "$LOCAL/pairs/$b.tmp" "$LOCAL/pairs/$b"; fi
  done
  stuck=0
  while [ ! -f "$DRIVE/STUDENT_DONE" ]; do
    before=$(ls "$LOCAL"/student_ckpt/ckpt_*.pt "$DRIVE"/student_ckpt/*.manifest.json 2>/dev/null | tail -2 | tr '\n' ' ')
    python -u distill_v3.py train --teacher "$TEACHER" --crnn artifacts/crnn.safetensors --pair_dir "$LOCAL/pairs" \
      --checkpoint_dir "$LOCAL/student_ckpt" --checkpoint_mirror "$DRIVE/student_ckpt" --out_dir "$DRIVE/student" \
      --batch_size $S_BATCH --micro_batch $S_MICRO --lr $S_LR --max_steps $S_STEPS --val_every $S_VAL_EVERY --val_limit $VAL_LIMIT \
      --heartbeat "$DRIVE/status.json" --done_marker "$DRIVE/STUDENT_DONE" && break
    wait_drive || { log "Drive gone 15 min; exiting for remount"; exit 1; }
    after=$(ls "$LOCAL"/student_ckpt/ckpt_*.pt "$DRIVE"/student_ckpt/*.manifest.json 2>/dev/null | tail -2 | tr '\n' ' ')
    if [ "$after" != "$before" ]; then stuck=0; log "student died after progress; resuming";
    elif [ "$S_MICRO" -gt 1 ]; then S_MICRO=$((S_MICRO / 2)); stuck=0; log "student failed, no progress -> S_MICRO=$S_MICRO";
    else stuck=$((stuck + 1)); log "student failed at micro 1 ($stuck/$MAX_RETRIES)"; fi
    [ "$stuck" -ge "$MAX_RETRIES" ] && { beat error student_stuck; exit 1; }
    beat student_retry "S_MICRO=$S_MICRO"; sleep 20
  done
fi
STUDENT="$DRIVE/student/student_ema_fp16.safetensors"
[ -f "$STUDENT" ] || { log "missing $STUDENT"; beat error no_student; exit 1; }
[ -f "$DRIVE/TRAIN_DONE" ] || { python -c "import json,time,os; json.dump(dict(time=time.strftime('%F %T'), crnn=json.load(open('$DRIVE/CRNN_DONE')), crnn_ft=json.load(open('$DRIVE/CRNN_FT_DONE')) if os.path.exists('$DRIVE/CRNN_FT_DONE') else None, crnn_gate=json.load(open('$DRIVE/CRNN_GATE.json')) if os.path.exists('$DRIVE/CRNN_GATE.json') else None, teacher=json.load(open('$DRIVE/TEACHER_DONE')), student='$STUDENT'), open('$DRIVE/TRAIN_DONE.tmp','w'), indent=1)" && mv "$DRIVE/TRAIN_DONE.tmp" "$DRIVE/TRAIN_DONE"; }

# ---------------------------------------------------------------- 4) Phase 5 export + eval + bench
beat export
python -u export_v3.py --student "$STUDENT" --out_dir "$LOCAL/export" --report "$LOCAL/export/export_report.json" --split_big || { beat error export; exit 1; }
for f in text_encoder.onnx text_encoder_int8.onnx unet_distilled_int8.onnx student_ema_fp16.safetensors export_report.json torch_ortfp32_ortint8.png; do
  copy_to_drive "$LOCAL/export/$f" "$DRIVE/onnx/$f"
done
for f in "$LOCAL"/export/unet_distilled.onnx.part* "$LOCAL"/export/unet_distilled.onnx.manifest.json; do
  [ -e "$f" ] && copy_to_drive "$f" "$DRIVE/onnx/$(basename "$f")"
done
beat eval
python -u eval_ocr.py models --unet "$LOCAL/export/unet_distilled_int8.onnx" --text "$LOCAL/export/text_encoder_int8.onnx" \
  --teacher "$TEACHER" --teacher_steps $EVAL_TEACHER_STEPS --assets "$LOCAL/assets" --tess v3 || { beat error eval; exit 1; }
python - <<PY || true
import json, os
p, g = "$LOCAL/assets/eval_ocr_v3.json", "$DRIVE/CRNN_GATE.json"
if os.path.exists(p) and os.path.exists(g):
    d = json.load(open(p)); d["crnn_gate"] = json.load(open(g)); json.dump(d, open(p, "w"), indent=1)
    print("[run_v3] eval_ocr_v3.json: recorded crnn_gate override =", d["crnn_gate"].get("override"))
PY
for f in "$LOCAL"/assets/*; do copy_to_drive "$f" "$DRIVE/assets/$(basename "$f")"; done
python -u bench_cpu.py --onnx_dir "$LOCAL/export" --out "$DRIVE/assets/bench_cpu_colab_NOT_TARGET_CPU.json" --reps $BENCH_REPS || true
# ---------------------------------------------------------------- 5) size check + optional HF push (token only from env)
beat hf_push
python - <<'PY' || true
import os, json, time
D = os.environ["DRIVE"]; big = []
for sub in ("onnx", "student", "teacher", "assets", "crnn", "teacher_ckpt", "student_ckpt"):
    p = os.path.join(D, sub)
    for f in (os.listdir(p) if os.path.isdir(p) else []):
        fp = os.path.join(p, f)
        if os.path.isfile(fp) and os.path.getsize(fp) >= 100 * 2**20: big.append((f"{sub}/{f}", os.path.getsize(fp)))
print("[size check] files >= 100 MB on Drive:", big or "none")
res = {"time": time.strftime("%F %T"), "pushed": False}
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
repo = os.environ.get("HF_REPO", "skyuu72/grinpress-v3")
if not tok:
    res["reason"] = "no HF_TOKEN in env; skipped"
else:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        for sub in ("onnx", "student", "assets"):
            if os.path.isdir(os.path.join(D, sub)):
                api.upload_folder(repo_id=repo, repo_type="model", folder_path=os.path.join(D, sub), path_in_repo=sub,
                                  allow_patterns=["*.onnx", "*.onnx.part*", "*.manifest.json", "*.safetensors", "*.json", "*.png"],
                                  commit_message=f"grinpress v3 Colab run: {sub}")
        for f in ("TRAIN_DONE", "CRNN_DONE", "CRNN_FT_DONE", "CRNN_GATE.json", "TEACHER_DONE", "teacher/teacher_val_history.json"):
            if os.path.exists(os.path.join(D, f)):
                api.upload_file(path_or_fileobj=os.path.join(D, f), path_in_repo="run/" + os.path.basename(f), repo_id=repo, repo_type="model")
        res.update(pushed=True, repo=repo)
    except Exception as e:  # never fatal; message only (never the token)
        res["reason"] = f"{type(e).__name__}: {str(e)[:300]}"
print("[hf push]", res)
json.dump(res, open(os.path.join(D, "HF_PUSH.json"), "w"), indent=1)
PY
python - <<PY
import json, time
s = json.load(open("$LOCAL/assets/eval_ocr_v3.json"))["summary"]
td = json.load(open("$DRIVE/TEACHER_DONE"))
json.dump({"finished": time.strftime("%F %T"), "teacher_best_grid48_exact": td.get("best_grid48_exact"),
           "teacher_gate_passed": (td.get("best_grid48_exact") or 0) > float("$T_STOP"),
           "crnn_gate": json.load(open("$DRIVE/CRNN_GATE.json")) if __import__("os").path.exists("$DRIVE/CRNN_GATE.json") else None,
           "student_grid48": {k: v.get("crnn") for k, v in s.items()}, "export": "onnx/", "assets": "assets/",
           "hf_push": json.load(open("$DRIVE/HF_PUSH.json")) if __import__("os").path.exists("$DRIVE/HF_PUSH.json") else None},
          open("$DRIVE/DONE.tmp", "w"), indent=1)
PY
mv "$DRIVE/DONE.tmp" "$DRIVE/DONE"
beat done
log "all done :3"
