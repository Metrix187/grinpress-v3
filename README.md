---
tags:
- text-to-image
- rectified-flow
- distillation
- onnx
- ocr
pipeline_tag: text-to-image
---

# grinpress v3 🐾 uwu :3

hiii, it's grinpress v3!! a teeny 128 px text-to-image puppy that draws newsprint-y **smiley faces with words next to
them** :3 *wags tail*. You tell it something like `a red smiley face with 'HELLO WORLD' next to it, face on the right,
on sepia paper` and it boops out a face (smiley, sad, meh, wink, angry, dead, uwu 🐶) with your text (up to 14 chars).

The big teacher (rectified flow, 50 Euler steps, CFG 5) got squished into a speedy **2- or 4-step student**. The student
is exported to ONNX (opset 17) in fp32 and dynamic INT8 (MatMul/Gemm only). It runs on a plain CPU, no GPU needed. Bork!

All numbers below are copied from the run's own eval files or measured on 2026-11-01. No made-up treats, promise 🐾

## the numbers (honest, no begging face)

Primary metric: **CRNN exact match, case-sensitive**, on the 48 held-out strings of grid48. Students are scored as the
shipped **INT8 ONNX** (`unet_distilled_int8.onnx` + `text_encoder_int8.onnx`). Sources: `assets/v3_run/DONE.json`,
`assets/eval_ocr_v3.json`, `assets/eval_compare_v3.json`.

| model | grid48 CRNN exact | CRNN char acc | tesseract exact |
|---|---|---|---|
| teacher (best ckpt, step 13000) | **1.0** (48/48) | – | – |
| student 4-step (INT8 ONNX) | **0.958** (46/48) | 0.991 | 0.729 |
| student 2-step (INT8 ONNX) | **0.9375** (45/48) | 0.983 | 0.708 |
| procedural reference renders | 0.979 (47/48) | 0.998 | 0.542 |

grid48 misses (so you can sniff them yourself, `assets/samples_{4,2}step_v3.png`):
- 4-step: `woof woof`→`woofwoof`, `...`→`.@.`
- 2-step: those two plus `GOOD MORNING!!`→`55Op MORNINGl`

grid14 teacher-vs-student (`assets/teacher_vs_student_v3.png`), CRNN exact: reference 1.0, teacher 0.929 (it drew
`GOOD MORNING!!!` with one extra `!`, silly teacher), student 4-step 1.0, student 2-step 1.0.

### the CRNN gate override (47/48) 🙈

The OCR judge is a small CRNN. The spec wanted it to read ≥ 98 % of the 48 reference renders before we trust it.
The stage-0b fine-tuned CRNN got **47/48 = 0.9792**. It missed `newspaper boy`, which it read as **`newspapee boy`**
(it's a *reference* render, so the CRNN fluffed it, not the model). That is just under 0.98, so the gate **failed**
and was **overridden with user approval** (`CRNN_GATE=0.979`). See `assets/v3_run/CRNN_GATE.json` and
`docs/WAIVER_crnn_gate.md`. Keep this in mind: every CRNN number here comes from a judge that is itself ~98 % accurate.

### little caveats I won't hide under the couch

- The exported student is the **final step-20000 EMA**, not the best validation checkpoint. The PyTorch val history
  (`assets/v3_run/student_val_history.json`) peaked at 4-step 0.979 (steps 2000/6000/10000) and 2-step 0.979 (step 14000).
- At step 20000 the PyTorch val said 4-step 0.9375 / 2-step 0.958. The final INT8 ONNX eval says 4-step 0.958 / 2-step 0.9375,
  so the order flips. Each gap is one image out of 48, so call 2-step and 4-step roughly tied.
- The INT8 quantisation error is small but real (`assets/v3_run/export_report.json`): velocity max diff 0.184, mean 0.0022,
  mean image diff 0.098 px values.

## CPU speed 🏃‍♀️💨 (target ≈ 0.65 s / image)

Full image at batch 1, 128 px: text encoder + all U-Net steps, onnxruntime CPU EP, after warm-up.

**Box CPU:** Intel Xeon (8 vCPU, 1 thread/core, AVX-512 VNNI + AMX-INT8), 15 GB RAM, onnxruntime 1.31.1, 8 threads.
This is a shared VM and a browser was idling on it, so expect a few % of noise.

| INT8 U-Net + INT8 text encoder | mean ± std (30 reps, `bench_cpu_mean.py`) | median (10 reps, `bench_cpu.py`) |
|---|---|---|
| **2-step** | **0.270 s ± 0.021** | 0.305 s |
| **4-step** | **0.540 s ± 0.040** | 0.534 s |

Both are under the 0.65 s target ✅. Other rows (`assets/bench_cpu_v3.json`, `assets/bench_cpu_v3_mean.json`): INT8 U-Net + fp32
text encoder 0.327 / 0.549 s mean; ORT fp32 0.325 / 0.561 s median; PyTorch fp32 0.289 / 0.549 s median. With 4 threads,
INT8+INT8 is 0.282 / 0.577 s mean.

Heads-up: on the Colab runtime CPU (Intel Xeon @ 2.20 GHz, 12 threads), the same INT8 combo was **0.536 s (2-step) /
0.988 s (4-step)** median, so 4-step **missed** the target there (`assets/bench_cpu_colab_NOT_TARGET_CPU.json`). The box CPU
has VNNI/AMX int8 instructions, which help a lot. Your speed depends heavily on your CPU!

## where the words break 🐾

I stress-tested 35 new words (short, long up to 14 chars, silly, mIxEd CaSe, punctuation + numbers) × all 7
expressions at 2 and 4 steps. Grids and notes are in `samples/` (`samples/BREAK_NOTES.md`). Quick version:

- CRNN exact overall: 4-step 0.971, 2-step 0.984. Mixed case and punctuation/numbers: 0 misses.
- Long 12-14-char phrases beside the face get tiny (~5-6 px glyphs) and are the weak spot (4-step 0.918). The CRNN
  only reads 75.5 % of the long *reference* renders anyway.
- The real glyph faults I could see are rare and character-level: a doubled `HHi!`, a malformed `m` (`snol bean`),
  a stray ink blob between words, and a squeezed word gap.

## files 📦

| file | what |
|---|---|
| `onnx/unet_distilled_int8.onnx` | INT8 student U-Net (89.5 MB), **use this one on CPU** |
| `onnx/text_encoder_int8.onnx` | INT8 text encoder (2.0 MB) |
| `onnx/unet_distilled.onnx` | fp32 U-Net (129.5 MB). In git/Drive it is stored as `.part00` + `.part01`: `cat unet_distilled.onnx.part* > unet_distilled.onnx` |
| `onnx/text_encoder.onnx` | fp32 text encoder (6.7 MB) |
| `student_ema_fp16.safetensors` | PyTorch EMA student, fp16 (68.4 MB) |
| `crnn_ft.safetensors` | the stage-0b fine-tuned CRNN OCR judge |
| `assets/` | eval JSONs, sample grids, bench results, run markers |
| `samples/` | varied-word stress grids + break notes |

ONNX interface: `text_encoder(input_ids int64[B,128], attention_mask int64[B,128]) -> context f32[B,128,256], context_mask`;
`unet(x_t f32[B,3,128,128], t f32[B], context, context_mask) -> v`. Sample with Euler on `t = 1 - i/steps`, `x -= (t_i - t_{i+1}) * v`,
starting from N(0,1) noise. See `export_v3.ort_sample`. Captions:
`a {color} {smiley|sad|meh|wink|angry|dead|uwu} face with '{text}' next to it[, face on the right | , text below][, on fresh paper | , on sepia paper]`.

```bash
python bench_cpu.py --onnx_dir onnx --reps 10            # median/min
python bench_cpu_mean.py --onnx_dir onnx --reps 30       # mean ± std
python eval_ocr.py models --unet onnx/unet_distilled_int8.onnx --text onnx/text_encoder_int8.onnx --crnn crnn_ft.safetensors
python samples_varied.py --onnx_dir onnx --crnn crnn_ft.safetensors --out samples
```

## the zoomies timeline (restarts + relaunch) 🔁

Training ran on Colab with Drive-resumable stages. Times are from `run.log`, UTC with ET in brackets:
- **Oct 16 05:01-11:55 UTC:** the stage-0 CRNN missed the 0.98 gate, so the runner kept stopping and relaunching
  (lots of ~70 s bounces in the log). The stage-0b fine-tune finished at 11:42 UTC with 47/48 (`newspapee boy`).
- **Oct 16 23:36 UTC (7:36 PM ET):** relaunch with the user-approved gate override (`CRNN_GATE=0.979`). From then on every
  relaunch skips the CRNN (`CRNN_FT_DONE`).
- **Oct 17 03:16 / 04:06 / 05:14 UTC:** teacher relaunches, each one resuming. **06:26 UTC (2:26 AM ET):** teacher done
  (step 13000, grid48 1.0).
- **Oct 17 11:40 and 21:26 UTC:** relaunches for teacher-pair generation (50k pairs, resumed shard-by-shard) and student
  distillation. The student got to step 13103 (ckpt mirrored to Drive), then the runtime went away with no error in the log.
- **Nov 1 01:49 UTC (Oct 31, 9:49 PM ET):** final relaunch, resuming from `ckpt_00013103.pt` and running to step 20000 (~1.1 s/it).
- **Nov 1 04:06 UTC (12:06 AM ET):** export, eval, `all done :3`. The in-run HF push was skipped (no token in that runtime).
  The weights were uploaded to HF afterwards.

Nothing was retrained from scratch. Every relaunch resumed from the latest checkpoint or marker on Drive. *happy tail wags* 🐾

made with lots of headpats. awoo~ 🐾 uwu :3
