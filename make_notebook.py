"""Build distill_colab_v3.ipynb from the repo sources (single source of truth; re-run after any code change).
The notebook embeds every source file (%%writefile) and the Phase-2 CRNN weights (base64, sha256-checked), so it
needs no GitHub/HF credentials on Colab. Robustness patterns follow grinpress-v2 distill_colab.ipynb @49675d9."""
import base64, hashlib, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["charset.py", "conditioning.py", "model.py", "generate_data.py", "generate_data_v2.py", "eval_grid.py", "gp_io.py",
           "crnn.py", "train_crnn.py", "ocr_tess.py", "eval_ocr.py", "train_teacher_v3.py", "distill_v3.py", "export_v3.py", "bench_cpu.py", "run_v3.sh"]
SRC = "/content/grinpress_v3"
DRIVE = "/content/drive/MyDrive/grinpress_v3"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def main():
    cells = [md(
        "# grinpress v3: unattended teacher fine-tune + reflow distillation on an A100 🐾\n"
        "**Runtime → GPU (A100) → Run all.** Re-running all cells is always safe:\n"
        "- the whole pipeline (`run_v3.sh`) runs as a **background process** (`nohup setsid`), so a closed tab or a cell interrupt does not stop it;\n"
        "- stages: **CRNN recognizer** (stage 0, Phase 2 on the GPU: scale channel + hard-class data, gate >= 98% on the 48 refs; `CRNN_FROM_REPO=1` uses the embedded weights instead) → teacher data pregen → **teacher fine-tune** (Phase 3, CRNN-validated on the 48-string grid, early stop > 95%) → **50k reflow pairs** → **student distill** (Phase 4) → **export + eval + bench** (Phase 5);\n"
        "- every stage resumes: atomic local checkpoints + async **sharded** Drive mirror, chunk-level pair resume, CUDA-OOM batch halving (persisted), Drive retry/wait;\n"
        "- `status.json` heartbeat, `run.log`, and the `CRNN_DONE` / `TEACHER_DONE` / `TRAIN_DONE` / `DONE` markers in `MyDrive/grinpress_v3/`. **Every Drive file is < 100 MB** (sharded ckpts, 1000-pair shards, fp16 weights, split fp32 ONNX);\n"
        "- the monitor cell tails the log, remounts Drive when the FUSE mount wedges, and relaunches the run if the process died before `DONE`.\n\n"
        "made with tail wags by the puppygirl CEO of Quantara (quantara.cv) :3"),
        code("# Mount Drive with retries (force_remount after a failure)\nimport os, time\nfrom google.colab import drive\n"
             "if not os.path.isdir('/content/drive/MyDrive'):\n    for i in range(6):\n        try:\n"
             "            drive.mount('/content/drive', force_remount=i > 0, timeout_ms=180000)\n"
             "            if os.path.isdir('/content/drive/MyDrive'):\n                break\n        except Exception as e:\n"
             "            print(f'drive.mount attempt {i+1} failed: {e}')\n            time.sleep(10 * (i + 1))\n"
             "assert os.path.isdir('/content/drive/MyDrive'), 'Drive did not mount; re-run this cell'\n"
             f"DRIVE = '{DRIVE}'\nos.makedirs(DRIVE, exist_ok=True)\nos.makedirs('{SRC}/artifacts', exist_ok=True)\nprint('Drive OK ->', DRIVE)"),
        md("## Sources (written fresh every run; identical to the GitHub repo `Metrix187/grinpress-v3`)")]
    for f in SOURCES:
        cells.append(code(f"%%writefile {SRC}/{f}\n" + open(os.path.join(ROOT, f)).read()))
    w = open(os.path.join(ROOT, "artifacts", "crnn.safetensors"), "rb").read()
    b64 = base64.b64encode(w).decode()
    cells.append(code("# Phase-2 CRNN recognizer weights (frozen OCR loss + evaluator), embedded so no credentials are needed\n"
                      "import base64, hashlib\nB64 = (\n" + "\n".join(f"    '{b64[i:i+4000]}'" for i in range(0, len(b64), 4000)) + "\n)\n"
                      "data = base64.b64decode(B64)\n"
                      f"assert hashlib.sha256(data).hexdigest() == '{hashlib.sha256(w).hexdigest()}', 'CRNN weights corrupted'\n"
                      f"open('{SRC}/artifacts/crnn.safetensors', 'wb').write(data); print('crnn.safetensors', len(data), 'bytes OK')"))
    cells.append(code("!apt-get -qq install -y tesseract-ocr > /dev/null\n"
                      "!pip install -q safetensors onnx onnxruntime onnxscript opencv-python-headless pytesseract huggingface_hub\n"
                      "!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
                      "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported() if torch.cuda.is_available() else '')"))
    cells.append(md("## Launch (background, idempotent)\nDefaults: 100k pregen samples; teacher batch 128 (micro 64), up to 40k steps, "
                    "validated every 1000 steps, early stop when the CRNN reads > 95% of the 48-grid exactly; 50k pairs (50 Euler steps, CFG 5, "
                    "GEN_BATCH 128); student batch 128 (micro 32), 20k steps. Micro-batches / GEN_BATCH halve themselves on CUDA OOM. "
                    "Stage 0 trains the CRNN first (120k samples, 30 epochs, ~0.5 h; REQUIRE_CRNN_GATE=1 stops the run if it reads < 98% of the 48 refs). "
                    "Optional: add a Colab Secret `HF_TOKEN` to push the final artifacts to the private HF repo skyuu72/grinpress-v3. "
                    "Estimated wall time on an A100 40GB: ~0.5 h CRNN + ~0.3 h pregen + 1-3 h teacher + ~2 h pairs + ~1-1.5 h student + ~0.3 h export/eval ≈ 5-7 h."))
    cells.append(code(
        "import os, subprocess\n"
        "RUN = dict(PREGEN_N='100000', T_BATCH='128', T_MICRO='64', T_STEPS='40000', T_LR='1e-4', T_VAL_EVERY='1000', T_STOP='0.95',\n"
        "           REQUIRE_CRNN_GATE='1', CRNN_FROM_REPO='0', REQUIRE_TEACHER_GATE='0', NUM_PAIRS='50000', SHARD='1000', GEN_BATCH='128', S_BATCH='128', S_MICRO='32',\n"
        "           S_STEPS='20000', S_LR='5e-5')\n"
        f"DRIVE = '{DRIVE}'; LOG = f'{{DRIVE}}/run.log'\n\n"
        "def running():\n"
        f"    return subprocess.run(['pgrep', '-f', 'bash {SRC}/run_v3.sh'], capture_output=True).returncode == 0\n\n"
        "def launch():\n"
        "    if os.path.exists(f'{DRIVE}/DONE'):\n        print('DONE marker found: the run is already finished 🐾'); return\n"
        "    if running():\n        print('run_v3.sh already running, not starting a second copy'); return\n"
        "    env = dict(os.environ, DRIVE=DRIVE, LOCAL='/content/grinpress_v3_local', **RUN)\n"
        "    try:  # optional: HF token from Colab Secrets (key icon, name HF_TOKEN, notebook access on); never stored in the notebook\n"
        "        from google.colab import userdata\n        t = userdata.get('HF_TOKEN')\n        if t: env['HF_TOKEN'] = t\n"
        "    except Exception as e:\n        print('no HF_TOKEN secret -> final HF push will be skipped (', type(e).__name__, ')')\n"
        "    print('HF push at the end:', 'enabled' if env.get('HF_TOKEN') else 'skipped (no token)')\n"
        f"    subprocess.Popen(f'nohup setsid bash {SRC}/run_v3.sh >> \"{{LOG}}\" 2>&1 &', shell=True, env=env, cwd='{SRC}')\n"
        "    print('launched in background; log ->', LOG)\n\nlaunch()"))
    cells.append(md("## Monitor until done (safe to interrupt or re-run; it relaunches the run if the process died)"))
    cells.append(code(
        "import time, json, os, subprocess, threading\nfrom IPython.display import clear_output\nfrom google.colab import drive\n\n"
        "def drive_ok(timeout=12):\n    \"\"\"listdir can hang forever on a wedged Drive FUSE; bound it with a short thread join.\"\"\"\n    box = []\n"
        "    def _ls():\n        try:\n            os.listdir(DRIVE); box.append(True)\n        except OSError:\n            box.append(False)\n"
        "    th = threading.Thread(target=_ls, daemon=True); th.start(); th.join(timeout)\n"
        "    if th.is_alive():\n        print('drive_ok: listdir hung >%ss (FUSE wedged?)' % timeout); return False\n    return bool(box and box[0])\n\n"
        "def remount():\n    for i in range(6):\n        try:\n            drive.mount('/content/drive', force_remount=True, timeout_ms=180000)\n"
        "            if drive_ok(): print('Drive remounted 🐾'); return True\n        except Exception as e:\n"
        "            print(f'remount attempt {i+1} failed: {e}')\n        time.sleep(15 * (i + 1))\n    return False\n\n"
        "relaunches, bad = 0, 0\nwhile True:\n    try:\n        if os.path.exists(f'{DRIVE}/DONE'): break\n"
        "        if not drive_ok():\n            bad += 1; print(time.strftime('%H:%M:%S'), f'Drive not reachable ({bad})')\n"
        "            if bad >= 3: remount(); bad = 0\n            time.sleep(60); continue\n        bad = 0\n"
        "        try: st = open(f'{DRIVE}/status.json').read()\n        except OSError: st = '(no heartbeat yet)'\n"
        "        try: tail = subprocess.run(['tail', '-n', '15', LOG], capture_output=True, text=True, timeout=60).stdout\n"
        "        except Exception: tail = ''\n        clear_output(wait=True)\n        print(time.strftime('%H:%M:%S'), 'status:', st); print(tail)\n"
        "        if not running() and not os.path.exists(f'{DRIVE}/DONE'):\n            if relaunches >= 20:\n"
        "                print('run keeps dying; check run.log 🐾'); break\n"
        "            relaunches += 1; print(f'process not running -> relaunch #{relaunches}'); launch()\n"
        "    except Exception as e:  # the monitor itself must never die\n        print('monitor error:', e)\n    time.sleep(60)\n"
        "if os.path.exists(f'{DRIVE}/DONE'):\n    print('DONE :3', open(f'{DRIVE}/DONE').read())"))
    cells.append(md("## Results"))
    cells.append(code(
        "from IPython.display import Image, display\nimport json, os\n"
        "for f in ('samples_4step_v3.png', 'samples_2step_v3.png', 'teacher_vs_student_v3.png'):\n"
        "    p = f'{DRIVE}/assets/{f}'\n    if os.path.exists(p): print(f); display(Image(p))\n"
        "for f in ('DONE', 'teacher/teacher_val_history.json', 'assets/eval_ocr_v3.json', 'onnx/export_report.json'):\n"
        "    p = f'{DRIVE}/{f}'\n    if os.path.exists(p): print(f, open(p).read()[:3000])"))
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 0}
    json.dump(nb, open(os.path.join(ROOT, "distill_colab_v3.ipynb"), "w"), indent=1)
    print("wrote distill_colab_v3.ipynb", len(cells), "cells")


if __name__ == "__main__":
    main()
