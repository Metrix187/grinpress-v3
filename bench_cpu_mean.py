"""Mean/std companion to bench_cpu.py (same captions, sessions and sampler via export_v3; batch 1, 128 px,
text encoder + all steps, 1 warm-up). bench_cpu.py reports median/min; this adds mean +- std over --reps.
  python bench_cpu_mean.py --onnx_dir onnx --reps 30 -> assets/bench_cpu_v3_mean.json"""
import argparse, json, os, time
import numpy as np
from bench_cpu import cpu_name
from export_v3 import ort_session, ort_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", default="onnx")
    ap.add_argument("--threads", type=int, nargs="+", default=[os.cpu_count()])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default="assets/bench_cpu_v3_mean.json")
    a = ap.parse_args()
    import onnxruntime as ort
    cap = "a red smiley face with 'HELLO WORLD' next to it, face on the right"
    combos = [("ORT int8 U-Net + int8 text encoder", "text_encoder_int8.onnx", "unet_distilled_int8.onnx"),
              ("ORT int8 U-Net + fp32 text encoder", "text_encoder.onnx", "unet_distilled_int8.onnx")]
    rows = []
    for th in a.threads:
        for name, tf, uf in combos:
            te, un = ort_session(os.path.join(a.onnx_dir, tf), th), ort_session(os.path.join(a.onnx_dir, uf), th)
            for steps in (2, 4):
                ort_sample(te, un, cap, steps, 0); ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter(); ort_sample(te, un, cap, steps, 0); ts.append(time.perf_counter() - t0)
                ts = np.array(ts)
                r = dict(path=name, threads=th, steps=steps, reps=a.reps, mean_s=float(ts.mean()), std_s=float(ts.std()),
                         median_s=float(np.median(ts)), min_s=float(ts.min()), max_s=float(ts.max()))
                rows.append(r)
                print(f"{name:36s} thr={th} {steps} steps: mean {r['mean_s']:.3f}s +- {r['std_s']:.3f} "
                      f"(median {r['median_s']:.3f}, min {r['min_s']:.3f}, max {r['max_s']:.3f})", flush=True)
    out = dict(cpu=cpu_name(), cores=os.cpu_count(), onnxruntime=ort.__version__, reps=a.reps,
               target_s=0.65, rows=rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
