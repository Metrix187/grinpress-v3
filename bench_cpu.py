"""CPU latency of the v3 student (batch 1, 128 px, full image = text encoder + all steps), median of --reps after warm-up.
  python bench_cpu.py --onnx_dir onnx [--student student_ema.safetensors] -> assets/bench_cpu_v3.json"""
import argparse, json, os, platform, time
import numpy as np, torch


def cpu_name():
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"): return l.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def bench(fn, reps):
    fn(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", default="onnx")
    ap.add_argument("--student", default="", help="optional: also time PyTorch fp32")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="assets/bench_cpu_v3.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import onnxruntime as ort
    from export_v3 import ort_session, ort_sample, torch_sample
    cap = "a red smiley face with 'HELLO WORLD' next to it, face on the right"
    d = a.onnx_dir; rows = []
    combos = [("ORT int8 U-Net + fp32 text encoder", "text_encoder.onnx", "unet_distilled_int8.onnx"),
              ("ORT int8 U-Net + int8 text encoder", "text_encoder_int8.onnx", "unet_distilled_int8.onnx"),
              ("ORT fp32", "text_encoder.onnx", "unet_distilled.onnx")]
    for name, tf, uf in combos:
        if not (os.path.exists(os.path.join(d, tf)) and os.path.exists(os.path.join(d, uf))): continue
        te, un = ort_session(os.path.join(d, tf), a.threads), ort_session(os.path.join(d, uf), a.threads)
        for steps in (2, 4):
            med, mn = bench(lambda: ort_sample(te, un, cap, steps, 0), a.reps)
            rows.append(dict(path=name, steps=steps, median_s=med, min_s=mn))
            print(f"{name:40s} {steps} steps: median {med:.3f}s (min {mn:.3f}s)", flush=True)
    if a.student:
        from distill_v3 import load_student
        s = load_student(a.student, "cpu")
        for steps in (2, 4):
            med, mn = bench(lambda: torch_sample(s, cap, steps, 0), a.reps)
            rows.append(dict(path="PyTorch fp32", steps=steps, median_s=med, min_s=mn))
    rep = dict(cpu=cpu_name(), cores=os.cpu_count(), threads=a.threads, reps=a.reps, torch=torch.__version__,
               onnxruntime=ort.__version__, onnx_dir=os.path.abspath(d), rows=rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
