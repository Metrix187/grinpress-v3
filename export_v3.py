"""Phase 5: export the v3 student to ONNX (fp32 + dynamic INT8), fp16 safetensors, and validate.

ONNX interface (index.html / bench / eval):
  text_encoder.onnx   : input_ids int64 [B,128] (ASCII code points, 0 = pad, 128 = unk), attention_mask int64 [B,128]
                        -> context float32 [B,128,256], context_mask int64 [B,128]
                        (quoted-span token-type + glyph-index features are computed INSIDE the graph from input_ids)
  unet_distilled.onnx : x_t float32 [B,3,128,128], t float32 [B], context, context_mask -> v_pred float32 [B,3,128,128]
  opset 17, dynamic batch. *_int8.onnx: onnxruntime quantize_dynamic, MatMul/Gemm only (Conv stays fp32).
  student_ema_fp16.safetensors: fp16 copy of the EMA student (~68 MB, under the 100 MB transfer cap).
Files > 95 MB (unet_distilled.onnx fp32 ~130 MB) are additionally split into .partNN + manifest with --split_big.
"""
import argparse, hashlib, json, os, time
import numpy as np, torch, torch.nn as nn
from conditioning import tokenize, MAX_LEN
from distill_v3 import load_student, student_schedule

CAPS = ["a red smiley face with 'HELLO WORLD' next to it, face on the right, on sepia paper",
        "a blue sad face with 'nyaa~' next to it", "a navy uwu face with 'GoodGirl' next to it, text below"]


class TextEncoderExport(nn.Module):
    def __init__(self, s):
        super().__init__(); self.cond = s.cond

    def forward(self, input_ids, attention_mask):
        m = attention_mask > 0
        x = self.cond.embed(input_ids)
        for layer in self.cond.encoder.layers:          # explicit loop: no nested-tensor fast path in the graph
            h = layer.norm1(x)
            x = x + layer.self_attn(h, h, h, key_padding_mask=~m, need_weights=False)[0]
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        return self.cond.norm(x), m.to(torch.int64)


class UNetExport(nn.Module):
    def __init__(self, s):
        super().__init__(); self.s = s
        w = s.w_embed(None, 1, "cpu")
        self.register_buffer("w_add", w.detach() if w is not None else torch.zeros(1, s.unet.tmlp[2].out_features))

    def forward(self, x_t, t, context, context_mask):
        return self.s.unet(x_t, t, context, context_mask > 0, temb_add=self.w_add)


def ort_session(path, threads=os.cpu_count()):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def ort_sample(te, un, caption, steps, seed, n=1):
    ids, mk = tokenize([caption] * n)
    ctx, cm = te.run(None, {"input_ids": ids.numpy(), "attention_mask": mk.long().numpy()})
    x = torch.randn(n, 3, 128, 128, generator=torch.Generator().manual_seed(seed)).numpy(); ts = student_schedule(steps)
    for i in range(steps):
        v = un.run(None, {"x_t": x, "t": np.full((n,), ts[i], np.float32), "context": ctx, "context_mask": cm})[0]
        x = x - (ts[i] - ts[i + 1]) * v
    return np.clip(x, -1, 1)


def torch_sample(student, caption, steps, seed, n=1):
    ids, mk = tokenize([caption] * n)
    with torch.inference_mode():
        ctx, m = student.cond(ids, mk); w_add = student.w_embed(None, n, "cpu")
        x = torch.randn(n, 3, 128, 128, generator=torch.Generator().manual_seed(seed)); ts = student_schedule(steps)
        for i in range(steps):
            x = x - (ts[i] - ts[i + 1]) * student.unet(x, torch.full((n,), ts[i]), ctx, m, temb_add=w_add)
    return x.clamp(-1, 1).numpy()


def split_big(path, limit=95 * 1000 * 1000):
    """path -> path.partNN (<= limit bytes) + path.manifest.json (sha256). Reassemble: cat path.part* > path."""
    data = open(path, "rb").read()
    if len(data) <= limit:
        return 0
    n = 0
    for n, i in enumerate(range(0, len(data), limit)):
        with open(f"{path}.part{n:02d}", "wb") as f: f.write(data[i:i + limit])
    json.dump({"n_parts": n + 1, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "reassemble": f"cat {os.path.basename(path)}.part* > {os.path.basename(path)}"}, open(path + ".manifest.json", "w"), indent=1)
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--out_dir", default="onnx")
    ap.add_argument("--report", default="")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--split_big", action="store_true", help="also write <95 MB parts of big files (Drive cap)")
    ap.add_argument("--fp16_out", default="", help="where to write student_ema_fp16.safetensors (default out_dir)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out_dir, exist_ok=True)
    s = load_student(a.student, "cpu")
    from safetensors.torch import save_file
    fp16 = a.fp16_out or os.path.join(a.out_dir, "student_ema_fp16.safetensors")
    save_file({k: v.half().contiguous() for k, v in s.state_dict().items()}, fp16)
    te, un = TextEncoderExport(s).eval(), UNetExport(s).eval()
    ids, mk = tokenize(CAPS[:2]); mk = mk.long()
    with torch.no_grad(): ctx, cm = te(ids, mk)
    tp, up = os.path.join(a.out_dir, "text_encoder.onnx"), os.path.join(a.out_dir, "unet_distilled.onnx")
    t0 = time.time()
    torch.onnx.export(te, (ids, mk), tp, opset_version=a.opset, dynamo=False,
                      input_names=["input_ids", "attention_mask"], output_names=["context", "context_mask"],
                      dynamic_axes={k: {0: "batch"} for k in ["input_ids", "attention_mask", "context", "context_mask"]})
    torch.onnx.export(un, (torch.randn(2, 3, 128, 128), torch.tensor([1.0, 0.5]), ctx, cm), up, opset_version=a.opset, dynamo=False,
                      input_names=["x_t", "t", "context", "context_mask"], output_names=["v_pred"],
                      dynamic_axes={k: {0: "batch"} for k in ["x_t", "t", "context", "context_mask", "v_pred"]})
    rep = {"opset": a.opset, "max_len": MAX_LEN, "export_s": round(time.time() - t0, 1)}
    # validation: batch 3 (dynamic batch), ORT fp32 vs torch
    ids, mk = tokenize(CAPS); x = torch.randn(3, 3, 128, 128, generator=torch.Generator().manual_seed(0)); t = torch.tensor([1.0, 0.75, 0.25])
    with torch.no_grad():
        ctx_t, cm_t = te(ids, mk.long()); ref_ctx, _ = s.cond(ids, mk); v_t = un(x, t, ctx_t, cm_t).numpy()
    ste, sun = ort_session(tp, a.threads), ort_session(up, a.threads)
    ctx_o, cm_o = ste.run(None, {"input_ids": ids.numpy(), "attention_mask": mk.long().numpy()})
    v_o = sun.run(None, {"x_t": x.numpy(), "t": t.numpy(), "context": ctx_o, "context_mask": cm_o})[0]
    rep["export_wrapper_vs_module_ctx_maxdiff"] = float(((ctx_t - ref_ctx) * mk[..., None]).abs().max())
    rep["ort_fp32_ctx_maxdiff"] = float(np.abs((ctx_o - ctx_t.numpy()) * mk[..., None].numpy()).max())
    rep["ort_fp32_v_maxdiff"] = float(np.abs(v_o - v_t).max())
    rep["context_mask_equal"] = bool((cm_o == cm_t.numpy()).all())
    img_t = torch_sample(s, CAPS[0], a.steps, 3); img_o = ort_sample(ste, sun, CAPS[0], a.steps, 3)
    rep["ort_fp32_image_maxdiff"] = float(np.abs(img_o - img_t).max())
    from onnxruntime.quantization import quantize_dynamic, QuantType
    q = {}
    for p in (tp, up):
        qp = p.replace(".onnx", "_int8.onnx")
        quantize_dynamic(p, qp, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"]); q[p] = qp
        for f in (p, qp): rep[f"size_MB_{os.path.basename(f)}"] = round(os.path.getsize(f) / 1e6, 1)
    teq, unq = ort_session(q[tp], a.threads), ort_session(q[up], a.threads)
    ctx_q, cm_q = teq.run(None, {"input_ids": ids.numpy(), "attention_mask": mk.long().numpy()})
    v_q = unq.run(None, {"x_t": x.numpy(), "t": t.numpy(), "context": ctx_q, "context_mask": cm_q})[0]
    rep["ort_int8_v_maxdiff"] = float(np.abs(v_q - v_t).max()); rep["ort_int8_v_meandiff"] = float(np.abs(v_q - v_t).mean())
    img_q = ort_sample(teq, unq, CAPS[0], a.steps, 3)
    rep["ort_int8_image_meandiff_pixels"] = float(np.abs(img_q - img_t).mean() * 127.5)
    rep[f"size_MB_{os.path.basename(fp16)}"] = round(os.path.getsize(fp16) / 1e6, 1)
    if a.split_big:
        rep["split_parts"] = {os.path.basename(p): split_big(p) for p in (tp, up, q[tp], q[up], fp16)}
    from PIL import Image
    to8 = lambda z: ((z[0].transpose(1, 2, 0) + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(np.concatenate([to8(img_t), to8(img_o), to8(img_q)], 1)).save(os.path.join(a.out_dir, "torch_ortfp32_ortint8.png"))
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(a.report or os.path.join(a.out_dir, "export_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
