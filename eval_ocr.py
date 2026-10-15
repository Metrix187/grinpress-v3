"""grinpress v3 OCR evaluation.

  python eval_ocr.py refs                       # Phase 2: tesseract v2 / tesseract v3 / CRNN on the 48 reference renders
  python eval_ocr.py models --unet onnx/unet_distilled_int8.onnx --text onnx/text_encoder_int8.onnx [--teacher ckpt]
        # Phase 5: 48-grid at 2 and 4 steps (+14-grid teacher vs student) -> assets/eval_ocr_v3.json, eval_compare_v3.json,
        #          samples_4step_v3.png, samples_2step_v3.png, teacher_vs_student_v3.png

Scores (per string): exact = case-sensitive exact match after whitespace normalisation; exact_ci = v2's metric
(upper-cased, stray face-edge glyph stripped); char_acc = 1 - Levenshtein/len on the exact_ci-normalised strings
(v2 definition), char_acc_cs = same on case-sensitive strings. Primary v3 metric: CRNN exact (case-sensitive).
"""
import argparse, json, os, random, re, sys, time
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[len(b)]


def norm_ci(s):
    s = re.sub(r"\s+", " ", s.strip().upper())
    return re.sub(r"^[)|}\].,'\u2018\u201c-] ", "", s)


def norm_cs(s):
    return re.sub(r"\s+", " ", s.strip())


def score_text(pred, gt):
    p, g = norm_ci(pred), norm_ci(gt)
    pc, gc = norm_cs(pred), norm_cs(gt)
    return dict(exact=pc == gc, exact_ci=p == g, char_acc=max(0.0, 1 - lev(p, g) / max(1, len(g))),
                char_acc_cs=max(0.0, 1 - lev(pc, gc) / max(1, len(gc))))


def summarize(rows, key):
    P = [r[key] for r in rows]
    return dict(n=len(P), exact=sum(p["exact"] for p in P) / len(P), exact_ci=sum(p["exact_ci"] for p in P) / len(P),
                char_acc=float(np.mean([p["char_acc"] for p in P])), char_acc_cs=float(np.mean([p["char_acc_cs"] for p in P])))


def fonts():
    from generate_data import ensure_fonts
    return sorted(ensure_fonts())


def reference_images(grid):
    from generate_data_v2 import render_v2
    F = fonts()
    return [render_v2(128, random.Random(seed), F, **{k: sp[k] for k in ("color", "expr", "text", "layout", "paper")})[0]
            for sp, _, seed in grid]


def ocr_all(imgs, layouts, crnn=None, tess=("v3",)):
    from crnn import recognize
    from ocr_tess import tess_v2, tess_v3
    out = {}
    if crnn is not None:
        out["crnn"] = recognize(crnn, imgs, layouts)
    if "v2" in tess:
        out["tess_v2"] = [tess_v2(im, l) for im, l in zip(imgs, layouts)]
    if "v3" in tess:
        out["tess_v3"] = [tess_v3(im, l) for im, l in zip(imgs, layouts)]
    if "v3s" in tess:
        out["tess_v3_sauvola"] = [tess_v3(im, l, "sauvola") for im, l in zip(imgs, layouts)]
    return out


FONT = ImageFont.load_default(size=13)


def grid_png(imgs, labels, cols, path, cell=192, title=None):
    rows = (len(imgs) + cols - 1) // cols; lh = 34; th = 30 if title else 0
    G = Image.new("RGB", (cols * cell, th + rows * (cell + lh)), "white"); d = ImageDraw.Draw(G)
    if title: d.text((6, 6), title, fill="black", font=ImageFont.load_default(size=16))
    for k, (im, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(k, cols); x0, y0 = c * cell, th + r * (cell + lh)
        G.paste(im.resize((cell, cell), Image.NEAREST), (x0, y0))
        for li, line in enumerate(lab.split("\n")[:2]):
            d.text((x0 + 3, y0 + cell + 2 + li * 15), line, fill="black", font=FONT)
    G.save(path); return path


# ------------------------------------------------------------------ Phase 2: references
def cmd_refs(a):
    from eval_grid import grid48
    from crnn import load_crnn
    g = grid48(); imgs = reference_images(g); lays = [sp["layout"] for sp, _, _ in g]; texts = [sp["text"] for sp, _, _ in g]
    crnn = load_crnn(a.crnn) if a.crnn and os.path.exists(a.crnn) else None
    t0 = time.time()
    preds = ocr_all(imgs, lays, crnn, tess=("v2", "v3", "v3s"))
    res = {"what": "OCR on the 48 procedural reference renders (generate_data_v2.render_v2, seeds 1000+i, byte-identical to v2)",
           "crnn": a.crnn if crnn is not None else None, "per_sample": [], "summary": {}}
    for i, (sp, cap, seed) in enumerate(g):
        row = dict(text=sp["text"], layout=sp["layout"], paper=sp["paper"], expr=sp["expr"], seed=seed)
        for k, v in preds.items(): row[k] = dict(ocr=v[i], **score_text(v[i], sp["text"]))
        res["per_sample"].append(row)
    for k in preds: res["summary"][k] = summarize(res["per_sample"], k)
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
    for k in preds:
        miss = [(r["text"], r[k]["ocr"]) for r in res["per_sample"] if not r[k]["exact"]]
        print(k, "misses (case-sensitive):", miss[:60])


# ------------------------------------------------------------------ Phase 5: models
def ort_sess(p, threads=os.cpu_count()):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])


def noise(seed):
    return torch.randn(1, 3, 128, 128, generator=torch.Generator().manual_seed(seed)).numpy()


def sched(n):
    return [1.0 - i / n for i in range(n + 1)]


def student_onnx(te, un, caps, seeds, steps, bs=8):
    from conditioning import tokenize
    out = []
    for b in range(0, len(caps), bs):
        c, sd = caps[b:b + bs], seeds[b:b + bs]
        ids, mk = tokenize(c)
        ctx, cm = te.run(None, {"input_ids": ids.numpy(), "attention_mask": mk.long().numpy()})
        x = np.concatenate([noise(s) for s in sd]); ts = sched(steps)
        for i in range(steps):
            v = un.run(None, {"x_t": x, "t": np.full((len(c),), ts[i], np.float32), "context": ctx, "context_mask": cm})[0]
            x = x - (ts[i] - ts[i + 1]) * v
        out.append(np.clip(x, -1, 1))
    return np.concatenate(out)


@torch.no_grad()
def teacher_images(teacher_path, caps, seeds, steps=50, cfg=5.0, bs=8, device="cpu"):
    from train_teacher_v3 import load_flow, sample_cfg
    T = load_flow(teacher_path, device)
    out = []
    for b in range(0, len(caps), bs):
        z = torch.from_numpy(np.concatenate([noise(s) for s in seeds[b:b + bs]])).to(device)
        out.append(sample_cfg(T, caps[b:b + bs], z, steps, cfg).cpu().numpy())
    return np.concatenate(out)


def to_img(x):
    return Image.fromarray(((x.transpose(1, 2, 0) + 1) * 127.5).round().clip(0, 255).astype(np.uint8))


def cmd_models(a):
    from eval_grid import grid48, grid14
    from crnn import load_crnn
    crnn = load_crnn(a.crnn)
    tess = tuple(t for t in a.tess.split(",") if t)
    os.makedirs(a.assets, exist_ok=True)
    te, un = ort_sess(a.text), ort_sess(a.unet)
    g = grid48(); caps = [c for _, c, _ in g]; seeds = [s for *_, s in g]; lays = [sp["layout"] for sp, _, _ in g]
    imgs, res = {}, {"unet": os.path.basename(a.unet), "text_encoder": os.path.basename(a.text), "n": len(g),
                     "captions_truncated": 0, "primary_metric": "crnn exact (case-sensitive)", "per_sample": [], "summary": {}}
    from conditioning import MAX_LEN
    res["captions_truncated"] = sum(len(c) > MAX_LEN for c in caps)
    t0 = time.time()
    for steps in a.steps:
        imgs[f"s{steps}"] = [to_img(z) for z in student_onnx(te, un, caps, seeds, steps)]
    imgs["ref"] = reference_images(g)
    preds = {k: ocr_all(v, lays, crnn, tess) for k, v in imgs.items()}
    for i, (sp, cap, seed) in enumerate(g):
        row = dict(sp, caption=cap, seed=seed)
        for k in imgs:
            row[k] = {m: dict(ocr=p[i], **score_text(p[i], sp["text"])) for m, p in preds[k].items()}
        res["per_sample"].append(row)
    for k in imgs:
        res["summary"][k] = {m: summarize([{"x": r[k][m]} for r in res["per_sample"]], "x") for m in preds[k]}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.assets, "eval_ocr_v3.json"), "w"), indent=1)
    for steps in a.steps:
        lab = [f"'{sp['text']}' -> '{r[f's{steps}']['crnn']['ocr']}'\n{sp['expr']}/{sp['layout']}/{sp['paper']}"
               for (sp, _, _), r in zip(g, res["per_sample"])]
        grid_png(imgs[f"s{steps}"], lab, 8, os.path.join(a.assets, f"samples_{steps}step_v3.png"),
                 title=f"grinpress v3 student ({os.path.basename(a.unet)}), {steps} steps, seeds 1000+i (label: text -> CRNN read)")
    print(json.dumps(res["summary"], indent=1))
    # ---- 14-string teacher vs student comparison
    g14 = grid14(); c14 = [c for _, c, _ in g14]; s14 = [s for *_, s in g14]; l14 = [sp["layout"] for sp, _, _ in g14]
    rows = {"ref": reference_images(g14)}
    if a.teacher:
        rows["teacher"] = [to_img(z) for z in teacher_images(a.teacher, c14, s14, a.teacher_steps, a.cfg, device=a.device)]
    for steps in a.steps:
        rows[f"s{steps}"] = [to_img(z) for z in student_onnx(te, un, c14, s14, steps)]
    cmp = {}
    for k, ims in rows.items():
        p = ocr_all(ims, l14, crnn, tess)
        cmp[k] = {m: dict(summarize([{"x": score_text(o, sp["text"])} for o, (sp, _, _) in zip(v, g14)], "x"), ocr=v) for m, v in p.items()}
    json.dump(cmp, open(os.path.join(a.assets, "eval_compare_v3.json"), "w"), indent=1)
    all_imgs, labels = [], []
    for k, ims in rows.items():
        for i, im in enumerate(ims):
            all_imgs.append(im); labels.append(f"'{g14[i][0]['text']}' {k}\n-> '{cmp[k]['crnn']['ocr'][i]}'")
    grid_png(all_imgs, labels, len(g14), os.path.join(a.assets, "teacher_vs_student_v3.png"), cell=160,
             title="rows: " + " | ".join(rows) + "  (same noise per column; label: CRNN read)")
    print(json.dumps({k: {m: {x: y for x, y in v.items() if x != "ocr"} for m, v in d.items()} for k, d in cmp.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refs")
    r.add_argument("--crnn", default=os.path.join(ROOT, "artifacts", "crnn.safetensors"))
    r.add_argument("--out", default=os.path.join(ROOT, "assets", "eval_ocr_refs_phase2.json"))
    m = sub.add_parser("models")
    m.add_argument("--unet", required=True)
    m.add_argument("--text", required=True)
    m.add_argument("--teacher", default="")
    m.add_argument("--teacher_steps", type=int, default=50)
    m.add_argument("--cfg", type=float, default=5.0)
    m.add_argument("--steps", type=int, nargs="+", default=[4, 2])
    m.add_argument("--crnn", default=os.path.join(ROOT, "artifacts", "crnn.safetensors"))
    m.add_argument("--tess", default="v3", help="comma list of tesseract variants to also run (v2,v3,v3s) or ''")
    m.add_argument("--assets", default=os.path.join(ROOT, "assets"))
    m.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    {"refs": cmd_refs, "models": cmd_models}[a.cmd](a)


if __name__ == "__main__":
    main()
