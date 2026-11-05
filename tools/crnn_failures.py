"""CRNN ref48 failure analysis: det vs GT box, box IoU, px-per-char, failure class. Usage:
  python tools/crnn_failures.py CKPT(.pt training ckpt or .safetensors) [out.json]"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from crnn import CRNN, load_crnn, crop_text, decode
from train_crnn import reference_set, to_tensor
from eval_ocr import score_text

def load(p):
    if p.endswith(".safetensors"): return load_crnn(p)
    st = torch.load(p, map_location="cpu", weights_only=False)
    sd = st["model"]
    # train_crnn.py saves no cfg, so read the architecture off the weights: cnn.{0,2,4,7,10} are the five cbr
    # blocks (c1..c4, cs), cnn.0's input dim is in_ch, and seq.N counts the Res1d layers
    w = tuple(sd[f"cnn.{i}.0.weight"].shape[0] for i in (0, 2, 4, 7, 10)); in_ch = sd["cnn.0.0.weight"].shape[1]
    seq = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("seq."))
    m = CRNN(w, seq, in_ch); m.load_state_dict(sd); return m.eval()

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy; return ix*iy/max(u, 1e-6)

def classify(gt, pred, pred_gt):
    c = []
    if pred.lower() == gt.lower(): c.append("case")
    if pred_gt == gt: c.append("detector")   # right with GT box, wrong with detector box
    if any(ch in gt for ch in "!?.,:;%$#&@*+-=~<>/()^_") and "".join(ch for ch in pred if ch.isalnum() or ch == " ") == "".join(ch for ch in gt if ch.isalnum() or ch == " "): c.append("punct")
    if len(gt) >= 10: c.append("long")
    return c or ["other"]

m = load(sys.argv[1]); imgs, lay, texts, gt, det = reference_set()
with torch.no_grad():
    x = to_tensor(imgs)
    pd = decode(m(crop_text(x, torch.from_numpy(det)))); pg = decode(m(crop_text(x, torch.from_numpy(gt))))
rows = []
for i, t in enumerate(texts):
    w = gt[i][2]-gt[i][0]; h = gt[i][3]-gt[i][1]
    rows.append(dict(i=i, gt=t, pred_det=pd[i], pred_gtbox=pg[i], layout=["left","right","below"][lay[i]], len=len(t),
                     gt_box=[round(float(v),1) for v in gt[i]], det_box=[round(float(v),1) for v in det[i]],
                     iou=round(float(iou(gt[i], det[i])), 3), text_h_px=round(float(h),1), px_per_char=round(float(w)/len(t),2),
                     ok_det=bool(pd[i] == t), ok_gt=bool(pg[i] == t)))
miss = [dict(r, cls=classify(r["gt"], r["pred_det"], r["pred_gtbox"])) for r in rows if not r["ok_det"]]
s = dict(ckpt=sys.argv[1], ref48_exact_det=sum(r["ok_det"] for r in rows)/48, ref48_exact_gtbox=sum(r["ok_gt"] for r in rows)/48,
         n_miss=len(miss), class_counts={}, mean_iou_all=float(np.mean([r["iou"] for r in rows])),
         mean_iou_miss=float(np.mean([r["iou"] for r in miss])) if miss else None,
         mean_px_per_char_all=float(np.mean([r["px_per_char"] for r in rows])),
         mean_px_per_char_miss=float(np.mean([r["px_per_char"] for r in miss])) if miss else None,
         text_h_px_all=float(np.mean([r["text_h_px"] for r in rows])),
         miss_by_layout={l: sum(r["layout"] == l for r in miss) for l in ("left", "right", "below")}, misses=miss, rows=rows)
for r in miss:
    for c in r["cls"]: s["class_counts"][c] = s["class_counts"].get(c, 0) + 1
if len(sys.argv) > 2: json.dump(s, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v for k, v in s.items() if k not in ("rows", "misses")}, indent=1))
for r in miss: print(f"  {r['gt']!r:18} det->{r['pred_det']!r:18} gtbox->{r['pred_gtbox']!r:18} {r['layout']:5} iou {r['iou']:.2f} h {r['text_h_px']:4.1f}px {r['px_per_char']:4.1f}px/ch {r['cls']}")
