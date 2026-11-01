"""Varied-word stress grids for the v3 INT8 ONNX student (not part of the official grid48/grid14 eval).

5 categories x 7 words, each word rendered with ALL 7 expressions (layout/paper/colour cycle per cell),
at 2 and 4 steps, scored with the fine-tuned CRNN (case-sensitive exact, eval_ocr.score_text). The procedural
reference render (generate_data_v2.render_v2, same spec+seed) is also OCR'd so CRNN misreads can be told
apart from model failures.
  python samples_varied.py --onnx_dir onnx --crnn artifacts/crnn_ft.safetensors --out samples
    -> samples/varied_{2,4}step_<category>.png, samples/varied_words_v3.json"""
import argparse, json, os, random, time
import numpy as np
from eval_ocr import ort_sess, student_onnx, to_img, grid_png, score_text, fonts
from eval_grid import COLORS, LAYS, PAPS, HELD_OUT
from generate_data_v2 import EXPRESSIONS_V2, caption_v2, MAX_TEXT
from charset import CHARSET

WORDS = {
    "short": ["yo", "pup", "Hi!", "zz", "a", "OMG", "ty"],
    "long": ["SNUGGLE BUDDY!", "tail wagging", "headpats pls", "BELLY RUBS NOW", "zoomies at 3am", "good pup vibes",
             "who let me out"],
    "silly": ["bonk", "boop!", "sniff sniff", "chimkin", "heckin", "blep", "smol bean"],
    "mixed_case": ["PuPpY", "sQuEaKy", "WaGgY", "BoRk BoRk", "tReAt", "McWoof", "iPup"],
    "punct_num": ["10/10", "24/7", "(o_o)", "99.9%", "1+1=2", "$3.50", "3:00 AM"],
}
for ws in WORDS.values():
    for w in ws:
        assert len(w) <= MAX_TEXT and all(c in CHARSET for c in w) and w not in HELD_OUT, w


def specs():
    out, k = [], 0
    for cat, ws in WORDS.items():
        for wi, w in enumerate(ws):
            for e, expr in enumerate(EXPRESSIONS_V2):
                sp = dict(category=cat, text=w, expr=expr, layout=LAYS[(wi + e) % 3], paper=PAPS[(wi + 2 * e) % 3],
                          color=COLORS[(k * 5 + e) % 12])
                out.append((sp, caption_v2(sp["color"], sp["expr"], sp["text"], sp["layout"], sp["paper"]), 3000 + k))
                k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", default="onnx")
    ap.add_argument("--crnn", default="artifacts/crnn_ft.safetensors")
    ap.add_argument("--out", default="samples")
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 2])
    a = ap.parse_args()
    from crnn import load_crnn, recognize
    from generate_data_v2 import render_v2
    os.makedirs(a.out, exist_ok=True)
    crnn = load_crnn(a.crnn)
    te = ort_sess(os.path.join(a.onnx_dir, "text_encoder_int8.onnx"))
    un = ort_sess(os.path.join(a.onnx_dir, "unet_distilled_int8.onnx"))
    S = specs(); caps = [c for _, c, _ in S]; seeds = [s for *_, s in S]; lays = [sp["layout"] for sp, _, _ in S]
    t0 = time.time(); imgs = {}
    for st in a.steps:
        imgs[f"s{st}"] = [to_img(z) for z in student_onnx(te, un, caps, seeds, st)]
    F = fonts()
    imgs["ref"] = [render_v2(128, random.Random(seed), F, **{k: sp[k] for k in ("color", "expr", "text", "layout", "paper")})[0]
                   for sp, _, seed in S]
    reads = {k: recognize(crnn, v, lays) for k, v in imgs.items()}
    rows = []
    for i, (sp, cap, seed) in enumerate(S):
        r = dict(sp, caption=cap, seed=seed)
        for k in imgs: r[k] = dict(ocr=reads[k][i], **score_text(reads[k][i], sp["text"]))
        rows.append(r)

    def agg(sel):
        return {k: dict(n=len(sel), exact=float(np.mean([r[k]["exact"] for r in sel])),
                        exact_ci=float(np.mean([r[k]["exact_ci"] for r in sel])),
                        char_acc_cs=float(np.mean([r[k]["char_acc_cs"] for r in sel]))) for k in imgs}
    summ = {"all": agg(rows)}
    for cat in WORDS: summ[f"cat:{cat}"] = agg([r for r in rows if r["category"] == cat])
    for e in EXPRESSIONS_V2: summ[f"expr:{e}"] = agg([r for r in rows if r["expr"] == e])
    for l in LAYS: summ[f"layout:{l}"] = agg([r for r in rows if r["layout"] == l])
    for lo, hi in ((1, 3), (4, 7), (8, 11), (12, 14)):
        sel = [r for r in rows if lo <= len(r["text"]) <= hi]
        if sel: summ[f"len:{lo}-{hi}"] = agg(sel)
    per_word = {}
    for cat, ws in WORDS.items():
        for w in ws:
            sel = [r for r in rows if r["text"] == w]
            per_word[w] = {k: sum(r[k]["exact"] for r in sel) for k in imgs}
    res = dict(what="varied-word stress test, INT8 ONNX student (unet_distilled_int8 + text_encoder_int8), CRNN fine-tuned "
                    "(case-sensitive exact). 35 words x 7 expressions. NOT the official grid48 metric.",
               crnn=os.path.basename(a.crnn), seconds=round(time.time() - t0, 1), summary=summ,
               per_word_exact_of_7=per_word, per_sample=rows)
    json.dump(res, open(os.path.join(a.out, "varied_words_v3.json"), "w"), indent=1)
    for st in a.steps:
        k = f"s{st}"
        for cat in WORDS:
            idx = [i for i, r in enumerate(rows) if r["category"] == cat]
            lab = [f"'{rows[i]['text']}'->'{rows[i][k]['ocr']}'{'' if rows[i][k]['exact'] else ' X'}\n"
                   f"{rows[i]['expr']}/{rows[i]['layout']}/{rows[i]['paper']}" for i in idx]
            grid_png([imgs[k][i] for i in idx], lab, 7, os.path.join(a.out, f"varied_{st}step_{cat}.png"), cell=176,
                     title=f"grinpress v3 INT8 ONNX, {st} steps, '{cat}' words x 7 expressions (label: text -> CRNN read, X = miss)")
    print(json.dumps({k: v for k, v in summ.items()}, indent=0))
    print(json.dumps(per_word))


if __name__ == "__main__":
    main()
