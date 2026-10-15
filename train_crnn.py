"""Phase 2: train the tiny conv-CTC recognizer on procedural crops from generate_data_v2 (box CPU).

  python train_crnn.py --n_train 20000 --epochs 15            # -> artifacts/crnn.safetensors, assets/crnn_train.json
  python train_crnn.py --eval_only --ckpt artifacts/crnn.safetensors

Data: render_v3 (v3 curriculum texts + 30% uniform random CHARSET strings, uniform layouts/papers/expressions,
50% ink-colour variety), stored as 128x128 grey uint8 + layout + text. The 48 eval strings are held out of training.
Gate (spec): >= 98% exact match on the 48 procedural reference renders (seeds 1000+i).
"""
import argparse, json, math, os, random, time
from multiprocessing import Pool
import numpy as np, torch, torch.nn.functional as F
from crnn import CRNN, crop_band, ctc_loss, decode, save_crnn, load_crnn, LAYOUT_ID
from charset import CHARSET

ROOT = os.path.dirname(os.path.abspath(__file__))
_FONTS = None


def fonts():
    global _FONTS
    if _FONTS is None:
        from generate_data import ensure_fonts
        _FONTS = sorted(ensure_fonts())
    return _FONTS


def random_charset_text(rng):
    n = rng.randint(1, 14)
    while True:
        t = "".join(rng.choice(CHARSET) for _ in range(n))
        t = " ".join(t.split())
        if t:
            return t


def _render(job):
    i, seed = job
    from generate_data_v2 import render_v3, sample_spec_v3
    from eval_grid import HELD_OUT
    rng = random.Random(seed * 1_000_003 + i)
    spec = sample_spec_v3(rng)
    if rng.random() < 0.3:
        t = random_charset_text(rng)
        spec["text"] = t if t not in HELD_OUT else spec["text"]
    img, meta, aux = render_v3(128, rng, fonts(), spec=spec, ink_variety=True)
    g = np.asarray(img.convert("L"))
    return g, LAYOUT_ID[meta["layout"]], meta["text"], aux["text_bbox"]


def make_set(n, seed, workers, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if len(d["texts"]) >= n:
            print(f"[data] cached {cache}"); return d["imgs"][:n], d["lay"][:n], list(d["texts"][:n])
    t0 = time.time()
    with Pool(workers) as p:
        out = p.map(_render, [(i, seed) for i in range(n)], chunksize=64)
    imgs = np.stack([o[0] for o in out]); lay = np.array([o[1] for o in out], np.int64); texts = [o[2] for o in out]
    print(f"[data] rendered {n} in {time.time()-t0:.0f}s")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, imgs=imgs, lay=lay, texts=np.array(texts, dtype=object))
    return imgs, lay, texts


def reference_set():
    """The 48 procedural reference renders of the v2/v3 eval grid (byte-identical to v2's reference_renders)."""
    from generate_data_v2 import render_v2
    from eval_grid import grid48
    imgs, lay, texts = [], [], []
    for sp, cap, seed in grid48():
        im = render_v2(128, random.Random(seed), fonts(), **{k: sp[k] for k in ("color", "expr", "text", "layout", "paper")})[0]
        imgs.append(np.asarray(im.convert("L"))); lay.append(LAYOUT_ID[sp["layout"]]); texts.append(sp["text"])
    return np.stack(imgs), np.array(lay), texts


def to_tensor(u8):
    return torch.from_numpy(u8).float().div_(127.5).sub_(1).unsqueeze(1)


def augment(crop, rng_t):
    B = crop.shape[0]
    # blur (sigma ~U(0.3,1.2) on half the batch)
    m = torch.rand(B) < 0.5
    if m.any():
        s = float(torch.empty(1).uniform_(0.3, 1.2)); k = 5
        ax = torch.arange(k) - k // 2; g = torch.exp(-ax.float() ** 2 / (2 * s * s)); g = g / g.sum()
        ker = (g[:, None] * g[None]).view(1, 1, k, k)
        crop[m] = F.conv2d(F.pad(crop[m], (2, 2, 2, 2), mode="replicate"), ker)
    # down-up resample (softness like a few-step sample)
    m = torch.rand(B) < 0.3
    if m.any():
        f = float(torch.empty(1).uniform_(0.45, 0.8))
        small = F.interpolate(crop[m], scale_factor=f, mode="bilinear", align_corners=False, antialias=True)
        crop[m] = F.interpolate(small, size=crop.shape[-2:], mode="bilinear", align_corners=False)
    # contrast / brightness / gamma-ish fade
    c = torch.empty(B, 1, 1, 1).uniform_(0.55, 1.15); b = torch.empty(B, 1, 1, 1).uniform_(-0.15, 0.15)
    crop = crop * c + b
    # noise
    crop = crop + torch.randn_like(crop) * torch.empty(B, 1, 1, 1).uniform_(0, 0.08) * (torch.rand(B, 1, 1, 1) < 0.6)
    return crop


def evaluate(model, imgs, lay, texts, bs=256):
    model.eval(); preds = []
    with torch.no_grad():
        for b in range(0, len(texts), bs):
            x = to_tensor(imgs[b:b + bs]); l = torch.from_numpy(lay[b:b + bs])
            preds += decode(model(crop_band(x, l)))
    from eval_ocr import score_text
    rows = [dict(gt=t, pred=p, **score_text(p, t)) for p, t in zip(preds, texts)]
    n = len(rows)
    return dict(n=n, exact=sum(r["exact"] for r in rows) / n, exact_ci=sum(r["exact_ci"] for r in rows) / n,
                char_acc=float(np.mean([r["char_acc"] for r in rows]))), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, nargs=5, default=[24, 48, 96, 128, 160])
    ap.add_argument("--seq_layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "crnn.safetensors"))
    ap.add_argument("--report", default=os.path.join(ROOT, "assets", "crnn_train.json"))
    ap.add_argument("--init", default="", help="continue from an existing crnn.safetensors")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    ref = reference_set()
    if a.eval_only:
        m = load_crnn(a.ckpt or a.out)
        s, rows = evaluate(m, *ref); print(json.dumps(s, indent=1))
        for r in rows:
            if not r["exact"]: print("  miss:", r["gt"], "->", r["pred"])
        return
    tr = make_set(a.n_train, a.seed, a.workers, os.path.join(a.cache_dir, f"crnn_train_s{a.seed}_n{a.n_train}.npz"))
    va = make_set(a.n_val, a.seed + 1000, a.workers, os.path.join(a.cache_dir, f"crnn_val_s{a.seed+1000}_n{a.n_val}.npz"))
    from eval_grid import HELD_OUT
    assert not (set(tr[2]) & HELD_OUT), "eval strings leaked into CRNN training data"
    m = load_crnn(a.init).train().requires_grad_(True) if a.init else CRNN(tuple(a.width), a.seq_layers)
    print(f"[crnn] params {sum(p.numel() for p in m.parameters())/1e6:.2f}M, train {len(tr[2])}, val {len(va[2])}")
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_ep = len(tr[2]) // a.bs; total = steps_per_ep * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    imgs, lay, texts = tr; t0 = time.time(); hist = []; step = 0
    for ep in range(a.epochs):
        m.train(); perm = np.random.default_rng(a.seed + ep).permutation(len(texts)); la = 0
        for k in range(steps_per_ep):
            idx = perm[k * a.bs:(k + 1) * a.bs]
            x = to_tensor(imgs[idx]); l = torch.from_numpy(lay[idx])
            jit = torch.empty(len(idx), 4).uniform_(-3.5, 3.5)
            crop = augment(crop_band(x, l, jit), None)
            loss = ctc_loss(m(crop), [texts[i] for i in idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step(); step += 1
            la = 0.98 * la + 0.02 * loss.item() if k else loss.item()
        sv, _ = evaluate(m, *va); sr, _ = evaluate(m, *ref)
        hist.append(dict(epoch=ep + 1, loss=round(la, 4), val=sv, ref48=sr, elapsed_s=int(time.time() - t0)))
        print(f"[crnn] ep {ep+1}/{a.epochs} loss {la:.4f} | val exact {sv['exact']:.4f} ci {sv['exact_ci']:.4f} | "
              f"ref48 exact {sr['exact']:.4f} ci {sr['exact_ci']:.4f} | {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); save_crnn(m, a.out)
    m2 = load_crnn(a.out)  # evaluate the fp16-stored weights actually shipped
    sv, _ = evaluate(m2, *va); sr, rows = evaluate(m2, *ref)
    rep = dict(args=vars(a), params=sum(p.numel() for p in m.parameters()), history=hist, final_val=sv, final_ref48=sr,
               ref48_misses=[r for r in rows if not r["exact"]], gate_ref48_exact_ge_0_98=sr["exact"] >= 0.98,
               train_seconds=int(time.time() - t0))
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("final_val", "final_ref48", "gate_ref48_exact_ge_0_98", "train_seconds")}, indent=1))
    for r in rep["ref48_misses"]: print("  miss:", r["gt"], "->", r["pred"])


if __name__ == "__main__":
    main()
