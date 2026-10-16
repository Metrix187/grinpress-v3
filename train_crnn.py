"""Phase 2: train the tiny conv-CTC recognizer on procedural crops from generate_data_v2 (box CPU).

  python train_crnn.py --n_train 20000 --epochs 15            # -> artifacts/crnn.safetensors, assets/crnn_train.json
  python train_crnn.py --eval_only --ckpt artifacts/crnn.safetensors

Data: render_v3 (v3 curriculum texts + 20% uniform random CHARSET strings, uniform layouts/papers/expressions,
50% ink-colour variety), stored as 128x128 grey uint8 + layout + text + GT text bbox + detected text bbox.
Training crops use the GT box or the detected box (50/50) with +-1.5 px jitter; evaluation uses detect_boxes only
(exactly what eval_ocr.py does on generated images). The 48 eval strings are held out of training.
Gate (spec): >= 98% exact match on the 48 procedural reference renders (seeds 1000+i).
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn.functional as F
from crnn import CRNN, crop_text, detect_boxes, ctc_loss, decode, save_crnn, load_crnn, LAYOUT_ID
from charset import CHARSET
from gp_io import worker_pool, Heartbeat

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


AMBIG = "cosuvwxzpk"          # glyphs whose lower/upper case differ (almost) only in size
CONFUSE = "ectdrnmhHIl10OoaGgqbB"  # observed / classic confusions (e/c, d/t, I/H, r/e, o/O)


def hard_text(rng):
    """v3 run-3 hard classes (from the run-2b ref48 failure analysis): case-ambiguous x-height strings, confusable
    glyphs, and long 10-14 char strings with spaces/punctuation at tiny font sizes."""
    k = rng.random()
    if k < 0.35:      # case-ambiguous: all lower / all upper / one-cap of size-only-case letters (+ a few others)
        n = rng.randint(2, 8); pool = AMBIG + "aeimnr"
        t = "".join(rng.choice(pool if rng.random() < 0.8 else AMBIG) for _ in range(n))
        r = rng.random(); t = t.upper() if r < 0.4 else (t if r < 0.8 else "".join(c.upper() if rng.random() < 0.4 else c for c in t))
    elif k < 0.6:     # confusables
        n = rng.randint(3, 12); t = "".join(rng.choice(CONFUSE if rng.random() < 0.7 else CHARSET) for _ in range(n))
    else:             # long 10-14, word-ish with spaces, mixed case + punctuation
        n = rng.randint(10, 14); t = ""
        while len(t) < n:
            w = "".join(rng.choice(CHARSET[:52] if rng.random() < 0.85 else CHARSET) for _ in range(rng.randint(2, 7)))
            r = rng.random(); w = w.upper() if r < 0.35 else (w.lower() if r < 0.7 else w)
            t = (t + " " + w) if t else w
        t = t[:n]
    t = " ".join(t.split())
    return t or "ok?"


HARD_FRAC = 0.0


def _render(job):
    i, seed = job[:2]
    hard_frac = job[2] if len(job) > 2 else 0.0
    from generate_data_v2 import render_v3, sample_spec_v3
    from eval_grid import HELD_OUT
    rng = random.Random(seed * 1_000_003 + i)
    spec = sample_spec_v3(rng)
    if rng.random() < 0.2:
        t = random_charset_text(rng)
        spec["text"] = t if t not in HELD_OUT else spec["text"]
    if hard_frac and rng.random() < hard_frac:
        t = hard_text(rng)
        spec["text"] = t if t not in HELD_OUT else spec["text"]
    img, meta, aux = render_v3(128, rng, fonts(), spec=spec, ink_variety=True)
    g = np.array(img.convert("L"))  # writable copy (np.asarray of a PIL image is read-only -> torch warning)
    lid = LAYOUT_ID[meta["layout"]]
    det = detect_boxes(torch.from_numpy(g).float().div(127.5).sub(1)[None, None], torch.tensor([lid]))[0].tolist()
    return g, lid, meta["text"], list(aux["text_bbox"]), det


RENDER_CHUNK = int(os.environ.get("GP_RENDER_CHUNK", 5000))  # progress line + resumable partial cache every RENDER_CHUNK samples


def _save_npz(path, **kw):
    with open(path + ".tmp", "wb") as f:
        np.savez(f, **kw)
    os.replace(path + ".tmp", path)


def make_set(n, seed, workers, cache, hard_frac=0.0, hb=None, what="train"):
    """-> (imgs uint8 [n,128,128], layout [n], texts, gt_boxes [n,4], det_boxes [n,4])
    Rendered in a spawn-context Pool of single-threaded workers (gp_io.worker_pool), streamed with imap; every
    RENDER_CHUNK samples a progress line is printed, the heartbeat is written and the chunk is saved to
    <cache>.parts/ so an interrupted render resumes instead of starting over."""
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if len(d["texts"]) >= n:
            print(f"[data] cached {cache}", flush=True)
            return d["imgs"][:n], d["lay"][:n], list(d["texts"][:n]), d["gt"][:n], d["det"][:n]
    t0 = time.time(); n_ch = math.ceil(n / RENDER_CHUNK)
    pdir = (cache + ".parts") if cache else ""
    if pdir: os.makedirs(pdir, exist_ok=True)
    parts = {}
    for c in range(n_ch):
        pp = os.path.join(pdir, f"part_{c:04d}.npz") if pdir else ""
        if pp and os.path.exists(pp):
            try:
                d = np.load(pp, allow_pickle=True)
                parts[c] = (d["imgs"], d["lay"], list(d["texts"]), d["gt"], d["det"])
            except Exception as e:  # noqa: BLE001  (truncated part from a killed run -> re-render it)
                print(f"[data] bad part {pp} ({e}); re-rendering", flush=True)
    if parts:
        print(f"[data] {what}: resuming, {len(parts)}/{n_ch} chunks of {RENDER_CHUNK} already rendered", flush=True)
    todo = [c for c in range(n_ch) if c not in parts]
    if todo:
        jobs = [(i, seed, hard_frac) for c in todo for i in range(c * RENDER_CHUNK, min(n, (c + 1) * RENDER_CHUNK))]
        done = sum(len(parts[c][2]) for c in parts)
        pool = worker_pool(workers)
        try:
            it = pool.imap(_render, jobs, chunksize=16); buf = []
            for c in todo:
                m = min(n, (c + 1) * RENDER_CHUNK) - c * RENDER_CHUNK
                buf = [next(it) for _ in range(m)]
                part = (np.stack([o[0] for o in buf]), np.array([o[1] for o in buf], np.int64), [o[2] for o in buf],
                        np.array([o[3] for o in buf], np.float32), np.array([o[4] for o in buf], np.float32))
                parts[c] = part; done += m
                if pdir:
                    _save_npz(os.path.join(pdir, f"part_{c:04d}.npz"), imgs=part[0], lay=part[1],
                              texts=np.array(part[2], dtype=object), gt=part[3], det=part[4])
                el = time.time() - t0
                print(f"[data] {what} {done}/{n} {el:.0f}s", flush=True)
                if hb: hb(force=True, phase="crnn_render", set=what, done=done, total=n, elapsed_s=int(el))
        finally:
            pool.close(); pool.join()
    cat = [parts[c] for c in range(n_ch)]
    imgs = np.concatenate([p[0] for p in cat]); lay = np.concatenate([p[1] for p in cat])
    texts = [t for p in cat for t in p[2]]
    gt = np.concatenate([p[3] for p in cat]); det = np.concatenate([p[4] for p in cat])
    print(f"[data] rendered {n} in {time.time()-t0:.0f}s", flush=True)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        _save_npz(cache, imgs=imgs, lay=lay, texts=np.array(texts, dtype=object), gt=gt, det=det)
        import shutil; shutil.rmtree(pdir, ignore_errors=True)
    return imgs, lay, texts, gt, det


def reference_set():
    """The 48 procedural reference renders of the v2/v3 eval grid (byte-identical to v2's reference_renders)."""
    from generate_data_v2 import render_v2
    from eval_grid import grid48
    imgs, lay, texts, gt = [], [], [], []
    for sp, cap, seed in grid48():
        im, _, aux = render_v2(128, random.Random(seed), fonts(), aux=True, **{k: sp[k] for k in ("color", "expr", "text", "layout", "paper")})
        imgs.append(np.asarray(im.convert("L"))); lay.append(LAYOUT_ID[sp["layout"]]); texts.append(sp["text"]); gt.append(aux["text_bbox"])
    imgs, lay = np.stack(imgs), np.array(lay)
    det = detect_boxes(to_tensor(imgs), torch.from_numpy(lay)).numpy()
    return imgs, lay, texts, np.array(gt, np.float32), det


def to_tensor(u8):
    return torch.from_numpy(u8).float().div_(127.5).sub_(1).unsqueeze(1)


def augment(crop, rng_t):
    """augment the grey channel only; the scale channel (if any) is passed through untouched."""
    if crop.shape[1] > 1:
        return torch.cat([augment(crop[:, :1].contiguous(), rng_t), crop[:, 1:]], 1)
    B, d = crop.shape[0], crop.device
    # blur (sigma ~U(0.3,1.2) on half the batch)
    m = torch.rand(B, device=d) < 0.5
    if m.any():
        s = float(torch.empty(1).uniform_(0.3, 1.2)); k = 5
        ax = torch.arange(k, device=d) - k // 2; g = torch.exp(-ax.float() ** 2 / (2 * s * s)); g = g / g.sum()
        ker = (g[:, None] * g[None]).view(1, 1, k, k)
        crop[m] = F.conv2d(F.pad(crop[m], (2, 2, 2, 2), mode="replicate"), ker)
    # down-up resample (softness like a few-step sample)
    m = torch.rand(B, device=d) < 0.3
    if m.any():
        f = float(torch.empty(1).uniform_(0.45, 0.8))
        small = F.interpolate(crop[m], scale_factor=f, mode="bilinear", align_corners=False, antialias=True)
        crop[m] = F.interpolate(small, size=crop.shape[-2:], mode="bilinear", align_corners=False)
    # contrast / brightness / gamma-ish fade
    c = torch.empty(B, 1, 1, 1, device=d).uniform_(0.55, 1.15); b = torch.empty(B, 1, 1, 1, device=d).uniform_(-0.15, 0.15)
    crop = crop * c + b
    # noise
    crop = crop + torch.randn_like(crop) * torch.empty(B, 1, 1, 1, device=d).uniform_(0, 0.08) * (torch.rand(B, 1, 1, 1, device=d) < 0.6)
    return crop


def evaluate(model, imgs, lay, texts, gt, det, bs=256, use="det"):
    model.eval(); preds = []
    boxes = det if use == "det" else gt
    dev = next(model.parameters()).device
    with torch.no_grad():
        for b in range(0, len(texts), bs):
            x = to_tensor(imgs[b:b + bs]).to(dev)
            preds += decode(model(crop_text(x, torch.from_numpy(boxes[b:b + bs]))))
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
    ap.add_argument("--in_ch", type=int, default=1, help="2 = grey + absolute-scale channel (fixes o/O, w/W case confusions)")
    ap.add_argument("--hard_frac", type=float, default=0.0, help="fraction of training texts from hard_text() (failure classes)")
    ap.add_argument("--device", default="", help="default: cuda if available (resolved only AFTER data rendering, so the parent never initialises CUDA before the render pool starts)")
    ap.add_argument("--heartbeat", default="", help="status.json path: written at render progress points and every epoch")
    ap.add_argument("--tag", default="", help="suffix for cache / ckpt names (keeps runs with different data apart)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "crnn.safetensors"))
    ap.add_argument("--report", default=os.path.join(ROOT, "assets", "crnn_train.json"))
    ap.add_argument("--init", default="", help="continue from an existing crnn.safetensors")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--resume_ckpt", default="", help="per-epoch training checkpoint (default cache/crnn_ckpt_*.pt)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    ref = reference_set()
    if a.eval_only:
        m = load_crnn(a.ckpt or a.out)
        s, rows = evaluate(m, *ref); print(json.dumps(s, indent=1))
        for r in rows:
            if not r["exact"]: print("  miss:", r["gt"], "->", r["pred"])
        return
    hf = f"_h{int(a.hard_frac*100)}" if a.hard_frac else ""
    hb = Heartbeat(a.heartbeat, 60)
    tr = make_set(a.n_train, a.seed, a.workers, os.path.join(a.cache_dir, f"crnn_train_s{a.seed}_n{a.n_train}{hf}.npz"), a.hard_frac, hb, "train")
    va = make_set(a.n_val, a.seed + 1000, a.workers, os.path.join(a.cache_dir, f"crnn_val_s{a.seed+1000}_n{a.n_val}.npz"), 0.0, hb, "val")
    a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")   # first CUDA touch: after rendering
    from eval_grid import HELD_OUT
    assert not (set(tr[2]) & HELD_OUT), "eval strings leaked into CRNN training data"
    m = load_crnn(a.init).train().requires_grad_(True) if a.init else CRNN(tuple(a.width), a.seq_layers, a.in_ch)
    dev = torch.device(a.device); m = m.to(dev)
    print(f"[crnn] params {sum(p.numel() for p in m.parameters())/1e6:.2f}M, train {len(tr[2])}, val {len(va[2])}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_ep = len(tr[2]) // a.bs; total = steps_per_ep * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    imgs, lay, texts, gtb, detb = tr; t0 = time.time(); hist = []; step = 0; ep0 = 0
    # per-epoch checkpoint + resume (an interrupted run restarts from the last finished epoch)
    ck = a.resume_ckpt or os.path.join(a.cache_dir, f"crnn_ckpt_s{a.seed}_n{a.n_train}_e{a.epochs}{hf}{a.tag}.pt")
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        hist, step, ep0 = st["hist"], st["step"], st["epoch"]; t0 -= st.get("elapsed", 0)
        torch.set_rng_state(st["rng"].cpu()); print(f"[crnn] resumed {ck} at epoch {ep0}", flush=True)
    for ep in range(ep0, a.epochs):
        m.train(); perm = np.random.default_rng(a.seed + ep).permutation(len(texts)); la = 0
        for k in range(steps_per_ep):
            idx = perm[k * a.bs:(k + 1) * a.bs]
            x = to_tensor(imgs[idx]).to(dev, non_blocking=True)
            pick = torch.rand(len(idx)) < 0.5
            boxes = torch.where(pick[:, None], torch.from_numpy(gtb[idx]), torch.from_numpy(detb[idx])).to(dev)
            jit = torch.empty(len(idx), 4).uniform_(-1.5, 1.5).to(dev)
            crop = augment(crop_text(x, boxes, jit), None)
            loss = ctc_loss(m(crop), [texts[i] for i in idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step(); step += 1
            hb(phase="crnn_train", epoch=ep + 1, epochs=a.epochs, step=step, total_steps=total, elapsed_s=int(time.time() - t0))
            la = 0.98 * la + 0.02 * loss.item() if k else loss.item()
        sv, _ = evaluate(m, *va); sr, _ = evaluate(m, *ref); srg, _ = evaluate(m, *ref, use="gt")
        hist.append(dict(epoch=ep + 1, loss=round(la, 4), val=sv, ref48=sr, ref48_gtbox=srg, elapsed_s=int(time.time() - t0)))
        print(f"[crnn] ep {ep+1}/{a.epochs} loss {la:.4f} | val exact {sv['exact']:.4f} ci {sv['exact_ci']:.4f} | "
              f"ref48 exact {sr['exact']:.4f} ci {sr['exact_ci']:.4f} gtbox {srg['exact']:.4f} | {time.time()-t0:.0f}s", flush=True)
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), hist=hist, step=step,
                        epoch=ep + 1, elapsed=time.time() - t0, rng=torch.get_rng_state()), ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        hb(force=True, phase="crnn_epoch", epoch=ep + 1, epochs=a.epochs, ref48=sr["exact"], val=sv["exact"], elapsed_s=int(time.time() - t0))
    m = m.cpu()
    os.makedirs(os.path.dirname(a.out), exist_ok=True); save_crnn(m, a.out)
    m2 = load_crnn(a.out)  # evaluate the fp16-stored weights actually shipped
    sv, _ = evaluate(m2, *va); sr, rows = evaluate(m2, *ref); srg, _ = evaluate(m2, *ref, use="gt")
    rep = dict(args=vars(a), params=sum(p.numel() for p in m.parameters()), history=hist, final_val=sv, final_ref48=sr,
               final_ref48_with_gt_boxes=srg, crop="text box (detect_boxes at eval), 40x200",
               ref48_misses=[r for r in rows if not r["exact"]], gate_ref48_exact_ge_0_98=sr["exact"] >= 0.98,
               train_seconds=int(time.time() - t0))
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("final_val", "final_ref48", "gate_ref48_exact_ge_0_98", "train_seconds")}, indent=1))
    for r in rep["ref48_misses"]: print("  miss:", r["gt"], "->", r["pred"])


if __name__ == "__main__":
    main()
