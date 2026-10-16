"""Phase 3: fine-tune the v1 teacher into grinpress-v3-teacher.

  pregen : render N procedural v3 samples (generate_data_v2.render_v3 curriculum) into local .npy shards
           (images uint8, text bbox, layout, caption, text). Deterministic per (seed, index) -> safe to regenerate;
           finished shards are skipped, so the stage resumes.
  train  : warm-start from skyuu72/grinpress (v1) into the v3 model (max_len 128, zero-init quote/glyph embeddings),
           rectified-flow loss  x_t = (1-t) x0 + t z1,  v = z1 - x0,  L_rf = mean((v_hat - v)^2 * (1 + 4 M))  (M = text bbox mask)
           + 0.1 * CTC(frozen CRNN(GT-text-box crop of x0_hat)) for samples with t < 0.5 (x0_hat = x_t - t v_hat), conditional rows only.
           Validation every --val_every steps: EMA teacher, 50 Euler steps, CFG 5, the exact 48-string eval grid
           (noise seeds 1000+i), scored by the CRNN; early stop once exact > --stop_exact (0.95). Also logs a disjoint
           random 48-string grid so the number isn't only measured on the (selection) grid.
Robustness: local-SSD ckpts + async sharded Drive mirror (<100 MB files), resume from newest, status heartbeat,
micro-batch halving on CUDA OOM (persisted), DONE marker.
"""
import argparse, copy, glob, json, math, os, random, time
import numpy as np, torch, torch.nn.functional as F
from conditioning import tokenize
from model import FlowModel, load_teacher_weights, load_state
from gp_io import (worker_pool, Heartbeat, AsyncMirror, atomic_save, save_json, load_json, list_ckpts, latest_resume_state,
                   is_oom, free_cuda, amp_ctx, write_marker, fake_oom_hook)

RES = 128
LAYOUTS = ["left", "right", "below"]


# ------------------------------------------------------------------ model helpers (also used by eval / distill)
def load_flow(path, device="cpu"):
    m = FlowModel()
    load_teacher_weights(m, path)
    return m.to(device).eval().requires_grad_(False)


def encode_both(model, ids, mask):
    ctx_c, m_c = model.cond(ids, mask)          # call on an eval()-mode model (no random CFG dropout)
    ctx_u, m_u = model.cond(ids, mask, force_uncond=True)
    return ctx_c, m_c, ctx_u, m_u


@torch.no_grad()
def cfg_velocity(model, x, t, conds, w):
    ctx_c, m_c, ctx_u, m_u = conds
    v = model.unet(torch.cat([x, x]), torch.cat([t, t]), torch.cat([ctx_c, ctx_u]), torch.cat([m_c, m_u]))
    v_c, v_u = v.float().chunk(2)
    w = w.view(-1, 1, 1, 1) if torch.is_tensor(w) else w
    return v_u + w * (v_c - v_u)


@torch.no_grad()
def sample_cfg(model, caps, z, steps=50, cfg=5.0, amp=None):
    """Euler t: 1 -> 0 with CFG (same schedule as v1 sample.py / v2 eval)."""
    dev = z.device
    ids, mk = tokenize(caps); ids, mk = ids.to(dev), mk.to(dev)
    with (amp or torch.autocast("cpu", enabled=False)):
        conds = encode_both(model, ids, mk)
        x = z.clone(); ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
        for i in range(steps):
            tb = ts[i].expand(len(z))
            x = x - (ts[i] - ts[i + 1]) * cfg_velocity(model, x, tb, conds, cfg)
    return x.float().clamp(-1, 1)


# ------------------------------------------------------------------ pregen
_FONTS = None


def _fonts():
    global _FONTS
    if _FONTS is None:
        from generate_data import ensure_fonts
        _FONTS = sorted(ensure_fonts())
    return _FONTS


def _render_job(job):
    i, seed = job
    from generate_data_v2 import render_v3
    rng = random.Random(seed * 1_000_003 + i)
    img, meta, aux = render_v3(RES, rng, _fonts())
    return np.asarray(img, np.uint8), aux["text_bbox"], LAYOUTS.index(meta["layout"]), meta["caption"], meta["text"]


def cmd_pregen(a):
    os.makedirs(a.data_dir, exist_ok=True)
    _fonts()
    n_sh = math.ceil(a.num_samples / a.shard_size); hb = Heartbeat(a.heartbeat, 60); t0 = time.time()
    with worker_pool(a.workers) as pool:   # spawn + 1 thread/worker (no fork after OpenMP/CUDA init)
        for s in range(n_sh):
            base = os.path.join(a.data_dir, f"shard_{s:04d}")
            if os.path.exists(base + ".json"):
                continue
            lo, hi = s * a.shard_size, min(a.num_samples, (s + 1) * a.shard_size)
            out = pool.map(_render_job, [(i, a.seed) for i in range(lo, hi)], chunksize=32)
            np.save(base + ".tmp.npy", np.stack([o[0] for o in out])); os.replace(base + ".tmp.npy", base + ".npy")
            meta = dict(bbox=[list(o[1]) for o in out], layout=[o[2] for o in out], caption=[o[3] for o in out], text=[o[4] for o in out])
            with open(base + ".json.tmp", "w") as f: json.dump(meta, f)
            os.replace(base + ".json.tmp", base + ".json")       # json last = shard complete
            hb(phase="pregen", shard=s + 1, n_shards=n_sh, elapsed_s=int(time.time() - t0))
            print(f"[pregen] shard {s+1}/{n_sh} ({hi-lo} samples, {time.time()-t0:.0f}s)", flush=True)
    hb(force=True, phase="pregen_done", n_shards=n_sh)


class ShardData:
    """Memory-mapped pregen shards; batches are drawn uniformly (the curriculum lives in the generator)."""
    def __init__(self, data_dir):
        self.imgs, self.meta = [], dict(bbox=[], layout=[], caption=[], text=[])
        for j in sorted(glob.glob(os.path.join(data_dir, "shard_*.json"))):
            self.imgs.append(np.load(j[:-5] + ".npy", mmap_mode="r"))
            m = json.load(open(j))
            for k in self.meta: self.meta[k] += m[k]
        assert self.imgs, f"no shards in {data_dir}"
        self.offsets = np.cumsum([0] + [len(x) for x in self.imgs])
        self.n = int(self.offsets[-1])
        self.bbox = np.array(self.meta["bbox"], np.int64); self.layout = np.array(self.meta["layout"], np.int64)

    def batch(self, idx):
        imgs = []
        for i in idx:
            s = int(np.searchsorted(self.offsets, i, side="right") - 1)
            imgs.append(self.imgs[s][i - self.offsets[s]])
        x0 = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        M = torch.zeros(len(idx), 1, RES, RES)
        for k, i in enumerate(idx):
            x0b, y0b, x1b, y1b = self.bbox[i]; M[k, :, y0b:y1b, x0b:x1b] = 1
        return (x0, M, [self.meta["caption"][i] for i in idx], [self.meta["text"][i] for i in idx], torch.from_numpy(self.layout[idx]),
                torch.from_numpy(self.bbox[idx]).float())


class Prefetch:
    """Background thread assembling batches from the memmap shards so the GPU does not wait on numpy."""
    def __init__(self, ds, bs, seed, depth=4):
        import queue, threading
        self.q = queue.Queue(depth); self.ds, self.bs = ds, bs
        self.rng = np.random.default_rng(seed)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            self.q.put(self.ds.batch(self.rng.integers(0, self.ds.n, self.bs)))

    def __next__(self):
        return self.q.get()


# ------------------------------------------------------------------ validation
def val_grids(n_random=48, seed=4242, limit=0):
    from eval_grid import grid48
    from generate_data_v2 import sample_spec_v3, caption_v2
    g = grid48()
    rng = random.Random(seed); rnd = []
    for i in range(n_random):
        sp = sample_spec_v3(rng)
        rnd.append((sp, caption_v2(sp["color"], sp["expr"], sp["text"], sp["layout"], sp["paper"]), 5000 + i))
    if limit:
        g, rnd = g[:limit], rnd[:limit]
    return {"grid48": g, "random48": rnd}


@torch.no_grad()
def validate(model, crnn, grids, dev, steps, cfg, amp, bs=48):
    from crnn import recognize
    from eval_ocr import score_text
    out = {}
    for name, g in grids.items():
        caps = [c for _, c, _ in g]; seeds = [s for *_, s in g]; texts = [sp["text"] for sp, _, _ in g]
        xs = []
        for b in range(0, len(g), bs):
            z = torch.cat([torch.randn(1, 3, RES, RES, generator=torch.Generator().manual_seed(s)) for s in seeds[b:b + bs]]).to(dev)
            xs.append(sample_cfg(model, caps[b:b + bs], z, steps, cfg, amp))
        x = torch.cat(xs)
        preds = recognize(crnn, x, [sp["layout"] for sp, _, _ in g])
        sc = [score_text(p, t) for p, t in zip(preds, texts)]
        out[name] = dict(exact=sum(s["exact"] for s in sc) / len(sc), exact_ci=sum(s["exact_ci"] for s in sc) / len(sc),
                         char_acc_cs=float(np.mean([s["char_acc_cs"] for s in sc])), reads=preds)
    return out, x


# ------------------------------------------------------------------ train
def logit_normal(n, dev):
    return torch.sigmoid(torch.randn(n, device=dev)).clamp(1e-4, 1.0)


def cmd_train(a):
    from crnn import load_crnn, crop_text, ctc_loss
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    os.makedirs(a.checkpoint_dir, exist_ok=True); os.makedirs(a.out_dir, exist_ok=True)
    hb = Heartbeat(a.heartbeat, a.heartbeat_sec); mirror = AsyncMirror(a.checkpoint_mirror, keep=2)
    model = FlowModel(); load_teacher_weights(model, a.init)
    model.cond.p_uncond = 0.0                                   # CFG dropout is done here (we need to know which rows)
    model.to(dev).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    crnn = load_crnn(a.crnn, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    ac, scaler = amp_ctx(dev, a.precision)
    step, best, hist = 0, -1.0, []
    st, src = (latest_resume_state(a.checkpoint_dir, a.checkpoint_mirror) if a.resume else (None, None))
    if st is not None:
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        if scaler and st.get("scaler"): scaler.load_state_dict(st["scaler"])
        step, best, hist = st["step"], st.get("best", -1.0), st.get("hist", [])
        print(f"[teacher] resumed {src} at step {step} (best grid48 exact {best:.3f})", flush=True)
        del st
    hist_path = os.path.join(a.out_dir, "teacher_val_history.json")
    stop = any(h.get("early_stop") for h in hist)
    if stop:
        print("[teacher] early-stop already reached in a previous run; finalising", flush=True)
    ds = ShardData(a.data_dir); it = Prefetch(ds, a.batch_size, a.seed + step)
    grids = val_grids(limit=a.val_limit)
    mpath = os.path.join(a.checkpoint_dir, "micro.json")
    micro = min(a.batch_size, a.micro_batch or a.batch_size, load_json(mpath, {}).get("micro_batch", 10 ** 9))
    print(f"[teacher] {ds.n} samples, batch {a.batch_size} (micro {micro}), lr {a.lr}, max_steps {a.max_steps}", flush=True)

    def save_ckpt():
        p = os.path.join(a.checkpoint_dir, f"ckpt_{step:08d}.pt")
        atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict() if scaler else None, "step": step, "best": best, "hist": hist,
                     "args": vars(a)}, p)
        for old in list_ckpts(a.checkpoint_dir)[:-2]:
            try: os.remove(old)
            except OSError: pass
        mirror(p)

    def export_ema(name):
        from safetensors.torch import save_file
        sd = {k: v.detach().float().cpu().contiguous() for k, v in ema.state_dict().items()}
        loc = os.path.join(a.checkpoint_dir, name + ".safetensors"); save_file(sd, loc + ".tmp"); os.replace(loc + ".tmp", loc)
        fp16 = os.path.join(a.out_dir, name + "_fp16.safetensors")     # 68 MB: under the 100 MB Drive cap
        def _w():
            save_file({k: v.half() for k, v in sd.items()}, fp16 + ".tmp"); os.replace(fp16 + ".tmp", fp16)
        from gp_io import drive_retry
        try: drive_retry(_w, f"write {os.path.basename(fp16)}")
        except Exception as err: print(f"[teacher] fp16 export failed ({err})", flush=True)
        return loc

    t_start, last_save, la, step0 = time.time(), time.time(), None, step
    while step < a.max_steps and not stop:
        x0, M, caps, texts, lay, boxes = next(it)
        x0, M, lay, boxes = x0.to(dev, non_blocking=True), M.to(dev), lay.to(dev), boxes.to(dev)
        B = x0.shape[0]
        ids, mk = tokenize(caps); ids, mk = ids.to(dev), mk.to(dev)
        z1 = torch.randn_like(x0); t = logit_normal(B, dev); tb = t.view(-1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * z1; v = z1 - x0
        drop = torch.rand(B, device=dev) < a.p_uncond
        wmap = 1 + a.mask_weight * M
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup_steps)
        while True:
            try:
                opt.zero_grad(set_to_none=True); ls = {"rf": 0.0, "ctc": 0.0}; fake_oom_hook(micro)
                for i in range(0, B, micro):
                    sl = slice(i, i + micro); frac = min(micro, B - i) / B
                    with ac:
                        ctx, m = model.cond(ids[sl], mk[sl])
                        nx, nm = model.cond.null(ctx.shape[0], ctx.shape[1], ctx.dtype, ctx.device)
                        d = drop[sl]
                        ctx = torch.where(d[:, None, None], nx, ctx); m = torch.where(d[:, None], nm, m)
                        vh = model.unet(xt[sl], t[sl], ctx, m)
                    l_rf = ((vh.float() - v[sl]) ** 2 * wmap[sl]).mean()
                    loss = l_rf
                    sel = (t[sl] < a.ctc_t_max) & ~d
                    if a.ctc_weight > 0 and sel.any():
                        x0h = (xt[sl][sel] - t[sl][sel].view(-1, 1, 1, 1) * vh[sel].float()).clamp(-1, 1)
                        lp = crnn(crop_text(x0h, boxes[sl][sel]))        # crop the GT text box
                        l_ctc = ctc_loss(lp, [texts[j] for j in range(sl.start, min(sl.stop, B)) if sel[j - sl.start]])
                        loss = loss + a.ctc_weight * l_ctc * sel.float().mean()
                        ls["ctc"] += float(l_ctc.detach()) * frac
                    ls["rf"] += float(l_rf.detach()) * frac
                    (scaler.scale(loss * frac) if scaler else loss * frac).backward()
                break
            except Exception as err:
                if not is_oom(err) or micro <= 1: raise
                opt.zero_grad(set_to_none=True); free_cuda(); micro = max(1, micro // 2)
                save_json({"micro_batch": micro, "reason": "oom"}, mpath)
                print(f"[teacher] CUDA OOM -> micro-batch {micro}, retrying step {step}", flush=True)
        if scaler:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            dec = min(a.ema_decay, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), model.parameters()): pe.lerp_(p.detach(), 1 - dec)
        step += 1
        la = ls["rf"] if la is None else 0.98 * la + 0.02 * ls["rf"]
        if step % a.log_every == 0 or step == 1:
            print(f"[teacher] step {step}/{a.max_steps} rf {ls['rf']:.4f} (avg {la:.4f}) ctc {ls['ctc']:.3f} "
                  f"{(time.time()-t_start)/max(1, step - step0):.2f}s/it", flush=True)
        hb(phase="teacher_train", step=step, max_steps=a.max_steps, rf=round(la, 5), ctc=round(ls["ctc"], 4), best=best)
        if a.val_every and step % a.val_every == 0:
            tv = time.time()
            r, x = validate(ema, crnn, grids, dev, a.val_steps, a.cfg, ac)
            rec = dict(step=step, grid48=r["grid48"]["exact"], grid48_ci=r["grid48"]["exact_ci"], random48=r["random48"]["exact"],
                       random48_ci=r["random48"]["exact_ci"], reads48=r["grid48"]["reads"], val_s=round(time.time() - tv, 1))
            if rec["grid48"] > best:
                best = rec["grid48"]; export_ema("teacher_v3_best"); rec["new_best"] = True
            if rec["grid48"] > a.stop_exact:
                rec["early_stop"] = True; stop = True
            hist.append(rec)
            try: save_json(hist, hist_path)
            except Exception as err: print(f"[teacher] history write failed ({err})", flush=True)
            print(f"[teacher] VAL step {step}: grid48 exact {rec['grid48']:.3f} (ci {rec['grid48_ci']:.3f}) | random48 exact "
                  f"{rec['random48']:.3f} | best {best:.3f} | {rec['val_s']}s", flush=True)
            hb(force=True, phase="teacher_val", step=step, grid48=rec["grid48"], random48=rec["random48"], best=best)
        if time.time() - last_save > a.save_every_min * 60 or stop:
            save_ckpt(); last_save = time.time()
    if not os.path.exists(os.path.join(a.checkpoint_dir, f"ckpt_{step:08d}.pt")):
        save_ckpt()
    mirror.flush()
    final = export_ema("teacher_v3_last")
    best_local = os.path.join(a.checkpoint_dir, "teacher_v3_best.safetensors")
    chosen = best_local if os.path.exists(best_local) else final
    import shutil
    shutil.copy(chosen, os.path.join(a.checkpoint_dir, "teacher_v3.safetensors"))
    from safetensors.torch import load_file, save_file
    from gp_io import drive_retry
    fp16 = os.path.join(a.out_dir, "teacher_v3_fp16.safetensors")       # the teacher used by Phase 4 (68 MB)
    sd16 = {k: v.half() for k, v in load_file(chosen).items()}
    drive_retry(lambda: (save_file(sd16, fp16 + ".tmp"), os.replace(fp16 + ".tmp", fp16)), "write teacher_v3_fp16")
    write_marker(a.done_marker, step=step, best_grid48_exact=best, teacher=os.path.basename(chosen)) if a.done_marker else None
    hb(force=True, phase="teacher_done", step=step, best=best)
    print(f"[teacher] done at step {step}; best grid48 exact {best:.3f}; teacher -> {chosen}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = sub.add_parser("pregen")
    p.add_argument("--data_dir", default="teacher_data")
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--shard_size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--heartbeat", default="")
    t = sub.add_parser("train")
    t.add_argument("--init", default="weights/v1_teacher.safetensors")
    t.add_argument("--crnn", default="artifacts/crnn.safetensors")
    t.add_argument("--data_dir", default="teacher_data")
    t.add_argument("--checkpoint_dir", default="teacher_ckpt")
    t.add_argument("--checkpoint_mirror", default="")
    t.add_argument("--out_dir", default="teacher_out", help="small outputs (fp16 weights, val history); Drive in Colab")
    t.add_argument("--batch_size", type=int, default=128)
    t.add_argument("--micro_batch", type=int, default=0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup_steps", type=int, default=500)
    t.add_argument("--max_steps", type=int, default=40000)
    t.add_argument("--ema_decay", type=float, default=0.9995)
    t.add_argument("--p_uncond", type=float, default=0.1)
    t.add_argument("--mask_weight", type=float, default=4.0)
    t.add_argument("--ctc_weight", type=float, default=0.1)
    t.add_argument("--ctc_t_max", type=float, default=0.5)
    t.add_argument("--val_every", type=int, default=1000)
    t.add_argument("--val_steps", type=int, default=50)
    t.add_argument("--val_limit", type=int, default=0, help="smoke tests: only the first N strings of each val grid")
    t.add_argument("--cfg", type=float, default=5.0)
    t.add_argument("--stop_exact", type=float, default=0.95)
    t.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    t.add_argument("--save_every_min", type=float, default=10.0)
    t.add_argument("--heartbeat", default="")
    t.add_argument("--heartbeat_sec", type=float, default=60)
    t.add_argument("--done_marker", default="")
    t.add_argument("--log_every", type=int, default=100)
    t.add_argument("--resume", type=int, default=1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=dev)
    a = ap.parse_args()
    {"pregen": cmd_pregen, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
