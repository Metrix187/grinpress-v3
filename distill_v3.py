"""Phase 4: v3 reflow pairs from the v3 teacher + 2-4 step student distillation.

  gen_pairs : z1 ~ N(0,I) (stored as per-sample seed), x0 = TeacherODE(z1) with 50 Euler steps + CFG 5, captions from
              the v3 curriculum (sample_spec_v3). Shards of --shard_size pairs (1000 -> ~49 MB, under the 100 MB Drive cap),
              each built from atomic chunk files (crash loses at most one chunk), batch halved on CUDA OOM and persisted.
              Per pair we also store the layout and a text bbox measured on the teacher image (dark-ink bbox inside the
              layout's text band; falls back to the band) -> text mask M.
  train     : student = teacher copy (+ zero-init guidance embedding, w=5 baked in). Per batch
                t ~ 60% {1.0, 0.75, 0.5, 0.25} / 40% logit-normal;  x_t = (1-t) x0 + t z1;  v = z1 - x0;  x0_hat = x_t - t v_hat
                L = mean((v_hat - v)^2 (1 + 4M))                           weighted reflow MSE
                  + l_sobel * mean(|Sobel(x0_hat) - Sobel(x0)| * M) / mean(M)  edge sharpness inside the text box
                  + l_ctc * CTC(frozen CRNN(band crop of x0_hat))          (rows with t <= --ctc_t_max)
              on --p_rollout (25%) of steps, also a 2-step rollout z1 -(t=1)-> x_0.5 -(t=0.5)-> x0_2 with gradients through
              both steps: + l_roll * mean((x0_2 - x0)^2 (1 + 4M)) + l_ctc * CTC(CRNN(x0_2)).
  sample    : few-step Euler with the student.
Robustness identical to train_teacher_v3 (local ckpts + async sharded Drive mirror, resume, heartbeat, OOM halving).
"""
import argparse, copy, glob, json, math, os, random, time
import numpy as np, torch, torch.nn.functional as F
from conditioning import tokenize
from model import StudentFlowModel, load_teacher_weights, load_state
from gp_io import (Heartbeat, AsyncMirror, atomic_save, save_json, load_json, list_ckpts, latest_resume_state,
                   is_oom, free_cuda, amp_ctx, write_marker, drive_retry)
from train_teacher_v3 import load_flow, encode_both, cfg_velocity, LAYOUTS, val_grids

RES = 128


def noise_from_seeds(seeds):
    out = torch.empty(len(seeds), 3, RES, RES)
    for i, s in enumerate(seeds):
        out[i] = torch.randn(3, RES, RES, generator=torch.Generator().manual_seed(int(s)))
    return out


def student_schedule(n):
    return [1.0 - i / n for i in range(n + 1)]


class TimeSampler:
    """p_discrete from the few-step grid points, else logit-normal(0,1)."""
    def __init__(self, p_discrete=0.6, points=(1.0, 0.75, 0.5, 0.25)):
        self.p, self.points = p_discrete, torch.tensor(points)

    def __call__(self, n, device):
        t = torch.sigmoid(torch.randn(n))
        d = torch.rand(n) < self.p
        t[d] = self.points[torch.randint(len(self.points), (int(d.sum()),))]
        return t.clamp(1e-3, 1.0).to(device)


@torch.no_grad()
def ink_bbox(x8, layout_ids, pad=2):
    """Text bbox on teacher outputs: dark pixels (luma < 0.55 x band median) inside the layout band.
    x8 uint8 [n,3,128,128]; returns int boxes [n,4] (x0,y0,x1,y1)."""
    from crnn import BANDS
    lum = (x8.float() * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1)
    out = []
    for i in range(len(x8)):
        bx0, by0, bx1, by1 = [int(v) for v in BANDS[int(layout_ids[i])].tolist()]
        reg = lum[i, by0:by1, bx0:bx1]
        ys, xs = torch.nonzero(reg < 0.55 * reg.median(), as_tuple=True)
        if len(xs) < 8:
            out.append([bx0, by0, bx1, by1]); continue
        q = lambda v, p: int(torch.quantile(v.float(), p))
        out.append([max(0, bx0 + q(xs, 0.01) - pad), max(0, by0 + q(ys, 0.01) - pad),
                    min(RES, bx0 + q(xs, 0.99) + 1 + pad), min(RES, by0 + q(ys, 0.99) + 1 + pad)])
    return torch.tensor(out, dtype=torch.int16)


# ------------------------------------------------------------------ gen_pairs
def shard_specs(seed, s, n):
    from generate_data_v2 import sample_spec_v3, caption_v2
    rng = random.Random(seed * 7919 + s); sp = [sample_spec_v3(rng) for _ in range(n)]
    caps = [caption_v2(p["color"], p["expr"], p["text"], p["layout"], p["paper"]) for p in sp]
    return sp, caps


def cmd_gen_pairs(a):
    dev = torch.device(a.device)
    teacher = load_flow(a.teacher, dev)
    os.makedirs(a.out_dir, exist_ok=True)
    hb = Heartbeat(a.heartbeat, 60); ac, _ = amp_ctx(dev, a.precision)
    n_shards = math.ceil(a.num_pairs / a.shard_size)
    bpath = os.path.join(a.out_dir, "gen_batch.json")
    batch = min(a.batch_size, load_json(bpath, {}).get("batch_size", a.batch_size))
    t0 = time.time()
    for s in range(n_shards):
        path = os.path.join(a.out_dir, f"pairs_{s:04d}.pt")
        if os.path.exists(path):
            continue
        n = min(a.shard_size, a.num_pairs - s * a.shard_size)
        specs, caps = shard_specs(a.seed, s, n)
        seeds = [a.seed * 10_000_000 + s * a.shard_size + i for i in range(n)]
        lay = torch.tensor([LAYOUTS.index(p["layout"]) for p in specs])
        pdir = os.path.join(a.out_dir, f"partial_{s:04d}")
        drive_retry(lambda: os.makedirs(pdir, exist_ok=True), f"mkdir {os.path.basename(pdir)}")
        chunks = {}
        for f in glob.glob(os.path.join(pdir, "chunk_*.pt")):
            b0, b1 = map(int, os.path.basename(f)[6:-3].split("_")); chunks[b0] = (b1, f)
        b = 0
        while b in chunks: b = chunks[b][0]
        if b: print(f"[gen] shard {s}: resuming at {b}/{n}", flush=True)
        while b < n:
            e = min(n, b + batch)
            try:
                z = noise_from_seeds(seeds[b:e]).to(dev)
                ids, mk = tokenize(caps[b:e]); ids, mk = ids.to(dev), mk.to(dev)
                with torch.inference_mode(), ac:
                    conds = encode_both(teacher, ids, mk)
                    x = z; ts = torch.linspace(1.0, 0.0, a.steps + 1, device=dev)
                    for i in range(a.steps):
                        x = x - (ts[i] - ts[i + 1]) * cfg_velocity(teacher, x, ts[i].expand(len(z)), conds, a.cfg)
                    x8 = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
                del z, x, conds
            except Exception as err:
                if not is_oom(err) or batch <= 1: raise
                free_cuda(); batch = max(1, batch // 2); save_json({"batch_size": batch, "reason": "oom"}, bpath)
                print(f"[gen] CUDA OOM -> batch {batch}, retrying chunk [{b},{e})", flush=True); continue
            cp = os.path.join(pdir, f"chunk_{b:06d}_{e:06d}.pt")
            atomic_save({"x0": x8}, cp); chunks[b] = (e, cp)
            if not os.path.exists(bpath): save_json({"batch_size": batch}, bpath)
            b = e
            hb(phase="gen_pairs", shard=s, n_shards=n_shards, done_in_shard=b, shard_size=n, batch=batch, elapsed_s=int(time.time() - t0))
            print(f"[gen] shard {s}: {b}/{n} (batch {batch}, {time.time()-t0:.0f}s)", flush=True)
        order, k = [], 0
        while k < n: order.append(chunks[k][1]); k = chunks[k][0]
        x0 = torch.cat([drive_retry(lambda f=f: torch.load(f, weights_only=True)["x0"], f"load {os.path.basename(f)}") for f in order])
        assert len(x0) == n
        atomic_save({"x0": x0, "seeds": torch.tensor(seeds), "captions": caps, "texts": [p["text"] for p in specs],
                     "layout": lay, "bbox": ink_bbox(x0, lay), "cfg": a.cfg, "steps": a.steps}, path)
        try:
            for f in glob.glob(os.path.join(pdir, "*")): os.remove(f)
            os.rmdir(pdir)
        except OSError as err:
            print(f"[gen] cleanup of {pdir} failed ({err}); ignoring", flush=True)
        print(f"[gen] wrote {path} ({n} pairs, {time.time()-t0:.0f}s)", flush=True)
    hb(force=True, phase="gen_pairs_done", n_shards=n_shards, batch=batch)


class PairData:
    def __init__(self, pair_dir):
        files = sorted(glob.glob(os.path.join(pair_dir, "pairs_*.pt"))); assert files, f"no pairs in {pair_dir}"
        x0, self.seeds, self.caps, self.texts, lay, bb = [], [], [], [], [], []
        for f in files:
            d = torch.load(f, weights_only=False)
            x0.append(d["x0"]); self.seeds += d["seeds"].tolist(); self.caps += d["captions"]; self.texts += d["texts"]
            lay.append(d["layout"]); bb.append(d["bbox"]); self.cfg = d["cfg"]
        self.x0, self.lay, self.bbox = torch.cat(x0), torch.cat(lay), torch.cat(bb).long()
        self.ids, self.mask = tokenize(self.caps)
        self.n = len(self.seeds)

    def batch(self, idx):
        x0 = self.x0[idx].float() / 127.5 - 1
        z1 = noise_from_seeds([self.seeds[i] for i in idx])
        M = torch.zeros(len(idx), 1, RES, RES)
        for k, i in enumerate(idx):
            x0b, y0b, x1b, y1b = self.bbox[i].tolist(); M[k, :, y0b:y1b, x0b:x1b] = 1
        return x0, z1, M, self.ids[idx], self.mask[idx], self.lay[idx], [self.texts[i] for i in idx]


class Prefetch:
    def __init__(self, ds, bs, seed, depth=4):
        import queue, threading
        self.q, self.ds, self.bs, self.rng = queue.Queue(depth), ds, bs, np.random.default_rng(seed)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            self.q.put(self.ds.batch(torch.from_numpy(self.rng.integers(0, self.ds.n, self.bs))))

    def __next__(self):
        return self.q.get()


def sobel(x):
    g = x.mean(1, keepdim=True)
    k = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), k.view(1, 1, 3, 3))
    gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), k.t().contiguous().view(1, 1, 3, 3))
    return torch.cat([gx, gy], 1)


@torch.no_grad()
def student_sample(student, caps, z, steps, w=None):
    dev = z.device
    ids, mk = tokenize(caps); ctx, m = student.cond(ids.to(dev), mk.to(dev))
    x = z.clone(); ts = student_schedule(steps)
    for i in range(steps):
        x = x - (ts[i] - ts[i + 1]) * student.forward_ctx(x, torch.full((len(z),), ts[i], device=dev), ctx, m, w=w).float()
    return x.clamp(-1, 1)


@torch.no_grad()
def validate_student(student, crnn, grid, dev, steps_list=(4, 2)):
    from crnn import recognize
    from eval_ocr import score_text
    caps = [c for _, c, _ in grid]; seeds = [s for *_, s in grid]; texts = [sp["text"] for sp, _, _ in grid]
    z = noise_from_seeds(seeds).to(dev); out = {}
    for n in steps_list:
        x = student_sample(student, caps, z, n)
        reads = recognize(crnn, x, [sp["layout"] for sp, _, _ in grid])
        sc = [score_text(p, t) for p, t in zip(reads, texts)]
        out[f"s{n}"] = dict(exact=sum(s["exact"] for s in sc) / len(sc), exact_ci=sum(s["exact_ci"] for s in sc) / len(sc))
    return out


def cmd_train(a):
    from crnn import load_crnn, crop_band, ctc_loss
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    os.makedirs(a.checkpoint_dir, exist_ok=True); os.makedirs(a.out_dir, exist_ok=True)
    hb = Heartbeat(a.heartbeat, a.heartbeat_sec); mirror = AsyncMirror(a.checkpoint_mirror, keep=2)
    student = StudentFlowModel(use_w_embed=True, w_default=a.w)
    load_teacher_weights(student, a.teacher)
    student.cond.p_uncond = 0.0
    student.to(dev).train()
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    crnn = load_crnn(a.crnn, dev)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    ac, scaler = amp_ctx(dev, a.precision)
    step, hist = 0, []
    st, src = (latest_resume_state(a.checkpoint_dir, a.checkpoint_mirror) if a.resume else (None, None))
    if st is not None:
        student.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        if scaler and st.get("scaler"): scaler.load_state_dict(st["scaler"])
        step, hist = st["step"], st.get("hist", []); print(f"[student] resumed {src} at step {step}", flush=True); del st
    ds = PairData(a.pair_dir); it = Prefetch(ds, a.batch_size, a.seed + step)
    tsamp = TimeSampler(a.p_discrete)
    grid = val_grids()["grid48"]
    mpath = os.path.join(a.checkpoint_dir, "micro.json")
    micro = min(a.batch_size, a.micro_batch or a.batch_size, load_json(mpath, {}).get("micro_batch", 10 ** 9))
    print(f"[student] {ds.n} pairs (cfg {ds.cfg}), batch {a.batch_size} (micro {micro}), max_steps {a.max_steps}", flush=True)
    rng = random.Random(a.seed + step)

    def save_ckpt():
        p = os.path.join(a.checkpoint_dir, f"ckpt_{step:08d}.pt")
        atomic_save({"model": student.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict() if scaler else None, "step": step, "hist": hist, "args": vars(a)}, p)
        for old in list_ckpts(a.checkpoint_dir)[:-2]:
            try: os.remove(old)
            except OSError: pass
        mirror(p)

    t_start, last_save, la, step0 = time.time(), time.time(), None, step
    while step < a.max_steps:
        x0, z1, M, ids, mk, lay, texts = next(it)
        x0, z1, M, ids, mk, lay = (v.to(dev, non_blocking=True) for v in (x0, z1, M, ids, mk, lay))
        B = x0.shape[0]
        t = tsamp(B, dev); tb = t.view(-1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * z1; v = z1 - x0; wmap = 1 + a.mask_weight * M
        rollout = rng.random() < a.p_rollout
        wv = torch.full((B,), float(ds.cfg), device=dev)
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup_steps)
        while True:
            try:
                opt.zero_grad(set_to_none=True); ls = dict(mse=0.0, sobel=0.0, ctc=0.0, roll=0.0)
                for i in range(0, B, micro):
                    sl = slice(i, i + micro); frac = min(micro, B - i) / B
                    with ac:
                        ctx, m = student.cond(ids[sl], mk[sl])
                        vh = student.forward_ctx(xt[sl], t[sl], ctx, m, w=wv[sl]).float()
                    l_mse = ((vh - v[sl]) ** 2 * wmap[sl]).mean()
                    x0h = xt[sl] - t[sl].view(-1, 1, 1, 1) * vh
                    Ms = M[sl]
                    l_sob = ((sobel(x0h) - sobel(x0[sl])).abs() * Ms).mean() / Ms.mean().clamp_min(1e-4)
                    loss = l_mse + a.l_sobel * l_sob
                    sel = t[sl] <= a.ctc_t_max
                    if a.l_ctc > 0 and sel.any():
                        lp = crnn(crop_band(x0h[sel].clamp(-1, 1), lay[sl][sel]))
                        l_ctc = ctc_loss(lp, [texts[j] for j in range(sl.start, min(sl.stop, B)) if sel[j - sl.start]])
                        loss = loss + a.l_ctc * l_ctc * sel.float().mean(); ls["ctc"] += float(l_ctc) * frac
                    if rollout:                                    # 2-step consistency 1.0 -> 0.5 -> 0.0
                        one = torch.ones(x0h.shape[0], device=dev)
                        with ac:
                            v1 = student.forward_ctx(z1[sl], one, ctx, m, w=wv[sl]).float()
                            xh = z1[sl] - 0.5 * v1
                            v2 = student.forward_ctx(xh, one * 0.5, ctx, m, w=wv[sl]).float()
                        x02 = xh - 0.5 * v2
                        l_roll = ((x02 - x0[sl]) ** 2 * wmap[sl]).mean()
                        loss = loss + a.l_roll * l_roll; ls["roll"] += float(l_roll) * frac
                        if a.l_ctc > 0:
                            loss = loss + a.l_ctc * ctc_loss(crnn(crop_band(x02.clamp(-1, 1), lay[sl])), texts[sl])
                    ls["mse"] += float(l_mse) * frac; ls["sobel"] += float(l_sob) * frac
                    (scaler.scale(loss * frac) if scaler else loss * frac).backward()
                break
            except Exception as err:
                if not is_oom(err) or micro <= 1: raise
                opt.zero_grad(set_to_none=True); free_cuda(); micro = max(1, micro // 2)
                save_json({"micro_batch": micro, "reason": "oom"}, mpath)
                print(f"[student] CUDA OOM -> micro-batch {micro}, retrying step {step}", flush=True)
        if scaler:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        with torch.no_grad():
            dec = min(a.ema_decay, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), student.parameters()): pe.lerp_(p.detach(), 1 - dec)
        step += 1
        la = ls["mse"] if la is None else 0.98 * la + 0.02 * ls["mse"]
        if step % a.log_every == 0 or step == 1:
            print(f"[student] step {step}/{a.max_steps} mse {ls['mse']:.4f} (avg {la:.4f}) sobel {ls['sobel']:.3f} "
                  f"ctc {ls['ctc']:.3f} roll {ls['roll']:.4f} {(time.time()-t_start)/max(1, step-step0):.2f}s/it", flush=True)
        hb(phase="student_train", step=step, max_steps=a.max_steps, mse=round(la, 5))
        if a.val_every and step % a.val_every == 0:
            r = validate_student(ema, crnn, grid, dev); r["step"] = step; hist.append(r)
            try: save_json(hist, os.path.join(a.out_dir, "student_val_history.json"))
            except Exception as err: print(f"[student] history write failed ({err})", flush=True)
            print(f"[student] VAL step {step}: 4-step exact {r['s4']['exact']:.3f} | 2-step exact {r['s2']['exact']:.3f}", flush=True)
        if time.time() - last_save > a.save_every_min * 60:
            save_ckpt(); last_save = time.time()
    save_ckpt()
    from safetensors.torch import save_file
    sd = {k: v.detach().float().cpu().contiguous() for k, v in ema.state_dict().items()}
    loc = os.path.join(a.checkpoint_dir, "student_ema.safetensors"); save_file(sd, loc + ".tmp"); os.replace(loc + ".tmp", loc)
    fp16 = os.path.join(a.out_dir, "student_ema_fp16.safetensors")
    drive_retry(lambda: (save_file({k: v.half() for k, v in sd.items()}, fp16 + ".tmp"), os.replace(fp16 + ".tmp", fp16)), "write student fp16")
    if a.done_marker: write_marker(a.done_marker, step=step)
    hb(force=True, phase="student_done", step=step)
    print(f"[student] done; EMA -> {loc} (+ fp16 {fp16})", flush=True)


def load_student(path, dev="cpu", w_default=5.0):
    s = StudentFlowModel(use_w_embed=True, w_default=w_default)
    sd = load_state(path)
    if not any(k.startswith("w_mlp") for k in sd): s.w_mlp = None
    load_teacher_weights(s, sd)
    return s.to(dev).eval().requires_grad_(False)


def cmd_sample(a):
    from PIL import Image
    s = load_student(a.checkpoint, a.device)
    z = noise_from_seeds([a.seed + i for i in range(a.num_images)]).to(a.device)
    x = student_sample(s, [a.caption] * a.num_images, z, a.steps)
    arr = ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    Image.fromarray(np.concatenate(list(arr), 1)).save(a.out); print("saved", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = sub.add_parser("gen_pairs")
    g.add_argument("--teacher", required=True)
    g.add_argument("--out_dir", default="pairs_v3")
    g.add_argument("--num_pairs", type=int, default=50000)
    g.add_argument("--shard_size", type=int, default=1000)
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--cfg", type=float, default=5.0)
    g.add_argument("--batch_size", type=int, default=128)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    g.add_argument("--heartbeat", default="")
    g.add_argument("--device", default=dev)
    t = sub.add_parser("train")
    t.add_argument("--teacher", required=True)
    t.add_argument("--crnn", default="artifacts/crnn.safetensors")
    t.add_argument("--pair_dir", default="pairs_v3")
    t.add_argument("--checkpoint_dir", default="student_ckpt")
    t.add_argument("--checkpoint_mirror", default="")
    t.add_argument("--out_dir", default="student_out")
    t.add_argument("--batch_size", type=int, default=128)
    t.add_argument("--micro_batch", type=int, default=0)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--warmup_steps", type=int, default=500)
    t.add_argument("--max_steps", type=int, default=20000)
    t.add_argument("--ema_decay", type=float, default=0.999)
    t.add_argument("--p_discrete", type=float, default=0.6)
    t.add_argument("--mask_weight", type=float, default=4.0)
    t.add_argument("--l_sobel", type=float, default=0.25)
    t.add_argument("--l_ctc", type=float, default=0.05)
    t.add_argument("--ctc_t_max", type=float, default=0.75)
    t.add_argument("--p_rollout", type=float, default=0.25)
    t.add_argument("--l_roll", type=float, default=1.0)
    t.add_argument("--w", type=float, default=5.0)
    t.add_argument("--val_every", type=int, default=2000)
    t.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    t.add_argument("--save_every_min", type=float, default=10.0)
    t.add_argument("--heartbeat", default="")
    t.add_argument("--heartbeat_sec", type=float, default=60)
    t.add_argument("--done_marker", default="")
    t.add_argument("--log_every", type=int, default=100)
    t.add_argument("--resume", type=int, default=1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=dev)
    s = sub.add_parser("sample")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--caption", default="a red smiley face with 'HELLO WORLD' next to it")
    s.add_argument("--steps", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--num_images", type=int, default=4)
    s.add_argument("--out", default="student.png")
    s.add_argument("--device", default=dev)
    a = ap.parse_args()
    {"gen_pairs": cmd_gen_pairs, "train": cmd_train, "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
