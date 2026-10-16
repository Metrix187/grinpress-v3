"""Hang-proof I/O + run-control helpers shared by every v3 GPU stage (ported from grinpress-v2 @49675d9).

* drive_retry      : each Drive FUSE attempt runs in a worker thread with a timeout; ENOENT/EIO/hangs are retried.
* atomic_save      : torch.save to tmp + os.replace (atomic on POSIX and Drive FUSE rename).
* save_sharded / load_sharded : big objects as <=95 MB parts + manifest (written LAST, atomically, with sha256) so
                     every Drive file stays under the 100 MB download cap; a half-written set is never "complete".
* Heartbeat        : status.json written from a daemon thread (skips if a previous write is still in flight).
* AsyncMirror      : copy local-SSD checkpoints to Drive in the background (sharded), keep the last N.
* OOM helpers      : is_oom / free_cuda; callers halve the (micro-)batch and persist it to a json.
"""
import glob, hashlib, io, json, os, shutil, threading, time
import torch

PART_BYTES = 95 * 1000 * 1000


def drive_retry(fn, what, tries=40, wait=15, attempt_timeout=120):
    for i in range(tries):
        box, err_box = [], []

        def _run():
            try:
                box.append(fn())
            except BaseException as e:  # noqa: BLE001
                err_box.append(e)

        th = threading.Thread(target=_run, daemon=True)
        th.start(); th.join(attempt_timeout)
        if th.is_alive():
            print(f"[io] {what} hung after {attempt_timeout}s; Drive FUSE stuck? retry {i+1}/{tries} in {wait}s", flush=True)
            if i == tries - 1:
                raise TimeoutError(f"{what} hung after {attempt_timeout}s x {tries}")
            time.sleep(wait); continue
        if err_box:
            err = err_box[0]
            if isinstance(err, RuntimeError) and "directory" not in str(err).lower() and "file" not in str(err).lower():
                raise err
            if not isinstance(err, (OSError, RuntimeError, TimeoutError)) or i == tries - 1:
                raise err
            print(f"[io] {what} failed ({err}); Drive hiccup? retry {i+1}/{tries} in {wait}s", flush=True)
            time.sleep(wait); continue
        return box[0] if box else None


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    # only re-create the leaf dir when its parent is visible (mount alive): a dead mount never silently becomes local disk
    if not os.path.isdir(parent) and os.path.isdir(os.path.dirname(parent)):
        os.makedirs(parent, exist_ok=True)


def atomic_save(obj, path):
    def _do():
        ensure_parent(path)
        torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)
    drive_retry(_do, f"save {os.path.basename(path)}")


def atomic_write_bytes(data, path):
    def _do():
        ensure_parent(path)
        with open(path + ".tmp", "wb") as f: f.write(data)
        os.replace(path + ".tmp", path)
    drive_retry(_do, f"write {os.path.basename(path)}")


def load_json(path, default):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(obj, path):
    atomic_write_bytes(json.dumps(obj, indent=1).encode(), path)


def save_sharded(obj, base):
    """Write obj as base.partNN (<=95 MB each) + base.manifest.json (last). Returns the part count."""
    buf = io.BytesIO(); torch.save(obj, buf); data = buf.getvalue()
    parts = [data[i:i + PART_BYTES] for i in range(0, len(data), PART_BYTES)] or [b""]
    for k, p in enumerate(parts):
        atomic_write_bytes(p, f"{base}.part{k:02d}")
    for stale in glob.glob(f"{base}.part*"):
        try:
            if int(stale.rsplit("part", 1)[1].split(".")[0]) >= len(parts): os.remove(stale)
        except (ValueError, OSError):
            pass
    save_json({"n_parts": len(parts), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "time": time.strftime("%F %T")}, f"{base}.manifest.json")
    return len(parts)


def sharded_exists(base):
    return os.path.exists(f"{base}.manifest.json")


def load_sharded(base, map_location="cpu"):
    man = drive_retry(lambda: json.load(open(f"{base}.manifest.json")), f"read {os.path.basename(base)} manifest")
    data = b"".join(drive_retry(lambda k=k: open(f"{base}.part{k:02d}", "rb").read(), f"read part {k}") for k in range(man["n_parts"]))
    if hashlib.sha256(data).hexdigest() != man["sha256"]:
        raise OSError(f"sha256 mismatch for {base}")
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)


def remove_sharded(base):
    for f in glob.glob(f"{base}.part*") + glob.glob(f"{base}.manifest.json"):
        try: os.remove(f)
        except OSError: pass


class Heartbeat:
    """status.json off the train thread; a wedged Drive FUSE must never stall training."""
    def __init__(self, path, every=60):
        self.path, self.every, self.last, self._writing = path, every, 0.0, False

    def __call__(self, force=False, **kw):
        if not self.path or (not force and time.time() - self.last < self.every) or self._writing:
            return
        self.last = time.time(); self._writing = True
        kw.update(time=time.strftime("%Y-%m-%d %H:%M:%S"), unix=int(time.time()))

        def _write():
            try:
                try:
                    ensure_parent(self.path)
                    with open(self.path + ".tmp", "w") as f: json.dump(kw, f)
                    os.replace(self.path + ".tmp", self.path)
                except OSError as err:
                    print(f"[hb] heartbeat write failed ({err}); will retry next beat", flush=True); self.last = 0.0
            finally:
                self._writing = False
        threading.Thread(target=_write, daemon=True).start()


class AsyncMirror:
    """Mirror local checkpoints to a Drive dir as sharded files in a daemon thread (one copy in flight; a request
    made while busy is queued as 'pending' and copied next, so the newest ckpt always reaches Drive).
    Call flush() before the process exits, otherwise the daemon thread dies with it."""
    def __init__(self, mirror_dir, keep=2):
        self.dir, self.keep, self.pending, self.th = mirror_dir, keep, None, None
        self.lock = threading.Lock()

    def __call__(self, local_path):
        if not self.dir:
            return
        with self.lock:
            self.pending = local_path
            if self.th is not None and self.th.is_alive():
                return
            self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()

    def _loop(self):
        while True:
            with self.lock:
                p, self.pending = self.pending, None
            if p is None:
                return
            self._copy(p)

    def _copy(self, local_path):
        name = os.path.basename(local_path)
        try:
            os.makedirs(self.dir, exist_ok=True)
            obj = torch.load(local_path, map_location="cpu", weights_only=False)
            save_sharded(obj, os.path.join(self.dir, name))
            bases = sorted({m[:-len(".manifest.json")] for m in glob.glob(os.path.join(self.dir, "ckpt_*.pt.manifest.json"))})
            for old in bases[:-self.keep]: remove_sharded(old)
            print(f"[mirror] {name} -> {self.dir}", flush=True)
        except Exception as err:  # noqa: BLE001
            print(f"[mirror] {name} failed ({err}); local ckpt is still safe", flush=True)

    def flush(self, timeout=1800):
        th = self.th
        if th is not None:
            th.join(timeout)
            if th.is_alive(): print("[mirror] flush timed out; Drive copy still in flight", flush=True)


def list_ckpts(d):
    return sorted(glob.glob(os.path.join(d, "ckpt_*.pt")), key=lambda p: int(os.path.basename(p)[5:-3]))


def latest_resume_state(local_dir, mirror_dir=""):
    """Newest of (local ckpt_*.pt, mirrored sharded ckpt). Returns (state_dict_obj or None, source)."""
    loc = list_ckpts(local_dir)
    best_step, best = -1, None
    if loc:
        best_step, best = int(os.path.basename(loc[-1])[5:-3]), ("local", loc[-1])
    if mirror_dir and os.path.isdir(mirror_dir):
        for m in glob.glob(os.path.join(mirror_dir, "ckpt_*.pt.manifest.json")):
            s = int(os.path.basename(m)[5:-len(".pt.manifest.json")])
            if s > best_step: best_step, best = s, ("mirror", m[:-len(".manifest.json")])
    if best is None:
        return None, None
    for kind, p in ([best] + ([("local", loc[-2])] if len(loc) > 1 else [])):
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False) if kind == "local" else load_sharded(p)
            return obj, p
        except Exception as err:  # noqa: BLE001 — a corrupt newest ckpt falls back to the previous one
            print(f"[resume] could not load {p} ({err}); trying older", flush=True)
    return None, None


_FAKE_OOM = {"left": int(os.environ.get("GP_FAKE_OOM", "0"))}


def fake_oom_hook(size):
    """Smoke-test hook: GP_FAKE_OOM=N raises a CUDA-OOM-like error on the first N calls with size > 1,
    so the halving/backoff paths can be exercised on a CPU box."""
    if _FAKE_OOM["left"] > 0 and size > 1:
        _FAKE_OOM["left"] -= 1
        raise RuntimeError("CUDA out of memory (simulated by GP_FAKE_OOM)")


def is_oom(e):
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def free_cuda():
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def amp_ctx(dev, precision):
    if dev.type != "cuda" or precision == "fp32":
        return torch.autocast("cpu", enabled=False), None
    dt = torch.bfloat16 if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()) else torch.float16
    return torch.autocast("cuda", dtype=dt), (torch.amp.GradScaler("cuda") if dt == torch.float16 else None)


def write_marker(path, **kw):
    try:
        save_json(dict(kw, finished=time.strftime("%F %T")), path)
    except Exception as err:  # noqa: BLE001
        print(f"[io] marker {path} failed ({err})", flush=True)
