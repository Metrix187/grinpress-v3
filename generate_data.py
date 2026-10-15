"""Synthetic 'vintage newspaper print' face+text dataset generator.

Each image: aged newsprint background, one halftone-filled face (smiley/sad/meh) on the
left, a short wood-type/serif text string on the right, distressed + ink-bled.
"""
import argparse, json, math, os, random, urllib.request
import numpy as np, cv2
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
FONT_URLS = [  # verified raw URLs (tried in order); at least one must succeed
    "https://raw.githubusercontent.com/google/fonts/main/ofl/oldstandardtt/OldStandard-Bold.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/abrilfatface/AbrilFatface-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/apache/ultra/Ultra-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/rye/Rye-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/oldstandardtt/OldStandard-Regular.ttf",
]

PALETTE = {
    "red": (200, 40, 40), "orange": (230, 120, 30), "yellow": (235, 195, 40),
    "green": (50, 150, 60), "teal": (30, 140, 140), "blue": (40, 80, 190),
    "navy": (25, 35, 110), "purple": (120, 50, 160), "pink": (230, 100, 160),
    "brown": (130, 80, 40), "gray": (120, 120, 120), "cyan": (40, 180, 220),
}
EXPRESSIONS = ["smiley", "sad", "meh"]
WORDS = ["NEWS", "HOT", "EXTRA", "WAR", "SALE", "HELLO", "OK", "YES", "NO", "WOW", "GOOD",
         "BAD", "MEH", "JOY", "FUN", "SAD", "DAILY", "TIMES", "POST", "FREE", "NEW", "BIG",
         "SOLD", "WIN", "LOVE", "HAPPY", "GLUM", "DEAL", "1923", "VOTE"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def ensure_fonts():
    """Download fonts with urllib if missing; return list of usable TTF paths."""
    os.makedirs(FONT_DIR, exist_ok=True)
    paths = []
    for url in FONT_URLS:
        p = os.path.join(FONT_DIR, url.rsplit("/", 1)[1])
        if not os.path.exists(p) or os.path.getsize(p) < 10000:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r, open(p + ".tmp", "wb") as f:
                    f.write(r.read())
                os.replace(p + ".tmp", p)
            except Exception as e:
                print(f"[font] failed {url}: {e}")
                continue
        try:
            ImageFont.truetype(p, 20)  # validate it parses
            paths.append(p)
        except Exception as e:
            print(f"[font] invalid {p}: {e}")
    if not paths:
        raise RuntimeError("Could not download any TTF font; check network access. "
                           "PIL default font is intentionally not used.")
    return paths


def random_text(rng):
    if rng.random() < 0.6:
        return rng.choice(WORDS)
    return "".join(rng.choice(LETTERS) for _ in range(rng.randint(1, 8)))


def newsprint(S, rng, fonts):
    """Aged paper: yellowed base, low-freq blotches, coarse fibers, faint background text."""
    nprng = np.random.default_rng(rng.randrange(2**31))
    base = np.array([228, 216, 186], np.float32) + nprng.uniform(-10, 10, 3)
    img = np.ones((S, S, 3), np.float32) * base
    low = cv2.resize(nprng.normal(0, 1, (8, 8)).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC)
    img += low[..., None] * np.array([6, 7, 12])            # uneven yellowing (more in blue channel)
    yy, xx = np.mgrid[0:S, 0:S] / S - 0.5
    img -= (xx**2 + yy**2)[..., None] * np.array([30, 40, 70])  # browned edges vignette
    img += nprng.normal(0, 4, (S, S, 1))                     # grain
    fib = np.zeros((S, S), np.uint8)
    for _ in range(int(S * S / 400)):                        # coarse fibers: short dark/light strokes
        x, y = nprng.integers(0, S, 2); a = nprng.uniform(0, math.pi); L = nprng.uniform(S/60, S/18)
        cv2.line(fib, (int(x), int(y)), (int(x + L*math.cos(a)), int(y + L*math.sin(a))),
                 int(nprng.integers(30, 90)), 1, cv2.LINE_AA)
    img -= cv2.GaussianBlur(fib, (0, 0), 0.6)[..., None].astype(np.float32) * 0.35
    # faint random background text (bleed-through from the other side)
    bt = Image.new("L", (S, S), 0); d = ImageDraw.Draw(bt)
    f = ImageFont.truetype(rng.choice(fonts), max(6, S // 22))
    y = -rng.randint(0, S // 20)
    while y < S:
        line = " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 7)))
                        for _ in range(12))
        d.text((-rng.randint(0, S // 8), y), line, fill=255, font=f)
        y += int(S / 18)
    bt = np.asarray(bt, np.float32) / 255
    img -= bt[..., None] * rng.uniform(14, 28)
    return img


def halftone(S, cx, cy, r, rng):
    """Halftone dot screen: 45° grid, dot radius ~ sqrt(ink coverage) with a shading gradient."""
    m = np.zeros((S, S), np.uint8)
    cell = S / rng.uniform(26, 34)
    ang = math.radians(45 + rng.uniform(-5, 5)); ca, sa = math.cos(ang), math.sin(ang)
    lx, ly = -0.5, -0.6  # light direction for shading
    n = int(S / cell) + 4
    for i in range(-n, n):
        for j in range(-n, n):
            u, v = i * cell, j * cell
            x, y = cx + u*ca - v*sa, cy + u*sa + v*ca
            dx, dy = (x - cx) / r, (y - cy) / r
            if dx*dx + dy*dy > 1.05: continue
            cov = np.clip(0.55 - 0.3*(dx*lx + dy*ly), 0.15, 0.9)      # coverage in [0,1]
            rad = cell * math.sqrt(cov / math.pi)                    # area ∝ coverage
            cv2.circle(m, (int(x*4), int(y*4)), max(1, int(rad*4)), 255, -1, cv2.LINE_AA, shift=2)
    circ = np.zeros_like(m); cv2.circle(circ, (int(cx), int(cy)), int(r), 255, -1, cv2.LINE_AA)
    return (m.astype(np.float32) / 255) * (circ.astype(np.float32) / 255)


def face_ink(S, cx, cy, r, expr, rng):
    m = np.zeros((S, S), np.uint8); th = max(2, int(r * 0.09))
    cv2.circle(m, (int(cx), int(cy)), int(r), 255, th, cv2.LINE_AA)
    er = max(2, int(r * 0.11))
    for sx in (-1, 1):
        cv2.circle(m, (int(cx + sx*r*0.35), int(cy - r*0.28)), er, 255, -1, cv2.LINE_AA)
    mw, my = r * 0.5, cy + r * 0.32
    pts = []
    for k in range(25):
        u = -1 + 2 * k / 24
        bend = {"smiley": 0.28, "sad": -0.28, "meh": 0.0}[expr]
        y = my + bend * r * (1 - u*u) - bend * r * 0.5           # parabola: up-curve smile, down frown
        pts.append((cx + u * mw, y + (0 if expr != "sad" else r * 0.12)))
    cv2.polylines(m, [np.round(np.array(pts) * 4).astype(np.int32)], False, 255, th, cv2.LINE_AA, shift=2)
    return m.astype(np.float32) / 255


def text_ink(S, text, font_path, x0, x1, cy, rng):
    img = Image.new("L", (S, S), 0); d = ImageDraw.Draw(img)
    size = int(S * 0.32)
    while size > 6:  # largest size fitting the right-hand region
        f = ImageFont.truetype(font_path, size)
        l, t, rr, b = d.textbbox((0, 0), text, font=f)
        if rr - l <= (x1 - x0) and b - t <= S * 0.4: break
        size -= 2
    w, h = rr - l, b - t
    d.text((x0 + (x1 - x0 - w) / 2 - l, cy - h / 2 - t), text, fill=255, font=f)
    return np.asarray(img, np.float32) / 255


def distress(a, rng, S):
    """Ink bleed (blur+soft-threshold spread) and distressed edges (multiplicative noise holes)."""
    nprng = np.random.default_rng(rng.randrange(2**31))
    a = cv2.GaussianBlur(a, (0, 0), S / 400 * rng.uniform(0.8, 1.5))
    a = np.clip(a * 1.25, 0, 1)                             # bleed: spreads ink slightly outward
    noise = cv2.resize(nprng.random((S // 6, S // 6)).astype(np.float32), (S, S), interpolation=cv2.INTER_LINEAR)
    speck = nprng.random((S, S)) > 0.04
    holes = np.clip((noise - 0.12) * 4, 0, 1) * speck       # worn patches + speckle dropouts
    return a * holes * rng.uniform(0.85, 1.0)


def render(res, rng, fonts):
    S = res * 2  # supersample then downsample for clean AA
    color_name = rng.choice(list(PALETTE)); rgb = np.array(PALETTE[color_name], np.float32)
    expr = rng.choice(EXPRESSIONS); text = random_text(rng)
    paper = newsprint(S, rng, fonts)
    cx, cy = S * rng.uniform(0.22, 0.27), S * rng.uniform(0.46, 0.54); r = S * rng.uniform(0.16, 0.19)
    col_a = distress(halftone(S, cx, cy, r * 0.97, rng), rng, S)
    blk_a = distress(np.maximum(face_ink(S, cx, cy, r, expr, rng),
                                text_ink(S, text, rng.choice(fonts), S*0.48, S*0.96, S*rng.uniform(0.46, 0.54), rng)), rng, S)
    # subtractive ink model: paper * (1 - a*(1 - ink/255))
    img = paper * (1 - col_a[..., None] * (1 - rgb / 255))
    img = img * (1 - blk_a[..., None] * (1 - np.array([25, 22, 20]) / 255))
    img = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (res, res), interpolation=cv2.INTER_AREA)
    meta = {"color": {"name": color_name, "rgb": [round(c / 255, 4) for c in PALETTE[color_name]]},
            "expression": expr, "text": text,
            "caption": f"a {color_name} {expr} face with '{text}' next to it"}
    return Image.fromarray(img), meta


def _work(job):
    i, seed, res, out, fonts = job
    rng = random.Random(seed * 1_000_003 + i)  # deterministic per-sample
    img, meta = render(res, rng, fonts)
    rel = f"images/{i:06d}.png"; img.save(os.path.join(out, rel))
    return {"image_path": rel, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--output_dir", default="data")
    ap.add_argument("--resolution", type=int, default=128, choices=[128, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    fonts = ensure_fonts(); print("fonts:", [os.path.basename(f) for f in fonts])
    os.makedirs(os.path.join(a.output_dir, "images"), exist_ok=True)
    jobs = [(i, a.seed, a.resolution, a.output_dir, fonts) for i in range(a.num_samples)]
    with open(os.path.join(a.output_dir, "metadata.jsonl"), "w") as fo, Pool(max(1, a.workers)) as pool:
        for k, rec in enumerate(pool.imap(_work, jobs, chunksize=16)):  # imap keeps order
            fo.write(json.dumps(rec) + "\n")
            if (k + 1) % 1000 == 0: print(f"{k+1}/{a.num_samples}", flush=True)
    print(f"wrote {a.num_samples} samples to {a.output_dir}")


if __name__ == "__main__":
    main()
