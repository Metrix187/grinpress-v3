"""grinpress v2/v3 dataset: 7 expressions, 3 layouts, 3 papers, text up to 14 chars.

v3 additions (render_v2 output for a given rng/seed is UNCHANGED, so the v2 eval reference renders are identical):
  * render_v2(..., aux=True) also returns {"text_mask": float32 [res,res] text-bbox mask, "text_bbox": (x0,y0,x1,y1)
    in output pixels, "target_ids": CRNN/CTC ids of the text (charset.py)}; optional ink_rgb for ink variety.
  * random_text_v3 / sample_spec_v3: the v3 curriculum (40% 9-14 chars, 30% mixed case + punctuation/emoticons,
    uniform over 7 expressions x 3 layouts x 3 papers). The 48+14 eval strings (eval_grid.py) and every word in
    them are held out of the v3 samplers.

Captions are backward compatible with v1: the base sentence is exactly the v1 template and the new
options are OPTIONAL suffixes that are omitted at their v1 defaults (face left, aged paper):
  a {color} {expression} face with '{text}' next to it[, face on the right | , text below][, on fresh paper | , on sepia paper]
The longest caption is 88 chars (> 64), so v2 conditioning uses max_len 96 (see V2_MAX_LEN / README).
"""
import argparse, json, math, os, random
from multiprocessing import Pool
import numpy as np, cv2
from PIL import Image
from generate_data import PALETTE, ensure_fonts, newsprint, halftone, text_ink, distress, WORDS, LETTERS

EXPRESSIONS_V2 = ["smiley", "sad", "meh", "wink", "angry", "dead", "uwu"]
LAYOUTS = {"left": "", "right": ", face on the right", "below": ", text below"}
PAPERS = {"aged": "", "fresh": ", on fresh paper", "sepia": ", on sepia paper"}
MAX_TEXT = 14
V2_MAX_LEN = 96


def caption_v2(color, expr, text, layout="left", paper="aged"):
    return f"a {color} {expr} face with '{text}' next to it{LAYOUTS[layout]}{PAPERS[paper]}"


def longest_caption():
    c = max(PALETTE, key=len); e = max(EXPRESSIONS_V2, key=len)
    return caption_v2(c, e, "W" * MAX_TEXT, max(LAYOUTS, key=lambda k: len(LAYOUTS[k])), max(PAPERS, key=lambda k: len(PAPERS[k])))


def random_text_v2(rng):
    r = rng.random()
    if r < 0.45:
        return rng.choice(WORDS)
    if r < 0.65:  # two words, still <= 14
        t = f"{rng.choice(WORDS)} {rng.choice(WORDS)}"
        return t if len(t) <= MAX_TEXT else rng.choice(WORDS)
    return "".join(rng.choice(LETTERS) for _ in range(rng.randint(1, MAX_TEXT)))


def _poly(m, pts, th):
    cv2.polylines(m, [np.round(np.array(pts) * 4).astype(np.int32)], False, 255, th, cv2.LINE_AA, shift=2)


def face_ink_v2(S, cx, cy, r, expr):
    """Outline + eyes + mouth for all 7 expressions (ink mask in [0,1])."""
    m = np.zeros((S, S), np.uint8); th = max(2, int(r * 0.09)); er = max(2, int(r * 0.11))
    cv2.circle(m, (int(cx), int(cy)), int(r), 255, th, cv2.LINE_AA)
    ex, ey = r * 0.35, cy - r * 0.28
    arc = lambda x0, y0, w, bend, n=21: [(x0 + w * (-1 + 2 * k / (n - 1)), y0 + bend * (1 - (-1 + 2 * k / (n - 1)) ** 2)) for k in range(n)]
    # eyes
    for sx in (-1, 1):
        x = cx + sx * ex
        if expr == "wink" and sx == 1:
            _poly(m, arc(x, ey + r * 0.05, er * 1.4, -r * 0.08), th)            # closed eye: ^ arc
        elif expr == "dead":
            d = er * 1.2
            cv2.line(m, (int(x - d), int(ey - d)), (int(x + d), int(ey + d)), 255, th, cv2.LINE_AA)
            cv2.line(m, (int(x - d), int(ey + d)), (int(x + d), int(ey - d)), 255, th, cv2.LINE_AA)
        elif expr == "uwu":
            _poly(m, arc(x, ey - r * 0.05, er * 1.4, r * 0.12), th)             # 'u' shaped closed eyes
        else:
            cv2.circle(m, (int(x), int(ey)), er, 255, -1, cv2.LINE_AA)
        if expr == "angry":                                                     # brows slanting down to the centre
            cv2.line(m, (int(x + sx * er * 1.8), int(ey - er * 2.6)), (int(x - sx * er * 1.6), int(ey - er * 1.4)), 255, th, cv2.LINE_AA)
    # mouth
    mw, my = r * 0.5, cy + r * 0.32
    if expr in ("smiley", "wink"):
        _poly(m, arc(cx, my - r * 0.14, mw, r * 0.28), th)
    elif expr in ("sad", "angry"):
        _poly(m, arc(cx, my + r * 0.26, mw * (0.8 if expr == "angry" else 1), -r * 0.28), th)
    elif expr == "meh":
        _poly(m, [(cx - mw, my), (cx + mw, my)], th)
    elif expr == "dead":                                                        # wobbly line
        _poly(m, [(cx - mw + 2 * mw * k / 30, my + r * 0.05 * math.sin(k / 30 * 4 * math.pi)) for k in range(31)], th)
    elif expr == "uwu":                                                         # 'w' mouth
        w = mw * 0.55
        _poly(m, arc(cx - w / 2, my, w / 2, r * 0.12) + arc(cx + w / 2, my, w / 2, r * 0.12), th)
    return m.astype(np.float32) / 255


def paper_v2(S, rng, fonts, kind):
    p = newsprint(S, rng, fonts)
    if kind == "fresh":     # brighter, less yellow: pull toward off-white
        p = 0.45 * p + 0.55 * np.array([246, 244, 236], np.float32)
    elif kind == "sepia":   # luminance -> sepia tone
        lum = p @ np.array([0.299, 0.587, 0.114], np.float32)
        p = np.clip(lum[..., None] * np.array([1.07, 0.88, 0.66], np.float32), 0, 255)
    return p


DEFAULT_INK = (25, 22, 20)


def render_v2(res, rng, fonts, color=None, expr=None, text=None, layout=None, paper=None, aux=False, ink_rgb=None):
    S = res * 2
    color = color or rng.choice(list(PALETTE)); rgb = np.array(PALETTE[color], np.float32)
    expr = expr or rng.choice(EXPRESSIONS_V2); text = text or random_text_v2(rng)
    layout = layout or rng.choices(list(LAYOUTS), weights=[0.5, 0.25, 0.25])[0]
    paper = paper or rng.choices(list(PAPERS), weights=[0.5, 0.25, 0.25])[0]
    bg = paper_v2(S, rng, fonts, paper)
    font = rng.choice(fonts)
    if layout == "below":
        r = S * rng.uniform(0.17, 0.2); cx, cy = S * rng.uniform(0.47, 0.53), S * rng.uniform(0.3, 0.34)
        tx0, tx1, tcy = S * 0.05, S * 0.95, S * rng.uniform(0.74, 0.78)
    else:
        r = S * rng.uniform(0.15, 0.18); cy = S * rng.uniform(0.46, 0.54)
        if layout == "left":
            cx = S * rng.uniform(0.2, 0.24); tx0, tx1 = S * 0.43, S * 0.97
        else:
            cx = S * rng.uniform(0.76, 0.8); tx0, tx1 = S * 0.03, S * 0.57
        tcy = S * rng.uniform(0.46, 0.54)
    col_a = distress(halftone(S, cx, cy, r * 0.97, rng), rng, S)
    t_ink = text_ink(S, text, font, tx0, tx1, tcy, rng)        # text_ink draws no random numbers
    blk_a = distress(np.maximum(face_ink_v2(S, cx, cy, r, expr), t_ink), rng, S)
    ink = np.array(ink_rgb if ink_rgb is not None else DEFAULT_INK)  # int array like v2 (float64 math, identical pixels)
    img = bg * (1 - col_a[..., None] * (1 - rgb / 255))
    img = img * (1 - blk_a[..., None] * (1 - ink / 255))
    img = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (res, res), interpolation=cv2.INTER_AREA)
    meta = {"color": {"name": color, "rgb": [round(c / 255, 4) for c in PALETTE[color]]}, "expression": expr,
            "text": text, "layout": layout, "paper": paper, "caption": caption_v2(color, expr, text, layout, paper)}
    if not aux:
        return Image.fromarray(img), meta
    return Image.fromarray(img), meta, text_aux(t_ink, res, text)


def text_aux(t_ink, res, text, pad=1):
    """Text-bbox mask at output resolution from the supersampled text ink, plus CTC target ids."""
    from charset import encode
    S = t_ink.shape[0]; sc = res / S
    ys, xs = np.nonzero(t_ink > 0.1)
    if len(xs) == 0:
        box = (0, 0, 0, 0)
    else:
        box = (max(0, int(np.floor(xs.min() * sc)) - pad), max(0, int(np.floor(ys.min() * sc)) - pad),
               min(res, int(np.ceil((xs.max() + 1) * sc)) + pad), min(res, int(np.ceil((ys.max() + 1) * sc)) + pad))
    m = np.zeros((res, res), np.float32); m[box[1]:box[3], box[0]:box[2]] = 1.0
    return {"text_mask": m, "text_bbox": box, "target_ids": encode(text)}


# ------------------------------------------------------------------ v3 text curriculum
_WORDS_V3 = """news extra daily times post press paper print ink sale deal free sold vote win love happy glum joy fun
good bad big new hot cold war peace yes nope maybe okay sure cool nice great super mega ultra tiny small large huge
dog dogs puppy pup pups doggo pupper hound kitty cat cats meow purr bunny fox wolf bear duck frog bee owl
boop bonk zoom nap naps treat treats snack snacks bone bones ball fetch walk walks sit stay paw paws tail tails
fluffy floof fuzzy sleepy hungry grumpy silly goofy derp smol chonk heck frick yay yippee hooray huzzah oops
hello howdy hiya bye ciao aloha thanks please sorry wait stop go run jump play read write call text mail email
morning evening night today tomorrow monday friday weekend summer winter spring autumn holiday party birthday
coffee tea cake pizza taco burger donut cookie apple banana cherry lemon mango peach berry candy sugar honey
city town street park beach river ocean forest garden house home office school store market bank cafe club
king queen boss chief captain hero star legend champ rookie team crew squad gang family friend buddy pal
money cash price cost rent bill tax gold silver stock trade market crash boom rally record update alert flash
weather sunny rainy storm snow windy foggy cloudy heat wave report story headline special edition issue volume
magnificent spectacular independence headquarters celebration announcement adventure wonderful fantastic
incredible remarkable delightful marvelous tremendous quarterly breakthrough everything something anywhere
programming computer internet keyboard telephone television radio camera rocket engine planet galaxy universe
""".split()


def _held_out():
    from eval_grid import HELD_OUT
    words = {w.lower() for s in HELD_OUT for w in s.split()}
    return HELD_OUT, words


def _word_pool():
    _, bad = _held_out()
    pool = [w for w in _WORDS_V3 if w.lower() not in bad]
    pool += [w.lower() for w in WORDS if w.lower() not in bad and w.lower() not in pool]
    return pool


EMOTICONS = [":)", ":(", ":D", ":P", ";)", ":O", "xD", "XD", "<3", "^_^", "^^", ">_<", "T_T", "-_-", "o_O", "O_o",
             "=)", ":-)", ":-(", "8)", "(:", "):", ":*", ">:(", "B)", "^w^", "~_~", "*_*", "@_@", ">.<", "o.o", ":v"]


def _case(rng, w, mode=None):
    mode = mode or rng.choice(["upper", "lower", "title", "random"])
    if mode == "upper": return w.upper()
    if mode == "lower": return w.lower()
    if mode == "title": return w[:1].upper() + w[1:].lower()
    return "".join(c.upper() if rng.random() < 0.4 else c.lower() for c in w)


def _rand_chars(rng, n, alphabet):
    s = "".join(rng.choice(alphabet) for _ in range(n))
    return s


def _clean(t):
    t = " ".join(t.split())
    return t[:MAX_TEXT].strip()


def _long_text(rng, pool):
    r = rng.random()
    if r < 0.45:                                  # 2-3 words, case-consistent (BREAKING NEWS-like)
        mode = rng.choices(["upper", "lower", "title", "random"], weights=[0.45, 0.3, 0.15, 0.1])[0]
        ws = [_case(rng, rng.choice(pool), mode) for _ in range(rng.randint(2, 3))]
        t = " ".join(ws)
        if rng.random() < 0.25: t += rng.choice(["!", "!!", "?", "?!", "...", "~", "."])
    elif r < 0.65:                                # one long word
        longw = [w for w in pool if len(w) >= 9] or pool
        t = _case(rng, rng.choice(longw))
    elif r < 0.85:                                # random chars
        alpha = rng.choice([UPPER_ALNUM, LOWER_ALNUM, MIXED_ALNUM, MIXED_ALNUM + "!?.-:%$#~ "])
        t = _rand_chars(rng, rng.randint(9, MAX_TEXT), alpha)
    else:                                         # number/word combos
        t = rng.choice([f"{rng.choice(pool).upper()} {rng.randint(0, 9999)}", f"{rng.randint(1, 99)} {rng.choice(pool)}s",
                        f"${rng.randint(1, 999)} {rng.choice(pool)}", f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d} {rng.choice(pool)}"])
    t = _clean(t)
    if len(t) < 9:
        t = _clean(t + " " + _case(rng, rng.choice(pool), "upper" if t.isupper() else "lower"))
    return t


def _mixed_text(rng, pool):
    r = rng.random()
    if r < 0.2: return rng.choice(EMOTICONS)
    if r < 0.4:                                   # random case word (HeLLo-like) / camel case (GoodGirl-like)
        if rng.random() < 0.5: return _clean(_case(rng, rng.choice(pool), "random"))
        return _clean("".join(_case(rng, rng.choice(pool), "title") for _ in range(2)))
    if r < 0.5:                                   # brand-ish (iPhone/McDonald-like)
        w = rng.choice(pool)
        return _clean(rng.choice(["i", "e", "Mc", "Mac", "O'"[:1]]) + w[:1].upper() + w[1:].lower())
    if r < 0.75:                                  # word + punctuation
        w = _case(rng, rng.choice(pool))
        return _clean(rng.choice([w + "!", w + "?", w + "?!", w + "!!", w + "...", w + "~", w + ".", "#" + w, "@" + w,
                                  w + " :)", w + "-" + _case(rng, rng.choice(pool))]))
    # numbers + symbols ($5, 50%, 12:30, 3.14, #1-like)
    n = rng.randint(0, 999)
    return _clean(rng.choice([f"${n}", f"{n}%", f"#{n}", f"{rng.randint(0, 23)}:{rng.randint(0, 59):02d}",
                              f"{rng.randint(0, 99)}.{rng.randint(0, 99)}", f"{n}+", f"~{n}", f"{n}/{rng.randint(1, 99)}",
                              f"<{n}>", f"{n}!", f"{rng.randint(1, 9)}+{rng.randint(1, 9)}={rng.randint(2, 18)}",
                              f"x{n}", f"-{n}", f"{n}st", f"({n})", f"A{n}", f"${n}.{rng.randint(0, 99):02d}"]))


def _base_text(rng, pool):
    r = rng.random()
    if r < 0.35: return _clean(rng.choice(pool).upper())
    if r < 0.6: return _clean(_case(rng, rng.choice(pool), rng.choice(["lower", "title"])))
    if r < 0.8: return _rand_chars(rng, rng.randint(1, 8), rng.choice([UPPER_ALNUM, LOWER_ALNUM, MIXED_ALNUM]))
    if r < 0.9: return str(rng.randint(0, 10 ** rng.randint(1, 4) - 1)).zfill(rng.choice([1, 1, 2, 3]))
    return _rand_chars(rng, rng.randint(1, 4), PUNCT_RICH)


UPPER_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LOWER_ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"
MIXED_ALNUM = UPPER_ALNUM + "abcdefghijklmnopqrstuvwxyz"
PUNCT_RICH = "!?.,:;%$#&@*+-=~<>/()^_" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_POOL = None


def random_text_v3(rng, kind=None):
    """kind in {long, mixed, base}; default draws the v3 curriculum mix 40/30/30. Never returns a held-out eval string."""
    global _POOL
    if _POOL is None: _POOL = _word_pool()
    held, _ = _held_out()
    from charset import CHARSET
    kind = kind or rng.choices(["long", "mixed", "base"], weights=[0.4, 0.3, 0.3])[0]
    for _ in range(50):
        t = {"long": _long_text, "mixed": _mixed_text, "base": _base_text}[kind](rng, _POOL)
        t = "".join(c for c in t if c in CHARSET).strip()
        if t and len(t) <= MAX_TEXT and "  " not in t and t not in held:
            return t
    return "PUP"


def sample_spec_v3(rng, text_kind=None):
    """Uniform over 12 colors x 7 expressions x 3 layouts x 3 papers; text from the v3 curriculum."""
    return dict(color=rng.choice(list(PALETTE)), expr=rng.choice(EXPRESSIONS_V2), text=random_text_v3(rng, text_kind),
                layout=rng.choice(list(LAYOUTS)), paper=rng.choice(list(PAPERS)))


INK_VARIANTS = [(25, 22, 20), (10, 10, 12), (40, 30, 25), (30, 28, 45), (55, 35, 25), (20, 30, 30)]


def render_v3(res, rng, fonts, spec=None, ink_variety=False):
    """v3 training sample: (PIL image, meta, aux) for a curriculum spec (drawn if None)."""
    spec = spec or sample_spec_v3(rng)
    ink = rng.choice(INK_VARIANTS) if ink_variety and rng.random() < 0.5 else None
    return render_v2(res, rng, fonts, aux=True, ink_rgb=ink, **spec)


def _work(job):
    i, seed, res, out, fonts = job
    img, meta = render_v2(res, random.Random(seed * 1_000_003 + i), fonts)
    rel = f"images/{i:06d}.png"; img.save(os.path.join(out, rel))
    return {"image_path": rel, **meta}


def preview(path, res, fonts):
    """Row per expression (7) x columns: 3 layouts on aged paper + fresh + sepia."""
    rng = random.Random(0); cols = [("left", "aged"), ("right", "aged"), ("below", "aged"), ("left", "fresh"), ("left", "sepia")]
    colors = list(PALETTE)
    grid = Image.new("RGB", (res * len(cols), res * len(EXPRESSIONS_V2)))
    for r_, e in enumerate(EXPRESSIONS_V2):
        for c_, (lay, pap) in enumerate(cols):
            img, _ = render_v2(res, rng, fonts, color=colors[(r_ * 5 + c_) % 12], expr=e,
                               text=["WOOF", "BARK BARK", "HI", "GOOD GIRL", "UWU"][c_], layout=lay, paper=pap)
            grid.paste(img, (c_ * res, r_ * res))
    grid.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--output_dir", default="data_v2")
    ap.add_argument("--resolution", type=int, default=128, choices=[128, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--preview", default="", help="only render a preview grid to this path")
    a = ap.parse_args()
    fonts = ensure_fonts()
    print(f"longest caption ({len(longest_caption())} chars): {longest_caption()}")
    if a.preview:
        preview(a.preview, a.resolution, fonts); print("saved", a.preview); return
    os.makedirs(os.path.join(a.output_dir, "images"), exist_ok=True)
    jobs = [(i, a.seed, a.resolution, a.output_dir, fonts) for i in range(a.num_samples)]
    with open(os.path.join(a.output_dir, "metadata.jsonl"), "w") as fo, Pool(max(1, a.workers)) as pool:
        for k, rec in enumerate(pool.imap(_work, jobs, chunksize=16)):
            fo.write(json.dumps(rec) + "\n")
    print(f"wrote {a.num_samples} samples to {a.output_dir}")


if __name__ == "__main__":
    main()
