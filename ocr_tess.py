"""Tesseract OCR for grinpress renders: the v2 baseline and the improved v3 preprocessing.

v2 baseline (grinpress-v2/eval/eval_samples.py): fixed loose crop -> grey -> 4x Lanczos -> tesseract --psm 7.
v3 improved:
  1. crop the per-layout text band (same boxes as the CRNN, + small margin), upscale 4x (cubic);
  2. estimate the paper colour (median of the brightest 60% of pixels) and score every pixel by its ink-vs-paper
     colour distance (luminance drop + 0.5 x chroma distance), so coloured halftone dots / faint bleed-through
     text score low and near-black ink scores high;
  3. binarise with Otsu on that distance map (Sauvola as an alternative, --binarize sauvola);
  4. clean up: drop specks, drop components touching the face-side border (face outline), tighten to the ink bbox,
     rescale so the text is ~56 px tall, pad with white;
  5. tesseract LSTM with a CHARSET whitelist, --psm 7 and --psm 8; keep the read with the higher mean confidence.
"""
import numpy as np, cv2, pytesseract
from PIL import Image
from charset import CHARSET
from crnn import BANDS, LAYOUT_ID

WHITELIST = CHARSET.replace(" ", "")


def tess_v2(img, layout):
    W = img.width
    box = {"left": (0.44, 0.25, 1.0, 0.75), "right": (0.0, 0.25, 0.56, 0.75), "below": (0.0, 0.58, 1.0, 0.95)}[layout]
    c = img.crop(tuple(int(v * W) for v in box)).convert("L")
    c = c.resize((c.width * 4, c.height * 4), Image.LANCZOS)
    return pytesseract.image_to_string(c, config="--psm 7").strip()


def _sauvola(g, win=31, k=0.2, R=128.0):
    g = g.astype(np.float64)
    m = cv2.boxFilter(g, -1, (win, win)); m2 = cv2.boxFilter(g * g, -1, (win, win))
    s = np.sqrt(np.maximum(m2 - m * m, 0))
    return g > m * (1 + k * (s / R - 1))


def binarize(img, layout, method="otsu", margin=2, up=4):
    a = np.asarray(img.convert("RGB"), np.float32)
    x0, y0, x1, y1 = [int(v) for v in BANDS[LAYOUT_ID[layout]].tolist()]
    x0, y0, x1, y1 = max(0, x0 - margin), max(0, y0 - margin), min(a.shape[1], x1 + margin), min(a.shape[0], y1 + margin)
    c = cv2.resize(a[y0:y1, x0:x1], None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    lum = c @ np.array([0.299, 0.587, 0.114], np.float32)
    bright = lum >= np.percentile(lum, 40)
    paper = np.median(c[bright], 0); plum = float(paper @ np.array([0.299, 0.587, 0.114], np.float32))
    chroma = np.linalg.norm((c - lum[..., None]) - (paper - plum), axis=-1)
    d = np.clip(plum - lum, 0, None) + 0.5 * chroma           # ink-vs-paper colour distance
    d8 = np.clip(d / max(d.max(), 1e-3) * 255, 0, 255).astype(np.uint8)
    if method == "sauvola":
        ink = ~_sauvola(255 - d8) & (d8 > 40)
    else:
        _, b = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); ink = b > 0
    ink = ink.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    H, W = ink.shape
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 6 * up * up // 4: continue                      # specks
        if layout == "left" and x == 0 and w < W * 0.25: continue  # face outline sliver at the left edge
        if layout == "right" and x + w >= W and w < W * 0.25: continue
        if layout == "below" and y == 0 and h < H * 0.5: continue
        keep[i] = True
    ink = keep[lab]
    if not ink.any():
        return None
    ys, xs = np.nonzero(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = 56.0 / max(crop.shape[0], 1)
    crop = cv2.resize(crop.astype(np.uint8) * 255, (max(1, int(crop.shape[1] * s)), 56), interpolation=cv2.INTER_AREA)
    out = 255 - cv2.copyMakeBorder(crop, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=0)
    return Image.fromarray(out)


def _read(b, psm):
    cfg = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST} -c preserve_interword_spaces=1"
    d = pytesseract.image_to_data(b, config=cfg, output_type=pytesseract.Output.DICT)
    words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if str(w).strip() and float(c) >= 0]
    if not words:
        return "", -1.0
    return " ".join(w for w, _ in words), float(np.mean([c for _, c in words]))


def tess_v3(img, layout, method="otsu"):
    b = binarize(img, layout, method)
    if b is None:
        return ""
    best = max((_read(b, p) for p in (7, 8)), key=lambda r: r[1])
    return best[0].strip()
