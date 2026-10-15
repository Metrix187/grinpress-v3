"""Euler ODE sampler for the rectified-flow model with classifier-free guidance.

Integrates dx/dt = v(x,t) from t=1 (noise) to t=0 (data):  x_{t-dt} = x_t - dt * v,
with CFG  v = v_u + s * (v_c - v_u).
"""
import argparse, os, re, sys
import numpy as np, torch
from PIL import Image
from conditioning import tokenize, MAX_LEN
from model import build_model
from generate_data import PALETTE, EXPRESSIONS

# v1 template plus the optional v2 suffixes (layout, paper); v1 prompts still parse identically.
PROMPT_RE = re.compile(r"^\s*an?\s+(?P<color>[a-z]+)\s+(?P<expr>[a-z]+)\s+face\s+with\s+"
                       r"['\"](?P<text>[^'\"]*)['\"]\s+next\s+to\s+it"
                       r"(?:\s*,\s*(?P<layout>face\s+on\s+the\s+right|text\s+below))?"
                       r"(?:\s*,\s*on\s+(?P<paper>fresh|sepia)\s+paper)?\s*\.?\s*$", re.I)
EXPRESSIONS_V2 = ["smiley", "sad", "meh", "wink", "angry", "dead", "uwu"]


def parse_prompt(p, v2=False):
    """Returns (color, expr, text, canonical_caption). v2=False: v1 rules (3 expressions, 1-8 chars, no suffixes).
    v2=True: 7 expressions, 1-14 chars, optional ', face on the right' / ', text below' and ', on fresh|sepia paper'."""
    m = PROMPT_RE.match(p)
    if not m:
        raise ValueError(f"Prompt must look like: a red smiley face with 'HELLO' next to it"
                         f"{'[, face on the right|, text below][, on fresh|sepia paper]' if v2 else ''}  (got: {p!r})")
    color, expr, text = m["color"].lower(), m["expr"].lower(), m["text"]
    layout = {"": "left", "face on the right": "right", "text below": "below"}[re.sub(r"\s+", " ", (m["layout"] or "").lower())]
    paper = (m["paper"] or "aged").lower()
    exprs, max_text = (EXPRESSIONS_V2, 14) if v2 else (EXPRESSIONS, 8)
    errs = []
    if color not in PALETTE: errs.append(f"unknown color {color!r}; choose from {sorted(PALETTE)}")
    if expr not in exprs: errs.append(f"unknown expression {expr!r}; choose from {exprs}")
    if not (1 <= len(text) <= max_text): errs.append(f"text must be 1-{max_text} characters (got {len(text)})")
    if not text.isascii(): errs.append("text must be ASCII")
    if not v2 and (layout != "left" or paper != "aged"):
        errs.append("layout/paper suffixes need the v2 model (parse_prompt(..., v2=True))")
    if errs: raise ValueError("; ".join(errs))
    # canonical caption, exactly the training template (suffixes only when non-default)
    cap = f"a {color} {expr} face with '{text}' next to it"
    cap += {"left": "", "right": ", face on the right", "below": ", text below"}[layout]
    cap += {"aged": "", "fresh": ", on fresh paper", "sepia": ", on sepia paper"}[paper]
    return color, expr, text, cap


@torch.no_grad()
def sample(model, caption, res, steps, cfg, seed, device, n=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 3, res, res, generator=g).to(device)
    ids, mask = tokenize([caption] * n, MAX_LEN); ids, mask = ids.to(device), mask.to(device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        tb = t.expand(n)
        # batch cond + uncond in one pass
        xin, tin = torch.cat([x, x]), torch.cat([tb, tb])
        ctx_c, m_c = model.cond(ids, mask)
        ctx_u, m_u = model.cond(ids, mask, force_uncond=True)
        v = model.unet(xin, tin, torch.cat([ctx_c, ctx_u]), torch.cat([m_c, m_u]))
        v_c, v_u = v.chunk(2)
        v = v_u + cfg * (v_c - v_u)
        x = x - dt * v                       # Euler step toward t=0 (data)
    return x.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--out", default="sample.png")
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--num_images", type=int, default=1, help=">1 saves a horizontal strip")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    try:
        color, expr, text, caption = parse_prompt(a.prompt)
    except ValueError as e:
        sys.exit(f"error: {e}")
    print(f"parsed: color={color} expression={expr} text={text!r}")
    model = build_model().to(a.device).eval()
    if a.checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(a.checkpoint, device=str(a.device))  # flat EMA state_dict (bf16 or fp32)
    else:
        ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
        sd = ck["model"] if (a.no_ema and "model" in ck) else ck.get("ema", ck.get("model"))
    model.load_state_dict({k: v.float() for k, v in sd.items()})  # upcast bf16 weights
    x = sample(model, caption, a.resolution, a.steps, a.cfg_scale, a.seed, a.device, a.num_images)
    imgs = ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    Image.fromarray(np.concatenate(list(imgs), axis=1)).save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
