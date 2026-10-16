"""Tiny conv-CTC text recognizer ("CRNN" without the RNN) for grinpress OCR eval + as a frozen loss in Phases 3/4.

Why no LSTM: Phases 3/4 backprop THROUGH the frozen recognizer in eval mode, and cuDNN RNN backward refuses to run
in eval mode. The sequence model is therefore a stack of residual 1-D convolutions over the width axis
(receptive field covers the whole crop), which is plenty for single-line procedural text.

Input: a TEXT-BOX crop, grey, 40x200 (aspect 5:1, aspect-preserving: the box is padded and then grown to 5:1 around
its centre, so tiny 14-char strings get ~2.5x more vertical resolution than a fixed band crop). Boxes come from
  * the generator's GT text bbox (training data, Phase-3 CTC loss: "crop GT text box"), or
  * detect_boxes(): dark-ink bbox inside the layout's text band (eval on references / generated images, and the bbox
    stored with every Phase-4 reflow pair). The layout bands were measured from 600 renders (left x54-126/y40-85,
    right x3-74/y43-88, below x5-123/y74-121).
The crop is a differentiable function of the image (F.grid_sample), so the SAME code path serves eval, CRNN training
and the CTC loss on x0_hat during teacher/student training.
(v3 run 1 used a fixed 48x160 band crop instead: 79.2% exact on the 48 refs; see assets/crnn_train_run1_band.json.)
"""
import math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from charset import N_CLASSES, BLANK, ctc_greedy, encode

H_IN, W_IN, RES = 40, 200, 128
ASPECT = W_IN / H_IN
LAYOUT_ID = {"left": 0, "right": 1, "below": 2}
BANDS = torch.tensor([[52, 36, 128, 92], [0, 38, 76, 94], [2, 70, 126, 124]], dtype=torch.float32)  # x0,y0,x1,y1 px


def to_gray(x):
    """[B,3,H,W] -> [B,1,H,W] (luma); single-channel input passes through."""
    if x.shape[1] == 1:
        return x
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def band_boxes(layout_ids, jitter=None):
    """layout_ids [B] -> boxes [B,4] px; jitter [B,4] px offsets added to (x0,y0,x1,y1)."""
    b = BANDS.to(layout_ids.device)[layout_ids]
    return b if jitter is None else b + jitter


def crop_boxes(x, boxes, out_hw=(H_IN, W_IN)):
    """Differentiable per-sample crop+resize. x [B,C,H,W] (any range), boxes [B,4] in pixel coords."""
    B, C, H, W = x.shape
    x0, y0, x1, y1 = boxes.unbind(1)
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = (x1 - x0) / W; theta[:, 0, 2] = (x0 + x1) / W - 1
    theta[:, 1, 1] = (y1 - y0) / H; theta[:, 1, 2] = (y0 + y1) / H - 1
    grid = F.affine_grid(theta, (B, C, *out_hw), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def with_scale(crop, eb):
    """Append a constant 'absolute scale' channel (expanded box height / 64 px) -> [B,2,H,W]. The aspect-normalised
    crop throws away absolute glyph size, which is the only cue separating o/O, w/W, s/S, ... in all-x-height strings.
    1-channel CRNNs (in_ch=1) simply ignore channel 1, so every caller can always pass the 2-channel crop."""
    s = ((eb[:, 3] - eb[:, 1]) / 64.0).to(crop.dtype).view(-1, 1, 1, 1).expand(-1, 1, *crop.shape[-2:])
    return torch.cat([crop, s], 1)


def crop_band(x, layout_ids, jitter=None):
    """x [B,3|1,128,128] in [-1,1] -> grey crop of the whole layout band (aspect-normalised) [B,2,40,200]."""
    eb = expand_boxes(band_boxes(layout_ids, jitter), pad_frac=0.0, pad_px=0.0)
    return with_scale(crop_boxes(to_gray(x), eb), eb)


def expand_boxes(boxes, pad_frac=0.12, pad_px=2.0, aspect=None, min_h=8.0):
    """Pad a text bbox, then grow it to the CRNN aspect ratio around its centre (never shrinks)."""
    aspect = aspect or ASPECT
    x0, y0, x1, y1 = boxes.float().unbind(1)
    w, h = (x1 - x0).clamp_min(1), (y1 - y0).clamp_min(1)
    ph = torch.maximum(h * (1 + 2 * pad_frac) + 2 * pad_px, torch.full_like(h, min_h))
    pw = w * (1 + 2 * pad_frac) + 2 * pad_px
    H = torch.maximum(ph, pw / aspect); W = H * aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return torch.stack([cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2], 1)


def crop_text(x, boxes, jitter=None):
    """x [B,3|1,128,128] in [-1,1], boxes [B,4] tight text bboxes (px) -> CRNN input [B,2,40,200] (grey + scale)."""
    b = boxes.float().to(x.device)
    if jitter is not None:
        b = b + jitter.to(x.device)
    eb = expand_boxes(b)
    return with_scale(crop_boxes(to_gray(x), eb), eb)


FACE_MARGIN = 4   # ignore this many px at the face-side edge of a band (face outline / halftone can poke in)


@torch.no_grad()
def detect_boxes(x, layout_ids, rel=0.45, q=0.005, min_drop=35.0):
    """Dark-ink text bbox inside each sample's layout band. x [B,3|1,128,128] in [-1,1] -> float boxes [B,4] (px).
    ink = luma < median - max(min_drop, rel * (median - darkest)), i.e. adaptive to faint tiny text;
    bbox from the q / 1-q coordinate quantiles; falls back to the band."""
    g = (to_gray(x.float()) + 1) * 127.5
    out = []
    for i in range(len(g)):
        lid = int(layout_ids[i]); bx0, by0, bx1, by1 = [int(v) for v in BANDS[lid].tolist()]
        if lid == 0: bx0 += FACE_MARGIN
        elif lid == 1: bx1 -= FACE_MARGIN
        else: by0 += FACE_MARGIN
        reg = g[i, 0, by0:by1, bx0:bx1]
        med = reg.median(); dark = torch.quantile(reg.flatten(), 0.002)
        ys, xs = torch.nonzero(reg < med - torch.clamp(rel * (med - dark), min=min_drop), as_tuple=True)
        if len(xs) < 6:
            out.append([bx0, by0, bx1, by1]); continue
        xs, ys = xs.float(), ys.float()
        out.append([bx0 + float(torch.quantile(xs, q)), by0 + float(torch.quantile(ys, q)),
                    bx0 + float(torch.quantile(xs, 1 - q)) + 1, by0 + float(torch.quantile(ys, 1 - q)) + 1])
    return torch.tensor(out, dtype=torch.float32, device=x.device)


def cbr(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Res1d(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.c1, self.b1 = nn.Conv1d(c, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c)
        self.c2, self.b2 = nn.Conv1d(c, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c)

    def forward(self, x):
        return F.relu(x + self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


class CRNN(nn.Module):
    """[B,1|2,40,200] grey crop (+ optional scale channel) -> log-probs [T=50,B,N_CLASSES] (CTC layout)."""
    def __init__(self, w=(24, 48, 96, 128, 160), seq_layers=3, in_ch=1):
        super().__init__()
        c1, c2, c3, c4, cs = w
        self.in_ch = in_ch
        self.cfg = dict(w=list(w), seq_layers=seq_layers, in_ch=in_ch)
        self.cnn = nn.Sequential(
            cbr(in_ch, c1), nn.MaxPool2d(2),                       # 20x100
            cbr(c1, c2), nn.MaxPool2d(2),                      # 10x50
            cbr(c2, c3), cbr(c3, c3), nn.MaxPool2d((2, 1)),    # 5x50
            cbr(c3, c4), cbr(c4, c4), nn.MaxPool2d((2, 1), ceil_mode=True),  # 3x50
            cbr(c4, cs, k=(3, 1), p=0))                        # 1x50
        self.seq = nn.Sequential(*[Res1d(cs) for _ in range(seq_layers)])
        self.head = nn.Conv1d(cs, N_CLASSES, 1)

    def forward(self, crop):
        # per-crop standardisation: robust to paper tone / ink darkness / generated-image contrast
        g = crop[:, :1]
        mu = g.mean((2, 3), keepdim=True); sd = g.std((2, 3), keepdim=True) + 1e-3
        g = (g - mu) / sd
        if self.in_ch == 2:
            sc = crop[:, 1:2] if crop.shape[1] > 1 else torch.zeros_like(g)
            g = torch.cat([g, sc], 1)
        h = self.cnn(g).squeeze(2)              # [B,C,50]
        return self.head(self.seq(h)).permute(2, 0, 1).log_softmax(-1)   # [T,B,K]


def ctc_loss(logp, texts_or_ids, reduction="mean"):
    """CTC on log-probs [T,B,K]; targets as strings or id lists. Mean over batch of per-sample loss / target length."""
    ids = [encode(t) if isinstance(t, str) else list(t) for t in texts_or_ids]
    tgt = torch.tensor([i for s in ids for i in s], dtype=torch.long, device=logp.device)
    tl = torch.tensor([len(s) for s in ids], dtype=torch.long, device=logp.device)
    il = torch.full((logp.shape[1],), logp.shape[0], dtype=torch.long, device=logp.device)
    return F.ctc_loss(logp.float(), tgt, il, tl, blank=BLANK, reduction=reduction, zero_infinity=True)


def decode(logp):
    return [ctc_greedy(s) for s in logp.argmax(-1).T.cpu().tolist()]


def save_crnn(m, path):
    from safetensors.torch import save_file
    import json
    sd = {k: v.detach().half().contiguous() if v.is_floating_point() else v.contiguous() for k, v in m.state_dict().items()}
    save_file(sd, path, metadata={"cfg": json.dumps(m.cfg)})


def load_crnn(path, device="cpu"):
    from safetensors import safe_open
    from safetensors.torch import load_file
    import json
    with safe_open(path, "pt") as f:
        cfg = json.loads(f.metadata()["cfg"])
    m = CRNN(tuple(cfg["w"]), cfg["seq_layers"], cfg.get("in_ch", 1))
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in load_file(path).items()})
    return m.to(device).eval().requires_grad_(False)


def pil_batch(imgs):
    """list of PIL RGB images -> [B,3,128,128] in [-1,1]."""
    a = np.stack([np.asarray(im.convert("RGB"), np.float32) for im in imgs]) / 127.5 - 1
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def recognize(model, x, layouts, boxes=None, bs=64):
    """x: [B,3,128,128] in [-1,1] tensor or list of PIL; layouts: list of 'left'/'right'/'below';
    boxes: optional [B,4] text bboxes (default: detect_boxes). -> list of strings."""
    if isinstance(x, (list, tuple)):
        x = pil_batch(x)
    dev = next(model.parameters()).device
    out = []
    for b in range(0, len(x), bs):
        xb = x[b:b + bs].to(dev).float()
        lid = torch.tensor([LAYOUT_ID[l] for l in layouts[b:b + bs]], device=dev)
        bx = detect_boxes(xb, lid) if boxes is None else torch.as_tensor(boxes[b:b + bs]).to(dev)
        out += decode(model(crop_text(xb, bx)))
    return out
