"""Tiny conv-CTC text recognizer ("CRNN" without the RNN) for grinpress OCR eval + as a frozen loss in Phases 3/4.

Why no LSTM: Phases 3/4 backprop THROUGH the frozen recognizer in eval mode, and cuDNN RNN backward refuses to run
in eval mode. The sequence model is therefore a stack of residual 1-D convolutions over the width axis
(receptive field covers the whole crop), which is plenty for single-line procedural text.

Input: a fixed per-layout "text band" crop of the 128x128 image (where generate_data_v2 can put text; measured
from 600 renders: left x54-126/y40-85, right x3-74/y43-88, below x5-123/y74-121), grey, resized to 48x160 with
F.grid_sample. The crop is a pure differentiable function of the image, so the SAME code path serves
the eval (PIL images), the CRNN training data and the CTC loss on x0_hat during teacher/student training.
"""
import math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from charset import N_CLASSES, BLANK, ctc_greedy, encode

H_IN, W_IN, RES = 48, 160, 128
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


def crop_band(x, layout_ids, jitter=None):
    """x [B,3|1,128,128] in [-1,1] -> grey band crop [B,1,48,160]."""
    return crop_boxes(to_gray(x), band_boxes(layout_ids, jitter))


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
    """[B,1,48,160] grey crop -> log-probs [T=40,B,N_CLASSES] (CTC layout)."""
    def __init__(self, w=(24, 48, 96, 128, 160), seq_layers=3):
        super().__init__()
        c1, c2, c3, c4, cs = w
        self.cfg = dict(w=list(w), seq_layers=seq_layers)
        self.cnn = nn.Sequential(
            cbr(1, c1), nn.MaxPool2d(2),                       # 24x80
            cbr(c1, c2), nn.MaxPool2d(2),                      # 12x40
            cbr(c2, c3), cbr(c3, c3), nn.MaxPool2d((2, 1)),    # 6x40
            cbr(c3, c4), cbr(c4, c4), nn.MaxPool2d((2, 1)),    # 3x40
            cbr(c4, cs, k=(3, 1), p=0))                        # 1x40
        self.seq = nn.Sequential(*[Res1d(cs) for _ in range(seq_layers)])
        self.head = nn.Conv1d(cs, N_CLASSES, 1)

    def forward(self, crop):
        # per-crop standardisation: robust to paper tone / ink darkness / generated-image contrast
        mu = crop.mean((2, 3), keepdim=True); sd = crop.std((2, 3), keepdim=True) + 1e-3
        h = self.cnn((crop - mu) / sd).squeeze(2)              # [B,C,40]
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
    m = CRNN(tuple(cfg["w"]), cfg["seq_layers"])
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in load_file(path).items()})
    return m.to(device).eval().requires_grad_(False)


def pil_batch(imgs):
    """list of PIL RGB images -> [B,3,128,128] in [-1,1]."""
    a = np.stack([np.asarray(im.convert("RGB"), np.float32) for im in imgs]) / 127.5 - 1
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def recognize(model, x, layouts, bs=64):
    """x: [B,3,128,128] in [-1,1] tensor or list of PIL; layouts: list of 'left'/'right'/'below'. -> list of strings."""
    if isinstance(x, (list, tuple)):
        x = pil_batch(x)
    dev = next(model.parameters()).device
    out = []
    for b in range(0, len(x), bs):
        lid = torch.tensor([LAYOUT_ID[l] for l in layouts[b:b + bs]], device=dev)
        out += decode(model(crop_band(x[b:b + bs].to(dev).float(), lid)))
    return out
