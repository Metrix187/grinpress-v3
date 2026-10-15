"""grinpress v3 character-level caption conditioning.

Changes vs v1/v2 (which used max_len 64 and silently truncated 21/48 eval prompts):
  * MAX_LEN = 128 (longest v2/v3 caption is 88 chars; headroom for free-form prompts).
  * Vocabulary: id = ASCII code point for every printable char (1..127), PAD = 0, and a DEDICATED UNK = 128
    (v1 mapped every non-ASCII char to '?', colliding with the real '?'). Every char of charset.CHARSET
    therefore has its own non-colliding id, identical to the v1 id (so v1 embedding rows warm-start 1:1).
  * Quoted-span features, computed from ids inside the module (so ONNX/index.html still only feed ids+mask):
      token_type: 0 = template text, 1 = the two quote delimiters of the FIRST '...' span, 2 = glyphs inside it
      glyph_idx : 0 outside the span, 1 + min(k, 15) for the k-th glyph inside (k = 0..15 relative index)
    Both are learned embeddings, ZERO-INITIALISED, so a v1 checkpoint loaded into this module behaves exactly like v1.
  * Unconditional (CFG) branch: the learned null sequence keeps its v1 length NULL_LEN = 64 and only those 64
    rows are attended to, so the uncond branch is also bit-for-bit v1 (padding it to 128 by repeating rows,
    as v2's loader did, changes the softmax).
"""
import math
import torch, torch.nn as nn

MAX_LEN = 128
NULL_LEN = 64
PAD, UNK = 0, 128
VOCAB = 130            # 0 pad, 1..127 ASCII, 128 unk, 129 reserved
QUOTE = ord("'")
N_GLYPH = 16           # relative glyph index 0..15 (clamped)


def tokenize(texts, max_len=MAX_LEN, strict=False):
    """List[str] -> (ids LongTensor [B,L], mask BoolTensor [B,L], True = real token).
    Non-ASCII / control chars -> UNK (128). strict=True raises on truncation or UNK instead."""
    ids = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        b = [ord(c) if 0 < ord(c) < 128 and (c.isprintable() or c == " ") else UNK for c in t]
        if strict and (len(b) > max_len or UNK in b):
            raise ValueError(f"caption needs truncation or has unknown chars: {t!r}")
        b = b[:max_len]
        if b:
            ids[i, :len(b)] = torch.tensor(b, dtype=torch.long)
    return ids, ids != PAD


def detokenize(ids):
    return ["".join("\ufffd" if int(k) == UNK else chr(int(k)) for k in row if int(k) != PAD) for row in ids]


def quote_features(ids):
    """ids [B,L] -> (token_type [B,L] in {0,1,2}, glyph_idx [B,L] in 0..16). Pure cumsum/compare ops (ONNX-friendly)."""
    q = (ids == QUOTE).long()
    c = torch.cumsum(q, 1)                                  # number of quotes seen up to and including this position
    inside = ((c == 1) & (q == 0)).long()                   # glyphs of the first quoted span
    delim = (q == 1) & (c <= 2)                             # its opening and closing quote
    k = torch.cumsum(inside, 1) - 1                         # relative glyph index within the span
    glyph = torch.where(inside == 1, 1 + torch.clamp(k, max=N_GLYPH - 1), torch.zeros_like(k))
    ttype = inside * 2 + delim.long()
    return ttype, glyph


def sinusoidal_1d(L, d):
    # PE[p,2i] = sin(p / 10000^(2i/d)), PE[p,2i+1] = cos(...). Positions 0..63 are identical to v1's 64-long table.
    pos = torch.arange(L).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(L, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TextConditioner(nn.Module):
    """Embeds chars -> per-token sequence [B,L,D] + mask [B,L]. No pooling.

    CFG dropout: with prob p_uncond a sample's sequence is replaced by the learned null sequence
    (its first NULL_LEN positions marked valid, the rest masked)."""
    def __init__(self, dim=256, layers=2, heads=4, max_len=MAX_LEN, p_uncond=0.1, quote_feats=True):
        super().__init__()
        self.max_len, self.p_uncond, self.quote_feats = max_len, p_uncond, quote_feats
        self.emb = nn.Embedding(VOCAB, dim)
        self.register_buffer("pe", sinusoidal_1d(max_len, dim), persistent=False)
        if quote_feats:
            self.type_emb = nn.Embedding(3, dim)
            self.glyph_emb = nn.Embedding(N_GLYPH + 1, dim)
            nn.init.zeros_(self.type_emb.weight); nn.init.zeros_(self.glyph_emb.weight)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.null_seq = nn.Parameter(torch.randn(NULL_LEN, dim) * 0.02)

    def embed(self, ids):
        x = self.emb(ids) + self.pe[None, :ids.shape[1]]
        if self.quote_feats:
            tt, g = quote_features(ids)
            x = x + self.type_emb(tt) + self.glyph_emb(g)
        return x

    def null(self, B, L, dtype, device):
        n = min(L, NULL_LEN)
        x = torch.zeros(B, L, self.null_seq.shape[1], dtype=dtype, device=device)
        x[:, :n] = self.null_seq[:n].to(dtype)
        m = torch.zeros(B, L, dtype=torch.bool, device=device); m[:, :n] = True
        return x, m

    def forward(self, ids, mask, force_uncond=False):
        x = self.norm(self.encoder(self.embed(ids), src_key_padding_mask=~mask))
        B, L = ids.shape
        if force_uncond:
            drop = torch.ones(B, dtype=torch.bool, device=ids.device)
        elif self.training and self.p_uncond > 0:
            drop = torch.rand(B, device=ids.device) < self.p_uncond
        else:
            return x, mask
        nx, nm = self.null(B, L, x.dtype, x.device)
        x = torch.where(drop[:, None, None], nx, x)
        mask = torch.where(drop[:, None], nm, mask)
        return x, mask
