"""The v3 evaluation grids: EXACTLY the v2 eval specs (grinpress-v2/eval/eval_samples.py + eval_compare.py).

grid48  : 48 strings, spec(i, s), noise / reference-render seed 1000+i (eval_ocr.json, samples_*step.png, reference_renders.png)
grid14  : 14 strings (PICK), expression EXPRS[i//2], layout left, paper aged, seed 2000+i (eval_compare.json)
These strings are HELD OUT from every v3 training text sampler (see generate_data_v2.random_text_v3).
"""
from generate_data import PALETTE
from generate_data_v2 import EXPRESSIONS_V2, LAYOUTS, PAPERS, caption_v2

STRINGS = [
    "hi", "ok", "no", "OK", "NO", "YES",
    "GOOD MORNING!!", "extraordinary", "BREAKING NEWS", "HELLO WORLD", "newspaper boy", "PUPPYGIRL CEO",
    "bork", "uwu", "nyaa~", "awoo", "woof woof", "BARK", "mlem", "owo", ">w<", ":3",
    "HeLLo", "Woof", "GoodGirl", "iPhone", "McDonald",
    "2026", "404", "1923", "3.14", "007", "12:30",
    "?!", "...", "wow!", "hey?", "a-ok", "#1", "50%", "$5",
    "NEWS", "HELLO", "SALE", "EXTRA", "LOVE", "VOTE", "HAPPY",
]
assert len(STRINGS) == 48 and len(set(STRINGS)) == 48 and all(len(s) <= 14 for s in STRINGS)
PICK = ["hi", "HELLO", "GOOD MORNING!!", "bork", "2026", "?!", "NEWS", "uwu", "extraordinary", "404", "LOVE", "nyaa~", "HeLLo", "VOTE"]
COLORS, LAYS, PAPS = list(PALETTE), list(LAYOUTS), list(PAPERS)
HELD_OUT = frozenset(STRINGS) | frozenset(PICK)


def spec48(i, s):
    return dict(text=s, expr=EXPRESSIONS_V2[i % 7], layout=LAYS[i % 3], paper=PAPS[(i // 3) % 3], color=COLORS[(i * 5) % 12])


def grid48():
    """[(spec, caption, seed)] for the 48-string grid."""
    out = []
    for i, s in enumerate(STRINGS):
        sp = spec48(i, s)
        out.append((sp, caption_v2(sp["color"], sp["expr"], sp["text"], sp["layout"], sp["paper"]), 1000 + i))
    return out


def grid14():
    out = []
    for i, s in enumerate(PICK):
        sp = dict(text=s, expr=EXPRESSIONS_V2[i // 2], layout="left", paper="aged", color=COLORS[(i * 5) % 12])
        out.append((sp, caption_v2(sp["color"], sp["expr"], sp["text"], sp["layout"], sp["paper"]), 2000 + i))
    return out


def all_eval_captions():
    return [c for _, c, _ in grid48()] + [c for _, c, _ in grid14()]
