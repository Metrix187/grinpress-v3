"""Phase 1 unit test: v3 tokenizer + quoted-span features on all 48 (+14) eval prompts.
Run: python tests/test_tokenizer.py  (writes assets/phase1_tokenizer_test.json)"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch
from conditioning import tokenize, detokenize, quote_features, MAX_LEN, PAD, UNK, VOCAB, TextConditioner
from charset import CHARSET
from eval_grid import grid48, grid14
from generate_data_v2 import longest_caption


def check_caption(cap, text, enc):
    ids, mk = tokenize([cap])
    n = int(mk.sum())
    tt, g = quote_features(ids)
    r = dict(caption=cap, n_chars=len(cap), n_tokens=n)
    r["lossless"] = detokenize(ids)[0] == cap
    r["no_unk"] = int((ids == UNK).sum()) == 0
    r["no_truncation"] = n == len(cap) and len(cap) <= MAX_LEN
    span = (tt[0] == 2).nonzero().flatten().tolist()
    r["span_text"] = "".join(chr(int(ids[0, p])) for p in span)
    r["span_matches_text"] = r["span_text"] == text
    gi = [int(g[0, p]) - 1 for p in span]
    r["glyph_idx"] = gi
    r["glyph_idx_distinct_0_to_n-1"] = gi == list(range(len(text)))
    r["delims_ok"] = int((tt[0] == 1).sum()) == 2
    # full positional signal = sinusoidal PE + glyph-index embedding (random weights so the test is meaningful
    # beyond the zero init); every glyph in the span must get a distinct encoding.
    with torch.no_grad():
        pos = enc.pe[span] + enc.glyph_emb(g[0, span])
        tok = enc.embed(ids)[0, span]
    d = torch.cdist(pos, pos) + torch.eye(len(span)) * 1e9 if span else torch.zeros(0)
    r["min_pairwise_pos_dist"] = float(d.min()) if len(span) > 1 else None
    r["distinct_pos_enc"] = len(span) <= 1 or float(d.min()) > 1e-3
    # repeated letters (LL, oo, !!...) must differ in glyph index and in the full token input
    reps = [(i, i + 1) for i in range(len(text) - 1) if text[i] == text[i + 1]]
    r["repeated_pairs"] = [text[i] + text[j] for i, j in reps]
    r["repeats_distinct"] = all(gi[i] != gi[j] and float((tok[i] - tok[j]).abs().max()) > 1e-4 for i, j in reps)
    r["ok"] = all(r[k] for k in ("lossless", "no_unk", "no_truncation", "span_matches_text",
                                  "glyph_idx_distinct_0_to_n-1", "delims_ok", "distinct_pos_enc", "repeats_distinct"))
    return r


def main():
    torch.manual_seed(0)
    enc = TextConditioner()
    torch.nn.init.normal_(enc.glyph_emb.weight, std=0.5); torch.nn.init.normal_(enc.type_emb.weight, std=0.5)
    # vocab: every charset char has a dedicated id, none collide with PAD/UNK or each other
    ids, _ = tokenize(list(CHARSET))
    cid = ids[:, 0].tolist()
    vocab_ok = len(set(cid)) == len(CHARSET) and PAD not in cid and UNK not in cid and max(cid) < VOCAB
    unk_ok = tokenize(["é"])[0][0, 0].item() == UNK and tokenize(["?"])[0][0, 0].item() != UNK
    res = {"max_len": MAX_LEN, "vocab": VOCAB, "charset_size": len(CHARSET), "charset_dedicated_ids": vocab_ok,
           "non_ascii_maps_to_dedicated_unk": unk_ok, "longest_possible_caption": len(longest_caption()), "grids": {}}
    for name, grid in (("grid48", grid48()), ("grid14", grid14())):
        rows = [check_caption(c, sp["text"], enc) for sp, c, _ in grid]
        res["grids"][name] = dict(n=len(rows), n_ok=sum(r["ok"] for r in rows),
                                  truncated_at_64_before=sum(r["n_chars"] > 64 for r in rows),
                                  truncated_now=sum(not r["no_truncation"] for r in rows),
                                  unk=sum(not r["no_unk"] for r in rows),
                                  max_tokens=max(r["n_tokens"] for r in rows),
                                  repeated_letter_prompts={r["span_text"]: dict(pairs=r["repeated_pairs"], glyph_idx=r["glyph_idx"])
                                                           for r in rows if r["repeated_pairs"]},
                                  failures=[r for r in rows if not r["ok"]])
    lc = check_caption(longest_caption(), "W" * 14, enc)
    res["longest_caption_ok"] = lc["ok"]
    res["PASS"] = vocab_ok and unk_ok and lc["ok"] and all(g["n_ok"] == g["n"] for g in res["grids"].values())
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "assets", "phase1_tokenizer_test.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "grids"}, indent=1))
    for k, g in res["grids"].items():
        print(k, {a: b for a, b in g.items() if a not in ("failures",)})
    assert res["PASS"], "tokenizer test FAILED"
    print("PASS")


if __name__ == "__main__":
    main()
