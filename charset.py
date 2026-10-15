"""Shared character set for v3 text: what the generator may print, what the CRNN recognises, and what
gets a dedicated tokenizer id (the tokenizer covers all printable ASCII; this is the printable-text subset).

' and " are excluded (they delimit the quoted span in captions); backslash and backtick are excluded too.
"""
LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = LOWER.upper()
DIGITS = "0123456789"
PUNCT = " !?.,:;%$#&@*+-=~<>/()^_"
CHARSET = LOWER + UPPER + DIGITS + PUNCT
assert len(set(CHARSET)) == len(CHARSET)
BLANK = 0                                   # CTC blank
CHAR2ID = {c: i + 1 for i, c in enumerate(CHARSET)}
ID2CHAR = {i: c for c, i in CHAR2ID.items()}
N_CLASSES = len(CHARSET) + 1                # + blank


def encode(text):
    """text -> CRNN target ids (raises on chars outside CHARSET)."""
    return [CHAR2ID[c] for c in text]


def decode(ids):
    return "".join(ID2CHAR.get(int(i), "") for i in ids)


def ctc_greedy(seq):
    """argmax id sequence [T] -> string (collapse repeats, drop blanks)."""
    out, prev = [], None
    for k in seq:
        k = int(k)
        if k != prev and k != BLANK:
            out.append(k)
        prev = k
    return decode(out)
