# where the text breaks 🐾 (varied-word stress test)

`python samples_varied.py --onnx_dir <onnx> --crnn crnn_ft.safetensors --out samples` (run 2026-11-01 on the box, CPU).
INT8 ONNX student (`unet_distilled_int8.onnx` + `text_encoder_int8.onnx`), 35 words NOT in grid48/grid14
(5 categories x 7) x all 7 expressions = 245 images per step count; layout/paper/colour cycle per cell, seeds 3000+k.
Scored with the stage-0b fine-tuned CRNN (sha256 be23291aeb97…, case-sensitive exact). The procedural reference
render of the same spec is also OCR'd as a "can the CRNN even read this layout" control.
Raw data: `varied_words_v3.json`. Grids: `varied_{2,4}step_{short,long,silly,mixed_case,punct_num}.png`.

| CRNN exact (char acc) | 4-step | 2-step | reference render |
|---|---|---|---|
| all 245 | 0.971 (0.997) | 0.984 (0.998) | 0.943 (0.993) |
| short (1-3 ch) | 0.980 | 0.980 | 1.000 |
| long (12-14 ch) | 0.918 | 0.959 | 0.755 |
| silly | 0.959 | 0.980 | 1.000 |
| mixed case | 1.000 | 1.000 | 0.980 |
| punct + numbers | 1.000 | 1.000 | 0.980 |

Per expression (4-step / 2-step): smiley 0.943/0.971, sad 0.943/0.971, meh 0.971/1.0, wink 0.971/0.943,
angry 1.0/1.0, dead 1.0/1.0, uwu 0.971/1.0. With n=35 per expression these differences are 1-2 images, i.e. noise.

## misses (CRNN) + what my eyes say when zoomed in 3x

| steps | text | CRNN read | expr/layout/paper | visual verdict |
|---|---|---|---|---|
| 4 + 2 | `Hi!` | `HHi!` | wink/below/sepia | **real model failure**: a doubled, smeared H is actually drawn (same seed, both step counts) |
| 4 | `smol bean` | `snol bean` | meh/below/fresh | **real-ish**: the `m` is malformed (broken middle stroke) |
| 4 + 2 | `sniff sniff` | `sniffi sniff` | sad/left/fresh | **real artifact**: a stray pink blob between the words that the CRNN reads as `i` |
| 4 | `who let me out` | `wholet me out` | sad/right/sepia | **partly real**: word gap between `who` and `let` is squeezed to ~1 px |
| 4 + 2 | `zoomies at 3am` | `zoonies/zoouies at 3am` | smiley,uwu/right/fresh | looks fine to me; tiny glyphs (~6 px x-height) → most likely CRNN misread |
| 4 / 2 | `SNUGGLE BUDDY!` | `SNUGGLE BEDDY!` / `BUD@Y!` | smiley,wink/left/aged | legible when zoomed; CRNN misread on tiny text |

## honest summary

- Mixed case (`PuPpY`, `sQuEaKy`, `BoRk BoRk`, `McWoof`, `iPup`…) and punctuation + numbers (`10/10`, `24/7`, `(o_o)`, `99.9%`,
  `1+1=2`, `$3.50`, `3:00 AM`) are rock solid: 0 misses in 98 images per step count. Short words miss only once, and that's the
  `HHi!` case below.
- Where it breaks: **long phrases (12-14 chars) in the left/right layouts**. At 128 px the text band is ~76 px wide, so
  14 chars → ~5-6 px glyphs. They're crisp-ish but tiny, word gaps get squeezed, and the CRNN itself only reads
  75.5 % of the long *reference* renders. So the long-string number is limited by OCR as much as by the model.
- Rare real glyph faults: letter doubling (`HHi!`), a malformed `m`, and stray ink blobs. I saw no wrong *words*.
  The faults are character-level.
- Expressions all render recognisably (smiley/sad/meh/wink/angry/dead/uwu) and none of them hurts the text. At this
  sample size the angry/dead 100 % is luck, not a trend.
- On this set 2-step scored ≥ 4-step. The official grid48 has the opposite order (4-step 0.958 > 2-step 0.9375), and
  both gaps are only a few images. Treat 2-step and 4-step as roughly equal on text.
