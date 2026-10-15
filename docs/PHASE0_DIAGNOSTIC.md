# Phase 0 diagnostic: grinpress-v2 codebase (inspected 2026-10-15, v2 HEAD b8fa0ec)

## Files
| file | role |
|---|---|
| `conditioning.py` | char tokenizer `tokenize()` + `TextConditioner` (2-layer pre-norm Transformer, d=256, 4 heads, sinusoidal PE, learned `null_seq` for CFG) |
| `model.py` | pixel-space U-Net (ch 64, mults 1/2/3/4, self+cross attn at 32/16 px levels), `FlowModel`, `StudentFlowModel` (+ zero-init guidance embedding w), `load_teacher_weights` |
| `generate_data.py` / `generate_data_v2.py` | procedural newsprint renderer (v1: 3 expressions, UPPER A-Z/0-9 <= 8 chars; v2: 7 expressions, 3 layouts, 3 papers, <= 14 chars, but the text sampler is still UPPER A-Z/0-9 + 30 upper-case WORDS) |
| `train.py`, `sample.py` | v1 teacher training (logit-normal t, lr 2e-4, EMA 0.9999) and CFG Euler sampler |
| `distill_reflow.py` | v2 gen_pairs (50 Euler + CFG 5, chunk resume, OOM halving) + reflow / online student training |
| `export_and_infer_cpu.py` | ONNX export (`TextEncoderExport` re-implements the conditioner forward), ORT validation, int8 dynamic quant (MatMul/Gemm), bench |
| `run_v2.sh`, `distill_colab.ipynb` | unattended Colab pipeline (status.json heartbeat, atomic ckpts + async Drive mirror, chunk resume, OOM halving, Drive retry/wait, DONE) |
| `eval/eval_samples.py`, `eval/eval_compare.py`, `eval/bench_cpu.py` | 48-string grid (seeds 1000+i), 14-string teacher/student grid (seeds 2000+i), tesseract `--psm 7` OCR |
| `index.html` | browser client; its JS tokenizer hard-codes `MAX_LEN = 64` |

## Tokenizer (the real one)
* Vocabulary: raw ASCII bytes, `VOCAB = 128`; id 0 = PAD. Each printable ASCII char already has its own id, **but** every
  non-ASCII char is mapped to `ord("?")`, i.e. it collides with the real `?` (no dedicated UNK).
* `MAX_LEN = 64` (conditioning.py:12); `tokenize()` silently truncates `[:max_len]`.
* `generate_data_v2.py` defines `V2_MAX_LEN = 96` and its docstring says v2 uses 96, **but nothing imports it**: every
  loader (train, distill_reflow, sample, export, eval, index.html) uses conditioning.MAX_LEN = 64. Longest v2 caption = 88
  chars; on the 48-string eval grid **21/48 captions exceed 64** (eval_ocr.json: `captions_truncated_at_64: 21`), max 85.
  Truncation cuts the layout/paper suffix and, for long quoted strings, the closing quote.

## Model
* v1 teacher (`skyuu72/grinpress/model.safetensors`, bf16, sha256 f41b00…, identical to v2's teacher.safetensors):
  **33,863,299 params** (U-Net 32.23M, text conditioner 1.63M). v3 model: 33,868,931 (+5,632: 2 vocab rows, 3 token-type
  rows, 17 glyph-index rows).

## Context / mask flow
1. `tokenize(captions)` -> `ids [B,64]` int64, `mask = ids != 0` (True = real token).
2. `TextConditioner`: `emb(ids) + PE[:L]` -> TransformerEncoder with `src_key_padding_mask = ~mask` -> LayerNorm -> `ctx [B,L,256]`.
   CFG: rows dropped (train p=0.1, or `force_uncond`) become `null_seq [64,256]` with mask all-True.
3. U-Net `AttnBlock`: pixel self-attn, then cross-attn `q = pixels`, `k,v = Linear(ctx)`, `attn_mask = mask[:,None,None,:]`
   (padding keys never attended); timestep FiLM in every ResBlock.
4. Teacher sampling batches cond+uncond (CFG `v_u + w(v_c - v_u)`); the v2 student bakes w=5 via `temb_add`.
5. ONNX: `text_encoder.onnx (input_ids, attention_mask [B,64]) -> (context [B,64,256], context_mask)`, then
   `unet_distilled.onnx (x_t, t, context, context_mask)`; context computed once per prompt.

## Where the 64 limit lives
`conditioning.py` (`MAX_LEN = 64`, `tokenize` default, `TextConditioner(max_len=64)` -> `pe` buffer and `null_seq [64,256]`),
`model.load_teacher_weights` (pads null_seq by repeating its last row when max_len grows, which changes the uncond softmax),
`export_and_infer_cpu.py` (docstring + traced example shapes), `sample.py`/`eval/eval_samples.py` (import MAX_LEN;
`captions_truncated_at_64`), `index.html` (`const MAX_LEN = 64`), the shipped ONNX graphs (sequence dim traced at 64).

## Other findings relevant to v3
* The v2 student was distilled from **v1 captions only** (`distill_reflow.random_caption` uses v1 EXPRESSIONS + `random_text`),
  so v2 options were never learned (as the v2 README says).
* OCR ceiling: tesseract on the clean procedural refs = 43.75% exact_ci (re-measured here, identical to v2).
* Measured A100 throughput from the v2 run.log: teacher pair gen ~1300 s / 10k pairs (100 NFE each, bf16, batch 128);
  student training 0.13 s/it at batch 128 (micro 32).
