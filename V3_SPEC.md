# grinpress-v3 spec (from user, Oct 15 2026)
Goal: exact-match text rendering -> ~100% on the 48-string v2 eval grid, 2-4 step CPU ONNX INT8 inference <= ~0.65s.

Bottlenecks: (1) tokenizer max_len=64 truncates 21/48 prompts; (2) teacher domain gap (v1 teacher only knew smiley/sad/meh, UPPER A-Z/0-9 <=8 chars, face-left, aged paper); (3) char duplication/kerning collapse (SALE->SAEE, HeLLo->HHLLO, BARK->BRKR, HELLO WORLD->HELOWOLL); (4) OCR evaluator ceiling (tesseract 43.8% on clean refs); (5) 2/4-step distillation softening (teacher 42.9%, 4-step 35.7%, 2-step 21.4% on 14-string grid).

PHASE 1 tokenizer/positional: max_len 64->128 in conditioning.py, loaders, ONNX export. Every char in generate_data_v2.py vocab (a-z, A-Z, 0-9, ?!.:%$-~>< etc) gets a dedicated non-colliding id. Add is_quoted_text token-type embedding and/or relative glyph-index embedding 0..15 for chars inside '...'. Unit test: all 48 prompts tokenize losslessly, zero <unk>, zero truncation, distinct positional encodings on the quoted string.

PHASE 2 OCR: eval_ocr.py preprocessing for tesseract (grayscale, Sauvola/Otsu using ink-vs-paper color distance, --psm 7/8, vocab whitelist). Train a tiny CRNN/CTC recognizer on ~20k procedural crops from generate_data_v2.py (also used as loss in Phase 3/4). Gate: >=98% exact match on the 48 procedural reference renders before proceeding.

PHASE 3 teacher fine-tune (grinpress-v3-teacher): generator returns (image, prompt, text_bbox_mask, target_string_ids); oversample 40% 9-14 char, 30% mixed case+punct/emoticons (:3 >w< ?! $5 50%), uniform over 7 expressions, 3 layouts (left,right,below), 3 papers (aged,fresh,sepia). Loss: RF MSE (x_t=(1-t)x0+t z1, v=z1-x0) weighted by 1+4*text_bbox_mask; for t<0.5, x0_hat = x_t - t*v_hat, crop GT text box, frozen CRNN, +0.1*CTC. Warm-start from skyuu72/grinpress, expand positional embeddings cleanly. Train until teacher >95% exact on 48-string grid.

PHASE 4 student: 50k reflow pairs from v3 teacher (50 Euler steps, CFG 5) with chunking+OOM backoff. Timesteps 60% from {1.0,0.75,0.5,0.25}, 40% logit-normal. Loss = weighted reflow MSE ||(v_hat-v)*sqrt(1+4M)||^2 + Sobel-gradient L1 on x0_hat inside M + frozen CRNN CTC on x0_hat crops; 2-step consistency rollouts (1.0->0.5->0.0) on 25% of batches.

PHASE 5 export: onnx/text_encoder.onnx + _int8, onnx/unet_distilled.onnx (fp32, opset17, dynamic batch) + _int8 (MatMul/Gemm dynamic INT8, Conv fp32); student_ema_fp16.safetensors (<100MB). bench_cpu.py -> assets/bench_cpu_v3.json (~0.31s 2-step, ~0.63s 4-step). eval_ocr.py on 48 & 14 grids, 2/4 step INT8 -> assets/eval_ocr_v3.json, assets/eval_compare_v3.json. Grids samples_4step_v3.png, samples_2step_v3.png, teacher_vs_student_v3.png.

Rules: inspect codebase first and print diagnostic summary; clean modular code; keep checkpointing + OOM backoff; after each phase run verification and report exact metrics before next phase.
