# Answer-flip study — replication on LLaVA-Video-7B (Run-All)

A secondary analysis from the study plan: the same measures on a second model, so H1, H2a, H2b and H3 can be checked for replication. Each run handles **one** benchmark, chosen in the config cell.

| `BENCHMARK` | Questions | Expected time | Videos downloaded |
|---|---|---|---|
| `"videomme"` | 300 (100 videos) | 3–4 h | 5 × 5 GB zips |
| `"egoschema"` | 300 | 8–10 h | 5 × 21 GB zips (only the needed videos are kept) |
| `"longvideobench"` | 300 (75 per duration group) | 9–12 h (may need a second session) | 31 × 5 GB parts, streamed |

**Model.** LLaVA-Video-7B-Qwen2, in its official FlashVID evaluation setting: 64 frames, 13 × 13 tokens per frame (average pooling), a newline token after each frame. The original weights are converted once per session to the transformers format (about 10 min, done in the model cell).

**Methods.** FlashVID, HoliTom, VisionZip, uniform pooling and random drop at 5%, 10% and 25% of the 10,816 video tokens, plus the same four harmless noise passes as the main study.

**Two safety checks stop the run before wasting GPU hours:** (1) our token sequence through the model's own forward vs our prefill and our own decoder loop, (2) cached vs uncached answers on the first video.

## How to run
1. Settings → Accelerator **GPU T4 x2**, Internet **On**, HF_TOKEN in Add-ons → Secrets (already set).
2. `BENCHMARK = "videomme"`, `QUICK_TEST = True` → **Run All** (~40 min, uses the real model on 3 questions). Check: `Converted weights loaded … ✅`, `Compressor self-test OK ✅`, equivalence check `PASS`, `Cache check: PASS`, a few ✓ lines.
3. Then `QUICK_TEST = False` → **Save Version → Save & Run All (Commit)**, one benchmark per run (three runs).
4. If a run stops at the 12-hour limit: **Add Input** → that version's output, and run again. It continues where it stopped.
5. Send back from `flips_rep_llava_video_<benchmark>/`: `summary_<benchmark>.json`, `cells.csv`, `summary_<benchmark>.png`, `results.jsonl`, `cache_check.json`, `equivalence_check.json`, `manifest_<benchmark>.json`, `run_record_*.json` (+ `errors.log` if not empty).
