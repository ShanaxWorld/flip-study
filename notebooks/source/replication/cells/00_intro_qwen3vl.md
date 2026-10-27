# Answer-flip study — replication on Qwen3-VL-8B (Run-All)

A secondary analysis from the study plan: the same measures on a third model, from a different family, so H1, H2a, H2b and H3 can be checked for replication. Each run handles **one** benchmark, chosen in the config cell.

| `BENCHMARK` | Questions | Expected time | Videos downloaded |
|---|---|---|---|
| `"videomme"` | 300 (100 videos) | 1.5–2 h | 5 × 5 GB zips |
| `"egoschema"` | 300 | 3–4 h | 5 × 21 GB zips (only the needed videos are kept) |
| `"longvideobench"` | 300 (75 per duration group) | 3–4 h | 31 × 5 GB parts, streamed |

**Model.** Qwen3-VL-8B-Instruct in FlashVID's evaluation setting: 32 frames (16 temporal groups of 2 frames, each with a timestamp), at most 448 × 448 pixels per frame (FlashVID's baseline script), 3-D positions and DeepStack features handled exactly as in the official model.

**Methods.** FlashVID (official Qwen3-VL settings: 4 minimum segments, ADTS v1, pruning to 10% at layer 28), uniform pooling and random drop at 5%, 10% and 25%, plus the same four harmless noise passes as the main study.

**Two safety checks stop the run before wasting GPU hours:** (1) our pipeline vs the official processor + model forward, (2) cached vs uncached answers on the first video.

## How to run
1. Settings → Accelerator **GPU T4 x2**, Internet **On**, HF_TOKEN in Add-ons → Secrets (already set).
2. `BENCHMARK = "videomme"`, `QUICK_TEST = True` → **Run All** (~30 min, uses the real model on 3 questions). Check: `Compressor self-test OK ✅`, equivalence check `PASS`, `Cache check: PASS`, a few ✓ lines.
3. Then `QUICK_TEST = False` → **Save Version → Save & Run All (Commit)**, one benchmark per run (three runs).
4. If a run stops at the 12-hour limit: **Add Input** → that version's output, and run again. It continues where it stopped.
5. Send back from `flips_rep_qwen3vl_<benchmark>/`: `summary_<benchmark>.json`, `cells.csv`, `summary_<benchmark>.png`, `results.jsonl`, `cache_check.json`, `equivalence_check.json`, `manifest_<benchmark>.json`, `run_record_*.json` (+ `errors.log` if not empty).
