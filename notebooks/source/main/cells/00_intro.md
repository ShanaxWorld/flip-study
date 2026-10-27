# Answer-flip study — main run, LLaVA-OneVision-7B (Run-All)

One notebook for all five benchmarks. Each run handles **one** benchmark, chosen in the config cell. Everything it does follows the pre-registration (sampling seeds, methods, budgets, four noise passes).

| `BENCHMARK` | Questions | Expected time | Videos downloaded |
|---|---|---|---|
| `"videomme"` | about 810 (chunks 02, 08, 16, 18, 20) | 6–7 h | 5 × 5 GB zips |
| `"mvbench"` | about 475 (25 per task, 19 tasks) | 4–6 h | one small file per video |
| `"egoschema"` | 500 | 6–7 h | 5 × 21 GB zips (only the needed videos are kept) |
| `"longvideobench"` | 500 (125 per duration group) | 6–8 h | 31 × 5 GB parts, streamed one at a time |
| `"mlvu"` | 504 (72 per task, 7 tasks) | 6–8 h | one file per video |

**What happens per video:** 3 frame sets are decoded (original, and shifted by ½ and ¼ of a sampling interval), the 23 compressed versions are built once, and the system prompt + video tokens are run through the model **once per version**. Every question and option order then reuses that cached prefix. FlashVID and HoliTom act inside the model and depend on the question, so they run in full per question.

**Two safety checks stop the run before wasting GPU hours:** (1) our pipeline vs the official model forward, (2) cached vs uncached answers on the first video.

## How to run
1. Settings → Accelerator **GPU T4 x2**, Internet **On**, HF_TOKEN in Add-ons → Secrets (already set; MVBench, LongVideoBench and MLVU terms accepted).
2. First time only, a free CPU check of all five datasets: `DATA_CHECK_ONLY = True`, Accelerator **None** → **Run All** (~30–40 min, mostly one 21 GB EgoSchema download). It stops after cell 3 with "Data check finished" — that stop is expected. Send me `data_check.json`.
3. Then `DATA_CHECK_ONLY = False`, Accelerator GPU, `BENCHMARK = "videomme"`, `QUICK_TEST = True` → **Run All** (~25 min). Check: `Compressor self-test OK ✅`, equivalence check `PASS`, `Cache check: PASS`, a few ✓ lines.
4. Then `QUICK_TEST = False` → **Save Version → Save & Run All (Commit)**, one benchmark per run.
5. If a run stops at the 12-hour limit: open the next version, **Add Input** → the previous version's output, and run again. It continues where it stopped.
6. Send back from `flips_main_<benchmark>/`: `summary_<benchmark>.json`, `cells.csv`, `summary_<benchmark>.png`, `results.jsonl`, `cache_check.json`, `equivalence_check.json`, `manifest_<benchmark>.json`, `run_record_*.json` (+ `errors.log` if not empty).
