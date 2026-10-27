# Answer-flip study — revision run, Qwen3-VL-8B with V-CAST (Run-All)

This notebook adds **V-CAST** (NeurIPS 2026) at 5%, 10% and 25% of the visual tokens to the Qwen3-VL-8B replication, using the **official V-CAST code for Qwen3-VL**, on exactly the same 900 questions (300 each from Video-MME, EgoSchema and LongVideoBench). The four noise passes of these questions are already in the replication results, so only the full pass (to check that its answers match) and the three V-CAST passes are run.

| `BENCHMARK` | Questions | Expected time (T4 ×2) |
|---|---|---|
| `"videomme"` | 300 | about 1 h |
| `"egoschema"` | 300 | about 1 h |
| `"longvideobench"` | 300 | about 1–1.5 h |

## How to run
1. Settings → Accelerator **GPU T4 x2**, Internet **On**, HF_TOKEN in Add-ons → Secrets (same as before).
2. `BENCHMARK = "videomme"`, `QUICK_TEST = True` → **Run All** (~20 min). Check: `Compressor self-test OK ✅`, equivalence check `PASS`, `Cache check: PASS`, a few ✓ lines.
3. Then `QUICK_TEST = False` → **Save Version → Save & Run All (Commit)**, one benchmark per run.
4. Send back only `results.jsonl` from `flips_ext_qwen3vl_<benchmark>/` (and `errors.log` if it is not empty).
