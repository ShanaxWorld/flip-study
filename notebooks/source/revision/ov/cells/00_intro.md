# Answer-flip study — revision run, LLaVA-OneVision-7B (Run-All)

This notebook adds two things to the main study, on exactly the same questions (same sampling seed and manifests):

1. **V-CAST** (NeurIPS 2026), the strongest recent training-free method, at 5%, 10% and 25% of the visual tokens.
2. **Six more harmless passes** of the full model: frames shifted by ⅛ and ⅜ of a sampling interval, two further fixed option orders, and two rewordings of the instruction text.

The full model is run again so that its answers can be checked against the main study (they should match). Each run handles **one** benchmark, chosen in the config cell.

| `BENCHMARK` | Questions | Expected time (T4 ×2) |
|---|---|---|
| `"videomme"` | 639 | about 1.5–2 h |
| `"mvbench"` | 475 | about 2–3 h |
| `"egoschema"` | 500 | about 2–3 h |
| `"longvideobench"` | 500 | about 2–3 h |
| `"mlvu"` | 504 | about 2–3 h |

The same two safety checks as in the main study stop the run before wasting GPU hours: (1) our pipeline vs the official model forward, (2) cached vs uncached answers on the first video (now including a V-CAST pass).

## How to run
1. Settings → Accelerator **GPU T4 x2**, Internet **On**, HF_TOKEN in Add-ons → Secrets (same as before).
2. `BENCHMARK = "videomme"`, `QUICK_TEST = True` → **Run All** (~25 min). Check: `Compressor self-test OK ✅`, equivalence check `PASS`, `Cache check: PASS`, a few ✓ lines.
3. Then `QUICK_TEST = False` → **Save Version → Save & Run All (Commit)**, one benchmark per run.
4. If a run stops at the time limit: open the next version, **Add Input** → the previous version's output, and run again. It continues where it stopped.
5. Send back only `results.jsonl` from `flips_ext_<benchmark>/` (and `errors.log` if it is not empty).
