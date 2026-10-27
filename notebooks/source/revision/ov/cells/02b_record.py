# ── Run record + resume from an earlier session ──────────────────────
import json, datetime, glob, shutil
RES_PATH = os.path.join(OUT_DIR, "results.jsonl")
ERR_PATH = os.path.join(OUT_DIR, "errors.log")
RUN_RECORD = {
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "benchmark": BENCHMARK, "model": MODEL_ID, "frames": NUM_FRAMES, "quick_test": QUICK_TEST,
    "versions": {"transformers": _v("transformers"), "torch": _v("torch"), "python": sys.version.split()[0]},
    "analysis_plan": PREREG_URL, "sample_seed": SAMPLE_SEED,
    "videomme_chunks": VIDEOMME_CONFIRM_CHUNKS, "mvbench_per_task": MVBENCH_PER_TASK,
    "lvb_per_group": LVB_PER_GROUP, "mlvu_per_task": MLVU_PER_TASK,
    "study": "revision", "methods": METHOD_GRID, "vcast": VCAST,
    "noise": {"frame_shifts": NOISE_FRAME_SHIFTS, "option_orders": NOISE_OPTION_ORDERS, "prompts": NOISE_PROMPTS},
}
with open(os.path.join(OUT_DIR, f"run_record_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"), "w") as f:
    json.dump(RUN_RECORD, f, indent=2)

# If a previous session's results.jsonl for this benchmark is attached (Add Input), continue from it.
if not os.path.exists(RES_PATH):
    # matches e.g. .../flips_ext_mlvu/results.jsonl or a dataset named "flips-ext-mlvu" (never the main study's results)
    prev = [p for p in glob.glob("/kaggle/input/**/results.jsonl", recursive=True)
            if f"ext_{BENCHMARK}" in p.lower().replace("-", "_") and "quick" not in p.lower()]
    if prev:
        shutil.copy(prev[0], RES_PATH)
        print("Resuming from", prev[0])
print(json.dumps(RUN_RECORD, indent=1)[:1500])
