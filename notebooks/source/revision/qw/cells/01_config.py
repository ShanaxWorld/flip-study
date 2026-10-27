# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG — change only BENCHMARK and QUICK_TEST                    ║
# ╚══════════════════════════════════════════════════════════════════╝
BENCHMARK  = "videomme"   # one of: "videomme", "egoschema", "longvideobench"
QUICK_TEST = True         # True = ~30-min smoke test (real model, 4 questions). False = the real run.

MODEL = "qwen3vl"         # fixed for this notebook

# ── Replication sample: 300 questions per benchmark, drawn once (seed 20261015) from the main study's manifests.
#    Video-MME: 100 of the 213 confirmatory videos (all their questions); EgoSchema: 300 of 500;
#    LongVideoBench: 75 per duration group. The question IDs are fixed in REP_MANIFEST below (with their hash).
SAMPLE_SEED = 20261015
VIDEOMME_CONFIRM_CHUNKS = ["02", "08", "16", "18", "20"]
LVB_PER_GROUP = 125
MVBENCH_PER_TASK, MLVU_PER_TASK = 25, 72          # not used here (kept so the shared data code runs unchanged)
PREREG_URL = "https://claude.ai/code/artifact/65cb336e-3916-4101-9914-8dd1ca98e189"

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
NUM_FRAMES = 32                                     # as in the replication run (FlashVID's Qwen3-VL setting)
MIN_PIXELS, MAX_PIXELS = 50716, 200704              # per frame, as in the replication run
# Revision run: V-CAST (NeurIPS 2026) with its OFFICIAL Qwen3-VL code, on the same 900 questions as the replication.
METHOD_GRID = {"vcast": [0.05, 0.10, 0.25]}
VCAST = dict(min_k=1)                               # official defaults (temperature 0.7 is fixed inside the code)

# ── No new noise passes here: the four noise passes of these questions are already in the replication results ──
NOISE_FRAME_SHIFTS = {}
NOISE_OPTION_ORDERS = []

SEED = 0
MAX_RUNTIME_HOURS = 11.0
DATA_CHECK_ONLY = False
OUT_DIR  = f"/kaggle/working/flips_ext_{MODEL}_{BENCHMARK}"
WORK_DIR = "/tmp/astra_data"
QUICK_MAX_Q = 4 if QUICK_TEST else None
if QUICK_TEST:
    OUT_DIR = OUT_DIR + "_quick"
assert BENCHMARK in ("videomme", "egoschema", "longvideobench"), f"Unknown BENCHMARK {BENCHMARK}"
print("Model:", MODEL_ID, "| benchmark:", BENCHMARK, "| mode:", "QUICK TEST" if QUICK_TEST else "REAL RUN",
      "| frames:", NUM_FRAMES)
