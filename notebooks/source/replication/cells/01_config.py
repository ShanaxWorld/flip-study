# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG — change only BENCHMARK and QUICK_TEST                    ║
# ╚══════════════════════════════════════════════════════════════════╝
BENCHMARK  = "videomme"   # one of: "videomme", "egoschema", "longvideobench"
QUICK_TEST = True         # True = ~30-min smoke test (real model, 4 questions). False = the real run.

MODEL = "__MODEL__"       # fixed for this notebook: "llava_video" or "qwen3vl"

# ── Replication sample: 300 questions per benchmark, drawn once (seed 20261015) from the main study's manifests.
#    Video-MME: 100 of the 213 confirmatory videos (all their questions); EgoSchema: 300 of 500;
#    LongVideoBench: 75 per duration group. The question IDs are fixed in REP_MANIFEST below (with their hash).
SAMPLE_SEED = 20261015
VIDEOMME_CONFIRM_CHUNKS = ["02", "08", "16", "18", "20"]
LVB_PER_GROUP = 125
MVBENCH_PER_TASK, MLVU_PER_TASK = 25, 72          # not used here (kept so the shared data code runs unchanged)
PREREG_URL = "https://claude.ai/code/artifact/65cb336e-3916-4101-9914-8dd1ca98e189"

if MODEL == "llava_video":
    MODEL_ID = "lmms-lab/LLaVA-Video-7B-Qwen2"          # original weights, converted to the transformers format below
    HF_BASE  = "llava-hf/llava-onevision-qwen2-7b-ov-hf"  # same architecture; provides the transformers config
    NUM_FRAMES = 64                                     # FlashVID's LLaVA-Video setting (scripts/llava_vid.sh)
    METHOD_GRID = {"flashvid": [0.05, 0.10, 0.25], "holitom": [0.05, 0.10, 0.25], "visionzip": [0.05, 0.10, 0.25],
                   "uniform": [0.05, 0.10, 0.25], "random": [0.05, 0.10, 0.25]}
    FLASHVID = dict(alpha=0.7, token_selection_method="attn_div_v2", temporal_threshold=0.8, do_segment=True,
                    segment_threshold=0.9, min_segment_num=8, complementary_segment=True,
                    expansion=1.25, pruning_layer=20, llm_retention_ratio=0.3)
    HOLITOM = dict(T={0.05: 0.65, 0.10: 0.65, 0.25: 0.80}, beta=0.6, D=0.0, K=7, max_window=1024,
                   inner_layer=18, inner_r=0.5)
else:
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    NUM_FRAMES = 32                                     # FlashVID's Qwen3-VL setting (scripts/qwen3_vl.sh)
    MIN_PIXELS, MAX_PIXELS = 50716, 200704              # per frame, as in FlashVID's Qwen3-VL baseline script
    METHOD_GRID = {"flashvid": [0.05, 0.10, 0.25], "uniform": [0.05, 0.10, 0.25], "random": [0.05, 0.10, 0.25]}
    FLASHVID = dict(alpha=0.7, token_selection_method="attn_div", temporal_threshold=0.8, do_segment=True,
                    segment_threshold=0.9, min_segment_num=4, complementary_segment=True,
                    expansion=1.25, pruning_layer=28, llm_retention_ratio=0.1)

# ── Four noise passes of the full model (as in the main study) ──
NOISE_FRAME_SHIFTS = {"frame_shift": 0.5, "frame_shift_q": 0.25}   # fraction of one sampling interval
NOISE_OPTION_ORDERS = ["option_cycle", "option_reverse"]

SEED = 0
MAX_RUNTIME_HOURS = 11.0
DATA_CHECK_ONLY = False
OUT_DIR  = f"/kaggle/working/flips_rep_{MODEL}_{BENCHMARK}"
WORK_DIR = "/tmp/astra_data"
QUICK_MAX_Q = 4 if QUICK_TEST else None
if QUICK_TEST:
    OUT_DIR = OUT_DIR + "_quick"
assert BENCHMARK in ("videomme", "egoschema", "longvideobench"), f"Unknown BENCHMARK {BENCHMARK}"
print("Model:", MODEL_ID, "| benchmark:", BENCHMARK, "| mode:", "QUICK TEST" if QUICK_TEST else "REAL RUN",
      "| frames:", NUM_FRAMES)
