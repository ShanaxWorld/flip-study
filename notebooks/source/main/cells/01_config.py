# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG — change only BENCHMARK and QUICK_TEST                    ║
# ╚══════════════════════════════════════════════════════════════════╝
BENCHMARK  = "mlvu"       # one of: "videomme", "mvbench", "egoschema", "longvideobench", "mlvu"
QUICK_TEST = False        # True = ~25-min smoke test (tiny model, a few questions). False = the real run.
DATA_CHECK_ONLY = False   # True = CPU-only check of all five benchmarks' data (no GPU needed); stops after cell 3.

MODEL_ID_REAL  = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
MODEL_ID_QUICK = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
SIGLIP_ID      = "google/siglip-so400m-patch14-384"   # FastVID uses the original SigLIP pooling head
NUM_FRAMES     = 32

# ── Pre-registered sampling (do not change; see the pre-registration document) ──
SAMPLE_SEED = 20261015
VIDEOMME_CONFIRM_CHUNKS = ["02", "08", "16", "18", "20"]   # drawn at random from the 10 never-piloted chunks
MVBENCH_PER_TASK = 25                                       # 19 tasks (fine_grained_pose = NTU, excluded)
LVB_PER_GROUP = 125                                         # 4 duration groups
MLVU_PER_TASK = 72                                          # 7 multiple-choice tasks
PREREG_URL = "https://claude.ai/code/artifact/65cb336e-3916-4101-9914-8dd1ca98e189"

# ── Methods (23 compressed passes) ──
METHOD_GRID = {
    "flashvid":  [0.05, 0.10, 0.25],
    "fastvid":   [0.05, 0.10, 0.25],
    "holitom":   [0.05, 0.10, 0.25],
    "visionzip": [0.05, 0.10, 0.25],
    "scissor":   [0.05, 0.10, 0.25],
    "dycoke":    [0.9, 1.0],            # TTM pruning rate K (its lowest budgets: 35% and 28%)
    "uniform":   [0.05, 0.10, 0.25],
    "random":    [0.05, 0.10, 0.25],
}
FASTVID  = dict(c=8, tau=0.9, d=0.4, p=4, beta=0.6)
FLASHVID = dict(alpha=0.7, token_selection_method="attn_div_v2", temporal_threshold=0.8, do_segment=True,
                segment_threshold=0.9, min_segment_num=8, complementary_segment=True,
                expansion=1.25, pruning_layer=20, llm_retention_ratio=0.3)
HOLITOM  = dict(T={0.05: 0.65, 0.10: 0.65, 0.25: 0.80}, beta=0.6, D=0.0, K=7, max_window=1024,
                inner_layer=18, inner_r=0.5)
SCISSOR  = dict(epsilon=0.05, search_steps=12, refine_steps=6)

# ── Four noise passes of the full model ──
NOISE_FRAME_SHIFTS = {"frame_shift": 0.5, "frame_shift_q": 0.25}   # fraction of one sampling interval
NOISE_OPTION_ORDERS = ["option_cycle", "option_reverse"]

SEED = 0
MAX_RUNTIME_HOURS = 11.0
OUT_DIR  = f"/kaggle/working/flips_main_{BENCHMARK}"
WORK_DIR = "/tmp/astra_data"

if QUICK_TEST:
    MODEL_ID = MODEL_ID_QUICK
    NUM_FRAMES = 16
    VIDEOMME_CONFIRM_CHUNKS = ["02"]
    QUICK_MAX_Q = 6
    OUT_DIR = OUT_DIR + "_quick"
else:
    MODEL_ID = MODEL_ID_REAL
    QUICK_MAX_Q = None
assert BENCHMARK in ("videomme", "mvbench", "egoschema", "longvideobench", "mlvu"), f"Unknown BENCHMARK {BENCHMARK}"
print("Benchmark:", BENCHMARK, "| mode:", "QUICK TEST" if QUICK_TEST else "REAL RUN", "| model:", MODEL_ID,
      "| frames:", NUM_FRAMES)
