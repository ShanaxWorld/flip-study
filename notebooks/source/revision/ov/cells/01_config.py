# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG — change only BENCHMARK and QUICK_TEST                    ║
# ╚══════════════════════════════════════════════════════════════════╝
BENCHMARK  = "videomme"   # one of: "videomme", "mvbench", "egoschema", "longvideobench", "mlvu"
QUICK_TEST = True         # True = ~25-min smoke test (tiny model, a few questions). False = the real run.
DATA_CHECK_ONLY = False   # True = CPU-only check of all five benchmarks' data (no GPU needed); stops after cell 3.

MODEL_ID_REAL  = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
MODEL_ID_QUICK = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
SIGLIP_ID      = "google/siglip-so400m-patch14-384"   # kept so the shared encoder code runs unchanged
NUM_FRAMES     = 32

# ── The main study's sampling (do not change: the run must reach exactly the same questions) ──
SAMPLE_SEED = 20261015
VIDEOMME_CONFIRM_CHUNKS = ["02", "08", "16", "18", "20"]   # drawn at random from the 10 never-piloted chunks
MVBENCH_PER_TASK = 25                                       # 19 tasks (fine_grained_pose = NTU, excluded)
LVB_PER_GROUP = 125                                         # 4 duration groups
MLVU_PER_TASK = 72                                          # 7 multiple-choice tasks
PREREG_URL = "https://claude.ai/code/artifact/65cb336e-3916-4101-9914-8dd1ca98e189"

# ── Revision run: one more method and six more harmless passes on the SAME questions as the main study ──
METHOD_GRID = {"vcast": [0.05, 0.10, 0.25]}       # V-CAST (NeurIPS 2026), the strongest recent training-free baseline
VCAST = dict(min_k=1)                               # official defaults (temperature 0.7 is fixed inside the code)

# New harmless passes of the full model (the four original passes are already in the main study's results)
NOISE_FRAME_SHIFTS = {"frame_shift_e": 0.125, "frame_shift_3e": 0.375}   # 1/8 and 3/8 of one sampling interval
NOISE_OPTION_ORDERS = ["option_perm1", "option_perm2"]                    # two further fixed option permutations
NOISE_PROMPTS = ["prompt_alt1", "prompt_alt2"]                            # two rewordings of the instruction text

SEED = 0
MAX_RUNTIME_HOURS = 11.0
OUT_DIR  = f"/kaggle/working/flips_ext_{BENCHMARK}"
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
