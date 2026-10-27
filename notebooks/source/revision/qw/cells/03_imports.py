# ── Imports + hardware check ──────────────────────────────────────────
import json, math, time, random, shutil, zipfile, gc, heapq
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files
import transformers
from transformers import AutoTokenizer

print("torch", torch.__version__, "| transformers", transformers.__version__)
N_GPU = torch.cuda.device_count()
assert N_GPU > 0 or DATA_CHECK_ONLY, "No GPU found! Right panel > Settings > Accelerator > 'GPU T4 x2', then Run All again."
for i in range(N_GPU):
    print(f"GPU{i}: {torch.cuda.get_device_name(i)}")
free_gb = shutil.disk_usage(WORK_DIR).free / 1e9
print(f"Free disk for temp videos: {free_gb:.1f} GB")
if free_gb < 12:
    print("⚠️ Less than 12 GB free — a 5 GB video chunk needs ~11 GB while unzipping.")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Optional: a Hugging Face token stored as a Kaggle Secret named HF_TOKEN (Add-ons > Secrets).
# Not needed for these public repos, but harmless.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded from Kaggle Secrets")
except Exception:
    print("No HF_TOKEN secret (fine for public repos)")
