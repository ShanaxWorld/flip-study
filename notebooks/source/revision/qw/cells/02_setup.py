# ── Setup: folders, environment, small installs ──────────────────────
import os, sys, subprocess
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(WORK_DIR, "hf_home")      # keep HF cache on the big temp disk
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pip_install(*pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "WARN ") + "pip install " + " ".join(pkgs))

pip_install("hf_xet")   # faster Hugging Face downloads (harmless if already present)
pip_install("av")       # backup video decoder

# Same library version as the main study, so every model runs on identical library code.
# (Checked before transformers is imported, so no restart is needed.)
from importlib.metadata import version as _v, PackageNotFoundError
TRANSFORMERS_PIN = "5.16.1"
try:
    _have = _v("transformers")
except PackageNotFoundError:
    _have = None
if _have != TRANSFORMERS_PIN:
    print(f"transformers {_have} found -> installing {TRANSFORMERS_PIN} (the version used in the main study)")
    pip_install(f"transformers=={TRANSFORMERS_PIN}")
    assert _v("transformers") == TRANSFORMERS_PIN, "transformers pin failed — send me this log"
print("transformers", _v("transformers"), "| torch", _v("torch"), "| python", sys.version.split()[0])

if MODEL == "qwen3vl":
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
else:
    from transformers import LlavaOnevisionForConditionalGeneration  # noqa: F401
print("transformers OK")
