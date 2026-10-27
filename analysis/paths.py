"""Repository paths used by every analysis script."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "data" / "main")             # LLaVA-OneVision-7B, five benchmarks
REP = str(ROOT / "data" / "replication")       # LLaVA-Video-7B, Qwen3-VL-8B (+ LLaVA-OV on the shared questions)
MANIFESTS = str(ROOT / "data" / "manifests")
OUT = str(ROOT / "results")
FIG = str(ROOT / "figures")
