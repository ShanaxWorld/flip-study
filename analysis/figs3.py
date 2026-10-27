"""Revision figure: sensitivity to the margin δ and power to certify noise-equivalence."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
exec(open("figs.py").read().split("# ---------- Fig. 1")[0])
S = pd.read_csv(f"{OUT}/delta_sensitivity.csv", dtype={"setting": str}); P = pd.read_csv(f"{OUT}/power.csv", dtype={"setting": str})
MC = {"llava_ov": PAL[0], "llava_video": PAL[1], "qwen3vl": PAL[2]}; MK_ = {"llava_ov": "o", "llava_video": "s", "qwen3vl": "D"}
MN_ = {"llava_ov": "LLaVA-OV-7B", "llava_video": "LLaVA-Video-7B", "qwen3vl": "Qwen3-VL-8B"}
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9))
ax = axs[0]; ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for m in MC:
    d = S[(S.model == m) & (S.setting == "0.25")].sort_values("delta")
    ax.plot(100 * d.delta, d.noise_equivalent / d.cells, color=MC[m], lw=1.1, marker=MK_[m], ms=3.5, label=MN_[m])
ax.axhline(0.7, color=INK2, lw=0.7, ls=(0, (1, 2))); ax.axvline(3, color=GRID, lw=1.2, zorder=0)
ax.set_ylim(0, 1); ax.set_xlabel("margin δ (points)"); ax.set_ylabel("25% cells noise-equivalent", fontsize=6.5)
ax.text(2.05, 0.73, "H2b rule (70%)", fontsize=5.6, color=INK2)
ax.legend(fontsize=5.4, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
ax = axs[1]; ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
se500 = float(P[(P.model == "llava_ov") & (P.setting == "0.25")].apply(lambda r: r.se * np.sqrt(r.n / 500), axis=1).median())
n = np.linspace(100, 3000, 300)
for delta, c in [(0.02, PAL[1]), (0.03, INK), (0.05, PAL[2])]:
    ax.plot(n, norm.cdf(delta / (se500 * np.sqrt(500 / n)) - 1.96), color=c, lw=1.1, label=f"δ = {round(100 * delta)} points")
ax.axvline(500, color=GRID, lw=1.2, zorder=0); ax.axhline(0.8, color=INK2, lw=0.7, ls=(0, (1, 2)))
ax.set_ylim(0, 1); ax.set_xlabel("questions per benchmark"); ax.set_ylabel("power (true Δ = 0)", fontsize=6.5)
ax.legend(fontsize=5.4, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(w_pad=0.8); save(fig, "figF_margin_power")
print("SE at n=500 (LLaVA-OV, 25%, median over benchmarks):", round(se500, 4))
