"""Fig. 3 (excess flips by budget) and Fig. 7 (three models) redrawn with V-CAST, the method added after the
confirmatory analysis (run after ext_ov.py and ext_qw.py).
They replace the versions written by figs.py and figs2.py."""
from paths import ROOT, MAIN, REP, MANIFESTS  # noqa: F401
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("figs.py").read().split("# ---------- Fig. 1")[0])
MNAME = dict(MNAME, vcast="V-CAST (post hoc)")
ORDER = ORDER[:7] + ["vcast", "dycoke"]
COL["vcast"] = INK; MK["vcast"] = "h"

# ---------- Fig. 3: excess flips by budget, nine methods ----------
df = pd.concat([df, pd.read_csv(f"{OUT}/vcast_ov_cells.csv", dtype={"setting": str})])
pool = pd.read_csv(f"{OUT}/pooled_methods_with_vcast.csv", dtype={"setting": str})
fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.4), sharey=True)
panels = [(b, df[df.benchmark == b]) for b in BENCH] + [("pooled", pool)]
for ax, (b, d) in zip(axs.flat, panels):
    axfmt(ax)
    for i, m in enumerate(ORDER):
        g = d[d.method == m].copy(); g["x"] = g.setting.map(XPOS) * (1 + 0.032 * (i - 4))
        g = g.sort_values("x")
        ls = (0, (3, 1.5)) if m == "vcast" else "-"
        ax.plot(g.x, g.excess, color=COL[m], lw=1.0, alpha=0.9, ls=ls)
        ax.errorbar(g.x, g.excess, yerr=[g.excess - g.lo, g.hi - g.excess], fmt="none", ecolor=COL[m], elinewidth=0.8, capsize=0)
        ax.scatter(g.x, g.excess, s=34 if m == "dycoke" else 16, marker=MK[m], color=COL[m], edgecolors=SURF, linewidths=0.5,
                   zorder=3, label=MNAME[m])
    ax.set_title(BNAME.get(b, "Pooled over 5 benchmarks (post hoc)"), fontsize=8, color=INK, loc="left")
for ax in axs[1]: ax.set_xlabel("visual tokens kept (%)")
for ax in axs[:, 0]: ax.set_ylabel("excess flips Δ")
axs[0, 0].text(4.4, DELTA + 0.006, "δ", ha="left", va="bottom", fontsize=7, color=INK2)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=9, bbox_to_anchor=(0.5, -0.035), fontsize=6.4, handletextpad=0.2, columnspacing=0.7)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig1_excess_by_budget")

# ---------- Fig. 7: same questions, three models, with V-CAST ----------
MC = {"llava_ov": PAL[0], "llava_video": PAL[1], "qwen3vl": PAL[2]}
MN = {"llava_ov": "LLaVA-OneVision-7B", "llava_video": "LLaVA-Video-7B", "qwen3vl": "Qwen3-VL-8B"}
MMK = {"llava_ov": "o", "llava_video": "s", "qwen3vl": "D"}
X = pd.read_csv(f"{OUT}/xmodel_pooled_with_vcast.csv", dtype={"setting": str})
xpos = {"0.05": 5, "0.1": 10, "0.25": 25}
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.0), sharey=True)
for ax, m in zip(axs, ["flashvid", "random", "uniform", "vcast"]):
    ax.set_xscale("log"); ax.set_xticks([5, 10, 25]); ax.set_xticklabels(["5", "10", "25"]); ax.minorticks_off(); ax.set_xlim(4.2, 30)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.axhline(0, color=INK2, lw=0.7)
    ax.axhline(DELTA, color=INK2, lw=0.7, ls=(0, (1, 2)))
    for i, mod in enumerate(["llava_ov", "llava_video", "qwen3vl"]):
        g = X[(X.model == mod) & (X.method == m)].copy()
        g["x"] = g.setting.map(xpos) * (1 + 0.05 * (i - 1)); g = g.sort_values("x")
        ax.plot(g.x, g.excess, color=MC[mod], lw=1.0)
        ax.errorbar(g.x, g.excess, yerr=[g.excess - g.lo, g.hi - g.excess], fmt="none", ecolor=MC[mod], elinewidth=0.8)
        ax.scatter(g.x, g.excess, s=18, marker=MMK[mod], color=MC[mod], edgecolors=SURF, linewidths=0.5, zorder=3, label=MN[mod])
    ax.set_title({"vcast": "V-CAST (post hoc)"}.get(m, MNAME[m]), fontsize=8, color=INK, loc="left")
    ax.set_xlabel("visual tokens kept (%)")
axs[0].set_ylabel("excess flips Δ"); h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "figC_models")
print("figs4 done")
