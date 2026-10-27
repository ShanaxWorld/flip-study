from paths import ROOT, MAIN, REP, MANIFESTS  # noqa: F401
import json, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import BENCH, BNAME, MNAME, MAIN7, NK, DELTA
from paths import OUT, FIG
res = json.load(open(f"{OUT}/results_cross.json"))
df = pd.read_csv(f"{OUT}/cells_all.csv", dtype={"setting": str})
pool = pd.read_csv(f"{OUT}/pooled_methods.csv", dtype={"setting": str})
acc = pd.read_csv(f"{OUT}/accuracy_tests.csv", dtype={"setting": str})

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDER = ["flashvid", "fastvid", "holitom", "visionzip", "scissor", "uniform", "random", "dycoke"]
COL = dict(zip(ORDER, PAL)); MK = dict(zip(ORDER, ["o", "s", "D", "^", "v", "P", "X", "*"]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})
XPOS = {"0.05": 5, "0.1": 10, "0.25": 25, "K1.0": 28, "K0.9": 35}
def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight"); plt.close(fig)
def axfmt(ax):
    ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 35]); ax.set_xticklabels(["5", "10", "25", "35"])
    ax.minorticks_off(); ax.set_xlim(4.2, 42)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.axhline(0, color=INK2, lw=0.7); ax.axhline(DELTA, color=INK2, lw=0.7, ls=(0, (1, 2)))

# ---------- Fig. 1: excess flips by budget ----------
fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.3), sharey=True)
panels = [(b, df[df.benchmark == b]) for b in BENCH] + [("pooled", pool)]
for ax, (b, d) in zip(axs.flat, panels):
    axfmt(ax)
    for i, m in enumerate(ORDER):
        g = d[d.method == m].copy(); g["x"] = g.setting.map(XPOS) * (1 + 0.035 * (i - 3.5))
        g = g.sort_values("x")
        ax.plot(g.x, g.excess, color=COL[m], lw=1.0, alpha=0.9)
        ax.errorbar(g.x, g.excess, yerr=[g.excess - g.lo, g.hi - g.excess], fmt="none", ecolor=COL[m], elinewidth=0.8, capsize=0)
        ax.scatter(g.x, g.excess, s=34 if m == "dycoke" else 16, marker=MK[m], color=COL[m], edgecolors=SURF, linewidths=0.5, zorder=3, label=MNAME[m])
    t = BNAME.get(b, "Pooled over 5 benchmarks (post hoc)")
    ax.set_title(t, fontsize=8, color=INK, loc="left")
for ax in axs[1]: ax.set_xlabel("visual tokens kept (%)")
for ax in axs[:, 0]: ax.set_ylabel("excess flips Δ")
axs[0, 0].text(4.4, DELTA + 0.006, "δ", ha="left", va="bottom", fontsize=7, color=INK2)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.035), fontsize=7, handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig1_excess_by_budget")

# ---------- Fig. 2: accuracy change vs excess flips ----------
m = df.merge(acc[["benchmark", "method", "setting", "mcnemar_p", "acc_drop_pts"]], on=["benchmark", "method", "setting"])
CCOL = {"lossy": PAL[1], "inconclusive": PAL[0], "noise-equivalent": PAL[2]}
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.axhline(0, color=INK2, lw=0.7); ax.axhline(DELTA, color=INK2, lw=0.7, ls=(0, (1, 2))); ax.axvline(0, color=INK2, lw=0.7)
for c in ["noise-equivalent", "inconclusive", "lossy"]:
    g = m[m.cls == c]; sig = g.mcnemar_p < 0.05
    ax.scatter(-g.acc_drop_pts[sig], g.excess[sig], s=18, color=CCOL[c], edgecolors=SURF, linewidths=0.5, zorder=3)
    ax.scatter(-g.acc_drop_pts[~sig], g.excess[~sig], s=18, facecolors=SURF, edgecolors=CCOL[c], linewidths=1.0, zorder=3)
    ax.scatter([], [], s=18, color=CCOL[c], label=c)
ax.set_xlabel("accuracy change vs full model (points)"); ax.set_ylabel("excess flips Δ")
n_ns = res["accuracy_vs_fidelity"]["lossy_acc_not_significant"]; n_l = res["accuracy_vs_fidelity"]["lossy_cells"]
ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.1, columnspacing=1.0)
ax.text(0.98, 0.97, f"filled: accuracy change significant\nhollow: not significant (McNemar)\n\n{n_ns} of {n_l} lossy cells are hollow",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=INK2)
save(fig, "fig2_accuracy_vs_fidelity")

# ---------- Fig. 3: direction (harmful share) ----------
H3 = res["H3"]; rows = [(BNAME[b], v["lossy_hs"], v["noise_hs"], v["diff"], v["ci"]) for b, v in H3["per_benchmark"].items()]
rows.append(("Pooled", H3["lossy_harmful_share"], H3["noise_harmful_share"], H3["diff"], H3["ci"]))
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.0), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
y = np.arange(len(rows))[::-1]
ax = axs[0]; ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for yi, r in zip(y, rows):
    ax.plot([r[2], r[1]], [yi, yi], color=GRID, lw=2, zorder=1)
ax.scatter([r[2] for r in rows], y, s=20, color=PAL[0], edgecolors=SURF, linewidths=0.5, zorder=3, label="noise passes")
ax.scatter([r[1] for r in rows], y, s=20, color=PAL[1], edgecolors=SURF, linewidths=0.5, zorder=3, label="lossy cells")
ax.axvline(0.5, color=INK2, lw=0.7); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("harmful share"); ax.set_xlim(0.44, 0.70)
ax.legend(fontsize=6.3, loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.8)
ax = axs[1]; ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for yi, r in zip(y, rows):
    ax.plot(r[4], [yi, yi], color=INK2, lw=1.0)
    ax.scatter([r[3]], [yi], s=20 if r[0] != "Pooled" else 30, color=INK, zorder=3, marker="o" if r[0] != "Pooled" else "D")
ax.axvline(0, color=INK2, lw=0.7); ax.axvline(0.10, color=INK2, lw=0.7, ls=(0, (1, 2)))
ax.set_xlabel("difference (95% CI)"); ax.set_xticks([0, 0.1, 0.2])
fig.tight_layout(w_pad=0.6)
save(fig, "fig3_direction")

# ---------- Fig. 4: confidence quartiles ----------
H5 = res["H5"]["per_benchmark"]
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
for ax, key, title, c in [(axs[0], "noise_quartiles", "noise passes", PAL[0]), (axs[1], "compressed_le10_quartiles", "compressed, ≤10% budget", PAL[1])]:
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    Q = np.array([H5[b][key] for b in BENCH])
    for q in Q:
        ax.plot(range(1, 5), q, color=c, lw=0.8, alpha=0.35)
    ax.plot(range(1, 5), Q.mean(0), color=c, lw=2, marker="o", ms=4, mec=SURF, mew=0.6)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(["Q1\nleast sure", "Q2", "Q3", "Q4\nmost sure"], fontsize=6.5)
    ax.set_title(title, fontsize=7.5, color=INK, loc="left"); ax.set_ylim(0, 0.55)
axs[0].set_ylabel("answer-change rate")
fig.tight_layout(w_pad=0.5)
save(fig, "fig4_confidence")

# ---------- Fig. 5: noise fragility ----------
from core import Bench
fr = res["H1"]["fragility"]
NOISE_LBL = {"frame_shift": "frames +½ interval", "frame_shift_q": "frames +¼ interval", "option_cycle": "options cycled", "option_reverse": "options reversed"}
S = {b: json.load(open(f"{MAIN}/summary_{b}.json")) for b in BENCH}
fig, ax = plt.subplots(figsize=(3.5, 2.1))
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
w = 0.16; x = np.arange(len(BENCH))
for i, k in enumerate(NK):
    v = [next(n["flip"] for n in S[b]["noise"] if n["pass"] == k) for b in BENCH]
    ax.bar(x + (i - 1.5) * (w + 0.012), v, width=w, color=PAL[i], label=NOISE_LBL[k], zorder=2)
ax.scatter(x, [fr[b][0] for b in BENCH], marker="_", s=110, color=INK, linewidths=1.6, zorder=4, label="fragility (any of the 4)")
ax.errorbar(x, [fr[b][0] for b in BENCH], yerr=[[fr[b][0] - fr[b][1] for b in BENCH], [fr[b][2] - fr[b][0] for b in BENCH]], fmt="none", ecolor=INK, elinewidth=0.8, zorder=4)
ax.axhline(0.30, color=INK2, lw=0.7, ls=(0, (1, 2)))
ax.set_xticks(x); ax.set_xticklabels([BNAME[b].replace("LongVideoBench", "LongVideo-\nBench") for b in BENCH], fontsize=6.5); ax.set_ylabel("share of answers changed")
ax.set_ylim(0, 0.52); ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22), handletextpad=0.7, markerscale=0.8, columnspacing=0.8, handlelength=1.6)
save(fig, "fig5_fragility")
print("figures written")
