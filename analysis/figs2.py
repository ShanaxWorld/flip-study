from paths import ROOT, MAIN, REP, MANIFESTS  # noqa: F401
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
exec(open("figs.py").read().split("# ---------- Fig. 1")[0])
MC = {"llava_ov": PAL[0], "llava_video": PAL[1], "qwen3vl": PAL[2]}
MN = {"llava_ov": "LLaVA-OneVision-7B", "llava_video": "LLaVA-Video-7B", "qwen3vl": "Qwen3-VL-8B"}
MMK = {"llava_ov": "o", "llava_video": "s", "qwen3vl": "D"}

# ---------- Fig. A: the measure (schematic) ----------
fig, ax = plt.subplots(figsize=(7.16, 2.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")
FS = 6.2
def box(x, y, w, h, text, fc="#f4f3f0", ec=MUTED, fs=FS, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.3)
def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK2))
box(0.5, 12.5, 11.5, 8, "video\n+ question")
box(16, 24, 20, 6.5, "full model\nanswer $a_0(q)$", fc="#e8f0fb")
box(16, 13, 20, 9, "4 harmless passes:\nframes shifted by ½ or ¼\nof an interval; options\ncycled or reversed", fc="#eef6f1")
box(16, 1.5, 20, 9.5, "compressed pass $m$\n(method, budget)\nanswer $a_m(q)$", fc="#fdeee7")
for y in (27.2, 17.5, 6.2):
    arr(12.3, 16.5, 15.6, y)
box(40, 13, 24, 17.5, "for each held-out pass $k$:\n$S_k$ = questions on which the\nother three passes agree\nwith $a_0$\n\nflip rates on $S_k$:\n$F_m(S_k)$ and $F_{n_k}(S_k)$", fc="#ffffff")
arr(36.3, 27.2, 39.7, 25); arr(36.3, 17.5, 39.7, 19); arr(36.3, 6.2, 39.7, 14.5)
box(40, 1.5, 24, 9.5, "", fc="#ffffff")
ax.text(52, 7.9, r"$\Delta_m=\frac{1}{4}\,\sum_{k}\,[\,F_m(S_k)-F_{n_k}(S_k)\,]$", ha="center", va="center", fontsize=6.7, color=INK)
ax.text(52, 3.4, "excess flips beyond noise", ha="center", va="center", fontsize=FS, color=INK)
arr(52, 12.8, 52, 11.3)
box(68.5, 23.5, 31, 7, "95% interval of $\\Delta_m$\n(videos resampled, 2000 draws)", fc="#ffffff")
arr(64.3, 6.2, 68.2, 25)
for y, t, c in [(15.6, "lossy: lower bound > 0", PAL[1]), (8.6, "noise-equivalent: upper bound < δ", PAL[2]),
                (1.6, "inconclusive: neither", PAL[0])]:
    ax.add_patch(FancyBboxPatch((68.5, y), 31, 5, boxstyle="round,pad=0.2,rounding_size=0.6", fc="white", ec=c, lw=1.2))
    ax.text(70, y + 2.5, t, ha="left", va="center", fontsize=FS, color=INK)
arr(84, 23.3, 84, 21.0)
save(fig, "figA_measure")

# ---------- Fig. B: video length ----------
L = pd.read_csv(f"{OUT}/length_subgroups.csv", dtype={"group": str})
lab = {"short": "short", "medium": "medium", "long": "long", "15": "≤15 s", "60": "≤1 min",
       "600": "≤10 min", "3600": "≤1 h"}
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.3), sharey="row", gridspec_kw={"width_ratios": [3, 4]})
for j, b in enumerate(["videomme", "longvideobench"]):
    d = L[L.benchmark == b].reset_index(drop=True); x = np.arange(len(d))
    ax = axs[0, j]; ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.bar(x, d.fragility, width=0.55, color=GRID, zorder=2)
    ax.errorbar(x, d.fragility, yerr=[d.fragility - d.frag_lo, d.frag_hi - d.fragility], fmt="none", ecolor=INK2, elinewidth=0.8, zorder=3)
    ax.plot(x, d.full_acc, color=INK, lw=1.2, marker="o", ms=3.5, zorder=4, label="full-model accuracy")
    ax.set_xticks(x); ax.set_xticklabels([lab[g] for g in d.group], fontsize=6.2); ax.set_ylim(0, 0.85)
    ax.set_title(BNAME[b], fontsize=7.5, color=INK, loc="left")
    ax = axs[1, j]; ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.axhline(0, color=INK2, lw=0.7)
    for k, (s, c) in enumerate([("0.05", PAL[1]), ("0.1", PAL[0]), ("0.25", PAL[2])]):
        xx = x + (k - 1) * 0.2
        ax.errorbar(xx, d[f"excess_{s}"], yerr=[d[f"excess_{s}"] - d[f"excess_{s}_lo"], d[f"excess_{s}_hi"] - d[f"excess_{s}"]],
                    fmt="o", ms=3, color=c, elinewidth=0.8, label=f"{int(float(s)*100)}% kept")
    ax.set_xticks(x); ax.set_xticklabels([lab[g] for g in d.group], fontsize=6.2)
axs[0, 0].set_ylabel("fragility (bars)\naccuracy (line)", fontsize=6.5); axs[1, 0].set_ylabel("excess flips Δ\n(mean of 7 methods)", fontsize=6.5)
h, l = axs[1, 1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.55, -0.01), handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(h_pad=0.6, w_pad=0.4, rect=(0, 0.05, 1, 1))
save(fig, "figB_length")

# ---------- Fig. C: same questions, three models ----------
X = pd.read_csv(f"{OUT}/xmodel_pooled.csv", dtype={"setting": str})
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.0), sharey=True)
xpos = {"0.05": 5, "0.1": 10, "0.25": 25}
for ax, m in zip(axs, ["flashvid", "random", "uniform"]):
    ax.set_xscale("log"); ax.set_xticks([5, 10, 25]); ax.set_xticklabels(["5", "10", "25"]); ax.minorticks_off(); ax.set_xlim(4.2, 30)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.axhline(0, color=INK2, lw=0.7); ax.axhline(DELTA, color=INK2, lw=0.7, ls=(0, (1, 2)))
    for i, mod in enumerate(["llava_ov", "llava_video", "qwen3vl"]):
        g = X[(X.model == mod) & (X.method == m)].copy(); g["x"] = g.setting.map(xpos) * (1 + 0.05 * (i - 1)); g = g.sort_values("x")
        ax.plot(g.x, g.excess, color=MC[mod], lw=1.0)
        ax.errorbar(g.x, g.excess, yerr=[g.excess - g.lo, g.hi - g.excess], fmt="none", ecolor=MC[mod], elinewidth=0.8)
        ax.scatter(g.x, g.excess, s=18, marker=MMK[mod], color=MC[mod], edgecolors=SURF, linewidths=0.5, zorder=3, label=MN[mod])
    ax.set_title(MNAME[m], fontsize=8, color=INK, loc="left"); ax.set_xlabel("visual tokens kept (%)")
axs[0].set_ylabel("excess flips Δ"); h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "figC_models")

# ---------- Fig. D: direction across models ----------
C = pd.read_csv(f"{OUT}/cells_every_model.csv", dtype={"setting": str})
fn = json.load(open(f"{OUT}/final_numbers.json")); rc = json.load(open(f"{OUT}/results_cross.json"))
rows = []
import core, rep
for model, benches in [("llava_ov_main", core.BENCH), ("llava_video", ["videomme", "egoschema", "longvideobench"]), ("qwen3vl", ["videomme", "egoschema", "longvideobench"])]:
    for b in benches:
        d = C[(C.model == model) & (C.benchmark == b)]
        L_ = d[d.cls == "lossy"]; hs = L_.harmful.sum() / max(L_.harmful.sum() + L_.helpful.sum(), 1)
        rows.append((model, BNAME[b], hs))
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
# noise harmful shares per (model, benchmark)
noise = {}
for model in ["llava_ov_main"]:
    for b, v in rc["H3"]["per_benchmark"].items():
        noise[(model, BNAME[b])] = v["noise_hs"]
for model in ["llava_video", "qwen3vl"]:
    for b in ["videomme", "egoschema", "longvideobench"]:
        s = json.load(open(f"{rep.REP}/{model}_summary_{b}.json")); noise[(model, BNAME[b])] = s["noise_hs"]
y = 0; yt, yl = [], []
heads = {"llava_ov_main": "LLaVA-OneVision-7B (all questions)", "llava_video": "LLaVA-Video-7B (300 per benchmark)", "qwen3vl": "Qwen3-VL-8B (300 per benchmark)"}
for model in ["llava_ov_main", "llava_video", "qwen3vl"]:
    ax.text(0.352, y + 0.15, heads[model], fontsize=6.3, color=INK, fontweight="bold", va="bottom", ha="left", bbox=dict(fc="white", ec="none", pad=0.6))
    y -= 0.85
    for (mm, bn, hs) in [r for r in rows if r[0] == model]:
        nz = noise[(model, bn)]
        ax.plot([nz, hs], [y, y], color=GRID, lw=2.2, zorder=1)
        ax.scatter([nz], [y], s=16, color=PAL[0], edgecolors=SURF, linewidths=0.4, zorder=3)
        ax.scatter([hs], [y], s=16, color=PAL[1], edgecolors=SURF, linewidths=0.4, zorder=3)
        yt.append(y); yl.append(bn); y -= 1
    y -= 0.3
ax.axvline(0.5, color=INK2, lw=0.7)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.3); ax.set_xlim(0.35, 0.92); ax.set_ylim(y + 0.6, 1.0)
ax.set_xlabel("share of changed answers that went right → wrong")
ax.scatter([], [], s=16, color=PAL[0], label="harmless noise passes"); ax.scatter([], [], s=16, color=PAL[1], label="lossy compressed settings")
ax.legend(fontsize=6.2, loc="lower right", bbox_to_anchor=(0.92, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
save(fig, "figD_direction")

# ---------- Fig. E: cost (main model) ----------
D_ = pd.read_csv(f"{OUT}/cells_all.csv", dtype={"setting": str})
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.axhline(0, color=INK2, lw=0.7); ax.axhline(DELTA, color=INK2, lw=0.7, ls=(0, (1, 2)))
for m in ORDER:
    g = D_[D_.method == m]
    ax.scatter(100 * g.flops_ratio, g.excess, s=14, marker=MK[m], color=COL[m], edgecolors=SURF, linewidths=0.4, label=MNAME[m], alpha=0.95)
ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 35]); ax.set_xticklabels(["5", "10", "25", "35"]); ax.minorticks_off()
ax.set_xlabel("LLM prefill FLOPs (% of full model)"); ax.set_ylabel("excess flips Δ")
ax.legend(fontsize=5.6, ncol=2, loc="upper right", handletextpad=0.1, columnspacing=0.4)
save(fig, "figE_cost")
print("ok")
