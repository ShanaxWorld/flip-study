# ── Per-benchmark analysis (pre-registered measures; the cross-benchmark hypothesis tests are run afterwards) ──
import matplotlib.pyplot as plt
R = []
with open(RES_PATH) as f:
    for line in f:
        try:
            R.append(json.loads(line))
        except Exception:
            pass
R = [r for r in R if len(r.get("noise", {})) == 4]
n = len(R)
if n < 5:
    raise SystemExit(f"Only {n} results — nothing to analyse yet (normal in a quick test with very few questions).")
rng = np.random.default_rng(SEED)
vid = np.array([r["video"] for r in R]); videos = np.unique(vid)
v_rows = [np.flatnonzero(vid == v) for v in videos]
B = 2000
boot = [np.concatenate([v_rows[j] for j in rng.integers(0, len(videos), len(videos))]) for _ in range(B)]
full = np.array([r["full"]["pred"] for r in R]); fcor = np.array([r["full"]["correct"] for r in R])
margin = np.array([r["full"]["margin"] for r in R])
NK = ["frame_shift", "frame_shift_q", "option_cycle", "option_reverse"]
NF = {k: np.array([r["noise"][k]["pred"] for r in R]) != full for k in NK}
STABLE = {k: sum(NF[j] for j in NK if j != k) == 0 for k in NK}       # stable under the other three
DELTA = 0.03

def kl(p_lp, q_lp):
    p_lp, q_lp = np.array(p_lp), np.array(q_lp)
    return float((np.exp(p_lp) * (p_lp - q_lp)).sum())

def pass_arrays(get):
    ok = np.array([get(r) is not None for r in R])
    pred = np.array([get(r)["pred"] if o else "" for r, o in zip(R, ok)])
    cor = np.array([bool(get(r)["correct"]) if o else False for r, o in zip(R, ok)])
    return ok, pred, cor

def excess(flip, ok, idx):
    """Pre-registered Δ: mean over the 4 held-out noise passes of [method flips − held-out noise flips] on the
    questions stable under the other three noise passes (restricted to questions where the method ran)."""
    vals = []
    for k in NK:
        s = STABLE[k][idx] & ok[idx]
        if s.sum() == 0:
            return np.nan
        vals.append(flip[idx][s].mean() - NF[k][idx][s].mean())
    return float(np.mean(vals))

def describe(get, name):
    ok, pred, cor = pass_arrays(get)
    flip = (pred != full) & ok
    idx_all = np.arange(n)
    d = excess(flip, ok, idx_all)
    d_b = np.array([excess(flip, ok, b) for b in boot])
    lo, hi = np.nanpercentile(d_b, [2.5, 97.5])
    h, g = int((fcor & ~cor & ok).sum()), int((~fcor & cor & ok).sum())
    rel_b = [100 * cor[b][ok[b]].mean() / max(fcor[b][ok[b]].mean(), 1e-9) for b in boot]
    return {"pass": name, "coverage": float(ok.mean()), "acc": float(cor[ok].mean()),
            "rel_acc": float(100 * cor[ok].mean() / fcor[ok].mean()),
            "rel_ci": [float(np.percentile(rel_b, 2.5)), float(np.percentile(rel_b, 97.5))],
            "flip": float(flip[ok].mean()), "excess": d, "excess_ci": [float(lo), float(hi)],
            "class": "lossy" if lo > 0 else ("noise-equivalent" if hi < DELTA else "inconclusive"),
            "harmful": h, "helpful": g, "harmful_share": h / (h + g) if h + g else float("nan"),
            "kl": float(np.mean([kl(R[k]["full"]["logp"], get(R[k])["logp"]) for k in np.flatnonzero(ok)])),
            "flops_ratio": float(np.mean([get(R[k])["flops"] for k in np.flatnonzero(ok)]) /
                                 np.mean([R[k]["full"]["flops"] for k in np.flatnonzero(ok)])),
            "retention": float(np.mean([get(R[k]).get("retention", 1.0) for k in np.flatnonzero(ok)])),
            "retention_layer_avg": float(np.mean([get(R[k]).get("retention_layer_avg", get(R[k]).get("retention", 1.0))
                                                 for k in np.flatnonzero(ok)]))}

# ---------------- noise ----------------
fragility = float(np.any(np.stack([NF[k] for k in NK]), 0).mean())
fr_b = [np.any(np.stack([NF[k][b] for k in NK]), 0).mean() for b in boot]
noise_rows = []
for k in NK:
    ok, pred, cor = pass_arrays(lambda r, k=k: r["noise"][k])
    h, g = int((fcor & ~cor).sum()), int((~fcor & cor).sum())
    noise_rows.append({"pass": k, "flip": float(NF[k].mean()), "acc": float(cor.mean()),
                       "heldout_flip_on_stable": float(NF[k][STABLE[k]].mean()), "harmful": h, "helpful": g})
accs = [fcor.mean()] + [x["acc"] for x in noise_rows]
noise_range = 100 * (max(accs) - min(accs))
H_n = sum(x["harmful"] for x in noise_rows); G_n = sum(x["helpful"] for x in noise_rows)
print(f"=== {BENCHMARK}: {n} questions, {len(videos)} videos | full accuracy {fcor.mean():.3f} ===")
print(f"Fragility (changed by ≥1 of 4 noise passes): {fragility:.3f} "
      f"(95% CI {np.percentile(fr_b, 2.5):.3f}–{np.percentile(fr_b, 97.5):.3f})")
print(f"Noise accuracy range: {noise_range:.1f} points | pooled noise harmful share {H_n / max(H_n + G_n, 1):.2f}")
for x in noise_rows:
    print(f"   {x['pass']:15s} flips {x['flip']:.3f} | held-out flips on stable questions {x['heldout_flip_on_stable']:.3f}")

# ---------------- methods ----------------
rows = []
for m, settings in METHOD_GRID.items():
    for s in settings:
        key = f"K{s}" if m == "dycoke" else str(s)
        rows.append({"method": m, "setting": key,
                     **describe(lambda r, m=m, key=key: r["methods"].get(m, {}).get(key), f"{m}@{key}")})
cells = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = cells[["method", "setting", "coverage", "retention", "rel_acc", "flip", "excess", "excess_ci", "class",
              "harmful_share", "kl", "flops_ratio"]].copy()
show["excess_ci"] = show.excess_ci.apply(lambda c: f"{c[0]:+.3f}…{c[1]:+.3f}")
print("\nCells (excess = extra answer changes beyond held-out noise, on noise-stable questions):")
print(show.round(3).to_string(index=False))
cells.to_csv(os.path.join(OUT_DIR, "cells.csv"), index=False)

# ---------------- confidence quartiles ----------------
q = np.digitize(margin, np.quantile(margin, [0.25, 0.5, 0.75]))
noise_any = np.mean(np.stack([NF[k] for k in NK]), 0)
low_keys = [(m, (f"K{s}" if m == "dycoke" else str(s))) for m, v in METHOD_GRID.items() for s in v
            if m != "dycoke" and s <= 0.10]
cf_low = np.mean(np.stack([np.array([(r["methods"].get(m, {}).get(s) or r["full"])["pred"] != r["full"]["pred"]
                                     for r in R]) for m, s in low_keys]), 0)
quart = {"noise": [float(noise_any[q == i].mean()) for i in range(4)],
         "compressed_le10": [float(cf_low[q == i].mean()) for i in range(4)]}
print("\nFlip rate by full-model confidence quartile (Q1 = least sure):", {k: np.round(v, 3).tolist() for k, v in quart.items()})

summary = {"benchmark": BENCHMARK, "n": n, "n_videos": int(len(videos)), "model": MODEL_ID, "frames": NUM_FRAMES,
           "full_accuracy": float(fcor.mean()), "fragility": fragility,
           "fragility_ci": [float(np.percentile(fr_b, 2.5)), float(np.percentile(fr_b, 97.5))],
           "noise_accuracy_range_points": noise_range, "noise": noise_rows,
           "noise_harmful_share": H_n / max(H_n + G_n, 1), "confidence_quartiles": quart,
           "cells": rows, "time_hours": (time.time() - T_START) / 3600 if "T_START" in dir() else None}
with open(os.path.join(OUT_DIR, f"summary_{BENCHMARK}.json"), "w") as f:
    json.dump(summary, f, indent=2, default=float)
# ---------------- plot (after saving the numbers, so a plotting problem can never lose them) ----------------
try:
    col = {"flashvid": "#17becf", "fastvid": "#1f77b4", "holitom": "#9467bd", "visionzip": "#d62728",
           "scissor": "#8c564b", "dycoke": "#2ca02c", "uniform": "#7f7f7f", "random": "#ff7f0e"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for m, g in cells.groupby("method", sort=False):
        g = g.sort_values("retention_layer_avg")
        x = 100 * g.retention_layer_avg.values
        ax[0].errorbar(x, g.excess, yerr=[g.excess - g.excess_ci.str[0], g.excess_ci.str[1] - g.excess],
                       fmt="o-", color=col[m], label=m, capsize=2, ms=4)
        ax[1].plot(x, g.rel_acc, "o-", color=col[m], label=m)
    ax[0].axhline(0, c="k", lw=1); ax[0].axhline(DELTA, c="k", lw=0.8, ls=":")
    ax[0].set_xscale("log"); ax[0].set_xlabel("visual tokens kept (%)"); ax[0].set_ylabel("excess flips beyond noise")
    ax[0].set_title(f"{BENCHMARK}: extra answer changes beyond harmless noise"); ax[0].legend(fontsize=8, ncol=2)
    ax[1].axhline(100, c="k", lw=0.8, ls="--"); ax[1].set_xscale("log")
    ax[1].set_xlabel("visual tokens kept (%)"); ax[1].set_ylabel("relative accuracy (%)"); ax[1].set_title("Relative accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"summary_{BENCHMARK}.png"), dpi=150); plt.show()
except Exception as e:
    plt.close("all")
    print(f"(plot skipped: {type(e).__name__}: {e})")

if QUICK_TEST:
    print("\n(Quick test: numbers are meaningless — this only proves the pipeline works.)")
print("Files saved in", OUT_DIR)
