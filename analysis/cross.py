"""Cross-benchmark confirmatory tests (H1-H5), secondary and exploratory analyses, as written in the plan."""
from paths import ROOT, MAIN, REP, MANIFESTS  # noqa: F401
import json, pickle, numpy as np, pandas as pd
from scipy.stats import binomtest
from core import *

from paths import OUT; os.makedirs(OUT, exist_ok=True)
X = {b: Bench(b) for b in BENCH}
cells = {(b, m, s): X[b].describe(m, s) for b in BENCH for m, s in cell_keys()}
df = pd.DataFrame([{k: v for k, v in c.items() if k != "boot"} for c in cells.values()])
df.to_csv(f"{OUT}/cells_all.csv", index=False)
# independent bootstrap streams per benchmark for the pooled (cross-benchmark) intervals
ss = np.random.SeedSequence(SEED).spawn(len(BENCH))
PB = {}
for b, sq in zip(BENCH, ss):
    rng = np.random.default_rng(sq); x = X[b]
    PB[b] = [np.concatenate([x.v_rows[j] for j in rng.integers(0, len(x.videos), len(x.videos))]) for _ in range(B)]
res = {"n": {b: X[b].n for b in BENCH}, "videos": {b: int(len(X[b].videos)) for b in BENCH},
       "full_acc": {b: float(X[b].fcor.mean()) for b in BENCH}}

# coverage rule
cov_flag = df[df.coverage < 0.95][["benchmark", "method", "setting", "coverage"]]
res["coverage_min"] = float(df.coverage.min()); res["coverage_flagged"] = cov_flag.to_dict("records")
h2 = df[(df.method != "dycoke") & (df.coverage >= 0.80)]

# ---------- H1 ----------
fr = {b: X[b].fragility() for b in BENCH}
n1 = sum(fr[b][0] >= 0.30 for b in BENCH)
res["H1"] = {"fragility": fr, "n_benchmarks_ge_30": n1, "supported": n1 >= 4}

# ---------- H2 ----------
def share(budget, cls):
    sub = h2[h2.setting == budget]
    return int((sub.cls == cls).sum()), int(len(sub))
a = share("0.05", "lossy"); bb = share("0.25", "noise-equivalent")
res["H2a"] = {"lossy": a[0], "cells": a[1], "share": a[0] / a[1], "supported": a[0] / a[1] >= 0.70}
res["H2b"] = {"noise_equivalent": bb[0], "cells": bb[1], "share": bb[0] / bb[1], "supported": bb[0] / bb[1] >= 0.70,
              "classes_at_25": h2[h2.setting == "0.25"].cls.value_counts().to_dict()}
h2c = {}
for b in BENCH:
    sub = h2[(h2.benchmark == b) & (h2.setting == "0.1")]
    h2c[b] = {"noise_equivalent": sub[sub.cls == "noise-equivalent"].method.tolist(),
              "lossy": sub[sub.cls == "lossy"].method.tolist()}
    h2c[b]["met"] = bool(h2c[b]["noise_equivalent"] and h2c[b]["lossy"])
n2c = sum(v["met"] for v in h2c.values())
res["H2c"] = {"per_benchmark": h2c, "n_met": n2c, "supported": n2c >= 3}
# class table and decision-lossless budgets
lossless = {}
for b in BENCH:
    for m in MAIN7:
        cl = {s: cells[(b, m, s)]["cls"] for s in ["0.05", "0.1", "0.25"]}
        lb = "above 25%"
        for s in ["0.25", "0.1", "0.05"]:
            if cl[s] == "noise-equivalent":
                lb = s
            else:
                break
        lossless[(b, m)] = lb
res["decision_lossless_budget"] = {f"{b}/{m}": v for (b, m), v in lossless.items()}

# ---------- H3 ----------
lossy_keys = [(b, m, s) for (b, m, s), c in cells.items() if c["cls"] == "lossy" and c["coverage"] >= 0.80]
hv, gv, nh, ng = {}, {}, {}, {}
for b in BENCH:
    x = X[b]
    hv[b] = sum((x.fcor & ~x.cells[(m, s)]["cor"] & x.cells[(m, s)]["ok"]).astype(int) for bb_, m, s in lossy_keys if bb_ == b) if any(k[0] == b for k in lossy_keys) else np.zeros(x.n, int)
    gv[b] = sum((~x.fcor & x.cells[(m, s)]["cor"] & x.cells[(m, s)]["ok"]).astype(int) for bb_, m, s in lossy_keys if bb_ == b) if any(k[0] == b for k in lossy_keys) else np.zeros(x.n, int)
    nh[b] = sum((x.fcor & ~x.noise_cor[k]).astype(int) for k in NK)
    ng[b] = sum((~x.fcor & x.noise_cor[k]).astype(int) for k in NK)
def h3_stat(idx):
    H = sum(hv[b][idx[b]].sum() for b in BENCH); G = sum(gv[b][idx[b]].sum() for b in BENCH)
    Hn = sum(nh[b][idx[b]].sum() for b in BENCH); Gn = sum(ng[b][idx[b]].sum() for b in BENCH)
    return H / (H + G), Hn / (Hn + Gn)
c_hs, n_hs = h3_stat({b: np.arange(X[b].n) for b in BENCH})
diffs = np.array([np.subtract(*h3_stat({b: PB[b][i] for b in BENCH})) for i in range(B)])
lo3, hi3 = np.percentile(diffs, [2.5, 97.5])
per_b = {}
for b in BENCH:
    H, G, Hn, Gn = hv[b].sum(), gv[b].sum(), nh[b].sum(), ng[b].sum()
    db = [hv[b][i].sum() / max(hv[b][i].sum() + gv[b][i].sum(), 1) - nh[b][i].sum() / (nh[b][i].sum() + ng[b][i].sum()) for i in X[b].boot]
    per_b[b] = {"lossy_cells": sum(k[0] == b for k in lossy_keys), "lossy_hs": H / (H + G), "noise_hs": Hn / (Hn + Gn),
                "diff": H / (H + G) - Hn / (Hn + Gn), "ci": list(np.percentile(db, [2.5, 97.5]))}
res["H3"] = {"n_lossy_cells": len(lossy_keys), "lossy_harmful_share": c_hs, "noise_harmful_share": n_hs,
             "diff": c_hs - n_hs, "ci": [lo3, hi3], "supported": (c_hs - n_hs) >= 0.10 and lo3 > 0, "per_benchmark": per_b}

# ---------- H4 ----------
def mcnemar_exact(a, b):
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    return 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue, n01, n10
def holm(p, alpha=0.05):
    order = np.argsort(p); rej = np.zeros(len(p), bool)
    for i, j in enumerate(order):
        if p[j] <= alpha / (len(p) - i):
            rej[j] = True
        else:
            break
    return rej
pairs_out, n_sig, n_pairs = {}, 0, 0
for b in BENCH:
    x = X[b]; rows = []
    for i, m1 in enumerate(MAIN7):
        for m2 in MAIN7[i + 1:]:
            c1, c2 = x.cells[(m1, "0.25")], x.cells[(m2, "0.25")]
            both = c1["ok"] & c2["ok"]
            p, n01, n10 = mcnemar_exact(c1["cor"][both], c2["cor"][both])
            rows.append({"m1": m1, "m2": m2, "acc1": c1["cor"][both].mean(), "acc2": c2["cor"][both].mean(),
                         "n01": n01, "n10": n10, "p": p})
    rej = holm(np.array([r["p"] for r in rows]))
    for r, z in zip(rows, rej):
        r["significant_holm"] = bool(z)
    pairs_out[b] = rows; n_sig += int(rej.sum()); n_pairs += len(rows)
nr = {b: X[b].noise_range() for b in BENCH}
n4 = sum(v >= 1.0 for v in nr.values())
res["H4"] = {"noise_range": nr, "n_range_ge_1": n4, "sig_pairs": n_sig, "pairs": n_pairs, "share_sig": n_sig / n_pairs,
             "min_raw_p": {b: min(r["p"] for r in pairs_out[b]) for b in BENCH},
             "supported": n4 >= 4 and n_sig / n_pairs < 0.20}
pd.DataFrame([{"benchmark": b, **r} for b in BENCH for r in pairs_out[b]]).to_csv(f"{OUT}/h4_pairs.csv", index=False)

# ---------- H5 ----------
low_keys = [(m, s) for m in MAIN7 for s in ["0.05", "0.1"]]
h5 = {}
for b in BENCH:
    x = X[b]
    q = np.digitize(x.margin, np.quantile(x.margin, [0.25, 0.5, 0.75]))
    noise_any = np.mean(np.stack([x.NF[k] for k in NK]), 0)
    cf_low = np.mean(np.stack([np.where(x.cells[k]["ok"], x.cells[k]["flip"], False) for k in low_keys]), 0)
    qn = [float(noise_any[q == i].mean()) for i in range(4)]; qc = [float(cf_low[q == i].mean()) for i in range(4)]
    rn = qn[0] / qn[3] if qn[3] > 0 else np.inf; rc = qc[0] / qc[3] if qc[3] > 0 else np.inf
    h5[b] = {"noise_quartiles": qn, "compressed_le10_quartiles": qc, "ratio_noise": rn, "ratio_compressed": rc,
             "met": bool(rn >= 3 and rc >= 3)}
n5 = sum(v["met"] for v in h5.values())
res["H5"] = {"per_benchmark": h5, "n_met": n5, "supported": n5 >= 4}

# ---------- Secondary: KL beyond held-out noise KL ----------
kl_rows = []
for (b, m, s), c in cells.items():
    x = X[b]; cc = x.cells[(m, s)]
    d = x.excess_kl(cc["kl"], cc["ok"], np.arange(x.n))
    db = np.array([x.excess_kl(cc["kl"], cc["ok"], i) for i in x.boot])
    lo, hi = np.nanpercentile(db, [2.5, 97.5])
    kl_rows.append({"benchmark": b, "method": m, "setting": s, "excess_kl": d, "lo": lo, "hi": hi,
                    "kl_lossy": lo > 0, "flip_lossy": c["cls"] == "lossy"})
kd = pd.DataFrame(kl_rows); kd.to_csv(f"{OUT}/kl_excess.csv", index=False)
res["secondary_kl"] = {"agreement_lossy": float((kd.kl_lossy == kd.flip_lossy).mean()),
                       "kl_lossy_only": int((kd.kl_lossy & ~kd.flip_lossy).sum()),
                       "flip_lossy_only": int((~kd.kl_lossy & kd.flip_lossy).sum()), "cells": len(kd)}

# ---------- Secondary: subgroups ----------
sub_rows = []
for b in BENCH:
    x = X[b]
    for g in sorted(set(x.group)):
        idx = np.flatnonzero(x.group == g)
        for m, s in cell_keys():
            cc = x.cells[(m, s)]
            sub_rows.append({"benchmark": b, "group": g, "n": len(idx), "method": m, "setting": s,
                             "excess": x.excess(cc["flip"], cc["ok"], idx),
                             "rel_acc": 100 * cc["cor"][idx][cc["ok"][idx]].mean() / max(x.fcor[idx].mean(), 1e-9),
                             "fragility": float(np.any(np.stack([x.NF[k][idx] for k in NK]), 0).mean())})
pd.DataFrame(sub_rows).to_csv(f"{OUT}/subgroups.csv", index=False)

# ---------- Secondary: cost — cheapest noise-equivalent setting per benchmark ----------
cheap = {}
for b in BENCH:
    ne = df[(df.benchmark == b) & (df.cls == "noise-equivalent")].sort_values("flops_ratio")
    cheap[b] = ne.iloc[0][["method", "setting", "flops_ratio", "excess", "hi"]].to_dict() if len(ne) else None
res["secondary_cost_cheapest_noise_equivalent"] = cheap

# ---------- Added after the plan (labelled post hoc): per-method Δ pooled across benchmarks ----------
pool_rows = []
for m, s in cell_keys():
    d = np.mean([cells[(b, m, s)]["excess"] for b in BENCH])
    db = []
    for i in range(B):
        db.append(np.mean([X[b].excess(X[b].cells[(m, s)]["flip"], X[b].cells[(m, s)]["ok"], PB[b][i]) for b in BENCH]))
    lo, hi = np.nanpercentile(db, [2.5, 97.5])
    pool_rows.append({"method": m, "setting": s, "excess": d, "lo": lo, "hi": hi,
                      "cls": "lossy" if lo > 0 else ("noise-equivalent" if hi < DELTA else "inconclusive"),
                      "rel_acc_mean": np.mean([cells[(b, m, s)]["rel_acc"] for b in BENCH]),
                      "flops_ratio": np.mean([cells[(b, m, s)]["flops_ratio"] for b in BENCH])})
pool = pd.DataFrame(pool_rows); pool.to_csv(f"{OUT}/pooled_methods.csv", index=False)

# ---------- Exploratory ----------
# (a) calibration questions needed to certify a budget as noise-equivalent (normal approximation)
cert = []
for (b, m, s), c in cells.items():
    se = np.nanstd(c["boot"]); gap = DELTA - c["excess"]
    cert.append({"benchmark": b, "method": m, "setting": s, "excess": c["excess"], "se": se,
                 "n_needed": float(X[b].n * (1.96 * se / gap) ** 2) if gap > 0 else np.inf})
cert = pd.DataFrame(cert); cert.to_csv(f"{OUT}/certification_n.csv", index=False)
# (b) does the full model's margin predict which questions a compressor flips? (AUC of -margin)
from scipy.stats import mannwhitneyu
aucs = []
for (b, m, s), c in cells.items():
    x = X[b]; cc = x.cells[(m, s)]; ok = cc["ok"]; f = cc["flip"][ok]; mg = x.margin[ok]
    if 0 < f.sum() < len(f):
        U = mannwhitneyu(-mg[f], -mg[~f]).statistic
        aucs.append({"benchmark": b, "method": m, "setting": s, "auc": U / (f.sum() * (~f).sum())})
aucs = pd.DataFrame(aucs); aucs.to_csv(f"{OUT}/margin_auc.csv", index=False)
# (c) overlap of flipped questions across the 7 methods at each budget
ov = []
for b in BENCH:
    x = X[b]
    for s in ["0.05", "0.1", "0.25"]:
        F = np.stack([x.cells[(m, s)]["flip"] for m in MAIN7])
        k = F.sum(0); p = F.mean(1)
        # expected count of questions flipped by all 7 if methods flipped independently at their own rates
        ov.append({"benchmark": b, "setting": s, "any": float((k >= 1).mean()), "ge4": float((k >= 4).mean()),
                   "all7": float((k == 7).mean()), "all7_if_independent": float(np.prod(p)),
                   "noise_fragile_share_of_all7": float(np.any(np.stack([x.NF[kk] for kk in NK]), 0)[k == 7].mean()) if (k == 7).any() else np.nan})
ov = pd.DataFrame(ov); ov.to_csv(f"{OUT}/overlap.csv", index=False)
res["exploratory"] = {"margin_auc_mean": float(aucs.auc.mean()), "margin_auc_range": [float(aucs.auc.min()), float(aucs.auc.max())],
                      "overlap": ov.to_dict("records")}

# accuracy parity vs fidelity: lossy cells whose relative accuracy is within 2 points of the full model
par = df[(df.cls == "lossy") & (df.rel_acc >= 98)]
res["parity_but_lossy"] = {"n": int(len(par)), "of_lossy": int((df.cls == "lossy").sum()),
                           "cells": par[["benchmark", "method", "setting", "rel_acc", "excess", "lo"]].to_dict("records")}

# accuracy test vs fidelity: lossy cells whose accuracy does NOT differ significantly from the full model (exact McNemar)
acc_rows = []
for (b, m, s), c in cells.items():
    x = X[b]; cc = x.cells[(m, s)]; ok = cc["ok"]
    p, n01, n10 = mcnemar_exact(x.fcor[ok], cc["cor"][ok])
    acc_rows.append({"benchmark": b, "method": m, "setting": s, "cls": c["cls"], "rel_acc": c["rel_acc"],
                     "acc_drop_pts": 100 * (x.fcor[ok].mean() - cc["cor"][ok].mean()), "mcnemar_p": p})
ad = pd.DataFrame(acc_rows); ad.to_csv(f"{OUT}/accuracy_tests.csv", index=False)
L = ad[ad.cls == "lossy"]
res["accuracy_vs_fidelity"] = {"lossy_cells": int(len(L)), "lossy_acc_not_significant": int((L.mcnemar_p >= 0.05).sum()),
                               "lossy_acc_within_2pts": int((L.acc_drop_pts.abs() <= 2).sum()),
                               "all_cells_acc_significant": int((ad.mcnemar_p < 0.05).sum()), "cells": int(len(ad))}
json.dump(res, open(f"{OUT}/results_cross.json", "w"), indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
pickle.dump({"cells": cells}, open(f"{OUT}/cells.pkl", "wb"))
print("done")
