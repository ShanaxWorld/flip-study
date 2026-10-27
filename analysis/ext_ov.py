"""Revision runs on LLaVA-OneVision-7B (post hoc, added after the confirmatory analysis):
1. V-CAST at 5/10/25% scored with the planned measure (four harmless passes, leave-one-out), per benchmark and pooled;
2. six further harmless passes (frame shifts of 1/8 and 3/8 of an interval, two further option permutations, two
   rewordings of the instruction) and every cell re-scored against noise floors built from different pass sets.
The revision notebook reran the uncompressed pass; it reproduces the main study's answers and log-probabilities
exactly on every question, which is checked here before the records are merged."""
import json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
from paths import MAIN, OUT, ROOT
import core

REV = str(ROOT / "data" / "revision" / "llava_ov")
NEW = ["frame_shift_e", "frame_shift_3e", "option_perm1", "option_perm2", "prompt_alt1", "prompt_alt2"]
SETS = {"planned4": core.NK, "new6": NEW, "all10": core.NK + NEW,
        "frame4": ["frame_shift", "frame_shift_q", "frame_shift_e", "frame_shift_3e"],
        "option4": ["option_cycle", "option_reverse", "option_perm1", "option_perm2"],
        "prompt2": ["prompt_alt1", "prompt_alt2"],
        # prompt_alt2 fails the accuracy check that defines a harmless pass (pooled McNemar p < 0.05), so the
        # sets below leave it out
        "new5": [k for k in NEW if k != "prompt_alt2"], "eligible9": [k for k in core.NK + NEW if k != "prompt_alt2"]}
VC = [("vcast", s) for s in ["0.05", "0.1", "0.25"]]


def cls(lo, hi, delta=core.DELTA):
    return "lossy" if lo > 0 else ("noise-equivalent" if hi < delta else "inconclusive")


def load(b):
    core.DATA = MAIN
    X = core.Bench(b)
    ext = {r["qid"]: r for r in map(json.loads, open(f"{REV}/{b}_results.jsonl"))}
    assert set(ext) == {r["qid"] for r in X.R}, b
    for r in X.R:
        e = ext[r["qid"]]
        assert e["full"]["pred"] == r["full"]["pred"] and e["full"]["logp"] == r["full"]["logp"], (b, r["qid"])
        r["noise"].update(e["noise"]); r["methods"]["vcast"] = e["methods"]["vcast"]
    for k in NEW:
        X.NF[k] = np.array([r["noise"][k]["pred"] for r in X.R]) != X.full
        X.noise_cor[k] = np.array([bool(r["noise"][k]["correct"]) for r in X.R])
    for m, s in VC:
        get = lambda r: r["methods"]["vcast"][s]
        pred = np.array([get(r)["pred"] for r in X.R]); ok = np.ones(X.n, bool)
        X.cells[(m, s)] = dict(ok=ok, pred=pred, cor=np.array([bool(get(r)["correct"]) for r in X.R]), flip=pred != X.full,
                               kl=np.array([core.kl(r["full"]["logp"], get(r)["logp"]) for r in X.R]),
                               flops=np.array([get(r)["flops"] / r["full"]["flops"] for r in X.R]),
                               ret=np.array([get(r)["retention"] for r in X.R]))
    # bootstrap resamples as count matrices (same resamples as core.Bench, seed 0)
    X.C = np.stack([np.bincount(bi, minlength=X.n) for bi in X.boot]).astype(float)
    return X


def directed(X, m, s, idx):
    """Directed variant (Eq. 4 of the paper): right-to-wrong flips beyond the held-out pass's right-to-wrong flips."""
    c = X.cells[(m, s)]; harm = X.fcor & ~c["cor"] & c["ok"]; vals = []
    for k in core.NK:
        st = X.STABLE[k][idx] & c["ok"][idx]
        vals.append(harm[idx][st].mean() - (X.fcor & ~X.noise_cor[k])[idx][st].mean())
    return float(np.mean(vals))


def excess_vec(X, flip, ok, passes, W):
    """Leave-one-out Δ over a pass set, for every row of the weight matrix W (questions weighted by resample counts)."""
    vals = []
    for k in passes:
        st = ok.copy()
        for j in passes:
            if j != k:
                st &= ~X.NF[j]
        den = W @ st
        num = W @ (st * (flip.astype(float) - X.NF[k]))
        with np.errstate(invalid="ignore", divide="ignore"):
            vals.append(np.where(den > 0, num / den, np.nan))
    return np.mean(vals, 0)


if __name__ == "__main__":
    X = {b: load(b) for b in core.BENCH}
    one = {b: np.ones((1, X[b].n)) for b in core.BENCH}
    # consistency with the planned code path
    c0 = X["videomme"].describe("flashvid", "0.1")
    v0 = excess_vec(X["videomme"], X["videomme"].cells[("flashvid", "0.1")]["flip"], X["videomme"].cells[("flashvid", "0.1")]["ok"],
                    core.NK, X["videomme"].C)
    assert abs(excess_vec(X["videomme"], X["videomme"].cells[("flashvid", "0.1")]["flip"],
                          X["videomme"].cells[("flashvid", "0.1")]["ok"], core.NK, one["videomme"])[0] - c0["excess"]) < 1e-12
    assert np.allclose(np.nanpercentile(v0, [2.5, 97.5]), [c0["lo"], c0["hi"]])

    # ---- 1. V-CAST with the planned measure ----
    vrows = []
    for b in core.BENCH:
        for m, s in VC:
            d = {k: v for k, v in X[b].describe(m, s).items() if k != "boot"}
            x, c = X[b], X[b].cells[(m, s)]
            n01 = int((x.fcor & ~c["cor"]).sum()); n10 = int((~x.fcor & c["cor"]).sum())
            db = np.array([directed(x, m, s, i) for i in x.boot])
            d.update(acc_drop_pts=100 * (x.fcor.mean() - c["cor"].mean()),
                     mcnemar_p=binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0,
                     harm_excess=directed(x, m, s, np.arange(x.n)), harm_lo=float(np.percentile(db, 2.5)),
                     harm_hi=float(np.percentile(db, 97.5)))
            vrows.append(d)
    V = pd.DataFrame(vrows); V.to_csv(f"{OUT}/vcast_ov_cells.csv", index=False)
    # pooled across benchmarks, independent streams per benchmark (as cross.py)
    ss = np.random.SeedSequence(core.SEED).spawn(len(core.BENCH))
    PW = {}
    for b, sq in zip(core.BENCH, ss):
        rng = np.random.default_rng(sq); x = X[b]
        PW[b] = np.stack([np.bincount(np.concatenate([x.v_rows[j] for j in rng.integers(0, len(x.videos), len(x.videos))]),
                                      minlength=x.n) for _ in range(core.B)]).astype(float)
    pooled = []
    for m, s in VC:
        d = np.mean([V[(V.benchmark == b) & (V.setting == s)].excess.iloc[0] for b in core.BENCH])
        db = np.mean([excess_vec(X[b], X[b].cells[(m, s)]["flip"], X[b].cells[(m, s)]["ok"], core.NK, PW[b]) for b in core.BENCH], 0)
        lo, hi = np.nanpercentile(db, [2.5, 97.5])
        pooled.append(dict(method=m, setting=s, excess=d, lo=lo, hi=hi, cls=cls(lo, hi),
                           rel_acc_mean=V[V.setting == s].rel_acc.mean(), flops_ratio=V[V.setting == s].flops_ratio.mean()))
    P = pd.DataFrame(pooled)
    old = pd.read_csv(f"{OUT}/pooled_methods.csv")
    old = old[old.method != "vcast"]
    pd.concat([old, P]).to_csv(f"{OUT}/pooled_methods_with_vcast.csv", index=False)

    # ---- 2. every cell against six noise floors ----
    rows = []
    for b in core.BENCH:
        x = X[b]
        for (m, s), c in x.cells.items():
            r = dict(benchmark=b, method=m, setting=s)
            for name, passes in SETS.items():
                d = excess_vec(x, c["flip"], c["ok"], passes, one[b])[0]
                db = excess_vec(x, c["flip"], c["ok"], passes, x.C)
                lo, hi = np.nanpercentile(db, [2.5, 97.5])
                r.update({f"{name}": d, f"{name}_lo": lo, f"{name}_hi": hi, f"{name}_cls": cls(lo, hi),
                          f"{name}_se": float(np.nanstd(db))})
            rows.append(r)
        print(b, "done", flush=True)
    E = pd.DataFrame(rows); E.to_csv(f"{OUT}/noise_ext_cells.csv", index=False)

    # ---- per-benchmark pass statistics ----
    bs = []
    for b in core.BENCH:
        x = X[b]; allk = core.NK + NEW
        row = dict(benchmark=b, n=x.n, full_acc=x.fcor.mean())
        for k in allk:
            row[f"flip_{k}"] = x.NF[k].mean(); row[f"acc_{k}"] = x.noise_cor[k].mean()
        for name, passes in SETS.items():
            row[f"fragility_{name}"] = np.any(np.stack([x.NF[k] for k in passes]), 0).mean()
            row[f"heldout_{name}"] = np.mean([x.NF[k][np.all(np.stack([~x.NF[j] for j in passes if j != k]), 0)].mean()
                                              for k in passes])
        accs = [x.fcor.mean()] + [x.noise_cor[k].mean() for k in allk]
        row["noise_range_all10"] = 100 * (max(accs) - min(accs))
        bs.append(row)
    BS = pd.DataFrame(bs); BS.to_csv(f"{OUT}/noise_ext_bench.csv", index=False)

    # ---- is each pass harmless? paired accuracy test against the uncompressed pass, per benchmark and pooled ----
    ht = []
    for k in core.NK + NEW:
        H = G = 0
        for b in core.BENCH:
            x = X[b]; h = int((x.fcor & ~x.noise_cor[k]).sum()); g = int((~x.fcor & x.noise_cor[k]).sum()); H += h; G += g
            ht.append(dict(pass_=k, benchmark=b, acc_change_pts=100 * (x.noise_cor[k].mean() - x.fcor.mean()), flip=x.NF[k].mean(),
                           harmful=h, helpful=g, p=binomtest(h, h + g, 0.5).pvalue if h + g else 1.0))
        ht.append(dict(pass_=k, benchmark="pooled", acc_change_pts=np.mean([r["acc_change_pts"] for r in ht[-5:]]),
                       flip=np.mean([r["flip"] for r in ht[-5:]]), harmful=H, helpful=G, p=binomtest(H, H + G, 0.5).pvalue))
    HT = pd.DataFrame(ht).rename(columns={"pass_": "pass"}); HT.to_csv(f"{OUT}/noise_pass_accuracy.csv", index=False)
    print(HT[HT.benchmark == "pooled"].round(4).to_string())

    # ---- summary ----
    main = E[E.method != "dycoke"]
    sm = {}
    for name in SETS:
        d = main
        sm[name] = dict(agree_with_planned=float((d[f"{name}_cls"] == d.planned4_cls).mean()),
                        corr_with_planned=float(np.corrcoef(d.planned4, d[name])[0, 1]),
                        mean_shift_pts=float(100 * (d[name] - d.planned4).mean()),
                        median_se_pts=float(100 * d[f"{name}_se"].median()),
                        by_budget={s: d[d.setting == s][f"{name}_cls"].value_counts().to_dict() for s in ["0.05", "0.1", "0.25"]},
                        lossy_to_ne=int(((d.planned4_cls == "lossy") & (d[f"{name}_cls"] == "noise-equivalent")).sum()),
                        ne_to_lossy=int(((d.planned4_cls == "noise-equivalent") & (d[f"{name}_cls"] == "lossy")).sum()))
    # the planned hypotheses' cell sets (35 cells per budget, no V-CAST) under each floor
    h = main[main.method != "vcast"]
    for name in SETS:
        sm[name]["H2a_lossy_at_5"] = int((h[h.setting == "0.05"][f"{name}_cls"] == "lossy").sum())
        sm[name]["H2b_ne_at_25"] = int((h[h.setting == "0.25"][f"{name}_cls"] == "noise-equivalent").sum())
        sm[name]["cells_per_budget"] = int((h.setting == "0.25").sum())
    json.dump(sm, open(f"{OUT}/noise_ext_summary.json", "w"), indent=1)

    pd.set_option("display.width", 250)
    print(V[["benchmark", "setting", "retention", "rel_acc", "flip", "excess", "lo", "hi", "cls", "harmful_share", "flops_ratio"]].round(3).to_string())
    print(P.round(4).to_string())
    print(BS.round(3).T.to_string())
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_budget"} for k, v in sm.items()}, indent=1))
    for name in SETS:
        print(name, sm[name]["by_budget"])
