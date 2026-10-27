"""Robustness analyses added in revision (all post hoc, computed from the released per-question outputs):
1. sensitivity of the cell classes to the equivalence margin δ;
2. power: interval width per benchmark and budget, and questions needed to classify a cell;
3. the noise floor rebuilt from frame-shift passes only or option-order passes only, and a probability-based
   (soft) flip in place of the letter flip;
4. where the changes of noise-equivalent cells fall: noise-fragile vs. noise-robust questions.
"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import norm, binomtest
from paths import MAIN, OUT
import core, rep

DELTAS = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
FRAME, OPTION = ["frame_shift", "frame_shift_q"], ["option_cycle", "option_reverse"]
REP_B = ["videomme", "egoschema", "longvideobench"]


def load(model, b):
    if model == "llava_ov":
        core.DATA = MAIN
        core.METHOD_GRID = {"flashvid": [0.05, 0.10, 0.25], "fastvid": [0.05, 0.10, 0.25], "holitom": [0.05, 0.10, 0.25],
                            "visionzip": [0.05, 0.10, 0.25], "scissor": [0.05, 0.10, 0.25], "dycoke": [0.9, 1.0],
                            "uniform": [0.05, 0.10, 0.25], "random": [0.05, 0.10, 0.25]}
        return core.Bench(b)
    return rep.bench(model, b)


def probs(lp):
    p = np.exp(np.asarray(lp, float) - np.max(lp)); return p / p.sum()


def excess_sub(X, flip, ok, idx, passes, NF=None):
    """Δ with the leave-one-out construction restricted to a subset of noise passes.
    flip: per-question flip of the compressed pass (bool or soft); NF: per-pass noise flips (default X.NF)."""
    NF = NF or X.NF
    vals = []
    for k in passes:
        st = np.ones(len(idx), bool)
        for j in passes:
            if j != k:
                st &= ~X.NF[j][idx]          # stability is always defined by hard (letter) flips
        st &= ok[idx]
        if st.sum() == 0:
            return np.nan
        vals.append(np.mean(flip[idx][st]) - np.mean(NF[k][idx][st]))
    return float(np.mean(vals))


def interval(X, flip, ok, passes, NF=None):
    d = excess_sub(X, flip, ok, np.arange(X.n), passes, NF)
    db = np.array([excess_sub(X, flip, ok, b, passes, NF) for b in X.boot])
    lo, hi = np.nanpercentile(db, [2.5, 97.5])
    return d, lo, hi, float(np.nanstd(db))


def cls(lo, hi, delta=core.DELTA):
    return "lossy" if lo > 0 else ("noise-equivalent" if hi < delta else "inconclusive")


rows, frag_rows = [], []
models = [("llava_ov", core.BENCH), ("llava_video", REP_B), ("qwen3vl", REP_B)]
for model, benches in models:
    for b in benches:
        X = load(model, b)
        # soft flips: probability that a letter sampled from the pass differs from one sampled from the full model
        P0 = [probs(r["full"]["logp"]) for r in X.R]
        soft = lambda key_fn: np.array([1 - float(np.dot(p0, probs(key_fn(r)))) if key_fn(r) is not None else np.nan
                                        for p0, r in zip(P0, X.R)])
        NF_soft = {k: soft(lambda r, k=k: r["noise"][k]["logp"]) for k in core.NK}
        fragile = np.any(np.stack([X.NF[k] for k in core.NK]), 0)
        for m, s in core.cell_keys():
            c = X.cells[(m, s)]; ok, flip = c["ok"], c["flip"]
            d4, lo4, hi4, se4 = interval(X, flip, ok, core.NK)
            dF, loF, hiF, _ = interval(X, flip, ok, FRAME)
            dO, loO, hiO, _ = interval(X, flip, ok, OPTION)
            fs = soft(lambda r: (r["methods"].get(m, {}).get(s) or {}).get("logp"))
            dS, loS, hiS, _ = interval(X, fs, ok, core.NK, NF_soft)
            rows.append(dict(model=model, benchmark=b, method=m, setting=s, n=int(ok.sum()), excess=d4, lo=lo4, hi=hi4, se=se4,
                             cls=cls(lo4, hi4), frame_only=dF, frame_lo=loF, frame_hi=hiF, frame_cls=cls(loF, hiF),
                             option_only=dO, option_lo=loO, option_hi=hiO, option_cls=cls(loO, hiO),
                             soft=dS, soft_lo=loS, soft_hi=hiS, soft_cls=cls(loS, hiS)))
            # where do the changes fall?
            cor = c["cor"]; harm = X.fcor & ~cor & ok; help_ = ~X.fcor & cor & ok
            nh_f = np.mean([(X.fcor & ~X.noise_cor[k])[fragile].sum() for k in core.NK])
            ng_f = np.mean([(~X.fcor & X.noise_cor[k])[fragile].sum() for k in core.NK])
            n01, n10 = int(harm.sum()), int(help_.sum())
            frag_rows.append(dict(model=model, benchmark=b, method=m, setting=s, cls=cls(lo4, hi4),
                                  n_fragile=int(fragile.sum()), n_robust=int((~fragile).sum()),
                                  harm_fragile=int(harm[fragile].sum()), help_fragile=int(help_[fragile].sum()),
                                  harm_robust=int(harm[~fragile].sum()), help_robust=int(help_[~fragile].sum()),
                                  noise_harm_fragile=nh_f, noise_help_fragile=ng_f,
                                  acc_drop_pts=100 * (X.fcor[ok].mean() - cor[ok].mean()),
                                  mcnemar_p=binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0,
                                  fragile_p=binomtest(int(harm[fragile].sum()), int(harm[fragile].sum() + help_[fragile].sum()), 0.5).pvalue
                                  if harm[fragile].sum() + help_[fragile].sum() else 1.0))
        print(model, b, "done", flush=True)

R = pd.DataFrame(rows); R.to_csv(f"{OUT}/robustness_cells.csv", index=False)
F = pd.DataFrame(frag_rows); F.to_csv(f"{OUT}/fragile_breakdown.csv", index=False)

# ---- 1. δ sensitivity (classes only change between noise-equivalent and inconclusive) ----
sens = []
for model in ["llava_ov", "llava_video", "qwen3vl"]:
    d = R[(R.model == model) & (R.method != "dycoke")]
    for delta in DELTAS:
        c = np.where(d.lo > 0, "lossy", np.where(d.hi < delta, "noise-equivalent", "inconclusive"))
        for s in ["0.05", "0.1", "0.25"]:
            msk = (d.setting == s).values
            sens.append(dict(model=model, delta=delta, setting=s, cells=int(msk.sum()),
                             noise_equivalent=int((c[msk] == "noise-equivalent").sum()),
                             inconclusive=int((c[msk] == "inconclusive").sum()), lossy=int((c[msk] == "lossy").sum()),
                             changed_vs_003=int((c[msk] != d.cls.values[msk]).sum())))
pd.DataFrame(sens).to_csv(f"{OUT}/delta_sensitivity.csv", index=False)

# ---- 2. power: SE of Δ per benchmark/budget (median over methods), MDE and questions needed ----
pw = []
for (model, b, s), g in R[R.method != "dycoke"].groupby(["model", "benchmark", "setting"]):
    n = int(g.n.median()); se = float(g.se.median())
    row = dict(model=model, benchmark=b, setting=s, n=n, se=se, halfwidth=1.96 * se, mde80=2.80 * se)
    for delta in [0.02, 0.03, 0.05]:
        row[f"power_eq_true0_d{delta}"] = float(norm.cdf(delta / se - 1.96))
        row[f"n80_true0_d{delta}"] = float(n * (2.80 * se / delta) ** 2)
    row["n80_true001_d0.03"] = float(n * (2.80 * se / 0.02) ** 2)
    pw.append(row)
pd.DataFrame(pw).to_csv(f"{OUT}/power.csv", index=False)

# ---- 3. noise-pass subsets and soft flips: agreement with the 4-pass letter classes ----
agree = {}
for model in ["llava_ov", "llava_video", "qwen3vl"]:
    d = R[R.model == model]
    agree[model] = {k: dict(agree=float((d[f"{k}_cls"] == d.cls).mean()),
                            lossy_both=int(((d[f"{k}_cls"] == "lossy") & (d.cls == "lossy")).sum()),
                            lossy_main=int((d.cls == "lossy").sum()), lossy_alt=int((d[f"{k}_cls"] == "lossy").sum()),
                            ne_main=int((d.cls == "noise-equivalent").sum()), ne_alt=int((d[f"{k}_cls"] == "noise-equivalent").sum()),
                            corr=float(np.corrcoef(d.excess, d[{"frame": "frame_only", "option": "option_only", "soft": "soft"}[k]])[0, 1]))
                    for k in ["frame", "option", "soft"]}
json.dump(agree, open(f"{OUT}/robustness_summary.json", "w"), indent=1)
print(json.dumps(agree, indent=1))
print(pd.DataFrame(sens).query("setting == '0.25'").to_string())
