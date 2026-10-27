"""All remaining numbers for the manuscript: subgroups, directed check, replication tables, same-question comparison."""
from paths import ROOT, MAIN, REP, MANIFESTS  # noqa: F401
import json, os, numpy as np, pandas as pd
from scipy.stats import binomtest
import core, rep
from paths import OUT
R = {}
rng_seed = core.SEED

# ---------- 1. same-question subsets of the main model (for the cross-model comparison) ----------
man = json.load(open(f"{MANIFESTS}/replication_manifests.json"))
for b in ["videomme", "egoschema", "longvideobench"]:
    keep = set(man[b]["qids"])
    with open(f"{rep.REP}/llava_ov_{b}_results.jsonl", "w") as f:
        for line in open(f"{core.DATA}/{b}_results.jsonl"):
            r = json.loads(line)
            if r["qid"] in keep:
                f.write(line)
rep.GRIDS["llava_ov"] = ["flashvid", "fastvid", "holitom", "visionzip", "scissor", "uniform", "random"]

def directed(X, m, s, idx):
    c = X.cells[(m, s)]
    harm = X.fcor & ~c["cor"] & c["ok"]
    vals = []
    for k in core.NK:
        st = X.STABLE[k][idx] & c["ok"][idx]
        nh = (X.fcor & ~X.noise_cor[k])
        vals.append(harm[idx][st].mean() - nh[idx][st].mean())
    return float(np.mean(vals))

# ---------- 2. every cell of every model with the directed check and the accuracy test ----------
rows = []
sets = [("llava_ov_main", b) for b in core.BENCH] + [(m, b) for m in ["llava_ov", "llava_video", "qwen3vl"]
                                                     for b in ["videomme", "egoschema", "longvideobench"]]
for model, b in sets:
    if model == "llava_ov_main":
        core.DATA = MAIN
        core.METHOD_GRID = {"flashvid": [0.05, 0.10, 0.25], "fastvid": [0.05, 0.10, 0.25], "holitom": [0.05, 0.10, 0.25],
                            "visionzip": [0.05, 0.10, 0.25], "scissor": [0.05, 0.10, 0.25], "dycoke": [0.9, 1.0],
                            "uniform": [0.05, 0.10, 0.25], "random": [0.05, 0.10, 0.25]}
        X = core.Bench(b)
    else:
        X = rep.bench(model, b)
    for m, s in core.cell_keys():
        d = X.describe(m, s)
        dd = directed(X, m, s, np.arange(X.n))
        db = np.array([directed(X, m, s, i) for i in X.boot])
        c = X.cells[(m, s)]
        n01 = int((X.fcor & ~c["cor"] & c["ok"]).sum()); n10 = int((~X.fcor & c["cor"] & c["ok"]).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        rows.append({"model": model, "benchmark": b, **{k: v for k, v in d.items() if k not in ("boot", "benchmark")},
                     "harm_excess": dd, "harm_lo": float(np.percentile(db, 2.5)), "harm_hi": float(np.percentile(db, 97.5)),
                     "acc_drop_pts": 100 * (X.fcor[c["ok"]].mean() - c["cor"][c["ok"]].mean()), "mcnemar_p": p,
                     "n": X.n, "full_acc_bench": float(X.fcor.mean()), "fragility_bench": X.fragility()[0]})
cells = pd.DataFrame(rows)
cells.to_csv(f"{OUT}/cells_every_model.csv", index=False)
ne = cells[cells.cls == "noise-equivalent"]
R["directed"] = {m: {"noise_equivalent": int((ne.model == m).sum()),
                     "direction_safe": int(((ne.model == m) & (ne.harm_hi < 0.015)).sum()),
                     "acc_loss_significant": int(((ne.model == m) & (ne.mcnemar_p < 0.05) & (ne.acc_drop_pts > 0)).sum())}
                 for m in ["llava_ov_main", "llava_video", "qwen3vl"]}

# ---------- 3. video-length subgroups (main study) with bootstrap CIs ----------
core.DATA = MAIN
core.METHOD_GRID = {m: [0.05, 0.10, 0.25] for m in core.MAIN7}
sub = []
for b, order in [("videomme", ["short", "medium", "long"]), ("longvideobench", ["15", "60", "600", "3600"])]:
    X = core.Bench(b)
    rng = np.random.default_rng(7)
    for g in order:
        idx = np.flatnonzero(X.group == g)
        vids = np.unique(X.vid[idx]); vrows = [idx[X.vid[idx] == v] for v in vids]
        boots = [np.concatenate([vrows[j] for j in rng.integers(0, len(vids), len(vids))]) for _ in range(1000)]
        frag = lambda ii: float(np.any(np.stack([X.NF[k][ii] for k in core.NK]), 0).mean())
        fb = [frag(i) for i in boots]
        rec = {"benchmark": b, "group": g, "n": len(idx), "videos": len(vids), "full_acc": float(X.fcor[idx].mean()),
               "fragility": frag(idx), "frag_lo": float(np.percentile(fb, 2.5)), "frag_hi": float(np.percentile(fb, 97.5))}
        for s in ["0.05", "0.1", "0.25"]:
            f = lambda ii: np.mean([X.excess(X.cells[(m, s)]["flip"], X.cells[(m, s)]["ok"], ii) for m in core.MAIN7])
            eb = [f(i) for i in boots]
            rec[f"excess_{s}"] = float(f(idx)); rec[f"excess_{s}_lo"] = float(np.nanpercentile(eb, 2.5))
            rec[f"excess_{s}_hi"] = float(np.nanpercentile(eb, 97.5))
        sub.append(rec)
pd.DataFrame(sub).to_csv(f"{OUT}/length_subgroups.csv", index=False)

# ---------- 4. replication verdicts (computed by rep_cross.py) ----------
R["verdicts"] = {m: json.load(open(f"{rep.REP}/{m}_verdicts.json")) for m in ["llava_video", "qwen3vl"]}

# ---------- 5. same questions, three models: fragility agreement ----------
def frag_flags(model, b):
    path = f"{rep.REP}/{model}_{b}_results.jsonl"
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["qid"]] = any(r["noise"][k]["pred"] != r["full"]["pred"] for k in core.NK)
    return out
agree = {}
for b in ["videomme", "egoschema", "longvideobench"]:
    F = {m: frag_flags(m, b) for m in ["llava_ov", "llava_video", "qwen3vl"]}
    q = sorted(set.intersection(*[set(v) for v in F.values()]))
    A = np.array([[F[m][x] for x in q] for m in F])
    agree[b] = {"n": len(q), "any": float(A.any(0).mean()), "all3": float(A.all(0).mean()),
                "all3_if_independent": float(np.prod(A.mean(1))), "rates": A.mean(1).round(3).tolist()}
R["fragility_agreement"] = agree
json.dump(R, open(f"{OUT}/final_numbers.json", "w"), indent=1, default=float)
print(json.dumps(R["directed"], indent=1)); print(json.dumps(agree, indent=1))
print(pd.DataFrame(sub).round(3).to_string())
