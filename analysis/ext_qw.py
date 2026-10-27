"""Revision runs, V-CAST on the 900 shared questions (post hoc):
Qwen3-VL-8B with V-CAST's official code, and LLaVA-OneVision-7B with our port (the same questions, taken from the
LLaVA-OV revision run). Scored with the planned measure; pooled over the three benchmarks exactly as xmodel.py.
The Qwen3-VL revision notebook reran the uncompressed pass; it reproduces the replication run's answers and
log-probabilities on every question, which is checked before merging."""
import json
import numpy as np, pandas as pd
from scipy.stats import binomtest
from paths import OUT, ROOT
import core, rep

REV = ROOT / "data" / "revision"
B3 = ["videomme", "egoschema", "longvideobench"]
rep.GRIDS["llava_ov"] = ["flashvid", "fastvid", "holitom", "visionzip", "scissor", "uniform", "random"]


def load(model, b):
    X = rep.bench(model, b)
    src = "qwen3vl" if model == "qwen3vl" else "llava_ov"
    ext = {r["qid"]: r for r in map(json.loads, open(REV / src / f"{b}_results.jsonl"))}
    for r in X.R:
        e = ext[r["qid"]]
        assert e["full"]["pred"] == r["full"]["pred"] and e["full"]["logp"] == r["full"]["logp"], (model, b, r["qid"])
        r["methods"]["vcast"] = e["methods"]["vcast"]
    for s in ["0.05", "0.1", "0.25"]:
        get = lambda r: r["methods"]["vcast"][s]
        pred = np.array([get(r)["pred"] for r in X.R])
        X.cells[("vcast", s)] = dict(ok=np.ones(X.n, bool), pred=pred, cor=np.array([bool(get(r)["correct"]) for r in X.R]),
                                    flip=pred != X.full, kl=np.array([core.kl(r["full"]["logp"], get(r)["logp"]) for r in X.R]),
                                    flops=np.array([get(r)["flops"] / r["full"]["flops"] for r in X.R]),
                                    ret=np.array([get(r)["retention"] for r in X.R]))
    return X


def directed(X, m, s, idx):
    c = X.cells[(m, s)]; harm = X.fcor & ~c["cor"] & c["ok"]; vals = []
    for k in core.NK:
        st = X.STABLE[k][idx] & c["ok"][idx]
        vals.append(harm[idx][st].mean() - (X.fcor & ~X.noise_cor[k])[idx][st].mean())
    return float(np.mean(vals))


if __name__ == "__main__":
    cells, pooled = [], []
    for model in ["qwen3vl", "llava_ov"]:
        Xs = {b: load(model, b) for b in B3}
        for b, X in Xs.items():
            for s in ["0.05", "0.1", "0.25"]:
                d = {k: v for k, v in X.describe("vcast", s).items() if k not in ("boot",)}
                c = X.cells[("vcast", s)]
                n01 = int((X.fcor & ~c["cor"]).sum()); n10 = int((~X.fcor & c["cor"]).sum())
                db = np.array([directed(X, "vcast", s, i) for i in X.boot])
                cells.append(dict(model=model, **d, acc_drop_pts=100 * (X.fcor.mean() - c["cor"].mean()),
                                  mcnemar_p=binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0,
                                  harm_excess=directed(X, "vcast", s, np.arange(X.n)),
                                  harm_lo=float(np.percentile(db, 2.5)), harm_hi=float(np.percentile(db, 97.5))))
        PB = {}
        for b, sq in zip(B3, np.random.SeedSequence(core.SEED).spawn(3)):
            g = np.random.default_rng(sq); X = Xs[b]
            PB[b] = [np.concatenate([X.v_rows[j] for j in g.integers(0, len(X.videos), len(X.videos))]) for _ in range(core.B)]
        for s in ["0.05", "0.1", "0.25"]:
            f = lambda idx: np.mean([Xs[b].excess(Xs[b].cells[("vcast", s)]["flip"], Xs[b].cells[("vcast", s)]["ok"], idx[b]) for b in B3])
            d = f({b: np.arange(Xs[b].n) for b in B3})
            db = [f({b: PB[b][i] for b in B3}) for i in range(core.B)]
            lo, hi = np.nanpercentile(db, [2.5, 97.5])
            acc = np.mean([Xs[b].cells[("vcast", s)]["cor"].mean() / Xs[b].fcor.mean() for b in B3]) * 100
            pooled.append(dict(model=model, method="vcast", setting=s, excess=d, lo=lo, hi=hi, rel_acc=acc,
                               cls="lossy" if lo > 0 else ("noise-equivalent" if hi < core.DELTA else "inconclusive")))
        print(model, "done", flush=True)
    C = pd.DataFrame(cells); C.to_csv(f"{OUT}/vcast_cells_900.csv", index=False)
    P = pd.DataFrame(pooled)
    old = pd.read_csv(f"{OUT}/xmodel_pooled.csv", dtype={"setting": str})
    allx = pd.concat([old[old.method != "vcast"], P]); allx.to_csv(f"{OUT}/xmodel_pooled_with_vcast.csv", index=False)
    pd.set_option("display.width", 250)
    print(C[["model", "benchmark", "setting", "rel_acc", "flip", "excess", "lo", "hi", "cls", "harmful_share", "acc_drop_pts",
             "mcnemar_p", "harm_excess", "harm_hi", "flops_ratio"]].round(3).to_string())
    for s in ["0.05", "0.1", "0.25"]:
        print(allx[allx.setting == s].sort_values(["model", "excess"]).round(3).to_string(index=False))
