"""Post hoc, added for the journal version:
1. cells whose excess-flip interval lies above 0 AND below the margin (detectably but slightly lossy);
2. a non-inferiority test of accuracy for every cell: the cell keeps accuracy if the upper end of the 95%
   video-bootstrap interval of the paired accuracy loss (full minus compressed) lies below a margin
   (2 or 3 points), using the same resamples as the excess-flip interval.
"""
import json
import numpy as np, pandas as pd
from paths import MAIN, OUT
import core, rep, ext_ov, ext_qw

MARGINS = [0.02, 0.03]


def acc_loss(X, c, idx):
    ok = c["ok"][idx]
    return float(X.fcor[idx][ok].mean() - c["cor"][idx][ok].mean())


rows = []
def run(model, b, X, keys):
    for m, s in keys:
        c = X.cells[(m, s)]
        d = X.describe(m, s)
        loss = acc_loss(X, c, np.arange(X.n))
        lb = np.array([acc_loss(X, c, i) for i in X.boot])
        hi = float(np.percentile(lb, 97.5)); lo = float(np.percentile(lb, 2.5))
        r = dict(model=model, benchmark=b, method=m, setting=s, excess=d["excess"], lo=d["lo"], hi=d["hi"], cls=d["cls"],
                 overlap=bool(d["lo"] > 0 and d["hi"] < core.DELTA), acc_loss=loss, acc_loss_lo=lo, acc_loss_hi=hi)
        for mg in MARGINS:
            r[f"noninf_{int(mg*100)}"] = bool(hi < mg)
        rows.append(r)

# LLaVA-OneVision main study (eight planned methods + V-CAST)
for b in core.BENCH:
    X = ext_ov.load(b)
    keys = list(core.cell_keys()) + ext_ov.VC
    run("llava_ov_main", b, X, keys)
    print(b, flush=True)
# replication models (V-CAST on Qwen3-VL)
for b in ["videomme", "egoschema", "longvideobench"]:
    X = rep.bench("llava_video", b); run("llava_video", b, X, list(core.cell_keys()))
    X = ext_qw.load("qwen3vl", b)
    keys = [(m, s) for m in rep.GRIDS["qwen3vl"] for s in ["0.05", "0.1", "0.25"]] + [("vcast", s) for s in ["0.05", "0.1", "0.25"]]
    run("qwen3vl", b, X, keys)
    print("rep", b, flush=True)

D = pd.DataFrame(rows); D.to_csv(f"{OUT}/noninferiority.csv", index=False)
ne = D[D.cls == "noise-equivalent"]
summ = {"cells": int(len(D)), "overlap_cells": D[D.overlap][["model", "benchmark", "method", "setting", "excess", "lo", "hi"]].to_dict("records"),
        "noise_equivalent": int(len(ne)),
        "ne_noninf_2": int(ne.noninf_2.sum()), "ne_noninf_3": int(ne.noninf_3.sum()),
        "by_model": {m: {"ne": int((ne.model == m).sum()), "noninf_2": int(ne[ne.model == m].noninf_2.sum()),
                         "noninf_3": int(ne[ne.model == m].noninf_3.sum())} for m in ["llava_ov_main", "llava_video", "qwen3vl"]},
        "all_cells_noninf_3_by_budget": {s: int(D[(D.setting == s)].noninf_3.sum()) for s in ["0.05", "0.1", "0.25"]},
        "median_halfwidth_acc_loss_pts": float(100 * np.median((D.acc_loss_hi - D.acc_loss_lo) / 2))}
json.dump(summ, open(f"{OUT}/noninferiority_summary.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(json.dumps(summ, indent=1))
print(ne[["model", "benchmark", "method", "setting", "excess", "hi", "acc_loss", "acc_loss_hi", "noninf_2", "noninf_3"]].round(3).to_string())
