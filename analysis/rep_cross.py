"""Replication verdicts (H1, H2a, H2b, H3) for one model over its available benchmarks, same rules as the main study."""
import sys, json, os, numpy as np
import core, rep
model = sys.argv[1]
benches = [b for b in ["videomme", "egoschema", "longvideobench"] if os.path.exists(f"{rep.REP}/{model}_{b}_results.jsonl")]
X, cells = {}, {}
for b in benches:
    X[b] = rep.bench(model, b)
    for m, s in core.cell_keys():
        cells[(b, m, s)] = X[b].describe(m, s)
fr = {b: X[b].fragility()[0] for b in benches}
h2a = [cells[(b, m, "0.05")]["cls"] == "lossy" for b in benches for m in rep.GRIDS[model]]
h2b = [cells[(b, m, "0.25")]["cls"] == "noise-equivalent" for b in benches for m in rep.GRIDS[model]]
# H3 pooled over lossy cells, video bootstrap within each benchmark (independent streams)
lossy = [k for k, c in cells.items() if c["cls"] == "lossy"]
hv, gv, nh, ng = {}, {}, {}, {}
for b in benches:
    x = X[b]
    ks = [(m, s) for (bb, m, s) in lossy if bb == b]
    hv[b] = sum(((x.fcor & ~x.cells[k]["cor"] & x.cells[k]["ok"]).astype(int) for k in ks), np.zeros(x.n, int))
    gv[b] = sum(((~x.fcor & x.cells[k]["cor"] & x.cells[k]["ok"]).astype(int) for k in ks), np.zeros(x.n, int))
    nh[b] = sum((x.fcor & ~x.noise_cor[k]).astype(int) for k in core.NK); ng[b] = sum((~x.fcor & x.noise_cor[k]).astype(int) for k in core.NK)
def stat(idx):
    H = sum(hv[b][idx[b]].sum() for b in benches); G = sum(gv[b][idx[b]].sum() for b in benches)
    Hn = sum(nh[b][idx[b]].sum() for b in benches); Gn = sum(ng[b][idx[b]].sum() for b in benches)
    return H / (H + G) - Hn / (Hn + Gn), H / (H + G), Hn / (Hn + Gn)
d, ls, ns = stat({b: np.arange(X[b].n) for b in benches})
streams = {b: np.random.default_rng(sq) for b, sq in zip(benches, np.random.SeedSequence(core.SEED).spawn(len(benches)))}
PB = {b: [np.concatenate([X[b].v_rows[j] for j in streams[b].integers(0, len(X[b].videos), len(X[b].videos))]) for _ in range(core.B)] for b in benches}
db = np.array([stat({b: PB[b][i] for b in benches})[0] for i in range(core.B)])
lo, hi = np.percentile(db, [2.5, 97.5])
need = int(np.ceil(0.8 * len(benches)))     # main rule "4 of 5" = 80% of benchmarks
out = {"model": model, "benchmarks": benches,
       "H1": {"fragility": fr, "n_ge_30": sum(v >= 0.30 for v in fr.values()), "needed": need,
              "replicated": sum(v >= 0.30 for v in fr.values()) >= need},
       "H2a": {"lossy": sum(h2a), "cells": len(h2a), "share": float(np.mean(h2a)), "replicated": float(np.mean(h2a)) >= 0.70},
       "H2b": {"noise_equivalent": sum(h2b), "cells": len(h2b), "share": float(np.mean(h2b)), "replicated": float(np.mean(h2b)) >= 0.70},
       "H3": {"lossy_cells": len(lossy), "lossy_hs": ls, "noise_hs": ns, "diff": d, "ci": [lo, hi],
              "replicated": bool(d >= 0.10 and lo > 0)}}
json.dump(out, open(f"{rep.REP}/{model}_verdicts.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=lambda o: round(float(o), 3)))
