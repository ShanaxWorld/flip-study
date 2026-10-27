"""Replication analysis: same per-benchmark measures as the main study (core.py), any model/method grid."""
from paths import ROOT, MAIN, REP, MANIFESTS  # noqa: F401
import sys, json, numpy as np
import core
REP = REP
GRIDS = {"llava_video": ["flashvid", "holitom", "visionzip", "uniform", "random"], "qwen3vl": ["flashvid", "uniform", "random"]}

def bench(model, b):
    core.DATA = REP
    core.METHOD_GRID = {m: [0.05, 0.10, 0.25] for m in GRIDS[model]}
    return core.Bench(f"{model}_{b}")

def report(model, b):
    X = bench(model, b)
    fr = X.fragility()
    cells = [X.describe(m, s) for m, s in core.cell_keys()]
    L = [c for c in cells if c["cls"] == "lossy"]
    h, g = sum(c["harmful"] for c in L), sum(c["helpful"] for c in L)
    nh = sum(int((X.fcor & ~X.noise_cor[k]).sum()) for k in core.NK); ng = sum(int((~X.fcor & X.noise_cor[k]).sum()) for k in core.NK)
    q = np.digitize(X.margin, np.quantile(X.margin, [0.25, 0.5, 0.75]))
    na = np.mean(np.stack([X.NF[k] for k in core.NK]), 0)
    cl = np.mean(np.stack([X.cells[(m, s)]["flip"] for m, s in core.cell_keys() if s in ("0.05", "0.1")]), 0)
    out = {"model": model, "benchmark": b, "n": X.n, "videos": int(len(X.videos)), "full_acc": float(X.fcor.mean()),
           "fragility": fr, "noise_range": X.noise_range(), "lossy_hs": h / max(h + g, 1), "noise_hs": nh / max(nh + ng, 1),
           "n_lossy": len(L), "q_noise": [float(na[q == i].mean()) for i in range(4)],
           "q_comp": [float(cl[q == i].mean()) for i in range(4)],
           "cells": [{k: v for k, v in c.items() if k != "boot"} for c in cells]}
    return out

if __name__ == "__main__":
    model, b = sys.argv[1], sys.argv[2]
    r = report(model, b)
    json.dump(r, open(f"{REP}/{model}_summary_{b}.json", "w"), indent=1, default=float)
    print(f"{model}/{b}: n={r['n']} videos={r['videos']} full acc {100*r['full_acc']:.1f}% | fragility {100*r['fragility'][0]:.1f}% "
          f"({100*r['fragility'][1]:.1f}-{100*r['fragility'][2]:.1f}) | noise range {r['noise_range']:.1f} pts")
    print(f"  harmful share: lossy cells {r['lossy_hs']:.3f} ({r['n_lossy']} cells) vs noise {r['noise_hs']:.3f}")
    print("  quartiles noise", np.round(r["q_noise"], 3), "| compressed<=10%", np.round(r["q_comp"], 3))
    for c in r["cells"]:
        print(f"  {c['method']:9s} {c['setting']:5s} ret {c['retention']:.3f} rel {c['rel_acc']:6.1f} flip {c['flip']:.3f} "
              f"Δ {c['excess']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}] {c['cls']:17s} hs {c['harmful_share']:.2f}")
