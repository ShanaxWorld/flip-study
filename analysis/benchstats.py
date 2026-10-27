from paths import ROOT, MAIN, REP, MANIFESTS  # noqa: F401
from paths import OUT
import json, numpy as np, pandas as pd, core, rep
rows = []
def stats(X, model, b):
    held = np.mean([X.NF[k][X.STABLE[k]].mean() for k in core.NK])
    nh = sum(int((X.fcor & ~X.noise_cor[k]).sum()) for k in core.NK); ng = sum(int((~X.fcor & X.noise_cor[k]).sum()) for k in core.NK)
    fr = X.fragility()
    return dict(model=model, benchmark=b, n=X.n, videos=len(X.videos), full_acc=X.fcor.mean(), fragility=fr[0], frag_lo=fr[1], frag_hi=fr[2],
                **{f"flip_{k}": X.NF[k].mean() for k in core.NK}, heldout=held, stable_share=np.mean([X.STABLE[k].mean() for k in core.NK]),
                noise_range=X.noise_range(), noise_hs=nh / (nh + ng),
                **{f"acc_{k}": X.noise_cor[k].mean() for k in core.NK})
for b in core.BENCH:
    core.DATA = MAIN
    X = core.Bench(b); rows.append(stats(X, "llava_ov", b))
for model in ["llava_video", "qwen3vl"]:
    for b in ["videomme", "egoschema", "longvideobench"]:
        rows.append(stats(rep.bench(model, b), model, b))
d = pd.DataFrame(rows); d.to_csv(f"{OUT}/bench_stats.csv", index=False)
pd.set_option("display.width", 250); print(d.round(3).to_string()); print(core.NK)
