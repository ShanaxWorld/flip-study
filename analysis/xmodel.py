"""Same 900 questions, three models: Δ per method/budget pooled over the three benchmarks (equal weight)."""
from paths import OUT
import json, numpy as np, pandas as pd
import core, rep
rep.GRIDS["llava_ov"] = ["flashvid", "fastvid", "holitom", "visionzip", "scissor", "uniform", "random"]
rows = []
for model in ["llava_ov", "llava_video", "qwen3vl"]:
    Xs = {b: rep.bench(model, b) for b in ["videomme", "egoschema", "longvideobench"]}
    PB = {}
    for b, sq in zip(Xs, np.random.SeedSequence(core.SEED).spawn(3)):
        g = np.random.default_rng(sq); X = Xs[b]
        PB[b] = [np.concatenate([X.v_rows[j] for j in g.integers(0, len(X.videos), len(X.videos))]) for _ in range(core.B)]
    for m in rep.GRIDS[model]:
        for s in ["0.05", "0.1", "0.25"]:
            f = lambda idx: np.mean([Xs[b].excess(Xs[b].cells[(m, s)]["flip"], Xs[b].cells[(m, s)]["ok"], idx[b]) for b in Xs])
            d = f({b: np.arange(Xs[b].n) for b in Xs})
            db = [f({b: PB[b][i] for b in Xs}) for i in range(core.B)]
            lo, hi = np.nanpercentile(db, [2.5, 97.5])
            acc = np.mean([Xs[b].cells[(m, s)]["cor"].mean() / Xs[b].fcor.mean() for b in Xs]) * 100
            rows.append({"model": model, "method": m, "setting": s, "excess": d, "lo": lo, "hi": hi, "rel_acc": acc,
                         "cls": "lossy" if lo > 0 else ("noise-equivalent" if hi < core.DELTA else "inconclusive")})
    print(model, "done")
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/xmodel_pooled.csv", index=False)
print(df.round(3).to_string())
