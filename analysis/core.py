"""Per-benchmark measures, computed exactly as the Kaggle notebook's analysis cell (same bootstrap, seed 0)."""
from paths import ROOT, MAIN, REP, MANIFESTS  # noqa: F401
import json, os
import numpy as np

DATA = MAIN
BENCH = ["videomme", "mvbench", "egoschema", "longvideobench", "mlvu"]
BNAME = {"videomme": "Video-MME", "mvbench": "MVBench", "egoschema": "EgoSchema",
         "longvideobench": "LongVideoBench", "mlvu": "MLVU"}
METHOD_GRID = {"flashvid": [0.05, 0.10, 0.25], "fastvid": [0.05, 0.10, 0.25], "holitom": [0.05, 0.10, 0.25],
               "visionzip": [0.05, 0.10, 0.25], "scissor": [0.05, 0.10, 0.25], "dycoke": [0.9, 1.0],
               "uniform": [0.05, 0.10, 0.25], "random": [0.05, 0.10, 0.25]}
MNAME = {"flashvid": "FlashVID", "fastvid": "FastVID", "holitom": "HoliTom", "visionzip": "VisionZip",
         "scissor": "LLaVA-Scissor", "dycoke": "DyCoke", "uniform": "Uniform pooling", "random": "Random drop"}
MAIN7 = [m for m in METHOD_GRID if m != "dycoke"]
NK = ["frame_shift", "frame_shift_q", "option_cycle", "option_reverse"]
SEED, B, DELTA = 0, 2000, 0.03

def cell_keys():
    for m, ss in METHOD_GRID.items():
        for s in ss:
            yield m, (f"K{s}" if m == "dycoke" else str(s))

def kl(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return float((np.exp(p) * (p - q)).sum())

class Bench:
    def __init__(self, name):
        self.name = name
        R = []
        with open(os.path.join(DATA, f"{name}_results.jsonl")) as f:
            for line in f:
                try:
                    R.append(json.loads(line))
                except Exception:
                    pass
        self.R = R = [r for r in R if len(r.get("noise", {})) == 4]
        self.n = n = len(R)
        rng = np.random.default_rng(SEED)
        self.vid = vid = np.array([r["video"] for r in R]); self.videos = videos = np.unique(vid)
        self.v_rows = [np.flatnonzero(vid == v) for v in videos]
        self.boot = [np.concatenate([self.v_rows[j] for j in rng.integers(0, len(videos), len(videos))]) for _ in range(B)]
        self.full = np.array([r["full"]["pred"] for r in R]); self.fcor = np.array([r["full"]["correct"] for r in R])
        self.margin = np.array([r["full"]["margin"] for r in R])
        self.group = np.array([str(r["group"]) for r in R]); self.task = np.array([str(r["task"]) for r in R])
        self.NF = {k: np.array([r["noise"][k]["pred"] for r in R]) != self.full for k in NK}
        self.STABLE = {k: sum(self.NF[j] for j in NK if j != k) == 0 for k in NK}
        self.noise_cor = {k: np.array([bool(r["noise"][k]["correct"]) for r in R]) for k in NK}
        self.noise_kl = {k: np.array([kl(r["full"]["logp"], r["noise"][k]["logp"]) for r in R]) for k in NK}
        self.cells = {}
        for m, s in cell_keys():
            get = lambda r: r["methods"].get(m, {}).get(s)
            ok = np.array([get(r) is not None for r in R])
            pred = np.array([get(r)["pred"] if o else "" for r, o in zip(R, ok)])
            cor = np.array([bool(get(r)["correct"]) if o else False for r, o in zip(R, ok)])
            klv = np.array([kl(r["full"]["logp"], get(r)["logp"]) if o else np.nan for r, o in zip(R, ok)])
            flops = np.array([get(r)["flops"] / r["full"]["flops"] if o else np.nan for r, o in zip(R, ok)])
            ret = np.array([get(r).get("retention_layer_avg", get(r).get("retention", 1.0)) if o else np.nan
                            for r, o in zip(R, ok)])
            self.cells[(m, s)] = dict(ok=ok, pred=pred, cor=cor, flip=(pred != self.full) & ok, kl=klv,
                                      flops=flops, ret=ret)

    # ---- the planned Δ on an index set (a bootstrap resample or a subgroup) ----
    def excess(self, flip, ok, idx):
        vals = []
        for k in NK:
            s = self.STABLE[k][idx] & ok[idx]
            if s.sum() == 0:
                return np.nan
            vals.append(flip[idx][s].mean() - self.NF[k][idx][s].mean())
        return float(np.mean(vals))

    def excess_kl(self, klv, ok, idx):
        vals = []
        for k in NK:
            s = self.STABLE[k][idx] & ok[idx]
            if s.sum() == 0:
                return np.nan
            vals.append(np.nanmean(klv[idx][s]) - self.noise_kl[k][idx][s].mean())
        return float(np.mean(vals))

    def describe(self, m, s):
        c = self.cells[(m, s)]
        ok, cor, flip = c["ok"], c["cor"], c["flip"]
        d = self.excess(flip, ok, np.arange(self.n))
        d_b = np.array([self.excess(flip, ok, b) for b in self.boot])
        lo, hi = np.nanpercentile(d_b, [2.5, 97.5])
        h, g = int((self.fcor & ~cor & ok).sum()), int((~self.fcor & cor & ok).sum())
        return dict(benchmark=self.name, method=m, setting=s, coverage=float(ok.mean()),
                    acc=float(cor[ok].mean()), full_acc=float(self.fcor[ok].mean()),
                    rel_acc=float(100 * cor[ok].mean() / self.fcor[ok].mean()),
                    flip=float(flip[ok].mean()), excess=d, lo=float(lo), hi=float(hi),
                    cls="lossy" if lo > 0 else ("noise-equivalent" if hi < DELTA else "inconclusive"),
                    harmful=h, helpful=g, harmful_share=h / (h + g) if h + g else np.nan,
                    kl=float(np.nanmean(c["kl"][ok])), flops_ratio=float(np.nanmean(c["flops"][ok])),
                    retention=float(np.nanmean(c["ret"][ok])), boot=d_b)

    def fragility(self):
        fr = float(np.any(np.stack([self.NF[k] for k in NK]), 0).mean())
        fr_b = [np.any(np.stack([self.NF[k][b] for k in NK]), 0).mean() for b in self.boot]
        return fr, float(np.percentile(fr_b, 2.5)), float(np.percentile(fr_b, 97.5))

    def noise_range(self):
        accs = [self.fcor.mean()] + [self.noise_cor[k].mean() for k in NK]
        return 100 * (max(accs) - min(accs))
