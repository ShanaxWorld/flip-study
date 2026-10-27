from paths import ROOT, MAIN, REP, MANIFESTS  # noqa: F401
import json, numpy as np, pandas as pd
from core import BENCH, BNAME, MNAME, MAIN7, NK
from paths import OUT
res = json.load(open(f"{OUT}/results_cross.json"))
df = pd.read_csv(f"{OUT}/cells_all.csv", dtype={"setting": str})
pool = pd.read_csv(f"{OUT}/pooled_methods.csv", dtype={"setting": str})
SYM = {"lossy": r"$^{\blacktriangle}$", "noise-equivalent": r"$^{\circ}$", "inconclusive": ""}
def cellfmt(r):
    v = 100 * r.excess
    return ("0.0" if abs(v) < 0.05 else f"{v:+.1f}") + SYM[r.cls]
# ---- Table: Δ (percentage points) per benchmark, method, budget ----
order = MAIN7 + ["dycoke"]
lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Excess flips $\Delta$ (percentage points) for every cell. $^{\blacktriangle}$ lossy (95\% CI above 0); $^{\circ}$ noise-equivalent (CI below $\delta=3$ points); unmarked: inconclusive. DyCoke runs at its own budgets (35\% and 28\%). The last column pools the five benchmarks with equal weight (post hoc).}",
         r"\label{tab:cells}", r"\small", r"\begin{tabular}{ll" + "r" * 6 + "}", r"\toprule",
         "Method & Budget & " + " & ".join(BNAME[b] for b in BENCH) + r" & Pooled \\", r"\midrule"]
for m in order:
    ss = ["K0.9", "K1.0"] if m == "dycoke" else ["0.05", "0.1", "0.25"]
    for i, s in enumerate(ss):
        lab = {"0.05": "5\\%", "0.1": "10\\%", "0.25": "25\\%", "K0.9": "35\\%", "K1.0": "28\\%"}[s]
        row = [cellfmt(df[(df.benchmark == b) & (df.method == m) & (df.setting == s)].iloc[0]) for b in BENCH]
        row.append(cellfmt(pool[(pool.method == m) & (pool.setting == s)].iloc[0]))
        lines.append((MNAME[m] if i == 0 else "") + f" & {lab} & " + " & ".join(row) + r" \\")
    lines.append(r"\addlinespace[2pt]")
lines[-1] = r"\bottomrule"; lines += [r"\end{tabular}", r"\end{table*}"]
open(f"{OUT}/table_cells.tex", "w").write("\n".join(lines))
# ---- Table: hypotheses ----
H = res
def yn(b): return "Yes" if b in (True, 1, 1.0, "True") else "No"
hyp = [("H1 Fragility", "fragility $\\geq$ 30\\% in $\\geq$4 of 5", f"{H['H1']['n_benchmarks_ge_30']:.0f} of 5 (34.9--43.3\\%)", yn(H['H1']['supported'])),
       ("H2a Damage at 5\\%", "$\\geq$70\\% of 35 cells lossy", f"{H['H2a']['lossy']} of 35 ({100*H['H2a']['share']:.0f}\\%)", yn(H['H2a']['supported'])),
       ("H2b No damage at 25\\%", "$\\geq$70\\% of 35 cells noise-equivalent", f"{H['H2b']['noise_equivalent']} of 35 ({100*H['H2b']['share']:.0f}\\%); {H['H2b']['classes_at_25']['inconclusive']} inconclusive, {H['H2b']['classes_at_25']['lossy']} lossy", yn(H['H2b']['supported'])),
       ("H2c Methods differ", "at 10\\%, one noise-equivalent and one lossy, $\\geq$3 of 5", f"{H['H2c']['n_met']} of 5", yn(H['H2c']['supported'])),
       ("H3 Direction", "harmful-share difference $\\geq$0.10, CI $>$ 0", f"{H['H3']['diff']:.3f} [{H['H3']['ci'][0]:.3f}, {H['H3']['ci'][1]:.3f}]", yn(H['H3']['supported'])),
       ("H4 Resolution", "noise range $\\geq$1 pt in $\\geq$4 of 5; $<$20\\% of pairs differ at 25\\%", f"{H['H4']['n_range_ge_1']:.0f} of 5; {H['H4']['sig_pairs']} of {H['H4']['pairs']} pairs", yn(H['H4']['supported'])),
       ("H5 Confidence", "Q1 $\\geq$3$\\times$ Q4 (noise and $\\leq$10\\%), $\\geq$4 of 5", f"{H['H5']['n_met']} of 5", yn(H['H5']['supported']))]
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Confirmatory hypotheses on LLaVA-OneVision-7B (2,618 questions, five benchmarks).}", r"\label{tab:hyp}", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.45cm}>{\raggedright\arraybackslash}p{2.75cm}>{\raggedright\arraybackslash}p{2.35cm}l}", r"\toprule", r"Hypothesis & Supported if & Result & Met \\", r"\midrule"]
lines += [" & ".join(h) + r" \\" for h in hyp]
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/table_hypotheses.tex", "w").write("\n".join(lines))
# ---- Table: benchmark summary ----
S = {b: json.load(open(f"{MAIN}/summary_{b}.json")) for b in BENCH}
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Benchmarks, full-model accuracy and the noise floor. Flip rates of the four harmless passes: frames shifted by $\frac12$ and $\frac14$ of a sampling interval, options cycled and reversed.}", r"\label{tab:bench}", r"\footnotesize",
         r"\setlength{\tabcolsep}{2.6pt}", r"\begin{tabular}{lrrrrrrrr}", r"\toprule", r"Benchmark & Qs & Videos & Acc. & $\frac12$ & $\frac14$ & cyc. & rev. & Fragility \\", r"\midrule"]
for b in BENCH:
    nz = {n["pass"]: n["flip"] for n in S[b]["noise"]}
    lines.append(f"{BNAME[b].replace('LongVideoBench', 'LongVideoB.')} & {res['n'][b]} & {res['videos'][b]} & {100*res['full_acc'][b]:.1f} & " + " & ".join(f"{100*nz[k]:.1f}" for k in NK) + f" & {100*res['H1']['fragility'][b][0]:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/table_benchmarks.tex", "w").write("\n".join(lines))
print(open(f"{OUT}/table_hypotheses.tex").read()); print(open(f"{OUT}/table_cells.tex").read()[:2500])
