#!/usr/bin/env bash
# Recomputes every number, table and figure of the paper from data/ (about 35 minutes on one CPU core).
set -e
cd "$(dirname "$0")"
python cross.py          # main study: hypotheses H1-H5, secondary and exploratory results -> results/
for m in llava_video qwen3vl; do
  for b in videomme egoschema longvideobench; do python rep.py $m $b; done   # per-benchmark replication summaries
  python rep_cross.py $m                                                       # replication verdicts
done
python final.py          # directed check, every-model cell table, video-length subgroups, fragility agreement
python xmodel.py         # three models on the same 900 questions
python benchstats.py     # benchmark-level noise statistics
python figs.py && python figs2.py && python tables.py   # figures/ and LaTeX tables
python revision.py       # δ sensitivity, power, noise-pass subsets, soft flips, fragile breakdown (about 15 minutes)
python figs3.py          # margin and power figure
python ext_ov.py         # revision runs: V-CAST on LLaVA-OneVision, six further harmless passes, noise-floor variants
python ext_qw.py         # revision runs: V-CAST on the 900 shared questions (Qwen3-VL official code, LLaVA-OV port)
python figs4.py          # Figs. 3 and 7 redrawn with V-CAST
python noninf.py         # journal version: overlap of classes, non-inferiority test of accuracy for every cell
