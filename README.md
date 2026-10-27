# Answer flips in training-free video token compression

Code and data for the paper

> S. M. Shah Nawaz Hossain and Sumaiya Tabassum Nimi, "Accuracy Parity Is Not Fidelity: Answer Flips in Training-Free Video Token Compression," submitted to *Multimedia Tools and Applications*, 2026.

Training-free token compression lets a video LLM answer from 5% to 25% of its visual tokens, and methods are usually compared by accuracy. This study asks how often compression changes the model's individual answers, compared with how often harmless input changes already do (sampling the frames a fraction of an interval later, or reordering the answer options). The main measure, **excess flips**, is the flip rate of a compressed pass minus that of a held-out harmless pass, on questions that the other harmless passes leave unchanged.

The repository contains every per-question output behind the paper (2,618 questions for LLaVA-OneVision-7B, 900 shared questions for LLaVA-Video-7B and Qwen3-VL-8B), the outputs of the later runs (V-CAST as a ninth method and six further harmless passes), the scripts that turn them into every number, table and figure, and the Kaggle notebooks that produced the outputs.

## Reproduce the paper's numbers (CPU, about 35 minutes)

```bash
pip install -r requirements.txt
bash analysis/run_all.sh
```

This rewrites `results/` and `figures/` from `data/`. Running it on the released data reproduces the files in this repository exactly (same bootstrap, seed 0).

## Layout

```
data/
  main/           LLaVA-OneVision-7B, one results file per benchmark (+ the per-run summaries saved on Kaggle)
  replication/    LLaVA-Video-7B and Qwen3-VL-8B on Video-MME, EgoSchema, LongVideoBench (300 questions each);
                  llava_ov_*_results.jsonl are the LLaVA-OneVision outputs on the same 900 questions
  revision/       runs added after the confirmatory analysis: llava_ov/ (V-CAST and six further harmless passes, 2,618 questions)
                  and qwen3vl/ (V-CAST with the official code, 900 questions); each repeats the full pass
  manifests/      question lists with SHA-256 hashes (main study, built from the results files; replication sample, as fixed before its runs)
analysis/         analysis scripts (run_all.sh runs them in order)
results/          every computed table (CSV/JSON) and the LaTeX tables
figures/          every figure (PDF and PNG)
notebooks/        Kaggle notebooks: main study, two replication notebooks, two notebooks for the later runs, the two pilot (port-and-gate) notebooks
  source/         the cell files and build scripts the notebooks are generated from
docs/             the analysis plan dated 15 October 2026
```

## Where each paper item comes from

| Paper item | File |
|---|---|
| Table 1 (hypotheses and outcomes) | `results/results_cross.json` (H1 to H5), `results/final_numbers.json` (replication verdicts) |
| Table 2 (methods) | V-CAST row: `results/vcast_ov_cells.csv` |
| Table 3 (benchmarks and noise floor) | `results/bench_stats.csv` |
| Table 4 (every LLaVA-OneVision cell) | `results/cells_all.csv`, `results/pooled_methods.csv`; V-CAST rows `results/vcast_ov_cells.csv`, `results/pooled_methods_with_vcast.csv` |
| Fig. 1 (the measure) | `figures/figA_measure.pdf` |
| Fig. 2 (noise floor) | `figures/fig5_fragility.pdf` |
| Fig. 3 (excess flips by budget, nine methods) | `figures/fig1_excess_by_budget.pdf` (drawn by `figs4.py`) |
| Fig. 4 (accuracy vs. excess flips) | `figures/fig2_accuracy_vs_fidelity.pdf`, `results/accuracy_tests.csv` |
| Fig. 5 (direction, three models) | `figures/figD_direction.pdf` |
| Fig. 6 (margin sensitivity and power) | `figures/figF_margin_power.pdf`, `results/delta_sensitivity.csv`, `results/power.csv` |
| Fig. 7 (three models, same questions, with V-CAST) | `figures/figC_models.pdf` (drawn by `figs4.py`), `results/xmodel_pooled_with_vcast.csv` |
| Fig. 8 (video length) | `figures/figB_length.pdf`, `results/length_subgroups.csv` |
| Further harmless passes (Section 5.7) | `results/noise_pass_accuracy.csv`, `results/noise_ext_cells.csv`, `results/noise_ext_bench.csv`, `results/noise_ext_summary.json` |
| V-CAST (Section 5.9) | `results/vcast_ov_cells.csv`, `results/vcast_cells_900.csv`, `results/pooled_methods_with_vcast.csv` |
| Non-inferiority of accuracy (Section 6.2) | `results/noninferiority.csv`, `results/noninferiority_summary.json` |
| Online Resource 1, Figs. S1 to S3 | `figures/figE_cost.pdf`, `fig3_direction.pdf`, `fig4_confidence.pdf` |
| Online Resource 1, Tables S2 to S11 | `results/cells_every_model.csv`, `pooled_methods.csv`, `xmodel_pooled.csv`, `length_subgroups.csv`, `subgroups.csv`, `overlap.csv`, `certification_n.csv` |
| Online Resource 1, Tables S12 to S15 (robustness) | `results/delta_sensitivity.csv`, `power.csv`, `robustness_cells.csv`, `robustness_summary.json`, `fragile_breakdown.csv` |
| Online Resource 1, Tables S16 to S19 (post hoc runs, non-inferiority) | `results/noise_pass_accuracy.csv`, `noise_ext_summary.json`, `noise_ext_bench.csv`, `vcast_ov_cells.csv`, `vcast_cells_900.csv`, `noninferiority.csv` |

## Data format

Each line of a `*_results.jsonl` file is one question:

| Field | Meaning |
|---|---|
| `qid`, `video`, `group`, `task` | question ID, video ID, sampling group (Video-MME duration, LongVideoBench duration group, MVBench/MLVU task) and task |
| `gt`, `n_opt` | correct option letter and number of options |
| `n_full` | number of visual tokens of the uncompressed model |
| `full` | the uncompressed pass |
| `noise` | the four harmless passes: `frame_shift` (half an interval), `frame_shift_q` (a quarter interval), `option_cycle`, `option_reverse`; option-order answers are already mapped back to the original labels |
| `methods[method][budget]` | each compressed pass (`budget` is the share of visual tokens kept, or `0.9`/`1.0` for DyCoke's K) |

Every pass records `pred` (answer letter), `logp` (log-probabilities of the option letters), `margin` (top minus runner-up), `correct`, `ntok` (visual tokens entering the language model), `vis_mean_layers` (visual tokens averaged over layers), `flops` (prefill FLOPs), `sec`, `sec_suffix` and `peak_gb` (timings and peak GPU memory on a Kaggle T4). Compressed passes also record `retention`, `retention_layer_avg` and, for LLaVA-Scissor, the threshold `tau` found for that video. No question text, option text or video is included.

The files in `data/revision/` (runs added after the confirmatory analysis) have the same format. Their `full` pass repeats the uncompressed model (identical to the released answers and log-probabilities), `methods` holds only `vcast`, and on LLaVA-OneVision `noise` holds the six further passes: `frame_shift_e` and `frame_shift_3e` (an eighth and three eighths of an interval), `option_perm1` and `option_perm2` (fixed permutations per number of options), and `prompt_alt1` and `prompt_alt2` (instruction rewordings). The Qwen3-VL revision files have no noise passes.

## Rerunning the GPU experiments

The notebooks run on Kaggle with two T4 GPUs and internet access; each benchmark is one session of up to 12 hours.

1. Accept the Hugging Face terms for MVBench (`OpenGVLab/MVBench`), MLVU (`MLVU/MVLU`) and LongVideoBench (`longvideobench/LongVideoBench`), and store a Hugging Face token as a Kaggle secret named `HF_TOKEN`.
2. Open `notebooks/flips_main_llavaov.ipynb`, set `BENCHMARK` in the configuration cell, and run all cells. `QUICK_TEST = True` runs a few questions first. The run resumes from `results.jsonl` if a session ends early.
3. The replication notebooks (`flips_rep_llava_video.ipynb`, `flips_rep_qwen3vl.ipynb`) and the notebooks for the later runs (`flips_ext_llavaov.ipynb`, `flips_ext_qwen3vl_vcast.ipynb`) work the same way; they also compare their full pass with the released answers. Their question samples are written into the notebook, with the same lists and SHA-256 hashes as `data/manifests/replication_manifests.json`.

Each run first checks our token pipeline against the library forward pass and checks that prefix caching changes no answer, and stops if either check fails. The pilot notebooks in `notebooks/pilot/` contain the port-and-gate runs that fixed the method settings. The notebooks are generated from `notebooks/source/*/cells/` with `python notebooks/source/main/build.py`, `python notebooks/source/replication/build.py` and `python notebooks/source/revision/build.py`.

## Licenses

Our code is MIT-licensed and the data in `data/` and `results/` are CC BY 4.0 (see `LICENSE`). Code copied from the compression methods keeps its original license (MIT, including V-CAST; Apache-2.0; and CC BY-NC 4.0 for two LLaVA-Scissor functions); see `THIRD_PARTY_NOTICES.md`. Model weights and benchmark videos are not redistributed.

## Citation

See `CITATION.cff`. A BibTeX entry will be added when the paper is published.
