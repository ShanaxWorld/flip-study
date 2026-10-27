# Analysis plan, dated 15 October 2026

The plan below was written on 15 October 2026, after the pilot runs and before any confirmatory run, and was sent to the supervising author as a dated PDF. Each Kaggle run also saved a timestamped run record and the hash of its question manifest. We reproduce the parts of the plan that define the study. The wording is shortened in places; every decision rule is given exactly as fixed.

## Status at the time of writing

The only data seen were the pilot runs on Video-MME chunks 01, 05, 10, 15 and 19 with LLaVA-OneVision-7B (606 questions from 202 videos). Earlier exploratory work had also used chunks 03, 07, 12, 14 and 17. All ten chunks were excluded from every confirmatory analysis. The pilot was used to choose the outcome measure, the noise passes and the thresholds, and those choices were then fixed.

## What the pilot showed

The uncompressed model changed its own answer about as often as good compressors did, so raw flip counts could not measure compression damage. Four harmless changes were tested on the full model: frames sampled half an interval later (19.5% of answers changed), a quarter interval later (19.3%), options cyclically reordered (20.6%) and options reversed (25.9%). Of all questions, 45.5% changed under at least one of the four and 3% under all four. Noise changes were balanced (48% right to wrong). Good compressors at 10% to 25% of tokens changed 14% to 20% of answers. On questions stable under three noise passes, the fourth pass still changed 7.7%; compression at 25% changed 4% to 8% of those questions and every method at 5% changed 12% to 25%. Damage leaned from right to wrong: 59% to 72% of right-or-wrong changes at 5% were harmful. The full model gave identical answers and log-probabilities on all 606 questions in two separate runs.

## Models, methods and noise passes

Primary model: LLaVA-OneVision-7B (llava-hf/llava-onevision-qwen2-7b-ov-hf), 32 uniformly sampled frames, 196 tokens per frame, fp16, greedy letter scoring. All confirmatory hypotheses are tested on this model. LLaVA-Video-7B and Qwen3-VL-8B are replication models and test H1 to H3 only. Every method passed the pilot gate (relative accuracy within 2 points of a published LLaVA-OneVision-7B Video-MME value, or a 95% interval covering it). Methods: FlashVID, FastVID, HoliTom, VisionZip and LLaVA-Scissor at 5%, 10% and 25%; DyCoke at K = 0.9 and 1.0 (35% and 28% of tokens); uniform pooling and random drop at 5%, 10% and 25%. That is 23 compressed passes per question. Four noise passes of the full model are run for every question: frames half an interval later, frames a quarter interval later, options shifted cyclically by one place, and options in reverse order, with answers mapped back before comparison.

## Benchmarks and sampling

Seed 20261015 for all sampling. Video-MME: all questions of the videos in chunks 02, 08, 16, 18 and 20, drawn at random from the ten chunks never used before (639 questions, 213 videos). MVBench: 25 questions per task over 20 tasks, with the NTU RGB+D videos removed because they are not distributed (about 500). EgoSchema: the public 500-question subset with answers. LongVideoBench: validation set, 125 questions per duration group. MLVU: development set, 72 questions per multiple-choice task (504). All questions use the video only and the pilot prompt. Questions with fewer than two options are removed before sampling (one MVBench question). A question is otherwise dropped only if its video cannot be decoded. Replication: 300 questions each from Video-MME, EgoSchema and LongVideoBench, drawn from the same manifests.

## Outcome measures

Primary: excess flips Δ<sub>m</sub> as defined in Eq. (3) of the paper: Δ<sub>m</sub> = ¼ Σ<sub>k</sub> [F<sub>m</sub>(S<sub>k</sub>) − F<sub>n<sub>k</sub></sub>(S<sub>k</sub>)], where S<sub>k</sub> holds the questions on which the three noise passes other than n<sub>k</sub> reproduce the full answer and F<sub>x</sub>(S) is the share of questions in S whose answer changes under pass x, with cells classified from the 95% interval as lossy (lower bound above 0), noise-equivalent (upper bound below δ = 0.03) or inconclusive. Decision-lossless budget: the smallest budget whose cell and all larger budgets' cells are noise-equivalent, else above 25%. Other measures: fragility (H1), harmful share (H3), noise accuracy range (H4), confidence quartiles of the full model's margin (H5), relative accuracy (descriptive), KL divergence (secondary), prefill FLOPs, wall time and peak memory (descriptive).

## Decision rules

"Cells at a budget" means the seven methods with that budget on five benchmarks, 35 cells.

- H1 Fragility: supported if fragility is at least 30% on at least 4 of 5 benchmarks.
- H2a Damage at low budgets: supported if at least 70% of the 35 cells at 5% are lossy.
- H2b No damage at 25%: supported if at least 70% of the 35 cells at 25% are noise-equivalent.
- H2c Methods differ: supported if, at 10%, on at least 3 of 5 benchmarks at least one method is noise-equivalent and another is lossy.
- H3 Direction: supported if the pooled harmful share over lossy cells minus the pooled harmful share over the four noise passes is at least 0.10, with a 95% interval above 0.
- H4 Resolution: supported if the noise accuracy range is at least 1.0 point on at least 4 of 5 benchmarks and, at 25%, fewer than 20% of method pairs differ significantly in accuracy.
- H5 Confidence: supported if, for pooled noise passes and for pooled compressed passes at 10% or less, the flip rate in the lowest confidence quartile is at least 3 times the highest, on at least 4 of 5 benchmarks.

The plan stated that the central claim needs H1, H2a, H2b and H3; that a failure of H2b would be reported as measurable damage even at 25%; and that a failure of H3 would mean excess flips are reported as undirected churn.

## Statistical analysis

All intervals are 95% percentile intervals from a cluster bootstrap over videos (2,000 resamples, seed 0). Δ and its interval use the same resamples for all terms. Pooled harmful shares pool counts over cells, then bootstrap videos within each benchmark. Pairwise accuracy uses exact McNemar tests between method pairs at 25% with Holm correction within each benchmark (21 pairs). No correction across hypotheses. A method that fails on a video is analysed on the questions where it ran; cells below 95% coverage are flagged and cells below 80% are excluded from H2 counts. Every cell of the main study had full coverage. Every benchmark runs to its full manifest before any hypothesis is evaluated, and gate refits, prompt changes or method tuning on confirmatory data are not allowed.

## Secondary and exploratory analyses

Secondary (planned, reported in full, not deciding the claims): replication of H1, H2a, H2b and H3 on LLaVA-Video-7B (FlashVID, HoliTom, VisionZip, uniform pooling, random drop) and Qwen3-VL-8B (FlashVID, uniform pooling, random drop); KL divergence beyond held-out noise KL; subgroups by video length and task type; excess flips against prefill FLOPs. Exploratory (labelled as such): calibration questions a Learn-then-Test procedure would need to certify a budget; whether the confidence margin predicts which questions a compressor flips; overlap of flipped questions across methods.

## Stopping conditions

A run stops, and the plan is revised before any result is read, if the equivalence check fails, if the prefix-cache check changes more than 1% of answers on its test video, or if full-model answers differ between two runs of the same input. None of these conditions occurred.
