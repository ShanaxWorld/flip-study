# Third-party code

The compression methods were run from their official code wherever possible. The functions listed below were copied
or ported into the notebook sources (`notebooks/source/*/cells/06*_*.py`, and `notebooks/source/revision/*/cells/06*.py` for V-CAST) and keep their original licenses. Every
copied block carries a comment naming its source file and listing any change we made (seeded tie-break noise, fp32
sums, memory-saving matrix products, or the per-video threshold search for LLaVA-Scissor).

| Method | Source | License | What is included |
|---|---|---|---|
| FlashVID (Fan et al., ICLR 2026) | github.com/Fanziyang-v/FlashVID | MIT, Copyright (c) 2026 Turbo ([text](licenses/FlashVID-MIT.txt)) | token selection, tree-based merging, ADTS v1/v2, inner-LLM pruning, copied verbatim |
| FastVID (Shen et al., NeurIPS 2025) | github.com/LunarShen/FastVID | MIT, Copyright (c) 2025 Shen Leqi ([text](licenses/FastVID-MIT.txt)) | LLaVA-OneVision compression, ported |
| HoliTom (Shao et al., NeurIPS 2025) | github.com/cokeshao/HoliTom | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | outer and inner-LLM merging, copied as plain functions |
| VisionZip (Yang et al., CVPR 2025) | LLaVA-OneVision video port in the HoliTom repository (`visionzip/llava_arch.py`) | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | dominant and contextual token selection |
| DyCoke (Tao et al., CVPR 2025) | github.com/KD-TAO/DyCoke | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | temporal token merging (`dycole_ttm`), copied |
| LLaVA-Scissor (Sun et al., 2025) | github.com/HumanMLLM/LLaVA-Scissor | CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/) | `UnionFind` and `approximate_components`, copied verbatim; non-commercial use only |
| V-CAST (Lin et al., NeurIPS 2026) | github.com/xinyouu/V-CAST, commit f15d276 (`compressor/v_cast/modeling_qwen3_vl_v_cast.py`) | MIT, Copyright (c) 2025 LMMs-Lab, as stated in the repository's LICENSE ([text](licenses/V-CAST-MIT.txt)) | curvature-based budget allocation and per-frame token selection, copied verbatim; used unchanged on Qwen3-VL and in our LLaVA-OneVision port (revision notebooks) |

Model weights and benchmark videos are not included. They are downloaded by the notebooks from Hugging Face under
their own licenses and terms (LLaVA-OneVision-7B, LLaVA-Video-7B-Qwen2, Qwen3-VL-8B-Instruct; Video-MME, MVBench,
EgoSchema, LongVideoBench, MLVU).
