# ══ V-CAST (Lin et al., NeurIPS 2026) — github.com/xinyouu/V-CAST, compressor/v_cast/modeling_qwen3_vl_v_cast.py ══
# MIT License (repository LICENSE). The functions between the two rulers are copied verbatim from the official file
# (commit f15d276, 3 Oct 2026). The official release supports Qwen3-VL only; LLaVA-OneVision is listed as "planned".
# Our LLaVA-OneVision port applies the same function to the 32 frames of pooled, projected tokens (32 x 196), keeps
# every selected token at its ORIGINAL position (V-CAST is coordinate-preserving: positions keep their gaps, as in
# the official Qwen3-VL path) and keeps LLaVA-OV's newline token. The budget is the share of the 6,272 visual tokens;
# temperature 0.7 and min_k = 1 are the official defaults.
from typing import List, Optional
import torch.nn.functional as F

# ─────────────────────────────── copied verbatim from V-CAST ───────────────────────────────
DEFAULT_RETAIN_RATIO = 0.25
DEFAULT_MIN_K = 1
DEFAULT_BUDGET_TEMP = 0.7


def _v_cast_print() -> bool:
    return True


def _tokens_per_frame(grid_thw_row: torch.Tensor, spatial_merge_size: int) -> tuple[int, int, int, int]:
    t = int(grid_thw_row[0].item())
    h = int(grid_thw_row[1].item())
    w = int(grid_thw_row[2].item())
    s = max(1, int(spatial_merge_size))
    resize_h = max(1, h // s)
    resize_w = max(1, w // s)
    return t, int(resize_h), int(resize_w), int(resize_h * resize_w)


def _compute_curvature(frame_reps: torch.Tensor) -> torch.Tensor:
    """
    frame_reps: [T, D], normalized
    returns curvature: [T]
    """
    t = int(frame_reps.shape[0])
    if t <= 1:
        return torch.ones((t,), device=frame_reps.device, dtype=torch.float32)
    v_in = frame_reps[1:-1] - frame_reps[:-2]
    v_out = frame_reps[2:] - frame_reps[1:-1]
    curv = 1.0 - F.cosine_similarity(v_in, v_out, dim=-1, eps=1e-6)
    ones = torch.ones((1,), device=frame_reps.device, dtype=curv.dtype)
    return torch.cat([ones, curv, ones], dim=0)


def _allocate_budget_per_frame(
    curvature: torch.Tensor,
    total_budget: int,
    *,
    min_k: int,
    max_k: int,
) -> torch.Tensor:
    """
    Allocate per-frame budgets summing to total_budget.
    curvature: [T] float
    returns: [T] long
    """
    t = int(curvature.shape[0])
    if t <= 0:
        return torch.zeros((0,), device=curvature.device, dtype=torch.long)

    total_budget = int(max(0, min(int(total_budget), int(t * max_k))))
    min_k = int(max(0, min(int(min_k), int(max_k))))

    if total_budget < t * min_k:
        min_k_eff = 0
    else:
        min_k_eff = min_k

    base = torch.full((t,), int(min_k_eff), device=curvature.device, dtype=torch.long)
    remaining = int(total_budget - int(base.sum().item()))
    if remaining <= 0:
        return base

    weights = curvature.float().clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-6)

    raw = weights * float(remaining)
    extra = torch.floor(raw).to(torch.long)
    max_extra = max_k - min_k_eff
    if max_extra < 0:
        max_extra = 0
    extra = torch.minimum(extra, torch.full_like(extra, int(max_extra)))
    alloc = base + extra

    remaining = int(total_budget - int(alloc.sum().item()))
    if remaining > 0:
        frac = raw - torch.floor(raw)
        frac = frac.masked_fill(alloc >= max_k, -1.0)
        for _ in range(remaining):
            idx = torch.argmax(frac)
            if frac[idx].item() < 0:
                break
            alloc[idx] += 1
            if alloc[idx] >= max_k:
                frac[idx] = -1.0
    elif remaining < 0:
        over = -remaining
        order = torch.argsort(weights, descending=False)
        for idx in order:
            if over <= 0:
                break
            if alloc[idx] > min_k_eff:
                alloc[idx] -= 1
                over -= 1

    return alloc


def _v_cast_compress_frames(
    frames: torch.Tensor,  # [T, HW, D]
    deepstack_frames: List[torch.Tensor],  # list of [T, HW, D_ds]
    retain_ratio: float,
    min_k: int,
    tag: str = "",
) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    t, tokens_per_frame, dim = frames.shape
    if t <= 0 or tokens_per_frame <= 0:
        keep_index = torch.zeros((0,), device=frames.device, dtype=torch.long)
        empty_main = frames.new_empty((0, dim))
        empty_ds = [ds.new_empty((0, ds.shape[-1])) for ds in deepstack_frames]
        return empty_main, empty_ds, keep_index

    if retain_ratio <= 0:
        retain_ratio = DEFAULT_RETAIN_RATIO

    frame_reps = F.normalize(frames.mean(dim=1), dim=-1, eps=1e-6)
    curvature = _compute_curvature(frame_reps)
    weights = torch.softmax(curvature.float() / float(DEFAULT_BUDGET_TEMP), dim=0)

    total_budget = int(round(float(t * tokens_per_frame) * float(retain_ratio)))
    total_budget = max(1, min(int(t * tokens_per_frame), total_budget))
    min_k = int(max(0, min(int(min_k), int(tokens_per_frame))))
    if min_k > 0:
        total_budget = max(total_budget, int(t * min_k))
    k_t = _allocate_budget_per_frame(
        weights,
        total_budget,
        min_k=min_k,
        max_k=tokens_per_frame,
    )

    tokens_out = []
    keep_indices = []
    deepstack_out = [[] for _ in deepstack_frames]

    for frame_idx in range(t):
        k = int(k_t[frame_idx].item())
        if k <= 0:
            continue

        frame_tokens = frames[frame_idx]
        rep = frame_reps[frame_idx]
        sim = F.cosine_similarity(frame_tokens, rep.unsqueeze(0), dim=-1, eps=1e-6)
        outlier = (1.0 - sim).float()
        norm = frame_tokens.float().norm(dim=-1)
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6)
        score = outlier + norm
        topk_idx = torch.topk(score, k=k, largest=True, sorted=False).indices
        topk_idx, _ = torch.sort(topk_idx)

        tokens_out.append(frame_tokens.index_select(0, topk_idx))
        keep_indices.append(topk_idx + frame_idx * tokens_per_frame)

        for ds_idx, ds in enumerate(deepstack_frames):
            ds_frame = ds[frame_idx]
            deepstack_out[ds_idx].append(ds_frame.index_select(0, topk_idx))

    if len(tokens_out) == 0:
        keep_index = torch.zeros((0,), device=frames.device, dtype=torch.long)
        empty_main = frames.new_empty((0, dim))
        empty_ds = [ds.new_empty((0, ds.shape[-1])) for ds in deepstack_frames]
        return empty_main, empty_ds, keep_index

    tokens_comp = torch.cat(tokens_out, dim=0)
    keep_index = torch.cat(keep_indices, dim=0).to(torch.long)
    deepstack_comp = [
        torch.cat(chunks, dim=0) if len(chunks) > 0 else ds.new_empty((0, ds.shape[-1]))
        for chunks, ds in zip(deepstack_out, deepstack_frames)
    ]
    return tokens_comp, deepstack_comp, keep_index
# ───────────────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def vcast_compress(X, r):
    """X: (T, L, D) projected video tokens. Returns kept tokens (K, D) and their original indices in [0, T*L)."""
    toks, _, keep = _v_cast_compress_frames(X, [], retain_ratio=r, min_k=VCAST["min_k"])
    return toks, keep

def compress_all(X, A, G, VA, VM, seed, only=None, skip=()):
    """Every (method, setting) -> dict(tokens, pos, n_full, newline, inner). Here only V-CAST."""
    T, L, D = X.shape
    out, failures = {}, []
    for r in METHOD_GRID.get("vcast", []):
        if (only is not None and "vcast" not in only) or "vcast" in skip:
            continue
        try:
            t, i = vcast_compress(X, r)
            out[("vcast", str(r))] = {"tokens": t, "pos": i.cpu().numpy(), "n_full": T * L, "newline": True, "inner": None}
        except Exception as e:
            failures.append(("vcast", str(r), f"{type(e).__name__}: {e}"))
    return out, failures

# ── self-test on smooth random "video" features (shapes, finiteness, budgets, positions) ──
def _fake_video(T, S, D, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(1, S, D, generator=g)
    drift = torch.cumsum(0.25 * torch.randn(T, 1, D, generator=g), 0)
    return (base + drift + 0.6 * torch.randn(T, S, D, generator=g)).to(dev, torch.float16)

_T, _L = NUM_FRAMES, POOLED * POOLED
_X = _fake_video(_T, _L, 64, EMB_DEVICE)
_comp, _fails = compress_all(_X, None, None, None, None, seed=0)
assert not _fails, _fails
assert len(_comp) == len(METHOD_GRID["vcast"]), "missing compressed passes"
for (m, s), c in _comp.items():
    t, p = c["tokens"], c["pos"]
    assert torch.isfinite(t.float()).all(), (m, s)
    assert len(p) == t.shape[0] and int(np.max(p)) < c["n_full"] and np.all(np.diff(p) > 0), (m, s)
    assert abs(t.shape[0] - round(float(s) * _T * _L)) <= 1, (m, s, t.shape[0])
    # the kept tokens are exactly the tokens at their original indices
    assert torch.equal(t, _X.reshape(-1, _X.shape[-1])[torch.as_tensor(p, device=_X.device)]), (m, s)
    print(f"  {m:6s} {s:5s} enters LLM with {t.shape[0]:5d} tokens = {t.shape[0] / (_T * _L):.3f}")
del _X, _comp
print(f"Compressor self-test OK ✅  ({len(METHOD_GRID['vcast'])} compressed passes)")
