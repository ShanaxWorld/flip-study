# ══ V-CAST (Lin et al., NeurIPS 2026) — github.com/xinyouu/V-CAST, compressor/v_cast/modeling_qwen3_vl_v_cast.py ══
# MIT License (repository LICENSE). The functions between the two rulers are copied verbatim from the official file
# (commit f15d276, 3 Oct 2026). As in the official Qwen3-VL path, the merged tokens of each temporal group are one
# "frame", the selected tokens keep their own 3-D positions and DeepStack features, and only video tokens are
# removed (vision start/end and timestamp tokens stay). The official code fixes the ratio at 25%; we pass 5%, 10%, 25%.
from typing import List, Optional

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

# ══ Compressors for Qwen3-VL (16 temporal groups of merged tokens, 3-D positions, DeepStack) ══
# Every method returns a spec (the assembled prefix). Kept tokens keep their own 3-D positions and DeepStack features.
# (FlashVID, uniform pooling and random drop were run in the replication notebook.) Uniform pooling averages tokens, DeepStack features and positions over each window.

def _best_grid_hw(n, H, W):
    best = None
    for h in range(1, H + 1):
        for w in range(1, W + 1):
            key = (abs(h * w - n), abs(h / max(w, 1) - H / W), -h * w)
            if best is None or key < best[0]:
                best = (key, h, w)
    return best[1], best[2]

def compress_all(enc, seed, only=None, skip=()):
    G, P, Hm, Wm = enc["G"], enc["P"], enc["Hm"], enc["Wm"]
    X = enc["X"].view(G, P, -1)
    DS = [d.view(G, P, -1) for d in enc["DS"]]
    gp = _group_pos(enc)                                      # (3, G, P)
    dev = X.device
    jobs = []
    for r in METHOD_GRID.get("uniform", []):
        def un(r=r):
            h, w = _best_grid_hw(int(P * r), Hm, Wm)
            pool = lambda t: F.adaptive_avg_pool2d(t.float().reshape(G, Hm, Wm, -1).permute(0, 3, 1, 2), (h, w)) \
                .permute(0, 2, 3, 1).reshape(G, h * w, -1)
            fx = pool(X).to(X.dtype)
            fds = [pool(d).to(d.dtype) for d in DS]
            fpos = F.adaptive_avg_pool2d(gp.permute(1, 0, 2).reshape(G, 3, Hm, Wm).float(), (h, w)).round().long()
            fpos = fpos.reshape(G, 3, h * w).permute(1, 0, 2)
            s = assemble(enc, [(fx[g], [d[g] for d in fds], fpos[:, g]) for g in range(G)])
            s.update(inner=None, nvis=G * h * w)
            return s
        jobs.append(("uniform", str(r), un))
    for r in METHOD_GRID.get("random", []):
        def rd(r=r):
            k = int(P * r)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            idx = torch.stack([torch.randperm(P, generator=gen)[:k].sort().values for _ in range(G)])
            groups = [(X[g][idx[g].to(dev)], [d[g][idx[g].to(dev)] for d in DS], gp[:, g][:, idx[g]]) for g in range(G)]
            s = assemble(enc, groups)
            s.update(inner=None, nvis=G * k)
            return s
        jobs.append(("random", str(r), rd))
    for r in METHOD_GRID.get("vcast", []):
        def vc(r=r):
            toks, dsc, keep = _v_cast_compress_frames(X, DS, retain_ratio=r, min_k=VCAST["min_k"])
            g_of = keep // P
            flat_pos = gp.reshape(3, G * P)
            groups = []
            for g in range(G):
                sel = g_of == g
                groups.append((toks[sel], [d[sel] for d in dsc], flat_pos[:, keep[sel].cpu()]))
            s = assemble(enc, groups)
            s.update(inner=None, nvis=int(toks.shape[0]))
            return s
        jobs.append(("vcast", str(r), vc))
    out, failures = {}, []
    for m, s, fn in jobs:
        if (only is not None and m not in only) or m in skip:
            continue
        try:
            out[(m, s)] = fn()
        except Exception as e:
            failures.append((m, s, f"{type(e).__name__}: {e}"))
    return out, failures

# ── self-test on a synthetic "video" (shapes, finiteness, budgets) ──
def _fake_enc(G=16, Hm=10, Wm=18, D=64, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = Hm * Wm
    base = torch.randn(1, P, D, generator=g)
    X = (base + torch.cumsum(0.25 * torch.randn(G, 1, D, generator=g), 0) + 0.6 * torch.randn(G, P, D, generator=g))
    pre = 10
    ids, pos, cur = [7] * pre, [list(range(pre))] * 3, pre
    ids, pos = list(ids), [list(p) for p in pos]
    for gi in range(G):
        ids += [5, VSTART_ID] + [VIDEO_TOKEN_ID] * P + [VEND_ID]
        for r in range(3):
            pos[r] += [cur, cur + 1]
        cur += 2
        hh, ww = torch.meshgrid(torch.arange(Hm), torch.arange(Wm), indexing="ij")
        pos[0] += [cur] * P; pos[1] += (cur + hh.flatten()).tolist(); pos[2] += (cur + ww.flatten()).tolist()
        cur += max(Hm, Wm)
        for r in range(3):
            pos[r] += [cur]
        cur += 1
    ids = torch.tensor(ids); pos = torch.tensor(pos)
    return {"ids": ids, "pos": pos, "vmask": ids == VIDEO_TOKEN_ID, "X": X.reshape(G * P, D).to(EMB_DEVICE, torch.float16),
            "DS": [torch.randn(G * P, D).to(EMB_DEVICE, torch.float16) for _ in range(N_DEEPSTACK)],
            "CA": torch.rand(G, P).to(EMB_DEVICE), "G": G, "Hm": Hm, "Wm": Wm, "P": P,
            "text_emb": torch.randn(len(ids), D).to(EMB_DEVICE, torch.float16), "next_pos": int(pos.max()) + 1,
            "n_full": G * P}

_enc = _fake_enc()
_comp, _fails = compress_all(_enc, seed=0)
assert not _fails, _fails
assert len(_comp) == sum(len(v) for v in METHOD_GRID.values()), "missing compressed passes"
for (m, s), c in _comp.items():
    assert torch.isfinite(c["embeds"].float()).all(), (m, s)
    assert c["embeds"].shape[0] == c["pos"].shape[1] == c["vmask"].shape[0], (m, s)
    assert int(c["vmask"].sum()) == c["nvis"] == c["ds"][0].shape[0], (m, s)
    print(f"  {m:9s} {s:5s} enters LLM with {c['nvis']:5d} visual tokens = {c['nvis'] / _enc['n_full']:.3f}"
          + (f" | inner {c['inner']['kind']} @ layer {c['inner']['layer']}" if c["inner"] else ""))
del _enc, _comp
print(f"Compressor self-test OK ✅  ({sum(len(v) for v in METHOD_GRID.values())} compressed passes)")
