# ══ Compressors for LLaVA-Video (64 frames x 13x13 tokens, a newline after each frame) ══
# Every method returns a spec: tokens (K, D) grouped by frame, their frame index, and an optional inner-LLM step.
# The per-frame newline layout is FlashVID's official LLaVA-Video setting; VisionZip's port supports it directly.
# HoliTom's official LLaVA-Video run uses grid newlines; here its window tokens are assigned to frames (the merged
# static tokens of a window go with its first frame, as in its grid code) so that all methods share one layout.

def _best_grid_hw(n, H, W):
    best = None
    for h in range(1, H + 1):
        for w in range(1, W + 1):
            key = (abs(h * w - n), abs(h / max(w, 1) - H / W), -h * w)
            if best is None or key < best[0]:
                best = (key, h, w)
    return best[1], best[2]

@torch.no_grad()
def uniform_pool_frames(X, r):
    """Per-frame average pooling of the 13x13 grid to the grid closest to int(169*r) tokens."""
    T, S, D = X.shape
    side = int(round(S ** 0.5))
    h, w = _best_grid_hw(int(S * r), side, side)
    g = X.float().view(T, side, side, D).permute(0, 3, 1, 2)
    g = F.adaptive_avg_pool2d(g, (h, w))
    return g.permute(0, 2, 3, 1).reshape(-1, D).to(X.dtype), h * w

@torch.no_grad()
def random_drop_frames(X, r, seed):
    T, S, D = X.shape
    k = int(S * r)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.stack([torch.randperm(S, generator=gen)[:k].sort().values for _ in range(T)]).to(X.device)
    return torch.gather(X, 1, idx.unsqueeze(-1).expand(-1, -1, D)).reshape(-1, D), k

def ht_holitom_parts(static_feat, dynamic_feat, dynamic_attn, static_pos, dynamic_pos, window_size, retain_ratio, D,
                     Beta, K, images_dtype, gen=None):
    """ht_holitom without flattening: returns (static tokens (n_s, D) or None, dynamic tokens (W, n_d, D))."""
    dynamic_feat, dynamic_pos = ht_merge_tokens_by_attention_density(dynamic_feat, dynamic_attn, dynamic_pos,
                                                                     retain_ratio, D, Beta, K, gen)
    if window_size == 1:
        return None, dynamic_feat.to(images_dtype)
    static_feat, static_pos = ht_merge_tokens_by_density(static_feat.unsqueeze(0), static_pos, retain_ratio, Beta, K, gen)
    return static_feat[0].to(images_dtype), dynamic_feat.to(images_dtype)

@torch.no_grad()
def holitom_compress_frames(X, attn, r, seed):
    tau = HOLITOM["T"][r]
    Beta, Dd, K, max_window_size = HOLITOM["beta"], HOLITOM["D"], HOLITOM["K"], HOLITOM["max_window"]
    gen = torch.Generator(device=X.device).manual_seed(seed)
    attn_weights = attn.to(X.device, X.dtype)
    batch_size, seq_len, embed_dim = X.shape
    normed = torch.nn.functional.normalize(X, p=2, dim=-1)
    feature_sim = torch.nn.functional.cosine_similarity(normed[:-1], normed[1:], dim=-1)
    selected_frames, total_reduced = ht_select_static_windows(feature_sim, batch_size, tau, max_window_size)
    total_tokens = batch_size * seq_len
    retain_ratio = min(r / ((total_tokens - total_reduced) / total_tokens), 1)
    sf, df, _, da, sp, dpos = ht_get_static_dynamic_features(X, attn_weights, selected_frames, feature_sim, tau)
    toks, frames = [], []
    for i, (start, end) in enumerate(selected_frames):
        st, dy = ht_holitom_parts(sf[i], df[i], da[i], sp[i], dpos[i], end - start + 1, retain_ratio, Dd, Beta, K,
                                  X.dtype, gen)
        for j in range(dy.shape[0]):
            f_tok = dy[j] if (j > 0 or st is None) else torch.cat([st, dy[j]], 0)
            toks.append(f_tok); frames.append(torch.full((f_tok.shape[0],), start + j, dtype=torch.long))
    return torch.cat(toks, 0), torch.cat(frames, 0).to(X.device)

def _spec(tokens, frame, T, inner=None):
    return {"tokens": tokens, "frame": frame.to(tokens.device), "T": T, "inner": inner, "nvis": int(tokens.shape[0])}

def compress_all(enc, seed, only=None, skip=()):
    """Every (method, setting) -> spec. Official code is kept as is; if it fails on a video, that (method, setting)
    is skipped for the video and reported — never silently replaced."""
    X, VA, VM = enc["X"], enc["VA"], enc["VM"]
    T, L, D = X.shape
    dev = X.device
    rep = lambda k: torch.arange(T, device=dev).repeat_interleave(k)
    jobs = []
    for r in METHOD_GRID.get("flashvid", []):
        def fv(r=r):
            cfg = FlashVidConfig(retention_ratio=r, **FLASHVID)
            toks, idx = flashvid_compression(X.clone(), VA.to(dev, X.dtype), cfg)
            inner = {"kind": "fastv", "layer": cfg.pruning_layer, "keep_ratio": cfg.llm_retention_ratio}
            return _spec(toks, idx // L, T, inner)
        jobs.append(("flashvid", str(r), fv))
    for r in METHOD_GRID.get("holitom", []):
        def ht(r=r):
            toks, fr = holitom_compress_frames(X, VA, r, seed)
            inner = {"kind": "holitom", "layer": HOLITOM["inner_layer"], "merge_ratio": HOLITOM["inner_r"]}
            return _spec(toks, fr, T, inner)
        jobs.append(("holitom", str(r), ht))
    for r in METHOD_GRID.get("visionzip", []):
        jobs.append(("visionzip", str(r), lambda r=r: _spec(visionzip_compress(X, VA, VM, r), rep(int(L * r)), T)))
    for r in METHOD_GRID.get("uniform", []):
        def un(r=r):
            t, k = uniform_pool_frames(X, r)
            return _spec(t, rep(k), T)
        jobs.append(("uniform", str(r), un))
    for r in METHOD_GRID.get("random", []):
        def rd(r=r):
            t, k = random_drop_frames(X, r, seed)
            return _spec(t, rep(k), T)
        jobs.append(("random", str(r), rd))
    out, failures = {}, []
    for m, s, fn in jobs:
        if (only is not None and m not in only) or m in skip:
            continue
        try:
            out[(m, s)] = fn()
        except Exception as e:
            failures.append((m, s, f"{type(e).__name__}: {e}"))
    return out, failures

# ── self-test on smooth random "video" features (shapes, finiteness, budgets, frame grouping) ──
def _fake_video(T, S, D, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(1, S, D, generator=g)
    drift = torch.cumsum(0.25 * torch.randn(T, 1, D, generator=g), 0)
    return (base + drift + 0.6 * torch.randn(T, S, D, generator=g)).to(dev, torch.float16)

_T, _L = NUM_FRAMES, POOLED * POOLED
_enc = {"X": _fake_video(_T, _L, 64, EMB_DEVICE), "VA": None, "VM": torch.randn(_T, _L, 16, device=EMB_DEVICE)}
_enc["VA"] = torch.rand(_T, _L, device=EMB_DEVICE); _enc["VA"] = _enc["VA"] / _enc["VA"].sum(1, keepdim=True)
_comp, _fails = compress_all(_enc, seed=0)
assert not _fails, _fails
assert len(_comp) == sum(len(v) for v in METHOD_GRID.values()), "missing compressed passes"
for (m, s), c in _comp.items():
    t = c["tokens"]
    assert torch.isfinite(t.float()).all(), (m, s)
    assert c["frame"].shape[0] == t.shape[0] and bool((c["frame"][1:] >= c["frame"][:-1]).all()), (m, s)
    print(f"  {m:9s} {s:5s} enters LLM with {t.shape[0]:5d} tokens = {t.shape[0] / (_T * _L):.3f} (+{_T} newlines)"
          + (f" | inner {c['inner']['kind']} @ layer {c['inner']['layer']}" if c["inner"] else ""))
del _enc, _comp
print(f"Compressor self-test OK ✅  ({sum(len(v) for v in METHOD_GRID.values())} compressed passes)")
