# ══ Compressors for Qwen3-VL (16 temporal groups of merged tokens, 3-D positions, DeepStack) ══
# Every method returns a spec (the assembled prefix). Kept tokens keep their own 3-D positions and DeepStack features.
# FlashVID follows its official Qwen3-VL code: compression over the temporal groups (each group = one "frame" for
# FlashVID), merged tokens placed at their anchor token's position, vision start/end tokens dropped, then FastV-style
# pruning to 10% at LLM layer 28. Uniform pooling averages tokens, DeepStack features and positions over each window.

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
    for r in METHOD_GRID.get("flashvid", []):
        def fv(r=r):
            cfg = FlashVidConfig(retention_ratio=r, **FLASHVID)
            toks, idx = flashvid_compression(X.clone(), enc["CA"].to(dev, X.dtype), cfg)
            g_of = idx // P
            flat_ds = [d.reshape(G * P, -1) for d in DS]
            flat_pos = gp.reshape(3, G * P)
            groups = []
            for g in range(G):
                sel = g_of == g
                gi = idx[sel]
                groups.append((toks[sel], [d[gi] for d in flat_ds], flat_pos[:, gi.cpu()]))
            s = assemble(enc, groups, drop_start_end=True)
            s.update(inner={"kind": "fastv", "layer": cfg.pruning_layer, "keep_ratio": cfg.llm_retention_ratio},
                     nvis=int(toks.shape[0]))
            return s
        jobs.append(("flashvid", str(r), fv))
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
