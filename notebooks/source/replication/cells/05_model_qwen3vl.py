# ── Model: Qwen3-VL-8B-Instruct with direct control over visual tokens ─────
# ── T4 memory fix + safe attention routing ─────────────────────────────
# 1) Expand grouped K/V heads ourselves so PyTorch can use the memory-efficient kernel (math kernel -> OOM on T4).
# 2) Never hand an explicit mask to the memory-efficient kernel: on T4 that path can crash with
#    "illegal memory access". A plain causal mask is converted to is_causal=True (identical maths).
# 3) Guard: if any NON-causal mask (e.g. transformers mistaking position gaps for packed sequences)
#    reaches a long attention call, stop loudly instead of computing something silently wrong.
from torch.nn.attention import sdpa_kernel, SDPBackend
if not hasattr(torch.nn.functional, "_astra_orig_sdpa"):
    torch.nn.functional._astra_orig_sdpa = torch.nn.functional.scaled_dot_product_attention
_orig_sdpa = torch.nn.functional._astra_orig_sdpa
_TRIL_CACHE = {}
def _plain_causal(mask, L, S):
    if L != S or mask.dim() != 4 or mask.shape[0] != 1 or mask.shape[1] != 1:
        return False
    key = (L, str(mask.device))
    if key not in _TRIL_CACHE:
        _TRIL_CACHE.clear()
        _TRIL_CACHE[key] = torch.ones(L, L, dtype=torch.bool, device=mask.device).tril()
    allowed = mask[0, 0] if mask.dtype == torch.bool else (mask[0, 0] == 0)
    return torch.equal(allowed, _TRIL_CACHE[key])

def _sdpa_safe(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kw):
    if enable_gqa and key.shape[-3] != query.shape[-3]:
        rep = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(rep, dim=-3)
        value = value.repeat_interleave(rep, dim=-3)
    if attn_mask is not None:
        L, S = query.shape[-2], key.shape[-2]
        if _plain_causal(attn_mask, L, S):
            attn_mask, is_causal = None, True
        elif L > 1024:
            raise RuntimeError("A non-causal attention mask reached a long attention call — stopping instead of "
                               "computing silently wrong results. Send me this error.")
        elif query.dtype == torch.float16:
            # short question block over a cached prefix: compute in fp32 (cheap at this size), so cached passes
            # match the fp32-accumulating kernel used for full passes as closely as possible
            m = attn_mask if attn_mask.dtype == torch.bool else attn_mask.float()
            return _orig_sdpa(query.float(), key.float(), value.float(), attn_mask=m, dropout_p=dropout_p,
                              is_causal=is_causal, scale=scale, **kw).to(query.dtype)
    return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)

torch.nn.functional.scaled_dot_product_attention = _sdpa_safe
NO_MATH = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
print("T4 attention fix applied (K/V heads expanded, causal masks routed to is_causal, math kernel disabled for the LLM)")


# Qwen3-VL specifics handled here: (1) 2 frames form one temporal patch, so 32 frames give 16 temporal groups, each
# introduced by a timestamp "<t seconds>" and wrapped in vision-start/end tokens; (2) 3-D (time, height, width) rotary
# positions; (3) DeepStack: features from three vision layers are added to the visual positions after LLM layers 0-2.
# Compression keeps each kept token's own 3-D position and its own DeepStack features (as FlashVID's official code).
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, apply_rotary_pos_emb_vision, repeat_kv
from transformers.video_utils import VideoMetadata
import torchvision.transforms.functional as TVF

max_memory = {i: "13GiB" for i in range(N_GPU)}
if N_GPU >= 2:
    max_memory[0] = "9GiB"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", max_memory=max_memory, attn_implementation="sdpa", dtype=torch.float16).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
tok = processor.tokenizer

def _get(root, paths):
    for p in paths:
        obj, ok = root, True
        for part in p.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError(f"None of {paths} found — send me the error and transformers version")

visual  = _get(model, ["model.visual", "visual"])
body    = _get(model, ["model.language_model", "language_model"])
lm_head = _get(model, ["lm_head"])
CFG, TCFG, VCFG = model.config, model.config.text_config, model.config.vision_config
VIDEO_TOKEN_ID = CFG.video_token_id
VSTART_ID, VEND_ID = CFG.vision_start_token_id, CFG.vision_end_token_id
MERGE = VCFG.spatial_merge_size
FACTOR = VCFG.patch_size * MERGE                         # 32 pixels per merged token side
NLAYERS, HID = TCFG.num_hidden_layers, TCFG.hidden_size
NHEADS, NKV = TCFG.num_attention_heads, TCFG.num_key_value_heads
HEAD_DIM = getattr(TCFG, "head_dim", None) or HID // NHEADS
INTER = TCFG.intermediate_size
EMB_DEVICE = body.embed_tokens.weight.device
LM_DEVICE = lm_head.weight.device
VDEV = next(visual.parameters()).device
N_DEEPSTACK = len(VCFG.deepstack_visual_indexes)
print(f"LLM layers={NLAYERS} hidden={HID} heads={NHEADS}/{NKV} head_dim={HEAD_DIM} | DeepStack layers {N_DEEPSTACK} | "
      f"{FACTOR}px per visual token, 2 frames per temporal group")

LETTERS = "ABCDEFGH"
LETTER_IDS = []
for L in LETTERS:
    ids = tok.encode(L, add_special_tokens=False)
    assert len(ids) == 1, f"Letter {L} is not a single token"
    LETTER_IDS.append(ids[0])

PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"

def _letter_list(n):
    L = list(LETTERS[:n])
    return L[0] if n == 1 else (f"{L[0]} or {L[1]}" if n == 2 else ", ".join(L[:-1]) + ", or " + L[-1])

def suffix_text(question, options):
    opts = "\n".join(str(o) for o in options)
    return ("\nSelect the best answer to the following multiple-choice question based on the video. "
            f"Respond with only the letter ({_letter_list(len(options))}) of the correct option.\n"
            f"{question}\n{opts}\nThe best answer is:<|im_end|>\n<|im_start|>assistant\n")

def build_suffix_ids(question, options):
    """The main study's prompt (identical text)."""
    return tok(suffix_text(question, options), add_special_tokens=False, return_tensors="pt").input_ids[0]

def smart_resize_frame(h, w, factor=None, min_pixels=None, max_pixels=None):
    """qwen-vl-utils smart_resize (per frame), with FlashVID's baseline pixel limits."""
    factor = factor or FACTOR
    min_pixels, max_pixels = min_pixels or MIN_PIXELS, max_pixels or MAX_PIXELS
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar

def prepare_video(frames, meta, prompt_tail=""):
    """Resize (bicubic, antialiased, as qwen-vl-utils) and run the official processor (patchify, normalise,
    timestamps). Returns the processor outputs for the text PREFIX + video (+ prompt_tail)."""
    h, w = frames[0].shape[:2]
    H, W = smart_resize_frame(h, w)
    v = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    v = TVF.resize(v, [H, W], interpolation=TVF.InterpolationMode.BICUBIC, antialias=True)
    v = v.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    fps = (meta or {}).get("fps") or 24.0
    idx = list((meta or {}).get("indices") or range(len(frames)))
    md = VideoMetadata(total_num_frames=int((meta or {}).get("total") or (max(idx) + 1)), fps=float(fps),
                       frames_indices=idx, width=W, height=H)
    return processor(text=[PREFIX + processor.video_token + prompt_tail], videos=[v], video_metadata=[md],
                     do_resize=False, do_sample_frames=False, cap_pixels_per_frame=False, return_tensors="pt")

# ── FlashVID needs the last vision block's attention (mean over heads and queries, per temporal group) ──
_LAST_ATTN = {}
def _capture_last_attn(mod, args, kwargs):
    hs = args[0] if args else kwargs["hidden_states"]
    _LAST_ATTN["h"] = hs
    _LAST_ATTN["pe"] = kwargs.get("position_embeddings")
    _LAST_ATTN["cu"] = kwargs.get("cu_seqlens")
_hook = visual.blocks[-1].attn.register_forward_pre_hook(_capture_last_attn, with_kwargs=True)

@torch.no_grad()
def _flashvid_cls_attention(attn_mod, n_groups):
    """FlashVID Qwen3VLVisionAttention_forward(return_logits=True) + the pooling in its vision forward."""
    hs, (cos, sin) = _LAST_ATTN["h"], _LAST_ATTN["pe"]
    seq = hs.shape[0]
    q, k, _ = attn_mod.qkv(hs).reshape(seq, 3, attn_mod.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q = q.transpose(0, 1).reshape(attn_mod.num_heads, n_groups, -1, attn_mod.head_dim).permute(1, 0, 2, 3)
    k = k.transpose(0, 1).reshape(attn_mod.num_heads, n_groups, -1, attn_mod.head_dim).permute(1, 0, 2, 3)
    out = []
    for i in range(n_groups):
        a = torch.matmul(q[i], k[i].transpose(-1, -2)) / attn_mod.head_dim ** 0.5
        a = torch.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        out.append(a.mean(0).mean(0))
    w = torch.stack(out, 0)                                   # (groups, patches per group)
    return w.view(n_groups, w.shape[-1] // (MERGE * MERGE), -1).mean(-1)

@torch.no_grad()
def encode_frames(frames, meta=None):
    """One frame set -> everything the passes need: the prefix token ids (with timestamps), their 3-D positions,
    the merged visual tokens (N, D), DeepStack features (3 x (N, D)), FlashVID's per-group attention (G, P)."""
    inp = prepare_video(frames, meta)
    ids = inp["input_ids"][0]
    grid = inp["video_grid_thw"]
    G, Hp, Wp = [int(x) for x in grid[0]]
    out = visual(inp["pixel_values_videos"].to(VDEV, torch.float16), grid_thw=grid.to(VDEV), return_dict=True)
    X = out.pooler_output if hasattr(out, "pooler_output") else out[0]
    DS = list(out.deepstack_features)
    CA = _flashvid_cls_attention(visual.blocks[-1].attn, G)
    _LAST_ATTN.clear()
    pos, _ = model.model.get_rope_index(ids[None], inp["mm_token_type_ids"], video_grid_thw=grid)
    pos = pos[:, 0]                                           # (3, L)
    vmask = ids == VIDEO_TOKEN_ID
    P = (Hp // MERGE) * (Wp // MERGE)
    assert int(vmask.sum()) == X.shape[0] == G * P, "visual token count mismatch"
    text_emb = body.embed_tokens(ids.to(EMB_DEVICE))
    return {"ids": ids, "pos": pos, "vmask": vmask, "X": X.to(EMB_DEVICE), "DS": [d.to(EMB_DEVICE) for d in DS],
            "CA": CA.to(EMB_DEVICE).float(), "G": G, "Hm": Hp // MERGE, "Wm": Wp // MERGE, "P": P,
            "text_emb": text_emb, "next_pos": int(pos.max()) + 1, "n_full": int(X.shape[0]),
            "pixel_values_videos": inp["pixel_values_videos"], "grid": grid}

def assemble(enc, groups, drop_start_end=False):
    """Builds a prefix from per-group visual tokens. groups[g] = (feat (k, D), [ds (k, D)] * 3, pos (3, k)).
    Text tokens (prompt, timestamps, vision start/end) keep their own embeddings and positions."""
    ids, pos, vm = enc["ids"], enc["pos"], enc["vmask"]
    dev = enc["text_emb"].device
    keep_text = ~vm
    if drop_start_end:
        keep_text &= (ids != VSTART_ID) & (ids != VEND_ID)
    vid_idx = torch.nonzero(vm).flatten()
    starts = vid_idx[::enc["P"]].tolist()                    # first index of each temporal group
    emb, pp, vmask, ds = [], [], [], [[] for _ in range(N_DEEPSTACK)]
    cur = 0
    for g in range(enc["G"]):
        s = starts[g]
        t = torch.arange(cur, s)
        t = t[keep_text[cur:s]]
        emb.append(enc["text_emb"][t.to(dev)]); pp.append(pos[:, t]); vmask.append(torch.zeros(len(t), dtype=torch.bool))
        f, dsl, gp = groups[g]
        emb.append(f.to(dev, enc["text_emb"].dtype)); pp.append(gp.cpu()); vmask.append(torch.ones(f.shape[0], dtype=torch.bool))
        for j in range(N_DEEPSTACK):
            ds[j].append(dsl[j])
        cur = s + enc["P"]
    t = torch.arange(cur, len(ids))
    t = t[keep_text[cur:]]
    emb.append(enc["text_emb"][t.to(dev)]); pp.append(pos[:, t]); vmask.append(torch.zeros(len(t), dtype=torch.bool))
    return {"embeds": torch.cat(emb, 0), "pos": torch.cat(pp, 1), "vmask": torch.cat(vmask, 0),
            "ds": [torch.cat(d, 0) for d in ds], "next_pos": enc["next_pos"]}

def _group_pos(enc):
    return enc["pos"][:, enc["vmask"]].view(3, enc["G"], enc["P"])

def full_spec(enc):
    G, P = enc["G"], enc["P"]
    gp = _group_pos(enc)
    X, DS = enc["X"].view(G, P, -1), [d.view(G, P, -1) for d in enc["DS"]]
    s = assemble(enc, [(X[g], [d[g] for d in DS], gp[:, g]) for g in range(G)])
    s.update(inner=None, nvis=enc["n_full"])
    return s

def llm_flops_layers(ns):
    """Prefill FLOPs of the LLM when layer l sees ns[l] tokens (q/o projections use heads x head_dim)."""
    dkv, dq = NKV * HEAD_DIM, NHEADS * HEAD_DIM
    return float(sum(2 * (2 * n * HID * dq + 2 * n * HID * dkv + 2 * n * n * dq + 3 * n * HID * INTER) for n in ns))

def llm_flops(n):
    return llm_flops_layers([n] * NLAYERS)

def _sync():
    for i in range(N_GPU):
        torch.cuda.synchronize(i)

@torch.no_grad()
def last_query_attention(layer, h, pe):
    """Attention of the last token over all tokens in this layer (FlashVID's output_attentions fallback)."""
    a = layer.self_attn
    dev = a.q_proj.weight.device
    h = h.to(dev)
    cos, sin = pe[0].to(dev), pe[1].to(dev)
    hn = layer.input_layernorm(h)
    q = a.q_norm(a.q_proj(hn[:, -1:]).view(1, 1, -1, HEAD_DIM)).transpose(1, 2)
    k = a.k_norm(a.k_proj(hn).view(1, hn.shape[1], -1, HEAD_DIM)).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
    _, k = apply_rotary_pos_emb(k, k, cos, sin)
    k = repeat_kv(k, NHEADS // NKV)
    w = torch.matmul(q, k.transpose(2, 3)) / HEAD_DIM ** 0.5
    return torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)

@torch.no_grad()
def apply_inner(h, pe, attn, inner, vmask):
    """FlashVID utils.fastv_prune with visual_pos_masks (its Qwen3-VL path)."""
    dev = h.device
    vm = vmask.to(dev)
    n_vis = int(vm.sum())
    n_keep = math.ceil(n_vis * inner["keep_ratio"])
    a = attn.mean(dim=1)[0, -1].to(dev)[vm]
    vis_idx = torch.nonzero(vm).flatten()
    top = torch.topk(a, k=n_keep).indices.sort().values
    keep = torch.sort(torch.cat([torch.nonzero(~vm).flatten(), vis_idx[top]])).values
    pe = (pe[0][..., keep.to(pe[0].device), :], pe[1][..., keep.to(pe[1].device), :])
    return h[:, keep], pe, n_keep

@torch.no_grad()
def forward_layers(x, pos3, vmask, ds, inner=None, nvis0=0):
    """Our own decoder loop (same calls as Qwen3VLTextModel.forward, incl. DeepStack); lets FlashVID prune inside."""
    pe = body.rotary_emb(x, pos3)
    h, attn, ns, nvis = x, None, [], []
    vis_now = nvis0
    vm_dev = vmask.to(x.device)
    for li, layer in enumerate(body.layers):
        if inner is not None and li == inner["layer"]:
            h, pe, vis_now = apply_inner(h, pe, attn, inner, vm_dev)
            attn = None
        if inner is not None and li == inner["layer"] - 1:
            attn = last_query_attention(layer, h, pe)
        ns.append(h.shape[1]); nvis.append(vis_now)
        h = layer(h, position_embeddings=pe, attention_mask=None, position_ids=None, use_cache=False)
        if isinstance(h, tuple):
            h = h[0]
        if ds is not None and li < len(ds):
            hd = h.device
            h = h.clone()
            h[0, vm_dev.to(hd)] = h[0, vm_dev.to(hd)] + ds[li].to(hd, h.dtype)
    h = body.norm(h.to(body.norm.weight.device))
    return h, ns, nvis

def _result(logits, n_opt, extra):
    lp = torch.log_softmax(logits[LETTER_IDS[:n_opt]], -1).cpu().numpy()
    if not np.all(np.isfinite(lp)):
        raise FloatingPointError("NaN/inf logits (fp16 overflow)")
    srt = np.sort(lp)[::-1]
    return {"pred": LETTERS[int(lp.argmax())], "logp": lp.round(4).tolist(), "margin": float(srt[0] - srt[1]), **extra}

def _seq(spec, suf_ids=None):
    E = body.embed_tokens
    x, pos, vm = spec["embeds"], spec["pos"], spec["vmask"]
    if suf_ids is not None:
        suf = E(suf_ids.to(EMB_DEVICE))
        n_q = suf.shape[0]
        x = torch.cat([x, suf.to(x.dtype)], 0)
        pos = torch.cat([pos, (spec["next_pos"] + torch.arange(n_q)).view(1, -1).expand(3, -1)], 1)
        vm = torch.cat([vm, torch.zeros(n_q, dtype=torch.bool)], 0)
    return x[None].to(EMB_DEVICE), pos[:, None, :].to(EMB_DEVICE), vm

@torch.no_grad()
def llm_pass(spec, suf_ids, n_opt, manual=False):
    x, pos3, vm = _seq(spec, suf_ids)
    for i in range(N_GPU):
        torch.cuda.reset_peak_memory_stats(i)
    _sync(); t0 = time.time()
    inner = spec.get("inner")
    with sdpa_kernel(NO_MATH):
        if inner is None and not manual:
            out = body(inputs_embeds=x, position_ids=pos3, visual_pos_masks=vm[None].to(EMB_DEVICE),
                       deepstack_visual_embeds=spec["ds"], use_cache=False)
            h = out[0][0, -1]
            ns, nvis = [x.shape[1]] * NLAYERS, [spec["nvis"]] * NLAYERS
        else:
            hh, ns, nvis = forward_layers(x, pos3, vm, spec["ds"], inner, nvis0=spec["nvis"])
            h = hh[0, -1]
    logits = lm_head(h.to(LM_DEVICE)[None])[0].float()
    _sync(); dt = time.time() - t0
    return _result(logits, n_opt, {"ntok": int(spec["nvis"]), "vis_mean_layers": float(np.mean(nvis)), "sec": float(dt),
                                   "flops": llm_flops_layers(ns),
                                   "peak_gb": round(max(torch.cuda.max_memory_allocated(i) for i in range(N_GPU)) / 1e9, 3)})

# ── Prefix caching ──
from transformers import DynamicCache

def _new_cache():
    try:
        return DynamicCache(config=TCFG)
    except TypeError:
        return DynamicCache()

def _crop(cache, n):
    extra = cache.get_seq_length() - n
    if extra > 0 and hasattr(cache, "crop"):
        cache.crop(-extra)
    if cache.get_seq_length() != n:
        for l in cache.layers:
            l.keys, l.values = l.keys[..., :n, :], l.values[..., :n, :]
    assert cache.get_seq_length() == n, "could not restore the cached prefix"

@torch.no_grad()
def prefill(spec):
    x, pos3, vm = _seq(spec)
    for i in range(N_GPU):
        torch.cuda.reset_peak_memory_stats(i)
    cache = _new_cache()
    _sync(); t0 = time.time()
    with sdpa_kernel(NO_MATH):
        body(inputs_embeds=x, position_ids=pos3, visual_pos_masks=vm[None].to(EMB_DEVICE),
             deepstack_visual_embeds=spec["ds"], past_key_values=cache, use_cache=True)
    _sync()
    return {"cache": cache, "n": int(x.shape[1]), "next_pos": spec["next_pos"], "sec": time.time() - t0,
            "nvis": int(spec["nvis"]), "peak": max(torch.cuda.max_memory_allocated(i) for i in range(N_GPU)) / 1e9}

@torch.no_grad()
def suffix_pass(P, suf_ids, n_opt):
    E = body.embed_tokens
    suf = E(suf_ids.to(EMB_DEVICE))[None]
    n_q = suf.shape[1]
    pos3 = (P["next_pos"] + torch.arange(n_q, device=EMB_DEVICE)).view(1, 1, -1).expand(3, 1, -1)
    am = torch.ones(1, P["n"] + n_q, dtype=torch.long, device=EMB_DEVICE)
    _sync(); t0 = time.time()
    with sdpa_kernel([SDPBackend.MATH]):
        out = body(inputs_embeds=suf, attention_mask=am, position_ids=pos3, past_key_values=P["cache"], use_cache=True)
    h = out[0][0, -1]
    logits = lm_head(h.to(LM_DEVICE)[None])[0].float()
    _sync(); dt = time.time() - t0
    _crop(P["cache"], P["n"])
    return _result(logits, n_opt, {"ntok": P["nvis"], "vis_mean_layers": float(P["nvis"]), "sec": float(P["sec"] + dt),
                                   "sec_suffix": float(dt), "flops": llm_flops(P["n"] + n_q),
                                   "peak_gb": round(P["peak"], 3)})

@torch.no_grad()
def preflight_attention(n_tok):
    dev = body.layers[0].self_attn.q_proj.weight.device
    q = torch.randn(1, NHEADS, n_tok, HEAD_DIM, device=dev, dtype=torch.float16)
    kv = torch.randn(1, NKV, n_tok, HEAD_DIM, device=dev, dtype=torch.float16)
    mask = torch.ones(n_tok, n_tok, dtype=torch.bool, device=dev).tril()[None, None]
    torch.cuda.reset_peak_memory_stats(dev)
    base = torch.cuda.memory_allocated(dev)
    try:
        with sdpa_kernel(NO_MATH):
            o1 = torch.nn.functional.scaled_dot_product_attention(q, kv, kv, is_causal=True, enable_gqa=True)
            o2 = torch.nn.functional.scaled_dot_product_attention(q, kv, kv, attn_mask=mask, enable_gqa=True)  # routed to is_causal
        torch.cuda.synchronize(dev)          # surface any asynchronous CUDA error HERE, not in a later cell
        assert torch.equal(o1, o2), "causal-mask routing changed the result"
    except RuntimeError as e:
        raise RuntimeError(f"Memory-efficient attention is not available here ({e}). Send me this message.")
    del o1, o2
    peak = (torch.cuda.max_memory_allocated(dev) - base) / 1e9
    del q, kv, mask
    torch.cuda.empty_cache()
    for i in range(N_GPU):
        free, total = torch.cuda.mem_get_info(i)
        print(f"GPU{i}: {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
    print(f"Attention preflight OK ✅  {n_tok} tokens -> extra peak memory {peak:.2f} GB")

preflight_attention(NUM_FRAMES // 2 * (MAX_PIXELS // FACTOR ** 2) + 600)
print("Model ready ✅")
