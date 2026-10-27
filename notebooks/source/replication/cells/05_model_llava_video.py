# ── Model: LLaVA-Video-7B with direct control over visual tokens ──────────
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


# ── 1) Convert the original LLaVA-Video weights (LLaVA-NeXT format) to the transformers LLaVA-OneVision format ──
# Same architecture as LLaVA-OneVision-7B (SigLIP-so400m with 26 layers, 2-layer GELU projector, Qwen2-7B, newline
# vector); only the parameter names differ. The converted copy lives on the temporary disk for this session.
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, LlavaOnevisionForConditionalGeneration
CONV_DIR = os.path.join(WORK_DIR, "llava_video_7b_hf")

def _rename(k):
    vt = "model.vision_tower.vision_tower.vision_model."
    if k.startswith(vt):
        return "model.vision_tower." + k[len(vt):]
    if k.startswith("model.mm_projector.0."):
        return "model.multi_modal_projector.linear_1." + k.split(".")[-1]
    if k.startswith("model.mm_projector.2."):
        return "model.multi_modal_projector.linear_2." + k.split(".")[-1]
    if k == "model.image_newline" or k.startswith("lm_head."):
        return k
    if k.startswith(("model.embed_tokens.", "model.layers.", "model.norm.")):
        return "model.language_model." + k[len("model."):]
    return None

def convert_llava_video():
    if os.path.exists(os.path.join(CONV_DIR, "DONE")):
        return
    t0 = time.time()
    os.makedirs(CONV_DIR, exist_ok=True)
    tmp = os.path.join(WORK_DIR, "lv_orig")
    idx = json.load(open(hf_hub_download(MODEL_ID, "model.safetensors.index.json", local_dir=tmp)))
    shards = sorted(set(idx["weight_map"].values()))
    new_map, skipped, vocab = {}, [], None
    for sh in shards:
        p = hf_hub_download(MODEL_ID, sh, local_dir=tmp)
        out = {}
        with safe_open(p, "pt") as f:
            for k in f.keys():
                nk = _rename(k)
                if nk is None:
                    skipped.append(k)
                    continue
                t = f.get_tensor(k)
                if nk == "model.language_model.embed_tokens.weight":
                    vocab = int(t.shape[0])
                out[nk] = t.to(torch.float16).contiguous()
        save_file(out, os.path.join(CONV_DIR, sh), metadata={"format": "pt"})
        new_map.update({k: sh for k in out})
        del out
        os.remove(p); _clean_cache(); gc.collect()
        print(f"  converted {sh} ({time.time() - t0:.0f}s)")
    if skipped:
        raise RuntimeError(f"LLaVA-Video weights with no mapping: {skipped[:5]} — send me this error")
    cfg = AutoConfig.from_pretrained(HF_BASE)
    cfg.text_config.vocab_size = vocab
    cfg.vision_config.vision_use_head = False          # the LLaVA-NeXT vision tower has no pooling head
    cfg.save_pretrained(CONV_DIR)
    with open(os.path.join(CONV_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": new_map}, f)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(CONV_DIR)
    shutil.rmtree(tmp, ignore_errors=True)
    open(os.path.join(CONV_DIR, "DONE"), "w").write("ok")
    print(f"LLaVA-Video converted in {time.time() - t0:.0f}s")

convert_llava_video()
max_memory = {i: "13GiB" for i in range(N_GPU)}
if N_GPU >= 2:
    max_memory[0] = "10GiB"
model, _info = LlavaOnevisionForConditionalGeneration.from_pretrained(
    CONV_DIR, device_map="auto", max_memory=max_memory, attn_implementation="sdpa", dtype=torch.float16,
    output_loading_info=True)
model = model.eval()
_bad_missing = [k for k in _info.get("missing_keys", []) if "post_layernorm" not in k]
_unexpected = list(_info.get("unexpected_keys", []))
if _bad_missing or _unexpected:
    raise RuntimeError(f"Conversion check failed: missing {_bad_missing[:5]}, unexpected {_unexpected[:5]}")
print("Converted weights loaded: no missing or unexpected parameters (the unused final vision layer norm aside) ✅")
tok = AutoTokenizer.from_pretrained(CONV_DIR)

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

vision_tower = _get(model, ["vision_tower", "model.vision_tower"])
projector    = _get(model, ["multi_modal_projector", "model.multi_modal_projector"])
NEWLINE      = _get(model, ["image_newline", "model.image_newline"]).data
lm_head      = _get(model, ["lm_head", "language_model.lm_head"])
body = None
for _, mod in model.named_modules():
    if hasattr(mod, "embed_tokens") and hasattr(mod, "layers") and hasattr(mod, "norm"):
        body = mod
        break
assert body is not None, "Could not find the language decoder"

CFG, VCFG, TCFG = model.config, model.config.vision_config, model.config.text_config
VFL = CFG.vision_feature_layer
VSS = getattr(CFG, "vision_feature_select_strategy", "full")
IMG_SIZE, PATCH = VCFG.image_size, VCFG.patch_size
SIDE = IMG_SIZE // PATCH
POOLED = SIDE // 2                                     # LLaVA-Video: average pooling, stride 2 -> 13 x 13
NLAYERS, HID = TCFG.num_hidden_layers, TCFG.hidden_size
NHEADS, NKV = TCFG.num_attention_heads, TCFG.num_key_value_heads
HEAD_DIM = getattr(TCFG, "head_dim", None) or HID // NHEADS
INTER = TCFG.intermediate_size
EMB_DEVICE = body.embed_tokens.weight.device
LM_DEVICE = lm_head.weight.device
IMG_MEAN, IMG_STD = [0.5] * 3, [0.5] * 3               # SigLIP normalisation (LLaVA-Video image processor)
VDEV = next(vision_tower.parameters()).device

# The vision layer that produces the features: its attention drives FlashVID / HoliTom / VisionZip
ENC_LAYERS = _get(vision_tower, ["vision_model.encoder.layers", "encoder.layers", "model.encoder.layers"])
assert isinstance(VFL, int)
_n_hs = len(ENC_LAYERS) + 1
_h_idx = VFL if VFL >= 0 else _n_hs + VFL
VZ_LAYER, VZ_IN = ENC_LAYERS[_h_idx - 1], _h_idx - 1
VZ_HEADS = VCFG.num_attention_heads
VZ_HD = VCFG.hidden_size // VZ_HEADS
print(f"LLM layers={NLAYERS} hidden={HID} heads={NHEADS}/{NKV} | vision layers {len(ENC_LAYERS)}, features from "
      f"layer {_h_idx} | {SIDE}x{SIDE} -> {POOLED}x{POOLED} tokens/frame + 1 newline per frame")

LETTERS = "ABCDEFGH"
LETTER_IDS = []
for L in LETTERS:
    ids = tok.encode(L, add_special_tokens=False)
    assert len(ids) == 1, f"Letter {L} is not a single token"
    LETTER_IDS.append(ids[0])

PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
PREFIX_IDS = tok(PREFIX, add_special_tokens=False, return_tensors="pt").input_ids[0]

def _letter_list(n):
    L = list(LETTERS[:n])
    return L[0] if n == 1 else (f"{L[0]} or {L[1]}" if n == 2 else ", ".join(L[:-1]) + ", or " + L[-1])

def build_suffix_ids(question, options):
    """The main study's prompt (identical text)."""
    opts = "\n".join(str(o) for o in options)
    s = ("\nSelect the best answer to the following multiple-choice question based on the video. "
         f"Respond with only the letter ({_letter_list(len(options))}) of the correct option.\n"
         f"{question}\n{opts}\nThe best answer is:<|im_end|>\n<|im_start|>assistant\n")
    return tok(s, add_special_tokens=False, return_tensors="pt").input_ids[0]

def preprocess(frames):
    arr = np.stack([np.asarray(Image.fromarray(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
                    for f in frames]).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(0, 3, 1, 2)
    m, s = torch.tensor(IMG_MEAN)[:, None, None], torch.tensor(IMG_STD)[:, None, None]
    return (x - m) / s

def get_2dPool(t):
    """LLaVA-NeXT get_2dPool with mm_spatial_pool_mode='average' (stride 2): (T, SIDE*SIDE, C) -> (T, 13*13, C)."""
    T, P, C = t.shape
    t = t.view(T, SIDE, SIDE, C).permute(0, 3, 1, 2).contiguous()
    t = F.avg_pool2d(t, 2)
    return t.permute(0, 2, 3, 1).reshape(T, -1, C)

@torch.no_grad()
def encode_frames(frames, meta=None, bs=8):
    """frames -> dict(X=(T, 169, D) projected+pooled tokens, VA=(T, 169) attention received in the feature layer
    (mean over heads and queries), VM=(T, 169, head_dim) keys averaged over heads)."""
    pix = preprocess(frames)
    pdev = next(projector.parameters()).device
    feats, vz_a, vz_m = [], [], []
    for i in range(0, pix.shape[0], bs):
        out = vision_tower(pixel_values=pix[i:i+bs].to(VDEV, torch.float16), output_hidden_states=True)
        h = out.hidden_states[VFL]
        if VSS == "default":
            h = h[:, 1:]
        xin = out.hidden_states[VZ_IN].to(VZ_LAYER.layer_norm1.weight.device)
        if VSS == "default":
            xin = xin[:, 1:]
        xin = VZ_LAYER.layer_norm1(xin)
        b, P, _ = xin.shape
        q = VZ_LAYER.self_attn.q_proj(xin).float().view(b, P, VZ_HEADS, VZ_HD).transpose(1, 2)
        k = VZ_LAYER.self_attn.k_proj(xin).float().view(b, P, VZ_HEADS, VZ_HD).transpose(1, 2)
        w = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * VZ_HD ** -0.5, dim=-1)
        vz_a.append(w.mean(dim=1).mean(dim=1))
        vz_m.append(k.mean(dim=1))
        del q, k, w
        feats.append(projector(h.to(pdev)))
    X = get_2dPool(torch.cat(feats).float()).to(torch.float16).contiguous()
    dev = X.device
    VA = get_2dPool(torch.cat(vz_a).unsqueeze(-1).to(dev))[..., 0]
    VM = get_2dPool(torch.cat(vz_m).to(dev))
    return {"X": X, "VA": VA, "VM": VM, "T": X.shape[0], "n_full": X.shape[0] * X.shape[1]}

def full_spec(enc):
    T, L, D = enc["X"].shape
    return {"tokens": enc["X"].reshape(-1, D), "frame": torch.arange(T, device=enc["X"].device).repeat_interleave(L),
            "T": T, "inner": None, "nvis": T * L}

def video_block(spec):
    """LLaVA-Video layout ('frame' newline, as in FlashVID's script): each frame's kept tokens, then the newline."""
    tok_, fr, T = spec["tokens"], spec["frame"].to(spec["tokens"].device), spec["T"]
    K, D = tok_.shape
    if K > 1:
        assert bool((fr[1:] >= fr[:-1]).all()), "tokens must be grouped by frame"
    counts = torch.bincount(fr, minlength=T)
    out = torch.empty(K + T, D, dtype=tok_.dtype, device=tok_.device)
    is_nl = torch.zeros(K + T, dtype=torch.bool, device=tok_.device)
    nl_at = torch.cumsum(counts, 0) + torch.arange(T, device=tok_.device)
    out[torch.arange(K, device=tok_.device) + fr] = tok_
    out[nl_at] = NEWLINE.to(tok_.device, tok_.dtype)
    is_nl[nl_at] = True
    return out, is_nl

def llm_flops_layers(ns):
    dkv = NKV * HEAD_DIM
    return float(sum(2 * (2 * n * HID * HID + 2 * n * HID * dkv + 2 * n * n * HID + 3 * n * HID * INTER) for n in ns))

def llm_flops(n):
    return llm_flops_layers([n] * NLAYERS)

def _sync():
    for i in range(N_GPU):
        torch.cuda.synchronize(i)

# ── inner-LLM operations (FlashVID: FastV-style pruning; HoliTom: similarity merging) ──
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

@torch.no_grad()
def last_query_attention(layer, h, pe):
    a = layer.self_attn
    dev = a.q_proj.weight.device
    h = h.to(dev)
    cos, sin = pe[0].to(dev), pe[1].to(dev)
    hn = layer.input_layernorm(h)
    q = a.q_proj(hn[:, -1:]).view(1, 1, -1, HEAD_DIM).transpose(1, 2)
    k = a.k_proj(hn).view(1, hn.shape[1], -1, HEAD_DIM).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
    _, k = apply_rotary_pos_emb(k, k, cos, sin)
    k = repeat_kv(k, NHEADS // NKV)
    w = torch.matmul(q, k.transpose(2, 3)) * a.scaling
    return torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)

@torch.no_grad()
def apply_inner(h, pos, pe, attn, inner, v0, vlen, is_nl):
    """The whole video block (tokens and per-frame newlines) is the visual span, as in both official codes for
    LLaVA-Video (FlashVID sets visual_token_length after adding the newlines; HoliTom uses the image feature length).
    Returns the new (h, pos, pe) and the number of non-newline visual tokens left."""
    dev = h.device
    a = attn.mean(dim=1)[0, -1].to(dev)
    N = h.shape[1]
    nl = is_nl.to(dev)
    if inner["kind"] == "fastv":                               # FlashVID utils.fastv_prune
        n_keep = math.ceil(vlen * inner["keep_ratio"])
        top = torch.topk(a[v0:v0 + vlen], k=n_keep).indices.sort().values + v0
        keep = torch.cat([torch.arange(v0, device=dev), top, torch.arange(v0 + vlen, N, device=dev)]).sort().values
    elif inner["kind"] == "holitom":                           # HoliTom modeling_qwen2
        n_top = round(vlen * (1 - inner["merge_ratio"]))
        top = a[v0:v0 + vlen].topk(n_top).indices + v0
        span = torch.arange(v0, v0 + vlen, device=dev)
        non_top = span[~torch.isin(span, top)]
        hs = h[0].float()
        nt, tp = hs[non_top], hs[top]
        sim = (nt @ tp.t()) / (nt.norm(dim=-1, keepdim=True) * tp.norm(dim=-1, keepdim=True).t())
        tgt = sim.argmax(dim=-1)
        m = torch.bincount(tgt, minlength=len(top))
        srt, perm = torch.sort(tgt, stable=True)
        first = torch.searchsorted(srt, srt)
        rank = torch.empty_like(tgt)
        rank[perm] = torch.arange(len(tgt), device=dev) - first + 1
        w = torch.pow(2.0, -(m[tgt] - rank + 1).float())
        new_top = tp * torch.pow(2.0, -m.float()).unsqueeze(1)
        new_top.index_add_(0, tgt, nt * w.unsqueeze(1))
        h = h.clone()
        h[0, top] = new_top.to(h.dtype)
        keep = torch.cat([torch.arange(v0, device=dev), top, torch.arange(v0 + vlen, N, device=dev)]).sort().values
    else:
        raise ValueError(inner["kind"])
    n_vis = int((~nl[top - v0]).sum())
    kp = keep.to(pos.device)
    pe = (pe[0][:, keep.to(pe[0].device)], pe[1][:, keep.to(pe[1].device)])
    return h[:, keep], pos[:, kp], pe, n_vis

@torch.no_grad()
def forward_layers(x, pos, inner=None, v0=0, vlen=0, is_nl=None, nvis0=0):
    """Our own decoder loop (same calls as Qwen2Model.forward); lets FlashVID/HoliTom act inside the LLM."""
    pe = body.rotary_emb(x, pos)
    h, attn, ns, nvis = x, None, [], []
    vis_now = nvis0
    for li, layer in enumerate(body.layers):
        if inner is not None and li == inner["layer"]:
            h, pos, pe, vis_now = apply_inner(h, pos, pe, attn, inner, v0, vlen, is_nl)
            attn = None
        if inner is not None and li == inner["layer"] - 1:
            attn = last_query_attention(layer, h, pe)
        ns.append(h.shape[1]); nvis.append(vis_now)
        h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe, use_cache=False)
        if isinstance(h, tuple):
            h = h[0]
    h = body.norm(h.to(body.norm.weight.device))
    return h, ns, nvis

def _result(logits, n_opt, extra):
    lp = torch.log_softmax(logits[LETTER_IDS[:n_opt]], -1).cpu().numpy()
    if not np.all(np.isfinite(lp)):
        raise FloatingPointError("NaN/inf logits (fp16 overflow)")
    srt = np.sort(lp)[::-1]
    return {"pred": LETTERS[int(lp.argmax())], "logp": lp.round(4).tolist(), "margin": float(srt[0] - srt[1]), **extra}

@torch.no_grad()
def llm_pass(spec, suf_ids, n_opt, manual=False, pre_ids=None):
    """One full prefill of [system prompt, video block, question]. Inner-LLM methods run our own decoder loop."""
    E = body.embed_tokens
    pre_ids = PREFIX_IDS if pre_ids is None else pre_ids
    pre, suf = E(pre_ids.to(EMB_DEVICE)), E(suf_ids.to(EMB_DEVICE))
    vb, is_nl = video_block(spec)
    x = torch.cat([pre, vb.to(EMB_DEVICE, pre.dtype), suf], 0)[None]
    pos = torch.arange(x.shape[1], device=EMB_DEVICE)[None]
    am = torch.ones(1, x.shape[1], dtype=torch.long, device=EMB_DEVICE)
    for i in range(N_GPU):
        torch.cuda.reset_peak_memory_stats(i)
    _sync(); t0 = time.time()
    inner = spec.get("inner")
    with sdpa_kernel(NO_MATH):
        if inner is None and not manual:
            out = body(inputs_embeds=x, attention_mask=am, position_ids=pos, use_cache=False)
            h = out[0][0, -1]
            ns, nvis = [x.shape[1]] * NLAYERS, [spec["nvis"]] * NLAYERS
        else:
            hh, ns, nvis = forward_layers(x, pos, inner, v0=pre.shape[0], vlen=vb.shape[0], is_nl=is_nl,
                                          nvis0=spec["nvis"])
            h = hh[0, -1]
    logits = lm_head(h.to(LM_DEVICE)[None])[0].float()
    _sync(); dt = time.time() - t0
    return _result(logits, n_opt, {"ntok": int(spec["nvis"]), "vis_mean_layers": float(np.mean(nvis)), "sec": float(dt),
                                   "flops": llm_flops_layers(ns),
                                   "peak_gb": round(max(torch.cuda.max_memory_allocated(i) for i in range(N_GPU)) / 1e9, 3)})

# ── Prefix caching: the system prompt + video block are run once per (video, pass); every question reuses them ──
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
def prefill(spec, pre_ids=None):
    E = body.embed_tokens
    pre_ids = PREFIX_IDS if pre_ids is None else pre_ids
    pre = E(pre_ids.to(EMB_DEVICE))
    vb, _ = video_block(spec)
    x = torch.cat([pre, vb.to(EMB_DEVICE, pre.dtype)], 0)[None]
    pos = torch.arange(x.shape[1], device=EMB_DEVICE)[None]
    am = torch.ones(1, x.shape[1], dtype=torch.long, device=EMB_DEVICE)
    for i in range(N_GPU):
        torch.cuda.reset_peak_memory_stats(i)
    cache = _new_cache()
    _sync(); t0 = time.time()
    with sdpa_kernel(NO_MATH):
        body(inputs_embeds=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True)
    _sync()
    return {"cache": cache, "n": int(x.shape[1]), "next_pos": int(x.shape[1]), "sec": time.time() - t0,
            "nvis": int(spec["nvis"]), "peak": max(torch.cuda.max_memory_allocated(i) for i in range(N_GPU)) / 1e9}

@torch.no_grad()
def suffix_pass(P, suf_ids, n_opt):
    E = body.embed_tokens
    suf = E(suf_ids.to(EMB_DEVICE))[None]
    n_q = suf.shape[1]
    pos = (P["next_pos"] + torch.arange(n_q, device=EMB_DEVICE))[None]
    am = torch.ones(1, P["n"] + n_q, dtype=torch.long, device=EMB_DEVICE)
    _sync(); t0 = time.time()
    with sdpa_kernel([SDPBackend.MATH]):
        out = body(inputs_embeds=suf, attention_mask=am, position_ids=pos, past_key_values=P["cache"], use_cache=True)
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

preflight_attention(NUM_FRAMES * (POOLED ** 2 + 1) + 400)
print("Model ready ✅")
