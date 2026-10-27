# ── Model: LLaVA-OneVision with direct control over visual tokens ──────
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

max_memory = {i: "13GiB" for i in range(N_GPU)}
if N_GPU >= 2:
    max_memory[0] = "10GiB"
_dtype_kw = {"dtype": torch.float16} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": torch.float16}
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", max_memory=max_memory, attn_implementation="sdpa", **_dtype_kw).eval()
tok = AutoTokenizer.from_pretrained(MODEL_ID)

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
POOLED = math.ceil(SIDE / 2)
NLAYERS, HID = TCFG.num_hidden_layers, TCFG.hidden_size
NHEADS, NKV = TCFG.num_attention_heads, TCFG.num_key_value_heads
HEAD_DIM = getattr(TCFG, "head_dim", None) or HID // NHEADS
INTER = TCFG.intermediate_size
EMB_DEVICE = body.embed_tokens.weight.device
LM_DEVICE = lm_head.weight.device
try:
    from transformers import AutoImageProcessor
    _ip = AutoImageProcessor.from_pretrained(MODEL_ID)
    IMG_MEAN, IMG_STD = list(_ip.image_mean), list(_ip.image_std)
except Exception:
    IMG_MEAN, IMG_STD = [0.5] * 3, [0.5] * 3

# ── FastVID needs the ORIGINAL SigLIP post-layernorm + attention-pooling head (as in the official code) ──
from transformers import SiglipVisionModel
_sig = SiglipVisionModel.from_pretrained(SIGLIP_ID, **_dtype_kw)
VDEV = next(vision_tower.parameters()).device
# transformers v4 nests these under .vision_model; v5 puts them directly on the model
SIG_POSTLN = _get(_sig, ["vision_model.post_layernorm", "post_layernorm", "model.post_layernorm"]).to(VDEV).eval()
SIG_HEAD = _get(_sig, ["vision_model.head", "head", "model.head"]).to(VDEV).eval()
for _need in ("probe", "attention", "layernorm", "mlp"):
    assert hasattr(SIG_HEAD, _need), (f"SigLIP head has no '{_need}' (type {type(SIG_HEAD).__name__}, "
                                      f"children {[n for n, _ in SIG_HEAD.named_children()]}) — send me this error")
assert isinstance(SIG_HEAD.attention, torch.nn.MultiheadAttention), \
    f"Unexpected SigLIP head attention type {type(SIG_HEAD.attention).__name__} — send me this error"
print("SigLIP pooling head loaded (post_layernorm + attention-pooling head) ✅")

# ── VisionZip needs the attention and keys of the vision layer that produces the features ──
# (official LLaVA-OV port in the HoliTom repo: attentions[-1].mean(heads).mean(queries) and keys.mean(heads))
ENC_LAYERS = _get(vision_tower, ["vision_model.encoder.layers", "encoder.layers", "model.encoder.layers"])
assert isinstance(VFL, int), "VisionZip port assumes a single vision feature layer"
_n_hs = len(ENC_LAYERS) + 1
_h_idx = VFL if VFL >= 0 else _n_hs + VFL
VZ_LAYER, VZ_IN = ENC_LAYERS[_h_idx - 1], _h_idx - 1
VZ_HEADS = VCFG.num_attention_heads
VZ_HD = VCFG.hidden_size // VZ_HEADS
print(f"VisionZip reads attention/keys from vision layer {_h_idx - 1} of {len(ENC_LAYERS)}")
del _sig; gc.collect()
print(f"LLM layers={NLAYERS} hidden={HID} heads={NHEADS}/{NKV} | {SIDE}x{SIDE} -> {POOLED}x{POOLED} tokens/frame")

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
    """The pilot prompt; for 4 options the text is identical to Steps 1 and 1b."""
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

@torch.no_grad()
def encode_video(pix, bs=8):
    """frames -> (T, POOLED*POOLED, D) projected tokens (exactly LLaVA-OV's video path, before the newline token),
    FastVID's per-frame SigLIP-head attention (T, P') and global feature (T, 1152),
    and VisionZip's pooled vision attention (T, P') and pooled keys (T, P', head_dim)."""
    pdev = next(projector.parameters()).device
    feats, attns, globs, vz_a, vz_m = [], [], [], [], []
    for i in range(0, pix.shape[0], bs):
        out = vision_tower(pixel_values=pix[i:i+bs].to(VDEV, torch.float16), output_hidden_states=True)
        h = out.hidden_states[VFL]
        if VSS == "default":
            h = h[:, 1:]
        # FastVID "vision abstract": original SigLIP post_layernorm + attention-pooling head on the same features
        z = SIG_POSTLN(h.to(VDEV))
        probe = SIG_HEAD.probe.repeat(z.shape[0], 1, 1)
        att_out, att_w = SIG_HEAD.attention(probe, z, z, need_weights=True)      # att_w: (b, 1, SIDE*SIDE)
        pooled = att_out + SIG_HEAD.mlp(SIG_HEAD.layernorm(att_out))
        globs.append(pooled[:, 0].float())
        attns.append(att_w.float())
        # VisionZip: self-attention of the feature-producing layer, recomputed (sdpa does not return weights)
        xin = out.hidden_states[VZ_IN].to(VZ_LAYER.layer_norm1.weight.device)
        if VSS == "default":
            xin = xin[:, 1:]
        xin = VZ_LAYER.layer_norm1(xin)
        b, P, _ = xin.shape
        q = VZ_LAYER.self_attn.q_proj(xin).float().view(b, P, VZ_HEADS, VZ_HD).transpose(1, 2)
        k = VZ_LAYER.self_attn.k_proj(xin).float().view(b, P, VZ_HEADS, VZ_HD).transpose(1, 2)
        w = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * VZ_HD ** -0.5, dim=-1)
        vz_a.append(w.mean(dim=1).mean(dim=1))          # (b, P): attention each token receives
        vz_m.append(k.mean(dim=1))                      # (b, P, head_dim): keys averaged over heads
        del q, k, w
        feats.append(projector(h.to(pdev)))
    def pool(t):                                        # (T, SIDE*SIDE, C) -> (T, POOLED*POOLED, C), = get_2dPool
        T, P, C = t.shape
        t = t.view(T, SIDE, SIDE, C).permute(0, 3, 1, 2).float()
        t = F.interpolate(t, size=(POOLED, POOLED), mode="bilinear")
        return t.permute(0, 2, 3, 1).reshape(T, POOLED * POOLED, C)
    f = pool(torch.cat(feats)).contiguous().to(torch.float16)
    dev = f.device
    a = pool(torch.cat(attns).view(-1, SIDE * SIDE, 1).to(dev))[..., 0]
    va = pool(torch.cat(vz_a).unsqueeze(-1).to(dev))[..., 0]
    vm = pool(torch.cat(vz_m).to(dev))
    return f, a, torch.cat(globs).to(dev), va, vm

def llm_flops_layers(ns):
    """Prefill FLOPs of the LLM when layer l sees ns[l] tokens."""
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
    """Attention of the LAST token over all tokens in this layer, (1, heads, 1, N) — exactly the row both official
    codes read (FlashVID computes it this way under flash/sdpa; HoliTom takes the last row of eager attention)."""
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
def apply_inner(h, pos, pe, attn, inner, v0, vlen, has_newline):
    """Returns pruned/merged (h, pos, pe) and the number of visual tokens left."""
    dev = h.device
    a = attn.mean(dim=1)[0, -1].to(dev)                       # mean over heads -> (N,)
    N = h.shape[1]
    if inner["kind"] == "fastv":                               # FlashVID utils.fastv_prune
        n_keep = math.ceil(vlen * inner["keep_ratio"])
        top = torch.topk(a[v0:v0 + vlen], k=n_keep).indices.sort().values + v0
        keep = torch.cat([torch.arange(v0, device=dev), top, torch.arange(v0 + vlen, N, device=dev)]).sort().values
        n_vis = n_keep
    elif inner["kind"] == "holitom":                           # HoliTom modeling_qwen2 (newline counted in the span)
        L = vlen + (1 if (has_newline and inner["include_newline"]) else 0)
        n_top = round(L * (1 - inner["merge_ratio"]))
        top = a[v0:v0 + L].topk(n_top).indices + v0
        span = torch.arange(v0, v0 + L, device=dev)
        non_top = span[~torch.isin(span, top)]                 # ascending, as in the official loop
        hs = h[0].float()
        nt, tp = hs[non_top], hs[top]
        sim = (nt @ tp.t()) / (nt.norm(dim=-1, keepdim=True) * tp.norm(dim=-1, keepdim=True).t())
        tgt = sim.argmax(dim=-1)
        # official: for each non-top token in order, h[t] = (h[t] + h[i]) / 2  -> closed form below (same maths)
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
        keep = torch.cat([torch.arange(v0, device=dev), top, torch.arange(v0 + L, N, device=dev)]).sort().values
        n_vis = n_top - (int(bool((top == v0 + vlen).any())) if L > vlen else 0)
    else:
        raise ValueError(inner["kind"])
    kp = keep.to(pos.device)
    pe = (pe[0][:, keep.to(pe[0].device)], pe[1][:, keep.to(pe[1].device)])
    return h[:, keep], pos[:, kp], pe, n_vis

@torch.no_grad()
def forward_layers(x, pos, inner=None, v0=0, vlen=0, has_newline=True):
    """Our own decoder loop (same calls as Qwen2Model.forward); lets FlashVID/HoliTom act inside the LLM."""
    pe = body.rotary_emb(x, pos)
    h, attn, ns, nvis = x, None, [], []
    vis_now = vlen
    for li, layer in enumerate(body.layers):
        if inner is not None and li == inner["layer"]:
            h, pos, pe, vis_now = apply_inner(h, pos, pe, attn, inner, v0, vlen, has_newline)
            attn = None
        if inner is not None and li == inner["layer"] - 1:
            attn = last_query_attention(layer, h, pe)
        ns.append(h.shape[1]); nvis.append(vis_now)
        h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe, use_cache=False)
        if isinstance(h, tuple):
            h = h[0]
    h = body.norm(h.to(body.norm.weight.device))
    return h, ns, nvis

@torch.no_grad()
def llm_pass(vis, vis_pos, n_video_full, suf_ids, n_opt, pre_ids=None, newline=True, inner=None, manual=False):
    """One prefill. vis: kept video tokens (K, D); vis_pos: their position offsets inside the video span;
    n_video_full: length of the video span for positions (the newline and question follow it).
    newline=False drops LLaVA-OV's video newline token (LLaVA-Scissor's flat path).
    inner: optional FlashVID/HoliTom inner-LLM operation (runs our own decoder loop)."""
    E = body.embed_tokens
    pre_ids = PREFIX_IDS if pre_ids is None else pre_ids
    pre, suf = E(pre_ids.to(EMB_DEVICE)), E(suf_ids.to(EMB_DEVICE))
    parts = [pre, vis.to(EMB_DEVICE, pre.dtype)]
    if newline:
        parts.append(NEWLINE.to(EMB_DEVICE, pre.dtype)[None])
    parts.append(suf)
    x = torch.cat(parts, 0)[None]
    n_pre, n_q = pre.shape[0], suf.shape[0]
    vp = torch.as_tensor(vis_pos, device=EMB_DEVICE, dtype=torch.long)
    nl = 1 if newline else 0
    pos = torch.cat([torch.arange(n_pre, device=EMB_DEVICE), n_pre + vp] +
                    ([torch.tensor([n_pre + n_video_full], device=EMB_DEVICE)] if newline else []) +
                    [n_pre + n_video_full + nl + torch.arange(n_q, device=EMB_DEVICE)])[None]
    am = torch.ones(1, x.shape[1], dtype=torch.long, device=EMB_DEVICE)
    for i in range(N_GPU):
        torch.cuda.reset_peak_memory_stats(i)
    _sync(); t0 = time.time()
    with sdpa_kernel(NO_MATH):
        if inner is None and not manual:
            out = body(inputs_embeds=x, attention_mask=am, position_ids=pos, use_cache=False)
            h = out[0][0, -1]
            ns, nvis = [x.shape[1]] * NLAYERS, [vis.shape[0]] * NLAYERS
        else:
            hh, ns, nvis = forward_layers(x, pos, inner, v0=n_pre, vlen=vis.shape[0], has_newline=newline)
            h = hh[0, -1]
    logits = lm_head(h.to(LM_DEVICE)[None])[0].float()
    _sync(); dt = time.time() - t0
    lp = torch.log_softmax(logits[LETTER_IDS[:n_opt]], -1).cpu().numpy()
    if not np.all(np.isfinite(lp)):
        raise FloatingPointError("NaN/inf logits (fp16 overflow)")
    srt = np.sort(lp)[::-1]
    return {"pred": LETTERS[int(lp.argmax())], "logp": lp.round(4).tolist(), "margin": float(srt[0] - srt[1]),
            "ntok": int(vis.shape[0]), "vis_mean_layers": float(np.mean(nvis)), "sec": float(dt),
            "flops": llm_flops_layers(ns),
            "peak_gb": round(max(torch.cuda.max_memory_allocated(i) for i in range(N_GPU)) / 1e9, 3)}

# ── Prefix caching: the system prompt + video tokens are run once per (video, pass); every question reuses them ──
from transformers import DynamicCache

def _new_cache():
    try:
        return DynamicCache(config=TCFG)
    except TypeError:
        return DynamicCache()

def _crop(cache, n):
    extra = cache.get_seq_length() - n
    if extra > 0 and hasattr(cache, "crop"):
        cache.crop(-extra)                      # negative = drop that many tokens (works in old and new versions)
    if cache.get_seq_length() != n:
        for l in cache.layers:
            l.keys, l.values = l.keys[..., :n, :], l.values[..., :n, :]
    assert cache.get_seq_length() == n, "could not restore the cached prefix"

@torch.no_grad()
def prefill(vis, vis_pos, n_video_full, newline=True, pre_ids=None):
    """Runs [system prompt, video tokens, (newline)] once and keeps the key/value cache."""
    E = body.embed_tokens
    pre_ids = PREFIX_IDS if pre_ids is None else pre_ids
    pre = E(pre_ids.to(EMB_DEVICE))
    parts = [pre, vis.to(EMB_DEVICE, pre.dtype)] + ([NEWLINE.to(EMB_DEVICE, pre.dtype)[None]] if newline else [])
    x = torch.cat(parts, 0)[None]
    n_pre = pre.shape[0]
    vp = torch.as_tensor(vis_pos, device=EMB_DEVICE, dtype=torch.long)
    pos = torch.cat([torch.arange(n_pre, device=EMB_DEVICE), n_pre + vp] +
                    ([torch.tensor([n_pre + n_video_full], device=EMB_DEVICE)] if newline else []))[None]
    am = torch.ones(1, x.shape[1], dtype=torch.long, device=EMB_DEVICE)
    for i in range(N_GPU):
        torch.cuda.reset_peak_memory_stats(i)
    cache = _new_cache()
    _sync(); t0 = time.time()
    with sdpa_kernel(NO_MATH):
        body(inputs_embeds=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True)
    _sync()
    return {"cache": cache, "n": int(x.shape[1]), "next_pos": int(n_pre + n_video_full + (1 if newline else 0)),
            "sec": time.time() - t0, "nvis": int(vis.shape[0]),
            "peak": max(torch.cuda.max_memory_allocated(i) for i in range(N_GPU)) / 1e9}

@torch.no_grad()
def suffix_pass(P, suf_ids, n_opt):
    """Question tokens on top of a cached prefix; the cache is cropped back afterwards so it can be reused.
    'sec' and 'flops' are reported for the pass as if run alone (prefix + suffix)."""
    E = body.embed_tokens
    suf = E(suf_ids.to(EMB_DEVICE))[None]
    n_q = suf.shape[1]
    pos = (P["next_pos"] + torch.arange(n_q, device=EMB_DEVICE))[None]
    am = torch.ones(1, P["n"] + n_q, dtype=torch.long, device=EMB_DEVICE)
    _sync(); t0 = time.time()
    # short query block over a long cache: explicit mask, small -> the plain math kernel is safe and cheap here
    with sdpa_kernel([SDPBackend.MATH]):
        out = body(inputs_embeds=suf, attention_mask=am, position_ids=pos, past_key_values=P["cache"], use_cache=True)
    h = out[0][0, -1]
    logits = lm_head(h.to(LM_DEVICE)[None])[0].float()
    _sync(); dt = time.time() - t0
    _crop(P["cache"], P["n"])
    lp = torch.log_softmax(logits[LETTER_IDS[:n_opt]], -1).cpu().numpy()
    if not np.all(np.isfinite(lp)):
        raise FloatingPointError("NaN/inf logits (fp16 overflow)")
    srt = np.sort(lp)[::-1]
    return {"pred": LETTERS[int(lp.argmax())], "logp": lp.round(4).tolist(), "margin": float(srt[0] - srt[1]),
            "ntok": P["nvis"], "vis_mean_layers": float(P["nvis"]), "sec": float(P["sec"] + dt),
            "sec_suffix": float(dt), "flops": llm_flops(P["n"] + n_q), "peak_gb": round(P["peak"], 3)}

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

preflight_attention(NUM_FRAMES * POOLED ** 2 + 400)
print("Model ready ✅")
