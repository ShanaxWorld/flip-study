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

