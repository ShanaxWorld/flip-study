# ── FlashVID, HoliTom and VisionZip: the same faithful ports used in the main study (see that notebook for the
#    line-by-line provenance). Only additions here: FlashVID's ADTS v1 token selection ("attn_div"), which its
#    Qwen3-VL script uses, and a HoliTom helper that assigns the merged tokens of each window to frames, needed for
#    LLaVA-Video's per-frame newline token. ──
# ══ Official method code shared by the replication models (same ports as the main study) ══
# Each compressor returns a dict: tokens (K, D) entering the LLM, pos (their position ids, contiguous for all three
# as in the official code), newline (whether LLaVA-OV's video newline token is appended), and inner (an optional
# inner-LLM operation applied by llm_pass at a given layer).

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# FlashVID (Fan et al., ICLR 2026 oral) — github.com/Fanziyang-v/FlashVID, flashvid/utils.py + token_selection.py
# MIT License, Copyright (c) the FlashVID authors. The functions below are copied verbatim (only the config class is
# re-declared here); the official LLaVA-OV path passes the bilinear-pooled features and the bilinear-pooled
# last-vision-layer attention (mean over heads and queries), then prunes inside the LLM at layer 20 (FastV-style,
# last-query attention of layer 19, keep 30%). Positions are contiguous before the LLM, original after pruning.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

@dataclass
class FlashVidConfig:
    retention_ratio: float = 0.25
    alpha: float = 0.7
    token_selection_method: str = "attn_div_v2"
    temporal_threshold: float = 0.8
    do_segment: bool = True
    segment_threshold: float = 0.9
    min_segment_num: int = 8
    complementary_segment: bool = True
    num_attn_div_tokens: Optional[int] = None
    num_sttm_tokens: Optional[int] = None
    visual_token_start_index: Optional[int] = None
    visual_token_length: Optional[int] = None
    expansion: float = 1.25
    pruning_layer: int = 20
    llm_retention_ratio: float = 0.3

def pairwise_cosine_distances(image_features):
    normed_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    similarities = torch.bmm(normed_features, normed_features.transpose(-1, -2))
    return 1.0 - similarities

def attn_div_v2_based_token_selection(features, cls_attention, num_retained_tokens):
    original_features = features
    features = features.float()
    pooled_features = features.mean(1)
    global_cls_attention = cls_attention.float() * 1e6
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)
    calibration_term1 = global_cls_attention.unsqueeze(1)
    local_cls_attention = torch.einsum("b n d, c d -> b c n", features, pooled_features).mean(1)
    calibration_term2 = local_cls_attention.unsqueeze(1)
    dist_matrix = dist_matrix * calibration_term1 * calibration_term2
    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)
    for i in range(1, num_retained_tokens):
        dist_sub_matrix = torch.gather(dist_matrix, dim=1,
                                       index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens))
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)
    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(original_features, dim=1,
                                     index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
    return selected_features, keep_indices

def attn_based_token_selection(features, cls_attention, num_retained_tokens):
    bsz, num_visual_tokens, feat_dim = features.shape
    topk_indices = torch.topk(cls_attention, k=num_retained_tokens, dim=-1).indices.sort().values
    selected_features = torch.gather(features, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
    return selected_features, topk_indices

def attn_div_based_token_selection(features, cls_attention, num_retained_tokens):
    # FlashVID ADTS v1 (token_selection.py), used by its Qwen3-VL script; copied verbatim
    original_features = features
    features = features.float()
    cls_attention = cls_attention.float() * 1e6
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)
    dist_matrix = dist_matrix * cls_attention.unsqueeze(1)
    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)
    for i in range(1, num_retained_tokens):
        dist_sub_matrix = torch.gather(dist_matrix, dim=1,
                                       index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens))
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        min_dist.scatter_(1, keep_indices[:, :i], -1)
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)
    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(original_features, dim=1,
                                     index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
    return selected_features, keep_indices

ALL_TOKEN_SELECTION_METHOD = {"attn_div_v2": attn_div_v2_based_token_selection, "attn_div": attn_div_based_token_selection,
                              "attn": attn_based_token_selection}

def flashvid_compression(video_features, cls_attention, flashvid_config):
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    if flashvid_config.do_segment:
        segment_lengths = fv_segment(video_features=video_features.mean(1),
                                     segment_threshold=flashvid_config.segment_threshold,
                                     min_segment_num=flashvid_config.min_segment_num,
                                     complementary_segment=flashvid_config.complementary_segment)
    else:
        segment_lengths = torch.tensor([num_frames], dtype=torch.long, device=video_features.device)
    num_segments = segment_lengths.shape[0]
    global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)
    token_budget = math.ceil(num_visual_tokens * flashvid_config.retention_ratio * flashvid_config.expansion)
    num_attn_div_tokens = math.ceil(token_budget * flashvid_config.alpha)
    num_sttm_tokens = token_budget - num_attn_div_tokens
    flashvid_config.num_attn_div_tokens = num_attn_div_tokens
    flashvid_config.num_sttm_tokens = num_sttm_tokens
    all_segment_features, all_segment_indices = [], []
    offset = 0
    for seg_idx in range(num_segments):
        seg_len = segment_lengths[seg_idx]
        segment_features = video_features[offset: offset + seg_len]
        segment_cls_attention = cls_attention[offset: offset + seg_len]
        segment_global_indices = global_indices.view(num_frames, num_visual_tokens)[offset: offset + seg_len]
        segment_features, segment_global_indices = fv_segment_compression(
            segment_features=segment_features, segment_global_indices=segment_global_indices,
            cls_attention=segment_cls_attention, flashvid_config=flashvid_config)
        all_segment_features.append(segment_features)
        all_segment_indices.append(segment_global_indices)
        offset += seg_len
    final_tokens = torch.cat(all_segment_features, dim=0)
    final_global_indices = torch.cat(all_segment_indices, dim=0)
    sorted_indices = final_global_indices.argsort()
    sorted_tokens = final_tokens[sorted_indices]
    flashvid_config.visual_token_length = sorted_tokens.shape[0]
    return sorted_tokens, final_global_indices[sorted_indices]

def fv_segment_compression(segment_features, segment_global_indices, cls_attention, flashvid_config):
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    if flashvid_config.alpha > 0:
        additional_kwargs = {"cls_attention": cls_attention} if "attn" in flashvid_config.token_selection_method else {}
        selected_features, selected_indices = ALL_TOKEN_SELECTION_METHOD[flashvid_config.token_selection_method](
            features=segment_features, num_retained_tokens=flashvid_config.num_attn_div_tokens, **additional_kwargs)
        selected_global_indices = segment_global_indices.gather(1, index=selected_indices).view(-1)
    else:
        selected_features = torch.tensor([]).to(segment_features)
        selected_indices = torch.tensor([]).to(segment_global_indices)
        selected_global_indices = torch.tensor([]).to(segment_global_indices)
    mask = torch.ones(num_frames, num_visual_tokens, dtype=torch.bool, device=segment_features.device)
    mask.scatter_(1, selected_indices, False)
    num_other_tokens = flashvid_config.num_sttm_tokens * num_frames
    if num_other_tokens > 0 and flashvid_config.temporal_threshold < 1.0:
        if num_frames > 1:
            temp_merged_token_list, temp_merged_indices_list = fv_spatiotemporal_compression(
                video_features=segment_features, temporal_threshold=flashvid_config.temporal_threshold,
                token_mask=mask, flashvid_config=flashvid_config)
            temp_merged_global_indices_list = [segment_global_indices.view(num_frames, -1)[i][temp_merged_indices]
                                               for i, temp_merged_indices in enumerate(temp_merged_indices_list)]
        else:
            temp_merged_token_list = [segment_features[0]]
            temp_merged_global_indices_list = [segment_global_indices[0]]
    else:
        temp_merged_token_list = []
        temp_merged_global_indices_list = []
    all_tokens = [selected_features.view(-1, feat_dim)]
    all_global_indices = [selected_global_indices]
    if num_other_tokens > 0:
        num_current_retained_tokens = sum(len(tokens) for tokens in temp_merged_token_list)
        adapative_contextual_ratio = num_other_tokens / num_current_retained_tokens
        if adapative_contextual_ratio < 1.0:
            num_frames_in_segment = len(temp_merged_token_list)
            max_num_tokens = max(len(tokens) for tokens in temp_merged_token_list)
            batched_tokens = torch.zeros((num_frames_in_segment, max_num_tokens, feat_dim),
                                         dtype=segment_features.dtype, device=segment_features.device)
            valid_token_mask = torch.zeros((num_frames_in_segment, max_num_tokens), dtype=torch.bool,
                                           device=segment_features.device)
            num_clusters_list, k_list = [], []
            for i, temp_merged_tokens in enumerate(temp_merged_token_list):
                num_tokens = len(temp_merged_tokens)
                batched_tokens[i, :num_tokens] = temp_merged_tokens
                valid_token_mask[i, :num_tokens] = True
                num_clusters = math.ceil(num_tokens * adapative_contextual_ratio)
                num_clusters_list.append(num_clusters)
                k_list.append(min(num_clusters, 7))
            cluster_indices_list, cluster_center_indices_list = fv_dpc_knn(
                features=batched_tokens, num_clusters=num_clusters_list, k=k_list, valid_token_mask=valid_token_mask)
            for i, (temp_merged_tokens, temp_merged_global_indices) in enumerate(
                    zip(temp_merged_token_list, temp_merged_global_indices_list)):
                num_clusters = num_clusters_list[i]
                if num_clusters > 0:
                    cluster_indices = cluster_indices_list[i][:len(temp_merged_tokens)]
                    cluster_center_indices = cluster_center_indices_list[i]
                    aggregated_tokens = torch.zeros((num_clusters, feat_dim), dtype=segment_features.dtype,
                                                    device=segment_features.device)
                    aggregated_tokens.scatter_add_(0, cluster_indices.unsqueeze(-1).expand(-1, feat_dim),
                                                   temp_merged_tokens)
                    cluster_counts = torch.bincount(cluster_indices, minlength=num_clusters).unsqueeze(-1).to(
                        segment_features.dtype)
                    aggregated_tokens = aggregated_tokens / cluster_counts
                    global_token_indices = temp_merged_global_indices[cluster_center_indices]
                else:
                    aggregated_tokens = temp_merged_tokens
                    global_token_indices = temp_merged_global_indices
                all_tokens.append(aggregated_tokens)
                all_global_indices.append(global_token_indices)
        else:
            for temp_merged_tokens, temp_merged_global_indices in zip(temp_merged_token_list,
                                                                      temp_merged_global_indices_list):
                all_tokens.append(temp_merged_tokens)
                all_global_indices.append(temp_merged_global_indices)
    segment_final_tokens = torch.cat(all_tokens, dim=0)
    segment_final_global_indices = torch.cat(all_global_indices, dim=0)
    return segment_final_tokens, segment_final_global_indices

def fv_segment(video_features, segment_threshold, min_segment_num, complementary_segment=True):
    num_frames, feat_dim = video_features.shape
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    transition_similarities = torch.sum(normed_video_features[:-1] * normed_video_features[1:], dim=-1)
    cut_indices = torch.where(transition_similarities < segment_threshold)[0]
    return fv_additional_segment(cut_indices=cut_indices, num_frames=num_frames, min_segment_num=min_segment_num,
                                 transition_similarities=transition_similarities,
                                 segment_threshold=segment_threshold, complementary_segment=complementary_segment)

def fv_additional_segment(cut_indices, num_frames, min_segment_num, transition_similarities, segment_threshold,
                          complementary_segment=True):
    num_segments = cut_indices.numel() + 1
    if num_segments < min_segment_num and complementary_segment:
        num_remaining_cut_indices = min_segment_num - num_segments
        transition_similarities[transition_similarities < segment_threshold] = 1.0
        complementary_cut_indices = torch.topk(transition_similarities,
                                               k=min(num_remaining_cut_indices, transition_similarities.shape[0]),
                                               largest=False).indices
        cut_indices = torch.cat([cut_indices, complementary_cut_indices]).sort().values
    padded_cut_indices = F.pad(cut_indices, (1, 1), value=0)
    padded_cut_indices[0] = -1
    padded_cut_indices[-1] = num_frames - 1
    return torch.diff(padded_cut_indices, n=1, dim=0)

@torch.no_grad()
def fv_dpc_knn(features, num_clusters, k=7, valid_token_mask=None):
    invalid_token_mask = ~valid_token_mask if valid_token_mask is not None else None
    bsz, seq_len, feat_dim = features.shape
    dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)
    if valid_token_mask is not None:
        dists = torch.masked_fill(dists, invalid_token_mask.unsqueeze(1).expand(-1, seq_len, -1), dists.max() + 1)
    max_k = max(k) if isinstance(k, list) else k
    nearest_dist = torch.topk(dists, k=max_k, dim=-1, largest=False).values
    if isinstance(k, list):
        density = torch.empty((bsz, seq_len), device=features.device, dtype=features.dtype)
        for i in range(bsz):
            density[i] = torch.mean(-(nearest_dist[i, :, :k[i]] ** 2), dim=-1).exp()
    else:
        density = torch.mean(-(nearest_dist ** 2), dim=-1).exp()
    if valid_token_mask is not None:
        density = torch.masked_fill(density, invalid_token_mask, 0.0)
    mask = density[:, None, :] > density[:, :, None]
    max_dist = dists.view(bsz, -1).max(dim=-1)[0].view(-1, 1, 1)
    modified_dists = torch.where(mask, dists, max_dist)
    dist, _ = torch.min(modified_dists, dim=-1)
    score = dist * density
    if isinstance(num_clusters, int):
        cluster_center_indices = torch.topk(score, k=num_clusters, dim=-1).indices
        dists = torch.gather(dists, dim=-1, index=cluster_center_indices.unsqueeze(1).expand(-1, seq_len, -1))
        cluster_indices = torch.argmin(dists, dim=-1)
        cluster_indices.scatter_(dim=-1, index=cluster_center_indices,
                                 src=torch.arange(num_clusters).to(cluster_indices).unsqueeze(0).expand(bsz, -1))
        return cluster_indices, cluster_center_indices
    cluster_indices_list, cluster_center_indices_list = [], []
    for i in range(bsz):
        k_i = num_clusters[i]
        if k_i == 0:
            cluster_center_indices_list.append(torch.tensor([], dtype=torch.long, device=features.device))
            cluster_indices_list.append(torch.zeros(seq_len, dtype=torch.long, device=features.device))
            continue
        cc_idx = torch.topk(score[i], k=k_i, dim=-1).indices
        cluster_center_indices_list.append(cc_idx)
        dists_i = torch.gather(dists[i], dim=-1, index=cc_idx.unsqueeze(0).expand(seq_len, -1))
        c_idx = torch.argmin(dists_i, dim=-1)
        c_idx.scatter_(dim=-1, index=cc_idx, src=torch.arange(k_i).to(c_idx))
        cluster_indices_list.append(c_idx)
    return cluster_indices_list, cluster_center_indices_list

def fv_spatiotemporal_compression(video_features, temporal_threshold, token_mask, flashvid_config):
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    lower_bound = (flashvid_config.num_attn_div_tokens + flashvid_config.num_sttm_tokens) * num_frames
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    cosine_similarities = torch.bmm(normed_video_features[1:], normed_video_features[:-1].transpose(1, 2))
    cosine_similarities[~token_mask[1:].unsqueeze(-1).expand(-1, -1, num_visual_tokens)] = -1.0
    cosine_similarities[~token_mask[:-1].unsqueeze(1).expand(-1, num_visual_tokens, -1)] = -1.0
    max_sims, max_sim_indices = torch.max(cosine_similarities, dim=-1)
    padded_max_sims = F.pad(max_sims, (0, 0, 1, 0), value=-1)
    padded_max_sim_indices = F.pad(max_sim_indices, (0, 0, 1, 0), value=-1)
    token_counts = torch.ones(num_frames, num_visual_tokens).to(video_features)
    mask = padded_max_sims > temporal_threshold
    retaining_token_mask = ~mask
    if retaining_token_mask.int().sum() < lower_bound:
        soft_threshold = padded_max_sims.view(-1).topk(k=(num_frames * num_visual_tokens) - lower_bound).values[-1]
        soft_threshold = max(soft_threshold, -1.0 + 1e-6)
        mask = padded_max_sims > soft_threshold
        retaining_token_mask = ~mask
    for frame_idx in range(num_frames - 1, -1, -1):
        frame_features = video_features[frame_idx]
        frame_token_counts = token_counts[frame_idx]
        frame_max_sim_indices = padded_max_sim_indices[frame_idx]
        tokens_to_merge = frame_features[~mask[frame_idx]]
        to_merge_token_counts = frame_token_counts[~mask[frame_idx]]
        if tokens_to_merge.numel() > 0:
            aggregated_tokens = tokens_to_merge / to_merge_token_counts.unsqueeze(-1).to(tokens_to_merge.dtype)
            video_features[frame_idx][~mask[frame_idx]] = aggregated_tokens
            token_counts[frame_idx][~mask[frame_idx]] = 1
        other_tokens = frame_features[mask[frame_idx]]
        if other_tokens.numel() > 0:
            anchor_token_indices = frame_max_sim_indices[mask[frame_idx]]
            aggregated_tokens = torch.zeros((num_visual_tokens, feat_dim), dtype=video_features.dtype,
                                            device=video_features.device)
            aggregated_tokens.scatter_add_(0, anchor_token_indices.unsqueeze(-1).expand(-1, feat_dim), other_tokens)
            aggregated_token_counts = torch.bincount(anchor_token_indices, minlength=num_visual_tokens).to(
                video_features.dtype)
            video_features[frame_idx - 1] += aggregated_tokens
            token_counts[frame_idx - 1] += aggregated_token_counts
            token_counts[frame_idx][mask[frame_idx]] = 0
    final_tokens, retained_token_indices = [], []
    for i in range(num_frames):
        frame_mask = retaining_token_mask[i] & token_mask[i]
        final_tokens.append(video_features[i][frame_mask])
        retained_token_indices.append(torch.where(frame_mask)[0])
    return final_tokens, retained_token_indices


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# HoliTom (Shao et al., NeurIPS 2025) — github.com/cokeshao/HoliTom, holitom/llava_arch.py + modeling_qwen2.py
# Apache-2.0. Methods copied as plain functions (self removed); settings from scripts/eval_ov-7b_holitom.sh:
# T=0.65 at 10% (0.80 otherwise), BETA=0.6 (beta weighting on, as the script leaves NO_BETA unset), D=0, K=7,
# inner-LLM merging at layer 18 with r=0.5 (HOLITOM_k=18, HOLITOM_r=0.5). Only edit: the 1e-6 density tie-break noise
# uses a seeded generator. LLaVA-OV uses mm_newline_position "one_token", so tokens are [static, dynamic] per window,
# one newline at the end, contiguous positions; the inner merge treats the newline as part of the image span (as there).
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def ht_cluster_dpc_knn(x, cluster_num, k=7, gen=None):
    with torch.no_grad():
        batch_size, seq_len, embed_dim = x.shape
        dist_matrix = torch.cdist(x.float(), x.float()) / (embed_dim ** 0.5)
        dist_nearest, index_nearest = torch.topk(dist_matrix, k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype, generator=gen) * 1e-6
        mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
        score = dist * density
        _, index_center = score.topk(cluster_num, dim=-1)
        return index_center, dist_matrix

def ht_select_static_windows(feature_sim, batch_size, tau, max_window_size):
    def get_pruned_static_count_vectorized(feature_sim, batch_size, tau):
        similarity_matrix = torch.ones((batch_size, batch_size, feature_sim.shape[1]), device=feature_sim.device)
        for start in range(batch_size - 1):
            cum_similarity = torch.cumprod(feature_sim[start:] > tau, dim=0)
            similarity_matrix[start, start + 1:start + 1 + len(cum_similarity)] = cum_similarity
        window_lengths = torch.arange(batch_size, device=feature_sim.device).unsqueeze(0) - \
            torch.arange(batch_size, device=feature_sim.device).unsqueeze(1)
        window_lengths = window_lengths.clamp(min=0)
        return (similarity_matrix.sum(dim=-1) * window_lengths).float()
    pruned_static_count = get_pruned_static_count_vectorized(feature_sim, batch_size, tau)
    dp = torch.zeros(batch_size, device=pruned_static_count.device)
    prev = torch.zeros(batch_size, dtype=torch.long, device=pruned_static_count.device)
    for i in range(batch_size):
        max_val = dp[i - 1] if i > 0 else 0
        best_j = i
        for window_size in range(2, min(i + 1, max_window_size) + 1):
            j = i - window_size
            current_val = (dp[j] if j >= 0 else 0) + pruned_static_count[j + 1, i]
            if current_val > max_val:
                max_val = current_val
                best_j = j + 1
        dp[i] = max_val
        prev[i] = best_j
    selected_frames = []
    i = batch_size - 1
    while i >= 0:
        selected_frames.append((prev[i].item(), i))
        i = prev[i].item() - 1
    selected_frames = selected_frames[::-1]
    return selected_frames, dp[-1].item()

def ht_merge_tokens_by_clustering(feat, target_indices, dist_matrix, cluster_num, Beta):
    batch_size, seq_len, embed_dim = feat.shape
    all_indices = torch.arange(seq_len, device=feat.device).unsqueeze(0).expand(batch_size, -1)
    non_target_indices = torch.zeros((batch_size, seq_len - cluster_num), dtype=torch.long, device=feat.device)
    for b in range(batch_size):
        non_target_mask = ~torch.isin(all_indices[b], target_indices[b])
        non_target_indices[b] = all_indices[b][non_target_mask]
    non_target_feat = torch.gather(feat, dim=1, index=non_target_indices.unsqueeze(-1).expand(-1, -1, feat.size(-1)))
    dist_matrix = torch.gather(dist_matrix, dim=1,
                               index=non_target_indices.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1)))
    dist_matrix = torch.gather(dist_matrix, dim=2, index=target_indices.unsqueeze(1).expand(-1, dist_matrix.size(1), -1))
    idx_cluster = torch.argmin(dist_matrix, dim=-1)
    cluster_tokens = []
    for b in range(batch_size):
        batch_tokens = []
        for i in range(cluster_num):
            mask = (idx_cluster[b] == i)
            if mask.any():
                cluster_means = non_target_feat[b][mask].mean(dim=0)          # NO_BETA unset -> beta weighting
                batch_tokens.append(Beta * feat[b][target_indices[b][i]] + (1 - Beta) * cluster_means)
            else:
                batch_tokens.append(feat[b][target_indices[b][i]])
        cluster_tokens.append(torch.stack(batch_tokens))
    return torch.stack(cluster_tokens)

def ht_merge_tokens_by_attention_density(feat, attn, pos, retain_ratio, D, Beta, K, gen=None):
    batch_size, seq_len, embed_dim = feat.shape
    dominant_num = round(math.ceil(seq_len * retain_ratio) * (1 - D))
    contextual_num = math.ceil(seq_len * retain_ratio) - dominant_num
    if dominant_num > 0:
        all_indices = attn.topk(dominant_num, dim=1).indices
        mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device).scatter_(1, all_indices, False)
        dominant_tokens = feat.masked_select(~mask.unsqueeze(-1)).view(batch_size, dominant_num, embed_dim)
        dominant_pos = pos.masked_select(~mask).view(batch_size, dominant_num)
    else:
        mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device)
        dominant_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
        dominant_pos = torch.empty((batch_size, 0), device=feat.device)
    if contextual_num > 0:
        feat_filtered = feat.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num, embed_dim)
        contextual_pos = pos.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num)
        target_indices, dist_matrix = ht_cluster_dpc_knn(feat_filtered, contextual_num, k=min(K, contextual_num), gen=gen)
        target_indices = torch.sort(target_indices, dim=-1)[0]
        contextual_pos = torch.stack([contextual_pos[b][target_indices[b]] for b in range(batch_size)])
        contextual_tokens = ht_merge_tokens_by_clustering(feat_filtered, target_indices, dist_matrix, contextual_num, Beta)
    else:
        contextual_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
        contextual_pos = torch.empty((batch_size, 0), device=feat.device)
    image_feat, image_pos = [], []
    for b in range(batch_size):
        image_feat.append(torch.cat([dominant_tokens[b], contextual_tokens[b]], dim=0))
        image_pos.append(torch.cat([dominant_pos[b], contextual_pos[b]], dim=0))
    return torch.stack(image_feat), torch.stack(image_pos)

def ht_merge_tokens_by_density(feat, pos, retain_ratio, Beta, K, gen=None):
    batch_size, seq_len, embed_dim = feat.shape
    cluster_num = round(seq_len * retain_ratio)
    if cluster_num > 0:
        target_indices, dist_matrix = ht_cluster_dpc_knn(feat, cluster_num, k=min(K, cluster_num), gen=gen)
        target_indices = torch.sort(target_indices, dim=-1)[0]
        image_pos = torch.stack([pos[b][target_indices[b]] for b in range(batch_size)])
        image_feat = ht_merge_tokens_by_clustering(feat, target_indices, dist_matrix, cluster_num, Beta)
    else:
        image_feat = torch.empty((batch_size, 0, embed_dim), device=feat.device)
        image_pos = torch.empty((batch_size, 0), device=feat.device)
    return image_feat, image_pos

def ht_holitom(static_feat, dynamic_feat, dynamic_attn, static_pos, dynamic_pos, window_size, retain_ratio, D, Beta,
               K, images_dtype, gen=None):
    # mm_newline_position == "one_token" branch (LLaVA-OneVision)
    if window_size == 1:
        dynamic_feat, dynamic_pos = ht_merge_tokens_by_attention_density(dynamic_feat, dynamic_attn, dynamic_pos,
                                                                         retain_ratio, D, Beta, K, gen)
        return dynamic_feat.flatten(0, 1).to(images_dtype)
    dynamic_feat, dynamic_pos = ht_merge_tokens_by_attention_density(dynamic_feat, dynamic_attn, dynamic_pos,
                                                                     retain_ratio, D, Beta, K, gen)
    static_feat, static_pos = ht_merge_tokens_by_density(static_feat.unsqueeze(0), static_pos, retain_ratio, Beta, K, gen)
    return torch.cat([static_feat.flatten(0, 1), dynamic_feat.flatten(0, 1)]).to(images_dtype)

def ht_get_static_dynamic_features(image_feat, attn_weights, selected_frames, feature_sim, tau):
    batch_size, seq_len, embed_dim = image_feat.shape
    static_feat_list, dynamic_feat_list, dynamic_attn_list, static_pos_list, dynamic_pos_list = [], [], [], [], []
    for start, end in selected_frames:
        all_indices = torch.arange(seq_len, device=image_feat.device).unsqueeze(0)
        if start == end:
            static_feat_list.append(torch.empty((0, embed_dim), device=image_feat.device))
            dynamic_feat_list.append(image_feat[start:end + 1])
            dynamic_attn_list.append(attn_weights[start:end + 1])
            static_pos_list.append(torch.empty((0, seq_len), device=image_feat.device))
            dynamic_pos_list.append(all_indices)
        else:
            windows_size = end - start + 1
            mask = torch.all(feature_sim[start:end, :] > tau, dim=0)
            static_feat_list.append(image_feat[start:end + 1, mask].mean(dim=0))
            dynamic_feat_list.append(image_feat[start:end + 1, ~mask])
            dynamic_attn_list.append(attn_weights[start:end + 1, ~mask])
            static_pos_list.append(all_indices[:, mask].expand(1, -1))
            dynamic_pos_list.append(all_indices[:, ~mask].expand(windows_size, -1))
    return static_feat_list, dynamic_feat_list, None, dynamic_attn_list, static_pos_list, dynamic_pos_list


# ── VisionZip (Yang et al., CVPR 2025) — LLaVA-OV video port from the HoliTom repo (visionzip/llava_arch.py) ──
# Per frame, after LLaVA-OV's 2x2 pooling: dominant tokens by received vision attention, contextual tokens by merging
# the rest onto evenly spaced targets using key similarity. Default dominant ratio 54/64 (as in that code).
# Output order per frame = [dominant (spatial order), contextual]; frames concatenated; contiguous positions.
@torch.no_grad()
def visionzip_compress(X, vz_attn, vz_metric, r, dominant_ratio=54 / 64):
    T, S, D = X.shape
    spatial = int(S * r)
    dom = int(spatial * dominant_ratio)
    ctx = spatial - dom
    hs = X.float()
    att = vz_attn.to(hs.device).float()
    met = vz_metric.to(hs.device).float()
    topk = att.topk(dom, dim=1).indices
    mask = torch.ones(T, S, dtype=torch.bool, device=hs.device).scatter_(1, topk, False)   # True = not dominant
    dominant = hs.masked_select(~mask.unsqueeze(-1)).view(T, dom, D)
    met_f = met[mask].view(T, S - dom, met.shape[2])
    hs_f = hs.masked_select(mask.unsqueeze(-1)).view(T, S - dom, D)
    met_n = met_f / met_f.norm(dim=-1, keepdim=True)
    step = max(1, (S - dom) // ctx)
    tgt = torch.arange(0, S - dom, step, device=hs.device)[:ctx]
    tgt_tok = met_n[:, tgt, :]
    rest = ~torch.isin(torch.arange(met_n.shape[1], device=hs.device), tgt)
    to_merge = met_n[:, rest, :]
    sim = torch.bmm(to_merge, tgt_tok.transpose(1, 2))
    onehot = torch.zeros(T, to_merge.shape[1], ctx, device=hs.device)
    onehot.scatter_(2, sim.argmax(dim=2).unsqueeze(-1), 1)
    counts = onehot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    agg = torch.bmm(onehot.transpose(1, 2), hs_f[:, rest, :]) / counts
    contextual = hs_f[:, tgt, :] + agg
    return torch.cat([dominant, contextual], dim=1).reshape(-1, D).to(X.dtype)

