# ── Safety check: our token pipeline must reproduce the official model forward ──
from transformers import AutoProcessor

def equivalence_check(frames, row):
    report = {"transformers": transformers.__version__}
    try:
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        opts = "\n".join(str(o) for o in row.options)
        text = ("Select the best answer to the following multiple-choice question based on the video. "
                "Respond with only the letter (A, B, C, or D) of the correct option.\n"
                f"{row.question}\n{opts}\nThe best answer is:")
        conv = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}]
        prompt = proc.apply_chat_template(conv, add_generation_prompt=True)
        inputs = proc(text=prompt, videos=[np.stack(frames)], return_tensors="pt")
        inputs = {k: (v.to(EMB_DEVICE, torch.float16) if torch.is_floating_point(v) else v.to(EMB_DEVICE))
                  for k, v in inputs.items()}
        n_opt = min(len(row.options), len(LETTERS))
        with torch.no_grad(), sdpa_kernel(NO_MATH):
            try:
                out = model(**inputs, logits_to_keep=1)
            except TypeError:
                out = model(**inputs, num_logits_to_keep=1)
        lp_off = torch.log_softmax(out.logits[0, -1].float()[LETTER_IDS[:n_opt]], -1).cpu().numpy()
        del out
        ids = inputs["input_ids"][0].cpu()
        vid_tok = getattr(CFG, "video_token_id", None) or getattr(CFG, "video_token_index", None)
        posi = (ids == vid_tok).nonzero().flatten()
        X = encode_video(inputs["pixel_values_videos"][0])[0]
        vis = X.reshape(-1, X.shape[-1])
        report["n_video_tokens_official"] = int(len(posi))
        report["n_video_tokens_ours"] = int(vis.shape[0] + 1)
        ours = llm_pass(vis, np.arange(vis.shape[0]), vis.shape[0], ids[posi[-1] + 1:], n_opt, pre_ids=ids[: posi[0]])
        lp_ours = np.array(ours["logp"])
        report["logp_official"] = lp_off.round(4).tolist()
        report["logp_ours_same_pixels"] = lp_ours.round(4).tolist()
        report["max_abs_gap"] = float(np.abs(lp_off - lp_ours).max())
        # our own decoder loop (used by FlashVID / HoliTom) must give the same answer as the library forward
        man = llm_pass(vis, np.arange(vis.shape[0]), vis.shape[0], ids[posi[-1] + 1:], n_opt, pre_ids=ids[: posi[0]],
                       manual=True)
        report["logp_ours_manual_loop"] = man["logp"]
        report["max_abs_gap_manual_loop"] = float(np.abs(np.array(man["logp"]) - lp_ours).max())
        report["pixel_mean_abs_diff_ours_vs_official"] = float(
            (preprocess(frames) - inputs["pixel_values_videos"][0].float().cpu()).abs().mean())
        ok = (report["n_video_tokens_official"] == report["n_video_tokens_ours"] and report["max_abs_gap"] <= 0.15
              and report["max_abs_gap_manual_loop"] <= 0.05)
        report["status"] = "PASS" if ok else "FAIL"
    except Exception as e:
        report["status"] = "SKIPPED"
        report["error"] = f"{type(e).__name__}: {e}"
    with open(os.path.join(OUT_DIR, "equivalence_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Equivalence check:", json.dumps(report, indent=2))
    if report["status"] == "FAIL" or (report["status"] == "SKIPPED" and not QUICK_TEST):
        raise RuntimeError("Equivalence check failed/skipped — stopping so no GPU hours are wasted. "
                           "Send me equivalence_check.json and the log.")
    gc.collect(); torch.cuda.empty_cache()


# ── Safety check 2: prefix caching must give the same answers as running each pass in full ──
def cache_check(Xf, comp, rows):
    """On the first video: full pass, option-reversed full pass and two compressed passes, cached vs uncached."""
    T, L, D = Xf.shape
    report = {"checks": []}
    picks = [("full", None)] + [k for k in comp if k in (("random", "0.1"), ("fastvid", "0.1"))]
    for key in picks:
        if key[0] == "full":
            vis, pos, nfull, nl = Xf.reshape(-1, D), np.arange(T * L), T * L, True
        else:
            c = comp[key]
            vis, pos, nfull, nl = c["tokens"], c["pos"], c["n_full"], c["newline"]
        P = prefill(vis, pos, nfull, newline=nl)
        for row in rows[:2]:
            n_opt = min(len(row.options), len(LETTERS))
            for opts in (row.options, reorder_options(row.options, "option_reverse")):
                suf = build_suffix_ids(row.question, opts)
                a = llm_pass(vis, pos, nfull, suf, n_opt, newline=nl)
                b = suffix_pass(P, suf, n_opt)
                report["checks"].append({"pass": "/".join(map(str, key)) if key[0] != "full" else "full",
                                         "qid": row.qid, "same_answer": a["pred"] == b["pred"],
                                         "gap": float(np.abs(np.array(a["logp"]) - np.array(b["logp"])).max()),
                                         "speedup_per_extra_question": round(a["sec"] / max(b["sec_suffix"], 1e-6), 1)})
        del P
    same = np.mean([c["same_answer"] for c in report["checks"]])
    report["max_gap"] = max(c["gap"] for c in report["checks"])
    report["same_answer_share"] = float(same)
    # pre-registered stop rule is about answers; the log-prob tolerance matches the equivalence check
    report["status"] = "PASS" if (same == 1.0 and report["max_gap"] <= 0.15) else "FAIL"
    with open(os.path.join(OUT_DIR, "cache_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Cache check:", report["status"], f"| max log-prob gap {report['max_gap']:.4f} | "
          f"same answers {100 * same:.0f}% | an extra question on a cached prefix is "
          f"~{np.median([c['speedup_per_extra_question'] for c in report['checks']]):.0f}x faster than a full pass")
    if report["status"] != "PASS":
        raise RuntimeError("Cache check failed — stopping so no GPU hours are wasted. Send me cache_check.json.")
    gc.collect(); torch.cuda.empty_cache()
