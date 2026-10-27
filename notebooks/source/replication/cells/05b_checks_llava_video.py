# ── Safety check 1: our pipeline must reproduce the model's own forward on the same video tokens ──
# LLaVA-Video's video path (average pooling, a newline after each frame) is not part of transformers, so the check
# runs the converted model's own forward on the token sequence we build and compares it with our prefill and with our
# own decoder loop (used by FlashVID and HoliTom). The pooling and newline layout follow the LLaVA-NeXT code above.
def equivalence_check(frame_sets, row):
    report = {"transformers": transformers.__version__}
    try:
        enc = encode_frames(frame_sets["orig"])
        spec = full_spec(enc)
        suf = build_suffix_ids(row.question, row.options)
        n_opt = min(len(row.options), len(LETTERS))
        E = body.embed_tokens
        vb, is_nl = video_block(spec)
        x = torch.cat([E(PREFIX_IDS.to(EMB_DEVICE)), vb.to(EMB_DEVICE), E(suf.to(EMB_DEVICE))], 0)[None]
        am = torch.ones(1, x.shape[1], dtype=torch.long, device=EMB_DEVICE)
        with torch.no_grad(), sdpa_kernel(NO_MATH):
            try:
                out = model(inputs_embeds=x, attention_mask=am, logits_to_keep=1)
            except TypeError:
                out = model(inputs_embeds=x, attention_mask=am, num_logits_to_keep=1)
        lp_off = torch.log_softmax(out.logits[0, -1].float()[LETTER_IDS[:n_opt]], -1).cpu().numpy()
        del out
        ours = llm_pass(spec, suf, n_opt)
        man = llm_pass(spec, suf, n_opt, manual=True)
        report["n_video_tokens"] = int(vb.shape[0])
        report["n_video_tokens_expected"] = int(NUM_FRAMES * (POOLED ** 2 + 1))
        report["newlines"] = int(is_nl.sum())
        report["logp_model_forward"] = lp_off.round(4).tolist()
        report["logp_ours"] = ours["logp"]
        report["max_abs_gap"] = float(np.abs(lp_off - np.array(ours["logp"])).max())
        report["logp_ours_manual_loop"] = man["logp"]
        report["max_abs_gap_manual_loop"] = float(np.abs(np.array(man["logp"]) - np.array(ours["logp"])).max())
        report["full_model_answer"] = ours["pred"]
        ok = (report["n_video_tokens"] == report["n_video_tokens_expected"] and report["newlines"] == NUM_FRAMES
              and report["max_abs_gap"] <= 0.15 and report["max_abs_gap_manual_loop"] <= 0.05)
        report["status"] = "PASS" if ok else "FAIL"
    except Exception as e:
        report["status"] = "SKIPPED"
        report["error"] = f"{type(e).__name__}: {e}"
    with open(os.path.join(OUT_DIR, "equivalence_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Equivalence check:", json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Equivalence check failed — stopping so no GPU hours are wasted. "
                           "Send me equivalence_check.json and the log.")
    gc.collect(); torch.cuda.empty_cache()


# ── Safety check 2: prefix caching must give the same answers as running each pass in full ──
def cache_check(enc, comp, rows):
    report = {"checks": []}
    picks = [("full", None)] + [k for k in comp if k in (("random", "0.1"), ("uniform", "0.1"), ("visionzip", "0.1"))][:2]
    for key in picks:
        spec = full_spec(enc) if key[0] == "full" else comp[key]
        P = prefill(spec)
        for row in rows[:2]:
            n_opt = min(len(row.options), len(LETTERS))
            for opts in (row.options, reorder_options(row.options, "option_reverse")):
                suf = build_suffix_ids(row.question, opts)
                a = llm_pass(spec, suf, n_opt)
                b = suffix_pass(P, suf, n_opt)
                report["checks"].append({"pass": "/".join(map(str, key)) if key[0] != "full" else "full",
                                         "qid": row.qid, "same_answer": a["pred"] == b["pred"],
                                         "gap": float(np.abs(np.array(a["logp"]) - np.array(b["logp"])).max()),
                                         "speedup_per_extra_question": round(a["sec"] / max(b["sec_suffix"], 1e-6), 1)})
        del P
    same = np.mean([c["same_answer"] for c in report["checks"]])
    report["max_gap"] = max(c["gap"] for c in report["checks"])
    report["same_answer_share"] = float(same)
    report["status"] = "PASS" if (same == 1.0 and report["max_gap"] <= 0.15) else "FAIL"
    with open(os.path.join(OUT_DIR, "cache_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Cache check:", report["status"], f"| max log-prob gap {report['max_gap']:.4f} | "
          f"same answers {100 * same:.0f}% | an extra question on a cached prefix is "
          f"~{np.median([c['speedup_per_extra_question'] for c in report['checks']]):.0f}x faster than a full pass")
    if report["status"] != "PASS":
        raise RuntimeError("Cache check failed — stopping so no GPU hours are wasted. Send me cache_check.json.")
    gc.collect(); torch.cuda.empty_cache()
