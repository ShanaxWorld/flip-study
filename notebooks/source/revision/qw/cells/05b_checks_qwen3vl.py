# ── Safety check 1: our pipeline must reproduce the official model forward (processor + Qwen3-VL forward) ──
def equivalence_check(frame_sets, row):
    report = {"transformers": transformers.__version__}
    try:
        frames, meta = frame_sets["orig"], frame_sets["_meta"]["orig"]
        n_opt = min(len(row.options), len(LETTERS))
        inp = prepare_video(frames, meta, prompt_tail=suffix_text(row.question, row.options))
        dev_in = {k: (v.to(VDEV, torch.float16) if torch.is_floating_point(v) else v.to(EMB_DEVICE)) for k, v in inp.items()}
        with torch.no_grad(), sdpa_kernel(NO_MATH):
            try:
                out = model(**dev_in, logits_to_keep=1)
            except TypeError:
                out = model(**dev_in, num_logits_to_keep=1)
        lp_off = torch.log_softmax(out.logits[0, -1].float()[LETTER_IDS[:n_opt]], -1).cpu().numpy()
        del out, dev_in
        enc = encode_frames(frames, meta)
        spec = full_spec(enc)
        suf = build_suffix_ids(row.question, row.options)
        report["prompt_tokens_identical"] = bool(torch.equal(inp["input_ids"][0], torch.cat([enc["ids"], suf])))
        report["n_video_tokens"] = int(enc["n_full"])
        report["temporal_groups"] = int(enc["G"]); report["tokens_per_group"] = int(enc["P"])
        ours = llm_pass(spec, suf, n_opt)
        man = llm_pass(spec, suf, n_opt, manual=True)
        report["logp_official"] = lp_off.round(4).tolist()
        report["logp_ours"] = ours["logp"]
        report["max_abs_gap"] = float(np.abs(lp_off - np.array(ours["logp"])).max())
        report["logp_ours_manual_loop"] = man["logp"]
        report["max_abs_gap_manual_loop"] = float(np.abs(np.array(man["logp"]) - np.array(ours["logp"])).max())
        report["full_model_answer"] = ours["pred"]
        ok = report["prompt_tokens_identical"] and report["max_abs_gap"] <= 0.15 and report["max_abs_gap_manual_loop"] <= 0.05
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
    picks = [("full", None)] + [k for k in comp if k in (("vcast", "0.1"),)]
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
