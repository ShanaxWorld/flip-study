# ── Per-benchmark summary of the revision run (the full analysis is done offline with the main study's results) ──
R = []
with open(RES_PATH) as f:
    for line in f:
        try:
            R.append(json.loads(line))
        except Exception:
            pass
print(f"{len(R)} questions in {RES_PATH}")
summary = {"benchmark": BENCHMARK, "n": len(R), "model": MODEL_ID}
if R:
    ref = REFERENCE_FULL.get(BENCHMARK, {})
    both = [r for r in R if r["qid"] in ref]
    same = [r["full"]["pred"] == ref[r["qid"]][0] for r in both]
    gaps = [float(np.max(np.abs(np.array(r["full"]["logp"]) - np.array(ref[r["qid"]][1:])))) for r in both
            if len(r["full"]["logp"]) == len(ref[r["qid"]]) - 1]
    summary["reference_matched"] = len(both)
    summary["full_same_answer_as_main_study"] = float(np.mean(same)) if same else None
    summary["full_max_logp_gap"] = float(np.max(gaps)) if gaps else None
    summary["full_median_logp_gap"] = float(np.median(gaps)) if gaps else None
    full_acc = np.mean([r["full"]["correct"] for r in R])
    summary["full_acc"] = float(full_acc)
    summary["noise_flip_rate"] = {k: float(np.mean([r["noise"][k]["pred"] != r["full"]["pred"] for r in R if k in r["noise"]]))
                                  for k in list(NOISE_FRAME_SHIFTS) + NOISE_OPTION_ORDERS + NOISE_PROMPTS}
    summary["noise_acc"] = {k: float(np.mean([r["noise"][k]["correct"] for r in R if k in r["noise"]]))
                            for k in list(NOISE_FRAME_SHIFTS) + NOISE_OPTION_ORDERS + NOISE_PROMPTS}
    summary["vcast"] = {}
    for s_ in [str(x) for x in METHOD_GRID["vcast"]]:
        rr = [r for r in R if s_ in r["methods"].get("vcast", {})]
        if rr:
            acc = np.mean([r["methods"]["vcast"][s_]["correct"] for r in rr])
            summary["vcast"][s_] = {"n": len(rr), "flip_rate": float(np.mean([r["methods"]["vcast"][s_]["pred"] != r["full"]["pred"] for r in rr])),
                                    "acc": float(acc), "rel_acc": float(100 * acc / np.mean([r["full"]["correct"] for r in rr])),
                                    "retention": float(np.mean([r["methods"]["vcast"][s_]["retention"] for r in rr]))}
with open(os.path.join(OUT_DIR, f"summary_ext_{BENCHMARK}.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
if not QUICK_TEST and summary.get("full_same_answer_as_main_study") is not None and summary["full_same_answer_as_main_study"] < 0.99:
    print("⚠️ The full model's answers differ from the main study on more than 1% of questions. "
          "Send results.jsonl anyway; the offline analysis will use this run's own full pass where needed.")
print("\n➡️  Send only results.jsonl from", OUT_DIR)
