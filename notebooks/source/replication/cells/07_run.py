# ── Main run: per video encode 3 frame sets, compress once, cache each prefix once, run every question on it ──
import zlib
done = set()
if os.path.exists(RES_PATH):
    with open(RES_PATH) as f:
        for line in f:
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                pass
print(f"{len(done)} questions already in results (resume)")

table = build_manifest()
T_START = time.time()
time_left = lambda: MAX_RUNTIME_HOURS * 3600 - (time.time() - T_START)

def video_groups():
    """Yields (video_key, local path, DataFrame of its questions still to run)."""
    if BENCHMARK == "videomme":
        finished = []                    # a chunk folder is deleted only once none of its videos can still be in use
        for ch in VIDEOMME_CONFIRM_CHUNKS:
            if time_left() < 1800:
                print("⏱️ Time budget reached — skipping remaining chunks"); return
            vids = videomme_fetch_chunk(ch)
            sub = table[(table.video_key.isin(vids.keys()) | table.video_alt.isin(vids.keys())) & ~table.qid.isin(done)]
            if QUICK_MAX_Q:
                sub = sub[sub.video_key.isin(sub.video_key.drop_duplicates().head(max(1, QUICK_MAX_Q // 3)))]
            if sub.empty:
                shutil.rmtree(os.path.join(WORK_DIR, f"vmme_{ch}"), ignore_errors=True)
                continue
            while len(finished) > 1:     # the main loop may still be reading the last video of the previous chunk
                shutil.rmtree(os.path.join(WORK_DIR, f"vmme_{finished.pop(0)}"), ignore_errors=True)
            print(f"Chunk {ch}: {len(sub)} questions, {sub.video_key.nunique()} videos")
            for vk, g in sub.groupby("video_key", sort=False):
                yield vk, vids.get(vk) or vids.get(g.video_alt.iloc[0]), g
            finished.append(ch)
            if QUICK_MAX_Q:
                return                                   # quick test: the first chunk with sample videos is enough
    else:
        sub = table[~table.qid.isin(done)]
        if QUICK_MAX_Q:
            per = max(1, QUICK_MAX_Q // max(1, sub.group.nunique()))
            sub = sub.groupby("group", group_keys=False).head(per).head(QUICK_MAX_Q)
        keys = sorted(sub.video_key.unique())
        print(f"{len(sub)} questions on {len(keys)} videos to run; downloading videos…")
        fetch = {"egoschema": egoschema_fetch, "longvideobench": longvideobench_fetch}[BENCHMARK]
        paths = fetch(keys)
        missing = [k for k in keys if k not in paths]
        for k in missing:
            with open(ERR_PATH, "a") as f:
                f.write(f"{k}\tmissing video\t{len(sub[sub.video_key == k])} questions dropped\n")
        if missing:
            print(f"⚠️ {len(missing)} videos could not be found/downloaded (listed in errors.log)")
        for vk, g in sub[sub.video_key.isin(paths.keys())].groupby("video_key", sort=False):
            yield vk, paths[vk], g

def _finish(res, row):
    res["correct"] = bool(res["pred"] == row.answer)
    return res

def run_video(fsets, rows, seed, first):
    enc = encode_frames(fsets["orig"], fsets["_meta"]["orig"])
    n_full = enc["n_full"]
    comp, fails = compress_all(enc, seed)
    if first:
        cache_check(enc, comp, rows)
    recs = {}
    for row in rows:
        recs[row.qid] = {"qid": row.qid, "video": row.video_key, "group": row.group, "task": row.task,
                         "gt": row.answer, "n_opt": min(len(row.options), len(LETTERS)), "n_full": n_full,
                         "noise": {}, "methods": {}}
    sufs = {row.qid: build_suffix_ids(row.question, row.options) for row in rows}
    # full model + option-order noise share one prefix
    P = prefill(full_spec(enc))
    for row in rows:
        n = recs[row.qid]["n_opt"]
        recs[row.qid]["full"] = _finish(suffix_pass(P, sufs[row.qid], n), row)
        for kind in NOISE_OPTION_ORDERS:
            res = suffix_pass(P, build_suffix_ids(row.question, reorder_options(row.options, kind)), n)
            recs[row.qid]["noise"][kind] = _finish(map_back(res, n, kind), row)
    del P
    # frame-shift noise
    for kind in NOISE_FRAME_SHIFTS:
        e = encode_frames(fsets[kind], fsets["_meta"][kind])
        P = prefill(full_spec(e))
        for row in rows:
            recs[row.qid]["noise"][kind] = _finish(suffix_pass(P, sufs[row.qid], recs[row.qid]["n_opt"]), row)
        del P, e
    for (m, s), c in comp.items():
        if c["inner"] is None:              # question-independent: cache the prefix
            P = prefill(c)
            outs = {row.qid: suffix_pass(P, sufs[row.qid], recs[row.qid]["n_opt"]) for row in rows}
            del P
        else:                               # FlashVID / HoliTom act inside the LLM and depend on the question
            outs = {row.qid: llm_pass(c, sufs[row.qid], recs[row.qid]["n_opt"]) for row in rows}
        for row in rows:
            res = _finish(outs[row.qid], row)
            res["retention"] = round(c["nvis"] / n_full, 4)
            res["retention_layer_avg"] = round(res["vis_mean_layers"] / n_full, 4)
            recs[row.qid]["methods"].setdefault(m, {})[s] = res
    return [recs[row.qid] for row in rows], fails

n_new, n_err, consec_fail, FIRST = 0, 0, 0, True
def _fail_guard():
    if consec_fail >= 6:
        raise RuntimeError("6 questions failed in a row — stopping instead of wasting GPU hours. Send me the log.")

reader = cf.ThreadPoolExecutor(1)
groups = video_groups()
nxt = next(groups, None)
fut = reader.submit(read_all_frame_sets, nxt[1]) if nxt else None
while nxt is not None:
    vk, path, g = nxt
    cur_fut = fut
    nxt = next(groups, None) if time_left() > 1200 else None
    if nxt is None and time_left() <= 1200:
        print("⏱️ Time budget reached — stopping after this video")
    fut = reader.submit(read_all_frame_sets, nxt[1]) if nxt else None       # decode the next video meanwhile
    t0 = time.time()
    rows = list(g.itertuples())
    try:
        fsets = cur_fut.result()
        if FIRST:
            equivalence_check(fsets, rows[0])
        recs, fails = run_video(fsets, rows, zlib.crc32(str(vk).encode()), FIRST)
        FIRST = False
    except RuntimeError as e:
        if "check failed" in str(e) or "Equivalence check" in str(e) or "Cache check" in str(e):
            raise
        n_err += len(rows); consec_fail += len(rows)
        with open(ERR_PATH, "a") as f: f.write(f"{vk}\tvideo\t{type(e).__name__}: {e}\n")
        print(f"  ✗ video {vk}: {type(e).__name__}: {e}"); _fail_guard(); continue
    except Exception as e:
        n_err += len(rows); consec_fail += len(rows)
        with open(ERR_PATH, "a") as f: f.write(f"{vk}\tvideo\t{type(e).__name__}: {e}\n")
        print(f"  ✗ video {vk}: {type(e).__name__}: {e}"); torch.cuda.empty_cache(); _fail_guard(); continue
    for m, s_, err in fails:
        with open(ERR_PATH, "a") as f: f.write(f"{vk}\tcompress {m}@{s_}\t{err}\n")
        print(f"  ! {m}@{s_} failed on video {vk} (skipped for this video): {err}")
    with open(RES_PATH, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
            done.add(r["qid"])
    n_new += len(recs); consec_fail = 0
    for r in recs:
        flips = sum(x["pred"] != r["full"]["pred"] for v in r["methods"].values() for x in v.values())
        nz = "".join("·" if x["pred"] == r["full"]["pred"] else "x" for x in r["noise"].values())
        print(f"  ✓ {r['qid']} [{r['group']}] gt={r['gt']} full={r['full']['pred']} noise[{nz}] "
              f"| {flips}/{sum(len(v) for v in r['methods'].values())} compressed changed")
    print(f"    video done in {time.time() - t0:.0f}s | {len(done)} questions total | {time_left() / 3600:.1f} h left")
    del fsets
    gc.collect(); torch.cuda.empty_cache()
reader.shutdown()
for _d in glob.glob(os.path.join(WORK_DIR, "vmme_*")):
    shutil.rmtree(_d, ignore_errors=True)
print(f"\nFinished: {n_new} new questions, {n_err} errors, {len(done)} total in {RES_PATH}")
