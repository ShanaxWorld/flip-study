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
        for ch in VIDEOMME_CONFIRM_CHUNKS:
            if time_left() < 1800:
                print("⏱️ Time budget reached — skipping remaining chunks"); return
            vids = videomme_fetch_chunk(ch)
            sub = table[(table.video_key.isin(vids.keys()) | table.video_alt.isin(vids.keys())) & ~table.qid.isin(done)]
            if QUICK_MAX_Q:
                sub = sub[sub.video_key.isin(sub.video_key.drop_duplicates().head(max(1, QUICK_MAX_Q // 3)))]
            print(f"Chunk {ch}: {len(sub)} questions, {sub.video_key.nunique()} videos")
            for vk, g in sub.groupby("video_key", sort=False):
                yield vk, vids.get(vk) or vids.get(g.video_alt.iloc[0]), g
            shutil.rmtree(os.path.join(WORK_DIR, f"vmme_{ch}"), ignore_errors=True)
    else:
        sub = table[~table.qid.isin(done)]
        if QUICK_MAX_Q:
            per = max(1, QUICK_MAX_Q // max(1, sub.group.nunique()))
            sub = sub.groupby("group", group_keys=False).head(per).head(QUICK_MAX_Q)
        keys = sorted(sub.video_key.unique())
        print(f"{len(sub)} questions on {len(keys)} videos to run; downloading videos…")
        fetch = {"mvbench": mvbench_fetch, "egoschema": egoschema_fetch,
                 "longvideobench": longvideobench_fetch, "mlvu": mlvu_fetch}[BENCHMARK]
        if BENCHMARK == "mlvu":
            # MLVU videos are large (~70-80 GB for the sample, more than Kaggle's disk): download 8 at a time
            # and delete each batch once no video in it can still be in use.
            yield from _batched_groups(sub, keys, fetch, batch=8)
            return
        paths = fetch(keys)
        missing = [k for k in keys if k not in paths]
        for k in missing:
            with open(ERR_PATH, "a") as f:
                f.write(f"{k}\tmissing video\t{len(sub[sub.video_key == k])} questions dropped\n")
        if missing:
            print(f"⚠️ {len(missing)} videos could not be found/downloaded (listed in errors.log)")
        for vk, g in sub[sub.video_key.isin(paths.keys())].groupby("video_key", sort=False):
            yield vk, paths[vk], g

def _remove_files(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass
    _clean_cache()

def _batched_groups(sub, keys, fetch, batch):
    prev, prev2 = [], []
    for i in range(0, len(keys), batch):
        if time_left() < 1800:
            print("⏱️ Time budget reached — not downloading more videos"); break
        ks = keys[i:i + batch]
        # When we get here, the main loop is still reading the last video of the previous batch,
        # so only the batch before that one is safe to delete.
        _remove_files(prev2)
        paths = fetch(ks)
        for k in ks:
            if k not in paths:
                with open(ERR_PATH, "a") as f:
                    f.write(f"{k}\tmissing video\t{len(sub[sub.video_key == k])} questions dropped\n")
        print(f"  batch {i // batch + 1}/{-(-len(keys) // batch)}: {len(paths)}/{len(ks)} videos | "
              f"free disk {shutil.disk_usage(WORK_DIR).free / 1e9:.0f} GB")
        for k in ks:
            if k in paths:
                yield k, paths[k], sub[sub.video_key == k]
        prev2, prev = prev, list(paths.values())

def _finish(res, row, T, L):
    res["correct"] = bool(res["pred"] == row.answer)
    return res

def run_video(Xf, A, G, VA, VM, Xshift, rows, seed, first):
    T, L, D = Xf.shape
    pool = cf.ThreadPoolExecutor(1)
    scissor_job = pool.submit(compress_all, Xf, A, G, VA, VM, seed, ("scissor",))       # CPU-heavy: runs alongside
    comp, fails = compress_all(Xf, A, G, VA, VM, seed, skip=("scissor",))
    if first:
        cache_check(Xf, comp, rows)
    recs = {}
    for row in rows:
        recs[row.qid] = {"qid": row.qid, "video": row.video_key, "group": row.group, "task": row.task,
                         "gt": row.answer, "n_opt": min(len(row.options), len(LETTERS)), "n_full": T * L,
                         "noise": {}, "methods": {}}
    sufs = {row.qid: build_suffix_ids(row.question, row.options) for row in rows}
    # full model + option-order noise share one prefix
    P = prefill(Xf.reshape(-1, D), np.arange(T * L), T * L)
    for row in rows:
        n = recs[row.qid]["n_opt"]
        recs[row.qid]["full"] = _finish(suffix_pass(P, sufs[row.qid], n), row, T, L)
        for kind in NOISE_OPTION_ORDERS:
            res = suffix_pass(P, build_suffix_ids(row.question, reorder_options(row.options, kind)), n)
            recs[row.qid]["noise"][kind] = _finish(map_back(res, n, kind), row, T, L)
    del P
    # frame-shift noise
    for kind, Xk in Xshift.items():
        P = prefill(Xk.reshape(-1, D), np.arange(T * L), T * L)
        for row in rows:
            recs[row.qid]["noise"][kind] = _finish(suffix_pass(P, sufs[row.qid], recs[row.qid]["n_opt"]), row, T, L)
        del P
    def run_comp(comp):
        for (m, s), c in comp.items():
            if c["inner"] is None:          # question-independent: cache the prefix
                P = prefill(c["tokens"], c["pos"], c["n_full"], newline=c["newline"])
                outs = {row.qid: suffix_pass(P, sufs[row.qid], recs[row.qid]["n_opt"]) for row in rows}
                del P
            else:                           # FlashVID / HoliTom act inside the LLM and depend on the question
                outs = {row.qid: llm_pass(c["tokens"], c["pos"], c["n_full"], sufs[row.qid], recs[row.qid]["n_opt"],
                                          newline=c["newline"], inner=c["inner"]) for row in rows}
            for row in rows:
                res = _finish(outs[row.qid], row, T, L)
                res["retention"] = round(c["tokens"].shape[0] / (T * L), 4)
                res["retention_layer_avg"] = round(res["vis_mean_layers"] / (T * L), 4)
                if "tau" in c:
                    res["tau"] = round(float(c["tau"]), 5)
                recs[row.qid]["methods"].setdefault(m, {})[s] = res
    run_comp(comp)
    comp2, fails2 = scissor_job.result()
    pool.shutdown()
    run_comp(comp2)
    return [recs[row.qid] for row in rows], fails + fails2

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
            equivalence_check(fsets["orig"], rows[0])
        Xf, A, G, VA, VM = encode_video(preprocess(fsets["orig"]))
        Xshift = {k: encode_video(preprocess(fsets[k]))[0] for k in NOISE_FRAME_SHIFTS}
        recs, fails = run_video(Xf, A, G, VA, VM, Xshift, rows, zlib.crc32(str(vk).encode()), FIRST)
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
    del Xf, A, G, VA, VM, Xshift, fsets
    gc.collect(); torch.cuda.empty_cache()
reader.shutdown()
print(f"\nFinished: {n_new} new questions, {n_err} errors, {len(done)} total in {RES_PATH}")
