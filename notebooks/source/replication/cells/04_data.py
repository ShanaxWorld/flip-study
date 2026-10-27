# ── Data: benchmark tables, pre-registered sampling, video download, frame reading ──
import io, tarfile, hashlib, re, concurrent.futures as cf
from huggingface_hub import list_repo_files

LETTERS = "ABCDEFGH"
_PREFIX = re.compile(r"^\s*\(?[A-Ha-h][\.\):]\s*")
_norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().rstrip(".").strip().lower()

def _opts(cands):
    """Plain candidate texts -> 'A. text' strings (any existing 'A.' / '(A)' prefix removed first)."""
    return [f"{LETTERS[i]}. {_PREFIX.sub('', str(c)).strip()}" for i, c in enumerate(cands)]

def _answer_letter(cands, answer):
    a = _norm(_PREFIX.sub("", str(answer)))
    hits = [i for i, c in enumerate(cands) if _norm(_PREFIX.sub("", str(c))) == a]
    return LETTERS[hits[0]] if len(hits) == 1 else None

def _hf(repo, fn, **kw):
    return hf_hub_download(repo, fn, repo_type="dataset", local_dir=os.path.join(WORK_DIR, repo.replace("/", "__")), **kw)

def _clean_cache():
    """Big archives leave a copy in the Hugging Face xet cache; remove it to keep disk free."""
    shutil.rmtree(os.path.join(os.environ.get("HF_HOME", ""), "xet"), ignore_errors=True)

def _sample(df, by, k, seed=SAMPLE_SEED):
    out = []
    for g, sub in df.groupby(by, sort=True):
        out.append(sub.sample(n=min(k, len(sub)), random_state=seed) if k else sub)
    return pd.concat(out).sort_values("qid", kind="stable").reset_index(drop=True)

# ═════ Video-MME (questions per chunk; videos come chunk by chunk) ═════
VMME_REPO = "lmms-eval/Video-MME"

def videomme_table():
    files = list_repo_files(VMME_REPO, repo_type="dataset")
    pq = [f for f in files if f.endswith(".parquet")]
    df = pd.concat([pd.read_parquet(_hf(VMME_REPO, f)) for f in pq], ignore_index=True)
    low = {c.lower(): c for c in df.columns}
    col = lambda *ns: next((low[n] for n in ns if n in low), None)
    out = pd.DataFrame({
        "video_key": df[col("videoid", "video_id")].astype(str),
        "video_alt": df[col("video_id", "videoid")].astype(str),
        "qid": df[col("question_id")].astype(str),
        "group": df[col("duration")].astype(str),
        "task": df[col("task_type")].astype(str),
        "question": df[col("question")].astype(str),
        "options": df[col("options")].apply(lambda o: [str(x) for x in list(o)]),
        "answer": df[col("answer")].astype(str).str.strip().str[0].str.upper(),
    })
    return out

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")

def videomme_fetch_chunk(chunk):
    fn = f"videos_chunked_{chunk}.zip"
    t0 = time.time()
    zpath = _hf(VMME_REPO, fn)
    vdir = os.path.join(WORK_DIR, f"vmme_{chunk}")
    os.makedirs(vdir, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            if m.filename.lower().endswith(VIDEO_EXT):
                with z.open(m) as src, open(os.path.join(vdir, os.path.basename(m.filename)), "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
    os.remove(zpath); _clean_cache()
    vids = {os.path.splitext(f)[0]: os.path.join(vdir, f) for f in os.listdir(vdir)}
    print(f"Video-MME chunk {chunk}: {len(vids)} videos in {time.time() - t0:.0f}s")
    return vids

# ═════ MVBench (video branch: pre-cut segments, one file per video) ═════
MVB_REPO, MVB_REV = "OpenGVLab/MVBench", "video"
MVB_FOLDERS = {   # lmms-eval DATA_LIST
    "object_interaction": "star/Charades_segment", "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment", "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation", "character_order": "perception/videos",
    "object_shuffle": "perception/videos", "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation", "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos", "scene_transition": "scene_qa/video",
    "state_change": "perception/videos", "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4", "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation", "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}   # fine_grained_pose (NTU RGB+D) is excluded by the pre-registration

def _read_json_any(path):
    txt = open(path, encoding="utf-8").read().strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]

MVB_FILES = None
def mvbench_table():
    global MVB_FILES
    MVB_FILES = list_repo_files(MVB_REPO, repo_type="dataset", revision=MVB_REV)
    rows, dropped = [], 0
    for task, folder in MVB_FOLDERS.items():
        jf = next((f for f in MVB_FILES if f.endswith(f"json/{task}.json")), f"json/{task}.json")
        for i, d in enumerate(_read_json_any(_hf(MVB_REPO, jf, revision=MVB_REV))):
            ans = _answer_letter(d["candidates"], d["answer"])
            if ans is None or len(d["candidates"]) > len(LETTERS):
                dropped += 1
                continue
            rows.append({"qid": f"{task}-{i:04d}", "video_key": f"{folder}/{d['video']}", "video_alt": "",
                         "group": task, "task": task, "question": str(d["question"]),
                         "options": _opts(d["candidates"]), "answer": ans})
    print(f"MVBench: {len(rows)} questions in {len(MVB_FOLDERS)} tasks ({dropped} dropped: answer not among candidates)")
    return pd.DataFrame(rows)

def mvbench_fetch(keys):
    all_files, by_base = set(MVB_FILES), {}
    for f in MVB_FILES:
        by_base.setdefault(os.path.basename(f), []).append(f)
    def one(k):
        cand = [k] if k in all_files else [f for f in by_base.get(os.path.basename(k), [])
                                         if f.startswith(k.split("/")[0])] or by_base.get(os.path.basename(k), [])
        if not cand:
            raise FileNotFoundError(k)
        return k, _hf(MVB_REPO, cand[0], revision=MVB_REV)
    return _parallel_fetch(keys, one)

# ═════ EgoSchema (public 500-question subset) ═════
EGO_REPO = "lmms-lab/egoschema"
def egoschema_table():
    files = list_repo_files(EGO_REPO, repo_type="dataset")
    pq = [f for f in files if f.startswith("Subset/") and f.endswith(".parquet")]
    df = pd.concat([pd.read_parquet(_hf(EGO_REPO, f)) for f in pq], ignore_index=True)
    rows = []
    for _, d in df.iterrows():
        opts = [str(o) for o in list(d["option"])]
        rows.append({"qid": str(d["question_idx"]), "video_key": str(d["video_idx"]), "video_alt": "",
                     "group": "ego", "task": "ego", "question": str(d["question"]), "options": _opts(opts),
                     "answer": LETTERS[int(str(d["answer"]).strip())]})
    print(f"EgoSchema subset: {len(rows)} questions")
    return pd.DataFrame(rows)

def egoschema_fetch(keys):
    need, got = set(keys), {}
    files = sorted(f for f in list_repo_files(EGO_REPO, repo_type="dataset") if f.endswith(".zip"))
    vdir = os.path.join(WORK_DIR, "ego_videos"); os.makedirs(vdir, exist_ok=True)
    for fn in files:
        if not need - set(got):
            break
        t0 = time.time()
        zpath = _hf(EGO_REPO, fn)
        with zipfile.ZipFile(zpath) as z:
            for m in z.infolist():
                stem = os.path.splitext(os.path.basename(m.filename))[0]
                if stem in need and stem not in got and m.filename.lower().endswith(VIDEO_EXT):
                    dst = os.path.join(vdir, os.path.basename(m.filename))
                    with z.open(m) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 24)
                    got[stem] = dst
        os.remove(zpath); _clean_cache()
        print(f"  {fn}: {len(got)}/{len(need)} videos found ({time.time() - t0:.0f}s, "
              f"free disk {shutil.disk_usage(WORK_DIR).free / 1e9:.0f} GB)")
    return got

# ═════ LongVideoBench (validation, video only) ═════
LVB_REPO = "longvideobench/LongVideoBench"
def longvideobench_table():
    files = list_repo_files(LVB_REPO, repo_type="dataset")
    if "lvb_val.json" in files:
        data = _read_json_any(_hf(LVB_REPO, "lvb_val.json"))
    else:
        data = pd.read_parquet(_hf(LVB_REPO, next(f for f in files if f.startswith("validation")))).to_dict("records")
    rows = []
    for d in data:
        cands = d.get("candidates")
        if cands is None:
            cands = [d[f"option{i}"] for i in range(5) if d.get(f"option{i}") not in (None, "N/A")]
        cands = [str(c) for c in list(cands)]
        rows.append({"qid": str(d["id"]), "video_key": os.path.basename(str(d["video_path"])), "video_alt": "",
                     "group": str(d["duration_group"]), "task": str(d.get("question_category", "")),
                     "question": str(d["question"]), "options": _opts(cands),
                     "answer": LETTERS[int(d["correct_choice"])]})
    print(f"LongVideoBench val: {len(rows)} questions")
    return pd.DataFrame(rows)

class _PartsReader(io.RawIOBase):
    """Reads videos.tar.part.aa, .ab, ... as one stream; downloads each part when needed, deletes it after."""
    def __init__(self, parts):
        self.parts, self.k, self.f, self.path = list(parts), -1, None, None
    def readable(self):
        return True
    def _next(self):
        if self.f:
            self.f.close(); os.remove(self.path); _clean_cache()
        self.k += 1
        if self.k >= len(self.parts):
            self.f = None
            return False
        t0 = time.time()
        self.path = _hf(LVB_REPO, self.parts[self.k])
        self.f = open(self.path, "rb")
        print(f"  {self.parts[self.k]} ready ({time.time() - t0:.0f}s)")
        return True
    def readinto(self, b):
        while True:
            if self.f is None and not self._next():
                return 0
            n = self.f.readinto(b)
            if n:
                return n
            if not self._next():
                return 0

def longvideobench_fetch(keys):
    need, got = set(keys), {}
    parts = sorted(f for f in list_repo_files(LVB_REPO, repo_type="dataset") if f.startswith("videos.tar.part"))
    vdir = os.path.join(WORK_DIR, "lvb_videos"); os.makedirs(vdir, exist_ok=True)
    reader = io.BufferedReader(_PartsReader(parts), buffer_size=1 << 24)
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for m in tar:
            b = os.path.basename(m.name)
            if m.isfile() and b in need and b not in got:
                with tar.extractfile(m) as src, open(os.path.join(vdir, b), "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
                got[b] = os.path.join(vdir, b)
            if not need - set(got):
                break
    return got

# ═════ MLVU (dev, the 7 multiple-choice tasks; one file per video) ═════
MLVU_REPO = "MLVU/MVLU"
MLVU_TASKS = ["1_plotQA", "2_needle", "3_ego", "4_count", "5_order", "6_anomaly_reco", "7_topic_reasoning"]
MLVU_FILES = None
def mlvu_table():
    global MLVU_FILES
    MLVU_FILES = list_repo_files(MLVU_REPO, repo_type="dataset")
    rows, dropped = [], 0
    for task in MLVU_TASKS:
        jf = next(f for f in MLVU_FILES if f.endswith(f"json/{task}.json"))
        for i, d in enumerate(_read_json_any(_hf(MLVU_REPO, jf))):
            ans = _answer_letter(d["candidates"], d["answer"])
            if ans is None or len(d["candidates"]) > len(LETTERS):
                dropped += 1
                continue
            rows.append({"qid": f"{task}-{i:04d}", "video_key": f"{task}/{d['video']}", "video_alt": "",
                         "group": task, "task": task, "question": str(d["question"]),
                         "options": _opts(d["candidates"]), "answer": ans})
    print(f"MLVU dev MC: {len(rows)} questions ({dropped} dropped)")
    return pd.DataFrame(rows)

def mlvu_fetch(keys):
    by_rel = {"/".join(f.split("/")[-2:]): f for f in MLVU_FILES if f.lower().endswith(VIDEO_EXT)}
    def one(k):
        if k not in by_rel:
            raise FileNotFoundError(k)
        return k, _hf(MLVU_REPO, by_rel[k])
    return _parallel_fetch(keys, one)

def _parallel_fetch(keys, one, workers=8):
    got, t0 = {}, time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, k): k for k in sorted(set(keys))}
        for i, fu in enumerate(cf.as_completed(futs)):
            try:
                k, p = fu.result()
                got[k] = p
            except Exception as e:
                with open(ERR_PATH, "a") as f:
                    f.write(f"{futs[fu]}\tdownload\t{type(e).__name__}: {e}\n")
            if (i + 1) % 100 == 0:
                print(f"  downloaded {i + 1}/{len(futs)} ({time.time() - t0:.0f}s)")
    print(f"Downloaded {len(got)}/{len(futs)} videos in {time.time() - t0:.0f}s")
    return got

# ═════ Manifest: the pre-registered sample, frozen to a file with its hash ═════
def _valid(df):
    """Pre-registration: questions with fewer than two options carry no information and are removed before sampling."""
    bad = df.options.apply(len) < 2
    if bad.any():
        print(f"Removed {int(bad.sum())} question(s) with fewer than two options: {df.qid[bad].tolist()}")
    return df[~bad]

def build_manifest():
    """The replication sample: the frozen question IDs in REP_MANIFEST (a subset of the main study's manifest)."""
    want = REP_MANIFEST[BENCHMARK]
    ids = sorted(want["qids"])
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == want["sha256"], "replication manifest corrupted"
    full = {"videomme": videomme_table, "egoschema": egoschema_table,
            "longvideobench": longvideobench_table}[BENCHMARK]()
    df = _valid(full)
    df = df[df.qid.isin(set(ids))].sort_values("qid", kind="stable").reset_index(drop=True)
    missing = sorted(set(ids) - set(df.qid))
    if missing:
        raise RuntimeError(f"{len(missing)} replication questions not found in the benchmark table, e.g. {missing[:3]}")
    with open(os.path.join(OUT_DIR, f"manifest_{BENCHMARK}.json"), "w") as f:
        json.dump({"benchmark": BENCHMARK, "seed": SAMPLE_SEED, "n": len(ids), "sha256": want["sha256"],
                   "qids": ids}, f)
    print(f"Replication manifest: {len(df)} questions, {df.video_key.nunique()} videos, sha256 {want['sha256'][:16]}…")
    print("Per group:", df.group.value_counts().sort_index().to_dict())
    return df

# ═════ Frames ═════
def _frame_indices(total, n, shift):
    """Uniform sampling as in LMMs-Eval (np.linspace over all frames). shift = fraction of one sampling
    interval to move every sample later (noise passes)."""
    idx = np.linspace(0, total - 1, n)
    if shift:
        idx = idx + shift * (total - 1) / max(n - 1, 1)
    return np.clip(idx, 0, total - 1).round().astype(int)

def _read_cv2(path, n, shift=0.0, meta=None):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or None
    frames, used = [], []
    if total > 0:
        for k in _frame_indices(total, n, shift):
            ok = False
            for off in (0, -1, 1, -5, 5, -20):
                kk = int(np.clip(k + off, 0, total - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, kk)
                ok, fr = cap.read()
                if ok:
                    break
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); used.append(kk)
    cap.release()
    if meta is not None:
        meta.update(fps=fps, total=total, indices=used)
    return frames

def _read_av(path, n, shift=0.0, meta=None):
    import av
    out, used = [], []
    with av.open(path) as c:
        s = c.streams.video[0]
        dur = float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / 1e6
        fps = float(s.average_rate) if s.average_rate else None
        end = max(dur - 0.5, 0)
        ts = np.linspace(0, end, n)
        if shift:
            ts = np.clip(ts + shift * end / max(n - 1, 1), 0, end)
        for t in ts:
            c.seek(int(t / s.time_base), stream=s, any_frame=False, backward=True)
            for fr in c.decode(s):
                out.append(fr.to_ndarray(format="rgb24")); used.append(int(round(t * (fps or 24))))
                break
    if meta is not None:
        meta.update(fps=fps, total=int(round(dur * (fps or 24))), indices=used)
    return out

def read_frames(path, n, shift=0.0, meta=None):
    """n frames sampled uniformly; meta (optional dict) receives fps, total frame count and the frame indices used."""
    frames = []
    try:
        frames = _read_cv2(path, n, shift, meta)
    except Exception as e:
        print("cv2 failed:", e)
    if len(frames) < n:
        try:
            frames = _read_av(path, n, shift, meta)
        except Exception as e:
            print("av failed:", e)
    if len(frames) == 0:
        raise RuntimeError(f"Could not decode {path}")
    if meta is not None:
        idx = list(meta.get("indices", []))[:len(frames)]
        while len(idx) < n:
            idx.append(idx[-1] if idx else 0)
        meta["indices"] = idx[:n]
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]

def read_all_frame_sets(path):
    """The three frame sets every video needs: original, and the two shifted noise passes (with their metadata)."""
    out, metas = {}, {}
    for k, s in [("orig", 0.0)] + list(NOISE_FRAME_SHIFTS.items()):
        metas[k] = {}
        out[k] = read_frames(path, NUM_FRAMES, shift=s, meta=metas[k])
    out["_meta"] = metas
    return out

# ═════ Option-order noise passes ═════
def _new_index(kind, j, n):
    """Which ORIGINAL option is shown at new position j."""
    return (j + 1) % n if kind == "option_cycle" else n - 1 - j

def reorder_options(options, kind):
    body = [_PREFIX.sub("", str(o)).strip() for o in options]
    n = len(body)
    return [f"{LETTERS[j]}. {body[_new_index(kind, j, n)]}" for j in range(n)]

def map_back(res, n, kind):
    """Express a reordered result in the original option letters."""
    lp_new = np.array(res["logp"])
    lp = np.empty(n)
    for j in range(n):
        lp[_new_index(kind, j, n)] = lp_new[j]
    srt = np.sort(lp)[::-1]
    out = dict(res)
    out.update(pred=LETTERS[int(lp.argmax())], logp=lp.round(4).tolist(), margin=float(srt[0] - srt[1]))
    return out


