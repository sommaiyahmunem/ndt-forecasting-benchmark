import os, glob, json
import numpy as np, pandas as pd

ROOT = "/content/drive/MyDrive/dtn_project"
FEATS = ["DL_bitrate","UL_bitrate","RSRP","RSRQ","SNR","CQI","Speed"]  # target first
CFG = {"4g": "FILL_IN_4G_PATH",
       "5g": f"{ROOT}/5G-production-dataset/5G-production-dataset"}
BOUNDS = {"DL_bitrate":(0,None),"UL_bitrate":(0,None),"CQI":(0,15),
          "RSRP":(-140,-40),"RSRQ":(-40,0)}
MAX_GAP, MIN_LEN = 5, 200
FALLBACK = []   # files whose timestamps could not be parsed

def list_csvs(root):
    out = []
    for f in glob.glob(f"{root}/**/*.csv", recursive=True):
        if "__MACOSX" in f or os.path.basename(f).startswith("._"): continue
        out.append(f)
    return sorted(out)

def read_trace(path):
    df = pd.read_csv(path, na_values=["-", " ", ""])
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    if ts.isna().mean() > 0.5:
        FALLBACK.append(path)
        ts = pd.Timestamp("2000-01-01") + pd.to_timedelta(range(len(df)), unit="s")
    df.index = ts
    df = df[~df.index.duplicated()].sort_index()
    df = df[[c for c in FEATS if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    for c, (lo, hi) in BOUNDS.items():
        if c in df: df[c] = df[c].clip(lo, hi)
    df = df.resample("1s").mean()
    df = df.interpolate(limit=MAX_GAP, limit_area="inside")
    ok = df.notna().all(axis=1)
    grp = (ok != ok.shift()).cumsum()
    best = max((g for _, g in df[ok].groupby(grp[ok])), key=len, default=None)
    return None if best is None or len(best) < MIN_LEN else best

def split(a, tr=0.7, va=0.1):
    n = len(a); i, j = int(n*tr), int(n*(tr+va))
    return a[:i], a[i:j], a[j:]

def process(name):
    out = f"{ROOT}/processed/{name}"; os.makedirs(out, exist_ok=True)
    FALLBACK.clear()
    traces, skipped = {}, 0
    for p in list_csvs(CFG[name]):
        d = read_trace(p)
        if d is None: skipped += 1; continue
        traces[os.path.relpath(p, CFG[name])[:-4].replace(os.sep, "__")] = d
    if not traces:
        raise RuntimeError(f"No usable traces for {name}. Check the path and column names.")
    cols = [c for c in FEATS if all(c in d for d in traces.values())]
    tr_all = np.vstack([split(d[cols].values)[0] for d in traces.values()])
    mu, sd = tr_all.mean(0), tr_all.std(0) + 1e-8      # scaler fitted on TRAIN only
    rows = []
    for k, d in traces.items():
        a, b, c = [(x - mu) / sd for x in split(d[cols].values)]
        np.savez(f"{out}/{k}.npz", train=a, val=b, test=c)
        rows.append(dict(trace=k, steps=len(d), train=len(a), val=len(b), test=len(c)))
    json.dump(dict(cols=cols, mean=mu.tolist(), std=sd.tolist()), open(f"{out}/scaler.json","w"))
    pd.DataFrame(rows).to_csv(f"{out}/summary.csv", index=False)
    print(name, "| kept", len(traces), "| skipped", skipped, "| features", cols)
    print("timestamp fallback used on", len(FALLBACK), "files")

def make_windows(arr, window=12, target_idx=0, context=None):
    if context is not None: arr = np.vstack([context[-window:], arr])
    w = np.lib.stride_tricks.sliding_window_view(arr, (window, arr.shape[1]))[:-1, 0]
    return w, arr[window:, target_idx]
