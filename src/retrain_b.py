"""Member B: replay a test trace in time order with a retraining strategy (none / fixed / ADWIN / KSWIN)."""
import copy, time
import numpy as np
import torch, torch.nn as nn
import neural

WINDOW = 12


def make_windows(arr, window=12, target_idx=0, context=None):
    """Same as Member A: context = end of the previous split, so val/test don't lose their first steps."""
    if context is not None: arr = np.vstack([context[-window:], arr])
    w = np.lib.stride_tricks.sliding_window_view(arr, (window, arr.shape[1]))[:-1, 0]
    return w, arr[window:, target_idx]


def make_detector(kind, **kw):
    """The drift monitor. It is fed one number per step (the absolute prediction error) and says 'drift' or not."""
    from river.drift import ADWIN, KSWIN
    if kind == "adwin": return ADWIN(delta=kw.get("delta", 0.002))
    if kind == "kswin": return KSWIN(alpha=kw.get("alpha", 0.005), window_size=kw.get("window_size", 100),
                                     stat_size=kw.get("stat_size", 30), seed=0)
    raise ValueError(kind)


def fine_tune(model, seg, window, epochs, lr, batch):
    """Continue training the current model on recent data (same Adam, MSE and gradient clipping as Member A)."""
    X, y = make_windows(seg, window)
    if len(X) == 0: return
    Xt = torch.as_tensor(X, dtype=torch.float32, device=neural.DEVICE)
    yt = torch.as_tensor(y, dtype=torch.float32, device=neural.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=neural.DEVICE)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); loss_fn(model(Xt[idx]), yt[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()


def replay(model0, X, y, arr_all, n0, strategy, window=WINDOW, det_kw=None, fixed_every=None, recent=600,
           ft_epochs=5, ft_lr=5e-4, ft_batch=128, block=256, max_retrains=50):
    """Replay one test trace in time order.
    X, y        : test windows and targets (scaled), as made by make_windows with the validation tail as context
    arr_all     : train+val+test rows of the chosen columns (scaled); n0 = number of train+val rows
    strategy    : 'none' | 'fixed' | 'adwin' | 'kswin'
    recent      : retrain on the last `recent` steps (None = all data so far)
    Returns predictions (scaled), the test-step numbers where a retrain happened, retraining seconds, capped flag."""
    model = copy.deepcopy(model0); n = len(y)
    preds = np.empty(n); alarms = []; retrain_s = 0.0; capped = False
    det = make_detector(strategy, **(det_kw or {})) if strategy in ("adwin", "kswin") else None
    every = fixed_every if strategy == "fixed" else None
    t = 0
    while t < n:
        e = min(n, t + block)
        p = neural.predict(model, X[t:e]).astype("float64")
        if det is None and not every:                 # nothing can fire: take the whole block
            preds[t:e] = p; t = e; continue
        fire = None
        for i in range(e - t):
            j = t + i
            if det is not None:
                det.update(float(abs(p[i] - y[j])))
                if det.drift_detected: fire = j; break
            elif (j + 1) % every == 0: fire = j; break
        stop = e if fire is None else fire + 1
        preds[t:stop] = p[:stop - t]                  # these steps were predicted by the model as it was
        if fire is not None:
            if len(alarms) >= max_retrains:
                capped = True; det = None; every = None
            else:
                alarms.append(fire)
                g = n0 + fire                          # row of arr_all that holds the target of step `fire`
                lo = 0 if recent is None else max(0, g + 1 - recent - window)
                t0 = time.time()
                fine_tune(model, arr_all[lo:g + 1], window, ft_epochs, ft_lr, ft_batch)
                if neural.DEVICE == "cuda": torch.cuda.synchronize()
                retrain_s += time.time() - t0
                if det is not None: det = make_detector(strategy, **(det_kw or {}))
        t = stop
    return preds, alarms, retrain_s, capped
