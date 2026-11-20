import time, copy, random
import numpy as np
import torch, torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class RNNNet(nn.Module):
    """LSTM, GRU or bidirectional LSTM: reads the window, predicts the next value."""
    def __init__(self, n_feat, hidden, kind):
        super().__init__()
        bi = kind == "bilstm"
        cell = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cell(n_feat, hidden, batch_first=True, bidirectional=bi)
        self.head = nn.Linear(hidden * (2 if bi else 1), 1)

    def forward(self, x):                       # x: (batch, window, features)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class ConvNet(nn.Module):
    """1D convolutional network over the window."""
    def __init__(self, n_feat, window, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(hidden * window, 1)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))         # (batch, hidden, window)
        return self.head(h.flatten(1)).squeeze(-1)


def make_model(kind, n_feat, window, hidden=64):
    if kind in ("lstm", "gru", "bilstm"): return RNNNet(n_feat, hidden, kind)
    if kind == "conv1d": return ConvNet(n_feat, window, hidden)
    raise ValueError(kind)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict(model, X, batch=4096):
    model.eval(); out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_model(model, Xtr, ytr, Xva, yva, lr=1e-3, max_epochs=30, patience=4, batch=512):
    """Same optimiser, loss and early stopping for every model."""
    model.to(DEVICE)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, bad, epochs = float("inf"), None, 0, 0
    t0 = time.time()
    n = len(Xt)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = float(np.mean((predict(model, Xva) - yva) ** 2))   # validation MSE (scaled units)
        epochs = ep + 1
        if val < best - 1e-6:
            best, bad, best_state = val, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    if DEVICE == "cuda": torch.cuda.synchronize()
    return dict(best_val=best, epochs=epochs, train_s=time.time() - t0)


def save_model(model, path, meta):
    torch.save(dict(state=model.state_dict(), meta=meta), path)


def load_model(path):
    ck = torch.load(path, map_location=DEVICE)
    m = ck["meta"]
    model = make_model(m["kind"], m["n_feat"], m["window"], m["hidden"])
    model.load_state_dict(ck["state"]); model.to(DEVICE).eval()
    return model, m
