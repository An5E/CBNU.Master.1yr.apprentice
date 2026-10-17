from pathlib import Path
import copy
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Configuration
# =========================
CSV_PATH = "./assets/ETTm2.csv"
TIME_COL = "date"
TARGET_COL = "OT"
FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", TARGET_COL]  # Add other numeric columns if needed

FREQ = "15min"
SEQ_LEN = 672       # 7 days of history
HORIZON = 96        # Forecast next 24 hours
BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 7
LR = 1e-3
SEED = 42

OUT = Path("results")
OUT.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", DEVICE)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

# =========================
# 2. Load and visualize data
# =========================
df = pd.read_csv(CSV_PATH, parse_dates=[TIME_COL])
df = df.sort_values(TIME_COL).set_index(TIME_COL)

if df.index.has_duplicates:
    raise ValueError("Duplicate timestamps found; resolve them first.")

df = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
df = df.asfreq(FREQ)

# Do not silently interpolate long gaps.
print("Missing values:\n", df.isna().sum())
df = df.interpolate(method="time", limit=4, limit_area="inside")
df = df.dropna()

if len(df) < 4 * (SEQ_LEN + HORIZON):
    raise ValueError("Not enough data for this configuration.")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df.index, df[TARGET_COL], linewidth=0.7)
ax.set_title("Time series overview")
ax.set_xlabel("Timestamp")
ax.set_ylabel(TARGET_COL)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "series_overview.png", dpi=150)
plt.close(fig)

# Split chronologically: 70% / 15% / 15%.
values = df.values.astype(np.float32)
n = len(values)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Fit scaler on training data only.
scaler = StandardScaler()
scaler.fit(values[:train_end])
z = scaler.transform(values).astype(np.float32)

# =========================
# 3. Window dataset
# =========================
class WindowDataset(Dataset):
    def __init__(self, data, first_target, last_target,
                 seq_len, horizon):
        self.data = data
        self.first = first_target
        self.last = last_target
        self.seq_len = seq_len
        self.horizon = horizon

        # Target interval is [first, last), context may
        # legitimately come from immediately preceding history.
        self.starts = list(range(
            max(first_target, seq_len),
            last_target - horizon + 1
        ))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t = self.starts[idx]
        x = self.data[t-self.seq_len:t]
        y = self.data[t:t+self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders(data, seq_len, horizon):
    train_ds = WindowDataset(
        data, seq_len, train_end, seq_len, horizon
    )
    val_ds = WindowDataset(
        data, train_end, val_end, seq_len, horizon
    )
    test_ds = WindowDataset(
        data, val_end, len(data), seq_len, horizon
    )

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise ValueError("Empty dataset; reduce SEQ_LEN/HORIZON.")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, val_loader, test_loader


# =========================
# 4. Models
# All models: [B, L, C] -> [B, H, C]
# =========================
class RecurrentModel(nn.Module):
    def __init__(self, kind, n_features, horizon,
                 hidden=64, layers=2, dropout=0.1):
        super().__init__()
        cls = {"RNN": nn.RNN, "LSTM": nn.LSTM,
               "GRU": nn.GRU}[kind]
        self.rnn = cls(
            n_features, hidden, num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.head(out[:, -1])
        return out.reshape(
            x.size(0), self.horizon, self.n_features
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(
            pos * div[:pe[:, 1::2].shape[1]]
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, n_features, horizon, d_model=64,
                 heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads,
            dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, horizon*n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x):
        h = self.encoder(self.pos(self.input_proj(x)))
        y = self.head(h[:, -1])
        return y.reshape(
            x.size(0), self.horizon, self.n_features
        )


class PatchTSTModel(nn.Module):
    """Compact patch-based Transformer; not official PatchTST."""
    def __init__(self, n_features, seq_len, horizon,
                 patch_len=16, stride=8, d_model=64,
                 heads=4, layers=2, dropout=0.1):
        super().__init__()
        if seq_len < patch_len:
            raise ValueError("seq_len must be >= patch_len")
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.horizon = horizon

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads,
            dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, layers)
        n_patches = (seq_len - patch_len) // stride + 1
        self.head = nn.Linear(
            n_patches*d_model, horizon
        )

    def forward(self, x):
        # [B,L,C] -> [B,C,L] -> [B,C,N,P]
        patches = x.transpose(1, 2).unfold(
            dimension=2, size=self.patch_len,
            step=self.stride
        )
        b, c, n, p = patches.shape
        h = self.patch_proj(patches.reshape(b*c, n, p))
        h = self.encoder(self.pos(h))
        h = h.reshape(b, c, -1)
        y = self.head(h)  # [B,C,H]
        return y.transpose(1, 2)  # [B,H,C]


class DLinearModel(nn.Module):
    def __init__(self, n_features, seq_len, horizon):
        super().__init__()
        self.linear = nn.Linear(seq_len, horizon)
        self.n_features = n_features

    def forward(self, x):
        # Per-channel direct linear forecast.
        y = self.linear(x.transpose(1, 2))
        return y.transpose(1, 2)


class PersistenceModel(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        return x[:, -1:, :].expand(
            -1, self.horizon, -1
        )


# =========================
# 5. Training / evaluation
# =========================
def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.set_grad_enabled(training):
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total += loss.item() * x.size(0)
        count += x.size(0)

    return total / count


def fit_model(model, train_loader, val_loader):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    best_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = PATIENCE
        else:
            patience_left -= 1

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1:02d}: "
                f"train={train_loss:.5f}, val={val_loss:.5f}"
            )
        if patience_left == 0:
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []

    # Warm-up before measuring inference time.
    for i, (x, _) in enumerate(loader):
        if i >= 2:
            break
        model(x.to(DEVICE))

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for x, y in loader:
        p = model(x.to(DEVICE)).cpu().numpy()
        ps.append(p)
        ys.append(y.numpy())

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return np.concatenate(ys), np.concatenate(ps), elapsed


def calculate_metrics(y_true, y_pred):
    # Inputs are inverse-transformed to original units.
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = np.mean(
        200 * np.abs(err) / np.maximum(denom, 1e-8)
    )
    return {"MAE": mae, "RMSE": rmse, "sMAPE_pct": smape}


# =========================
# 6. Benchmark
# =========================
def build_model(name, seq_len, horizon, n_features,
                patch_len=16):
    if name in ("RNN", "LSTM", "GRU"):
        return RecurrentModel(name, n_features, horizon)
    if name == "Transformer":
        return TransformerModel(n_features, horizon)
    if name == "PatchTST":
        return PatchTSTModel(
            n_features, seq_len, horizon,
            patch_len=patch_len
        )
    if name == "DLinear":
        return DLinearModel(n_features, seq_len, horizon)
    if name == "Persistence":
        return PersistenceModel(horizon)
    raise ValueError(f"Unknown model: {name}")


def run_experiment(name, seq_len=SEQ_LEN,
                   patch_len=16, seed=SEED):
    seed_everything(seed)
    train_loader, val_loader, test_loader = make_loaders(
        z, seq_len, HORIZON
    )
    model = build_model(
        name, seq_len, HORIZON, z.shape[1], patch_len
    )

    if name != "Persistence":
        model = fit_model(model, train_loader, val_loader)
    else:
        model = model.to(DEVICE)

    y, p, elapsed = predict(model, test_loader)

    # Inverse transform each feature to original units.
    y = scaler.inverse_transform(y.reshape(-1, z.shape[1])).reshape(y.shape)
    p = scaler.inverse_transform(p.reshape(-1, z.shape[1])).reshape(p.shape)

    metrics = calculate_metrics(
        y[:, :, 0], p[:, :, 0]
    )
    metrics.update({
        "model": name,
        "seq_len": seq_len,
        "patch_len": patch_len if name == "PatchTST" else None,
        "inference_seconds": elapsed,
        "n_test_windows": len(test_loader.dataset),
        "parameters": sum(
            q.numel() for q in model.parameters()
            if q.requires_grad
        ),
        "seed": seed,
    })

    # Plot one representative forecast window.
    k = min(10, len(y) - 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y[k, :, 0], label="Actual")
    ax.plot(p[k, :, 0], label="Forecast")
    ax.set_title(f"{name}: next {HORIZON} steps")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if name == "PatchTST":
        filename = (
            f"forecast_{name}_L{seq_len}_P{patch_len}.png"
        )
    else:
        filename = f"forecast_{name}_L{seq_len}.png"

    fig.savefig(
        OUT / filename,
        dpi=150
    )
    plt.close(fig)

    return metrics


if __name__ == "__main__":
    results = []

    # First run: core benchmark.
    for name in [
        "RNN", "LSTM",
        "GRU", "Transformer", "PatchTST"
    ]:
        print(f"\n===== {name} =====")
        results.append(run_experiment(name))

    # Ablation A: input context length.
    # Run these after the core benchmark.
    for length in [96, 288, 672, 1344]:
        print(f"\n===== GRU context={length} =====")
        results.append(run_experiment("GRU", seq_len=length))
        print(f"\n===== PatchTST context={length} =====")
        results.append(run_experiment("PatchTST", seq_len=length))

    # Ablation B: patch size, holding other settings fixed.
    for patch in [8, 16, 32]:
        print(f"\n===== PatchTST patch={patch} =====")
        results.append(
            run_experiment("PatchTST", patch_len=patch)
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "benchmark_results.csv", index=False)

    print("\n===== Sorted by RMSE =====")
    print(result_df.sort_values("RMSE").to_string(index=False))

    # Performance comparison plot.
    core = result_df[
        result_df["model"].isin([
            "Persistence", "DLinear", "RNN", "LSTM",
            "GRU", "Transformer", "PatchTST"
        ])
        & (result_df["seq_len"] == SEQ_LEN)
    ].drop_duplicates("model")

    ax = core.sort_values("RMSE").plot.bar(
        x="model", y="RMSE", legend=False,
        figsize=(10, 4), rot=30
    )
    ax.set_ylabel("RMSE (original units)")
    ax.set_title("Test-set RMSE comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close()