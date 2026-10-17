"""ETTm2: fixed feature set, OT-only multi-horizon forecasting."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

TIME_COL = 'date'
TARGET_COL = 'OT'
# H4 by default. Manually edit this list after reviewing the training-only heatmap.
FEATURE_COLS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
SEQ_LEN = 672
HORIZON = 96
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path):
    if TARGET_COL not in FEATURE_COLS:
        raise ValueError('TARGET_COL must appear in FEATURE_COLS for this implementation')
    df = pd.read_csv(csv_path, parse_dates=[TIME_COL]).sort_values(TIME_COL)
    if df[TIME_COL].duplicated().any():
        raise ValueError('Duplicate timestamps')
    df = df.set_index(TIME_COL).asfreq('15min')
    selected = df[FEATURE_COLS].apply(pd.to_numeric, errors='raise')
    if selected.isna().any().any():
        raise ValueError('Missing timestamps/values: resolve them before splitting (no future-aware interpolation)')
    n = len(selected)
    train_end, val_end = int(n * .70), int(n * .85)
    if train_end <= SEQ_LEN + HORIZON or val_end - train_end < HORIZON or n - val_end < HORIZON:
        raise ValueError('Insufficient rows for the chosen sequence and horizon')
    scaler = StandardScaler().fit(selected.iloc[:train_end])
    z = scaler.transform(selected).astype(np.float32)
    target_idx = FEATURE_COLS.index(TARGET_COL)
    return selected, z, scaler, target_idx, train_end, val_end


def save_correlation_heatmap(df, train_end, out_dir):
    # EDA only; no automated feature selection or test-set inspection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    corr = df.iloc[:train_end].corr(method='pearson')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_dir / 'train_correlation.csv')
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(corr.iloc[i, j]) > .65 else 'black')
    fig.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title('Training-only feature correlation (EDA)')
    fig.tight_layout()
    fig.savefig(out_dir / 'train_correlation_heatmap.png', dpi=160)
    plt.close(fig)


class WindowDataset(Dataset):
    def __init__(self, z, first_target, last_target, seq_len, horizon, target_idx):
        self.z = z
        self.starts = np.arange(max(first_target, seq_len), last_target - horizon + 1, dtype=np.int64)
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_idx = target_idx

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        t = int(self.starts[index])
        x = self.z[t-self.seq_len:t]
        y = self.z[t:t+self.horizon, self.target_idx:self.target_idx+1]
        return torch.from_numpy(x), torch.from_numpy(y)


class RecurrentModel(nn.Module):
    def __init__(self, kind, n_features, horizon, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[kind]
        self.encoder = cls(n_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        _, h = self.encoder(x)
        if isinstance(h, tuple):  # LSTM (h_n, c_n)
            h = h[0]
        return self.head(h[-1]).unsqueeze(-1)
