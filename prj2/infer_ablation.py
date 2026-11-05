"""Compare multiple trusted ablation .pt checkpoints at one common OT forecast origin.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

FULL_FEATURES = ('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT')
HORIZON = 1344

class RecurrentModel(nn.Module):
    """Matches original infer_ablation_ot.py (encoder + head)."""
    def __init__(self, kind, n_features, horizon, hidden, layers, dropout):
        super().__init__()
        cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[kind]
        self.encoder = cls(n_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        _, h = self.encoder(x)
        if isinstance(h, tuple):
            h = h[0]
        return self.head(h[-1]).unsqueeze(-1)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    required = {'state_dict', 'model', 'features', 'seq_len', 'params'}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f'Invalid ablation checkpoint: {path}')
    if checkpoint['model'] not in ('RNN', 'LSTM', 'GRU'):
        raise ValueError(f'Unsupported model in {path}')
    features = list(checkpoint['features'])
    if not features or len(set(features)) != len(features) or 'OT' not in features or not set(features).issubset(FULL_FEATURES):
        raise ValueError(f'Invalid features in {path}')
    if int(checkpoint['seq_len']) < 1:
        raise ValueError(f'Invalid seq_len in {path}')
    return checkpoint


def metrics(actual, prediction):
    error = prediction - actual
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.sum((actual - actual.mean())**2))
    r2 = float(1 - np.sum(error**2)/denominator) if denominator > 0 else None
    return {'MSE': mse, 'MAE': mae, 'RMSE': mse**.5, 'R2': r2}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pt', nargs='+', default=[], help='One or more trusted .pt files')
    p.add_argument('--pt-glob', action='append', default=[], help='Glob pattern; may be repeated (quote pattern)')
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='results/inference_compare')
    p.add_argument('--split', choices=('train','val','test'), default='test')
    p.add_argument('--origin-index', type=int, default=0)
    p.add_argument('--origin-time', help='Explicit forecast origin: YYYY-MM-DD HH:MM:SS')
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()
    if args.origin_index < 0:
        p.error('--origin-index must be >= 0')

    import glob
    paths = [Path(s) for s in args.pt]
    for pattern in args.pt_glob:
        paths.extend(Path(s) for s in sorted(glob.glob(pattern)))
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not paths:
        p.error('Specify --pt file1.pt [file2.pt ...] or --pt-glob ".../*.pt"')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoints = [(path, load_checkpoint(path)) for path in paths]

    raw = pd.read_csv(args.csv, parse_dates=['date']).sort_values('date')
    if raw['date'].duplicated().any():
        raise ValueError('Duplicate dates in CSV')
    raw = raw.set_index('date').asfreq('15min')
    raw = raw.loc[:, list(FULL_FEATURES)].apply(pd.to_numeric, errors='raise')
    if raw.isna().any().any():
        raise ValueError('Missing timestamps or values in CSV')
    n = len(raw)
    train_end, val_end = int(n*.70), int(n*.85)
    boundaries = {'train':(0,train_end), 'val':(train_end,val_end), 'test':(val_end,n)}
    split_start, split_end = boundaries[args.split]
    max_seq_len = max(int(ckpt['seq_len']) for _, ckpt in checkpoints)
    if args.origin_time:
        origin = int(raw.index.get_indexer([pd.Timestamp(args.origin_time)])[0])
        if origin < 0:
            raise ValueError('origin-time not found in CSV')
    else:
        origin = max(split_start, max_seq_len) + args.origin_index
    if origin < max(split_start,max_seq_len) or origin + HORIZON > split_end:
        raise ValueError('Common forecast origin outside selected split or insufficient history/horizon')
    future_time = raw.index[origin:origin+HORIZON]
    actual = raw['OT'].iloc[origin:origin+HORIZON].to_numpy(dtype=np.float64)
    history_time = raw.index[origin-max_seq_len:origin]
    historical = raw['OT'].iloc[origin-max_seq_len:origin].to_numpy(dtype=np.float64)
    device = torch.device('cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{args.split}_{origin}'
    predictions = pd.DataFrame({'date':future_time, 'actual_OT':actual})
    summary = []
    fig, ax = plt.subplots(figsize=(14,5.5))
    ax.plot(history_time, historical, color='#94a3b8', lw=1, label='Historical OT')
    ax.plot(future_time, actual, color='#111827', lw=2.2, label='Actual future OT')
    for path, ckpt in checkpoints:
        features = list(ckpt['features'])
        seq_len = int(ckpt['seq_len'])
        scaler = StandardScaler().fit(raw.iloc[:train_end][features])
        x = scaler.transform(raw.iloc[origin-seq_len:origin][features]).astype(np.float32)
        ti = features.index('OT')
        params = ckpt['params']
        model = RecurrentModel(ckpt['model'], len(features), HORIZON,
                               int(params['hidden_size']), int(params['num_layers']), float(params['dropout'])).to(device)
        try:
            model.load_state_dict(ckpt['state_dict'], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f'Checkpoint architecture mismatch: {path}. Verify training main.py.\n{exc}') from exc
        model.eval()
        with torch.inference_mode():
            z = model(torch.from_numpy(x).unsqueeze(0).to(device)).cpu().numpy()[0,:,0]
        pred = z.astype(np.float64)*float(scaler.scale_[ti])+float(scaler.mean_[ti])
        label = path.stem
        if label in predictions.columns:
            raise ValueError(f'Duplicate checkpoint stem: {label}; rename files to unique names')
        predictions[label] = pred
        score = metrics(actual,pred)
        summary.append({'checkpoint':str(path), 'label':label, 'model':ckpt['model'],
                        'features':'|'.join(features), 'seq_len':seq_len,
                        'best_epoch':ckpt.get('best_epoch'), **score})
        ax.plot(future_time,pred,lw=1.5,label=f'{label} (RMSE={score["RMSE"]:.3f})')
        print(f'{label}: RMSE={score["RMSE"]:.5f} MAE={score["MAE"]:.5f}')
    ax.axvline(future_time[0],ls='--',color='#64748b',lw=1,label='Forecast origin')
    ax.set(title=f'ETTm2 OT prediction comparison | {args.split} | {future_time[0]}',
           xlabel='Timestamp',ylabel='OT (original units)')
    ax.grid(alpha=.2)
    ax.legend(loc='best',fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    chart = out / f'{stem}_comparison.png'
    fig.savefig(chart,dpi=170)
    if args.show:
        plt.show()
    plt.close(fig)
    pred_csv = out / f'{stem}_predictions.csv'
    summary_csv = out / f'{stem}_metrics.csv'
    predictions.to_csv(pred_csv,index=False)
    pd.DataFrame(summary).sort_values('RMSE').to_csv(summary_csv,index=False)
    report = {'csv':str(Path(args.csv).resolve()),
              'csv_sha256':hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
              'split':args.split,'origin':str(future_time[0]),'horizon':HORIZON,
              'checkpoint_count':len(checkpoints),'results':summary}
    (out / f'{stem}_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Wrote {chart}, {pred_csv}, {summary_csv}')

if __name__ == '__main__':
    main()
