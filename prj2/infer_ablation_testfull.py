"""Full ETTm2 test-set inference for ablation checkpoints; no changes to existing scripts.

Reproduces ablation_rnn.py test windows and train-only StandardScaler.
Outputs two visualizations: overlapping-prediction mean and stitched 96-step blocks.
The full sliding-window RMSE is computed BEFORE aggregation (matches ablation evaluate).

Example:
 python infer_ablation_full_test.py --pt RNN_OT_ONLY_L672.pt LSTM_OT_ONLY_L672.pt GRU_OT_ONLY_L672.pt --csv ETTm2.csv --out results/full_test
 python infer_ablation_full_test.py --pt-glob 'checkpoints/*_OT_ONLY_*.pt' --csv ETTm2.csv
"""
import argparse
import glob
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

# Reuse the model definition and checkpoint validator, without modifying original script.
from infer_ablation import RecurrentModel, load_checkpoint

HORIZON = 96
FEATURES = ('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pt', nargs='+', default=[], help='Trusted ablation .pt checkpoints')
    p.add_argument('--pt-glob', action='append', default=[], help='Quoted glob for checkpoint paths; repeatable')
    p.add_argument('--csv', required=True, help='Same ETTm2.csv as ablation training')
    p.add_argument('--out', default='results/full_test')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--dpi', type=int, default=160)
    args = p.parse_args()
    if args.batch_size <= 0:
        p.error('--batch-size must be positive')
    paths = [Path(s).resolve() for s in args.pt]
    for pattern in args.pt_glob:
        paths.extend(Path(s).resolve() for s in sorted(glob.glob(pattern)))
    args.paths = list(dict.fromkeys(paths))
    if not args.paths:
        p.error('Supply --pt path1.pt path2.pt or --pt-glob ".../*.pt"')
    for path in args.paths:
        if not path.is_file():
            p.error(f'Checkpoint not found: {path}')
    return args


def load_data(path):
    raw = pd.read_csv(path, parse_dates=['date']).sort_values('date')
    if raw.date.isna().any() or raw.date.duplicated().any():
        raise ValueError('CSV has missing or duplicate timestamps')
    raw = raw.set_index('date').asfreq('15min')
    raw = raw[list(FEATURES)].apply(pd.to_numeric, errors='raise')
    if raw.isna().any().any():
        raise ValueError('CSV has missing timestamps or feature values')
    train_end = int(len(raw) * .70)
    val_end = int(len(raw) * .85)
    if val_end < 1 or len(raw) - val_end < HORIZON:
        raise ValueError('Insufficient test data')
    return raw, train_end, val_end


def make_model(ckpt, device):
    params = ckpt['params']
    model = RecurrentModel(ckpt['model'], len(ckpt['features']), HORIZON,
                           int(params['hidden_size']), int(params['num_layers']),
                           float(params['dropout'])).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model


def plot_series(dates, actual, series, title, ylabel, path, dpi):
    fig, ax = plt.subplots(figsize=(17, 5.8))
    
    ax.plot(dates, actual, color='#3ca69673', lw=1.0, alpha=.85, label='Actual OT')
    ax.fill_between(dates, actual, color="#3ca69673")
        
    for name, values in series.items():
        ax.plot(dates, values, lw=1, alpha=1, label=f'Model: {name}')
    ax.set(title=title, xlabel='Timestamp', ylabel=ylabel)
    ax.grid(alpha=.18)
    ax.legend(loc='best', fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    fig.show()
    plt.show()
    # plt.close(fig)


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    raw, train_end, test_start = load_data(args.csv)
    test_end = len(raw)
    test_len = test_end - test_start
    actual = raw['OT'].iloc[test_start:].to_numpy(dtype=np.float64)
    dates = raw.index[test_start:]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({'date': dates, 'actual_OT': actual})
    summary = []
    horizon_rows = []
    mean_curves = {}
    stitched_curves = {}
    all_labels = set()

    for path in args.paths:
        ckpt = load_checkpoint(path)
        label = path.stem
        if label in all_labels:
            raise ValueError(f'Duplicate checkpoint filename stem: {label}')
        all_labels.add(label)
        features = list(ckpt['features'])
        seq_len = int(ckpt['seq_len'])
        if seq_len > test_start:
            raise ValueError(f'{label}: history longer than data preceding test split')
        scaler = StandardScaler().fit(raw.iloc[:train_end][features])
        z = scaler.transform(raw[features]).astype(np.float32)
        ot_idx = features.index('OT')
        ot_mean, ot_scale = float(scaler.mean_[ot_idx]), float(scaler.scale_[ot_idx])
        model = make_model(ckpt, device)

        # Exactly the same forecast origins as ablation_rnn.WindowDataset test split.
        origins = np.arange(test_start, test_end-HORIZON+1, dtype=np.int64)
        sum_sq = 0.0
        sum_abs = 0.0
        sum_y = 0.0
        sum_y_sq = 0.0
        count = 0
        horizon_sq = np.zeros(HORIZON, dtype=np.float64)
        horizon_abs = np.zeros(HORIZON, dtype=np.float64)
        # Aggregated visual curve: all predictions for each timestamp, averaged.
        sum_pred = np.zeros(test_len, dtype=np.float64)
        n_pred = np.zeros(test_len, dtype=np.int64)
        # Stitched curve: nonoverlapping 96-step blocks, with tail handled separately.
        stitched = np.full(test_len, np.nan, dtype=np.float64)
        block_origins = set(origins[::HORIZON].tolist())
        # The last partial block must be forecast from the final valid origin.
        block_origins.add(int(origins[-1]))
        for first in range(0, len(origins), args.batch_size):
            batch_origins = origins[first:first+args.batch_size]
            x = np.stack([z[t-seq_len:t] for t in batch_origins])
            with torch.inference_mode():
                pred_z = model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float64)[:, :, 0]
            actual_z = np.stack([z[t:t+HORIZON, ot_idx] for t in batch_origins]).astype(np.float64)
            err = (pred_z - actual_z) * ot_scale
            actual_orig = actual_z * ot_scale + ot_mean
            pred_orig = pred_z * ot_scale + ot_mean
            sum_sq += float(np.square(err).sum())
            sum_abs += float(np.abs(err).sum())
            sum_y += float(actual_orig.sum())
            sum_y_sq += float(np.square(actual_orig).sum())
            count += err.size
            horizon_sq += np.square(err).sum(axis=0)
            horizon_abs += np.abs(err).sum(axis=0)
            for j, t in enumerate(batch_origins):
                start = int(t-test_start)
                sum_pred[start:start+HORIZON] += pred_orig[j]
                n_pred[start:start+HORIZON] += 1
                if int(t) in block_origins:
                    # Assign only still-unfilled positions, including final tail.
                    segment = stitched[start:start+HORIZON]
                    missing = np.isnan(segment)
                    segment[missing] = pred_orig[j, missing]
        if np.any(n_pred == 0) or np.isnan(stitched).any():
            raise RuntimeError(f'{label}: test timestamps left without predictions')
        mse = sum_sq/count
        denom = sum_y_sq - sum_y*sum_y/count
        mean_curve = sum_pred/n_pred
        mean_curves[label] = mean_curve
        stitched_curves[label] = stitched
        predictions[f'{label}__overlap_mean'] = mean_curve
        predictions[f'{label}__nonoverlap_96'] = stitched
        summary.append({
            'checkpoint': str(path), 'label': label, 'model': ckpt['model'],
            'features': '|'.join(features), 'seq_len': seq_len,
            'best_epoch': ckpt.get('best_epoch'), 'test_windows': len(origins),
            'evaluated_predictions': count,
            'test_MSE': mse, 'test_MAE': sum_abs/count,
            'test_RMSE': float(np.sqrt(mse)),
            'test_R2': float(1-sum_sq/denom) if denom > 0 else float('nan'),
            'overlap_mean_curve_RMSE': float(np.sqrt(np.mean(np.square(mean_curve-actual)))),
            'nonoverlap_96_curve_RMSE': float(np.sqrt(np.mean(np.square(stitched-actual))))
        })
        for h in range(HORIZON):
            horizon_rows.append({'label': label, 'model': ckpt['model'], 'horizon_step': h+1,
                                 'minutes_ahead': (h+1)*15,
                                 'RMSE': float(np.sqrt(horizon_sq[h]/len(origins))),
                                 'MAE': float(horizon_abs[h]/len(origins))})
        print(f'{label}: full sliding-window test RMSE = {np.sqrt(mse):.6f} '
              f'({len(origins)} windows x {HORIZON} steps)')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    predictions.to_csv(out/'test_predictions_visualization.csv', index=False, float_format='%.8f')
    pd.DataFrame(summary).sort_values('test_RMSE').to_csv(out/'test_metrics.csv', index=False)
    pd.DataFrame(horizon_rows).to_csv(out/'horizon_metrics.csv', index=False)
    plot_series(dates, actual, mean_curves,
                'ETTm2 Test OT | Mean of all overlapping 96-step predictions',
                'OT (original units)', out/'test_overlap_mean.png', args.dpi)
    plot_series(dates, actual, stitched_curves,
                'ETTm2 Test OT | Stitched non-overlapping 96-step predictions',
                'OT (original units)', out/'test_nonoverlap_96.png', args.dpi)
    fig, ax = plt.subplots(figsize=(12, 5))
    for label in mean_curves:
        part = [r for r in horizon_rows if r['label'] == label]
        ax.plot([r['horizon_step'] for r in part], [r['RMSE'] for r in part], label=f'Model: {label}')
    ax.set(title='ETTm2 Test | Horizon-wise RMSE', xlabel='Forecast horizon (15-min steps)',
           ylabel='RMSE (original OT units)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'test_horizon_rmse.png', dpi=args.dpi)
    fig.show()
    plt.show()
    # plt.close(fig)
    report = {'dataset': str(Path(args.csv).resolve()),
              'dataset_sha256': hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
              'train_end': train_end, 'test_start': test_start, 'test_end': test_end,
              'test_observations': test_len, 'forecast_horizon': HORIZON,
              'window_count': test_len-HORIZON+1,
              'metrics_definition': 'RMSE of every prediction in every sliding test window, in original OT units',
              'overlap_mean_definition': 'Arithmetic mean of all forecasts targeting each timestamp; visualization only',
              'nonoverlap_definition': 'Forecast origins spaced 96 samples; last incomplete block filled from final valid origin; visualization only',
              'results': summary}
    (out/'test_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Output directory: {out.resolve()}')

if __name__ == '__main__':
    main()
