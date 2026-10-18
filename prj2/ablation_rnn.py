"""A1 sequence length and A2 feature ablations for ETTm2 OT forecasting.

Run after grid_search_rnn.py. Grid results select hyperparameters by validation
RMSE; ablations re-train with fixed hyperparameters and common forecast origins.
"""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from main import (FEATURE_COLS, TARGET_COL, HORIZON, SEED, DEVICE,
                  RecurrentModel, WindowDataset, seed_everything)

MODELS = ('RNN', 'LSTM', 'GRU')
LENGTHS = (96, 192, 336, 672, 1344)
BASE_LENGTH = 672
FULL_FEATURES = ('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT')
WEIGHT_DECAY = 1e-4


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_csv(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_best(grid_dir):
    grid_dir = Path(grid_dir)
    manifest_path = grid_dir / 'experiment_config.json'
    results_path = grid_dir / 'grid_results.csv'
    if not manifest_path.exists() or not results_path.exists():
        raise FileNotFoundError('Grid search outputs missing: experiment_config.json and grid_results.csv required')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if list(manifest['features']) != list(FULL_FEATURES) or manifest['seq_len'] != BASE_LENGTH or manifest['horizon'] != HORIZON:
        raise ValueError('Grid search must use H4 (all 7 features), seq_len=672, horizon=96')
    results = pd.read_csv(results_path)
    selected = {}
    for model in MODELS:
        part = results[(results.model == model) & results.val_RMSE_OT.notna()].sort_values(['val_RMSE_OT', 'run_id'])
        if len(part) != len(manifest['grid']):
            raise ValueError(f'Grid search incomplete for {model}: {len(part)} / {len(manifest["grid"])}')
        best = part.iloc[0]
        selected[model] = dict(hidden_size=int(best.hidden_size), num_layers=int(best.num_layers),
                               learning_rate=float(best.learning_rate), dropout=float(best.dropout),
                               grid_run_id=str(best.run_id), grid_val_rmse=float(best.val_RMSE_OT))
    return manifest, selected


def prepare_data(csv_path, max_length):
    raw = pd.read_csv(csv_path, parse_dates=['date']).sort_values('date')
    if raw.date.duplicated().any():
        raise ValueError('Duplicate timestamps')
    raw = raw.set_index('date').asfreq('15min')
    raw = raw[list(FULL_FEATURES)].apply(pd.to_numeric, errors='raise')
    if raw.isna().any().any():
        raise ValueError('Missing timestamps or values')
    n = len(raw)
    train_end, val_end = int(n * .70), int(n * .85)
    if train_end <= max_length + HORIZON or val_end - train_end < HORIZON or n - val_end < HORIZON:
        raise ValueError('Insufficient samples')
    return raw, train_end, val_end


def make_loaders(raw, features, seq_len, train_end, val_end, batch, max_length):
    # Fixed origins for ALL experiments, including training. The longest window
    # determines earliest training origin; val/test origins are split boundaries.
    scaler = StandardScaler().fit(raw.iloc[:train_end][list(features)])
    z = scaler.transform(raw[list(features)]).astype(np.float32)
    ti = list(features).index(TARGET_COL)
    train = WindowDataset(z, max_length, train_end, seq_len, HORIZON, ti)
    val = WindowDataset(z, train_end, val_end, seq_len, HORIZON, ti)
    test = WindowDataset(z, val_end, len(z), seq_len, HORIZON, ti)
    # Verify all have identical forecast origins independent of seq_len.
    assert train.starts[0] == max_length and val.starts[0] == train_end and test.starts[0] == val_end
    kwargs = {'num_workers': 0, 'pin_memory': DEVICE.type == 'cuda'}
    return (DataLoader(train, batch_size=batch, shuffle=True, **kwargs),
            DataLoader(val, batch_size=batch, shuffle=False, **kwargs),
            DataLoader(test, batch_size=batch, shuffle=False, **kwargs),
            float(scaler.mean_[ti]), float(scaler.scale_[ti]), len(train))


def evaluate(model, loader, mean, scale):
    model.eval()
    sum_sq = sum_abs = sum_y = sum_y_sq = 0.
    count = 0
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(DEVICE, non_blocking=True)).cpu().numpy().astype(np.float64)
            actual = y.numpy().astype(np.float64)
            err = (pred - actual) * scale
            actual = actual * scale + mean
            sum_sq += float(np.square(err).sum())
            sum_abs += float(np.abs(err).sum())
            sum_y += float(actual.sum())
            sum_y_sq += float(np.square(actual).sum())
            count += err.size
    mse = sum_sq / count
    denom = sum_y_sq - sum_y * sum_y / count
    return {'MSE': mse, 'MAE': sum_abs/count, 'RMSE': mse**.5,
            'R2': 1 - sum_sq/denom if denom > 0 else float('nan')}


def fit(model, train_loader, val_loader, params, epochs, patience, mean, scale):
    optim = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=WEIGHT_DECAY)
    best = float('inf')
    best_state = None
    best_epoch = 0
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.
        count = 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_total += float(loss.item()) * y.numel()
            count += y.numel()
        val = evaluate(model, val_loader, mean, scale)
        history.append({'epoch': epoch, 'train_RMSE': (loss_total/count)**.5*scale,
                        'val_RMSE': val['RMSE']})
        if val['RMSE'] < best - 1e-8:
            best = val['RMSE']
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return best_epoch, history


def plot_results(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    for metric in ('val_RMSE', 'test_RMSE'):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name in MODELS:
            part = df[(df.model == name) & (df.study == 'A1')].sort_values('seq_len')
            ax.plot(part.seq_len, part[metric], marker='o', label=name)
        ax.axvline(BASE_LENGTH, ls='--', color='gray', alpha=.5)
        ax.set(xlabel='Input sequence length (15-minute steps)', ylabel=f'{metric} (OT units)',
               title=f'A1: {metric} by input length')
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(out / f'a1_{metric}.png', dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = .35
        for i, name in enumerate(MODELS):
            part = df[(df.model == name) & (df.seq_len == BASE_LENGTH)].set_index('features_id')
            ax.bar(i-width/2, part.loc['OT_ONLY', metric], width, label='OT only' if i==0 else None)
            ax.bar(i+width/2, part.loc['ALL', metric], width, label='All features' if i==0 else None)
        ax.set_xticks(range(3), MODELS)
        ax.set(ylabel=f'{metric} (OT units)', title=f'A2: {metric} by input features')
        ax.legend(); ax.grid(axis='y', alpha=.25); fig.tight_layout()
        fig.savefig(out / f'a2_{metric}.png', dpi=150); plt.close(fig)


def run(args):
    if args.batch < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError('batch, epochs, patience must be positive')
    grid_manifest, best_params = load_best(args.grid_dir)
    if args.epochs != int(grid_manifest['epochs']) or args.patience != int(grid_manifest['patience']):
        raise ValueError('Ablation epochs/patience must match grid search; use matching CLI arguments')
    raw, train_end, val_end = prepare_data(args.csv, max(LENGTHS))
    if len(raw) != grid_manifest['rows'] or train_end != grid_manifest['train_end'] or val_end != grid_manifest['val_end']:
        raise ValueError('CSV split does not match grid experiment')
    csv_hash = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {'csv_sha256': csv_hash, 'grid_manifest': grid_manifest, 'best_params': best_params,
                'seq_lengths': LENGTHS, 'full_features': FULL_FEATURES,
                'batch': args.batch, 'epochs': args.epochs, 'patience': args.patience,
                'seed': SEED, 'weight_decay': WEIGHT_DECAY,
                'train_origin_start': max(LENGTHS), 'horizon': HORIZON}
    mf = out / 'ablation_config.json'
    if mf.exists():
        if json.loads(mf.read_text(encoding='utf-8')) != json.loads(json.dumps(manifest)):
            raise ValueError('Existing output configuration differs. Use a new --out directory.')
    else:
        atomic_json(mf, manifest)
    runs_file = out / 'ablation_results.csv'
    rows = pd.read_csv(runs_file).to_dict('records') if runs_file.exists() else []
    ckpts = out / 'checkpoints'; ckpts.mkdir(exist_ok=True)
    histories = out / 'histories'; histories.mkdir(exist_ok=True)
    conditions = []# [('A1', 'ALL', list(FULL_FEATURES), length) for length in LENGTHS]
    conditions.append(('A2', 'OT_ONLY', [TARGET_COL], BASE_LENGTH))
    conditions.append(('A2', 'H1', ["HUFL","MUFL","LUFL",TARGET_COL], BASE_LENGTH))
    conditions.append(('A2', 'H2', ["HUFL","HULL",TARGET_COL], BASE_LENGTH))
    conditions.append(('A2', 'H3', ["HUFL","HULL","MUFL","MULL",TARGET_COL], BASE_LENGTH))
    conditions.append(('A2', 'H4', ["MULL","LUFL",TARGET_COL], BASE_LENGTH))
    for name in MODELS:
        params = best_params[name]
        for study, features_id, features, seq_len in conditions:
            run_id = f'{name}_{features_id}_L{seq_len}'
            ckpt = ckpts / f'{run_id}.pt'
            if any(r['run_id'] == run_id for r in rows) and ckpt.exists():
                print('SKIP', run_id, flush=True)
                continue
            seed_everything(SEED)
            train_loader, val_loader, test_loader, mean, scale, train_count = make_loaders(
                raw, features, seq_len, train_end, val_end, args.batch, max(LENGTHS))
            model = RecurrentModel(name, len(features), HORIZON, hidden=params['hidden_size'],
                                   layers=params['num_layers'], dropout=params['dropout']).to(DEVICE)
            if DEVICE.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            epoch, history = fit(model, train_loader, val_loader, params, args.epochs,
                                 args.patience, mean, scale)
            val_metrics = evaluate(model, val_loader, mean, scale)
            test_metrics = evaluate(model, test_loader, mean, scale)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated()/1024**2 if DEVICE.type == 'cuda' else 0.
            checkpoint = {'state_dict': {k: v.detach().cpu() for k,v in model.state_dict().items()},
                          'model': name, 'features': features, 'seq_len': seq_len,
                          'params': params, 'best_epoch': epoch}
            temp_ckpt = ckpt.with_suffix('.pt.tmp')
            torch.save(checkpoint, temp_ckpt); os.replace(temp_ckpt, ckpt)
            atomic_csv(histories / f'{run_id}.csv', history)
            row = {'run_id': run_id, 'model': name, 'study': study,
                   'features_id': features_id, 'features': '|'.join(features),
                   'seq_len': seq_len, 'best_epoch': epoch, 'train_windows': train_count,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'train_seconds': seconds, 'peak_vram_MB': peak,
                   **{f'val_{k}': v for k,v in val_metrics.items()},
                   **{f'test_{k}': v for k,v in test_metrics.items()}}
            rows = [r for r in rows if r['run_id'] != run_id] + [row]
            atomic_csv(runs_file, rows)
            print(f'{run_id}: val RMSE={val_metrics["RMSE"]:.4f}, test RMSE={test_metrics["RMSE"]:.4f}', flush=True)
    results = pd.DataFrame(rows)
    if len(results) == len(MODELS)*len(conditions):
        baseline = results[(results.features_id == 'ALL') & (results.seq_len == BASE_LENGTH)].set_index('model')
        for metric in ('val_RMSE', 'test_RMSE'):
            results[f'delta_{metric}_vs_baseline'] = results.apply(
                lambda r: r[metric] - baseline.loc[r['model'], metric], axis=1)
        atomic_csv(runs_file, results.to_dict('records'))
        atomic_csv(out / 'a1_sequence_length.csv', results[results.study == 'A1'].to_dict('records'))
        # A2 includes the shared baseline row (not separately re-trained).
        a2 = results[(results.features_id == 'OT_ONLY') | ((results.features_id == 'ALL') & (results.seq_len == BASE_LENGTH))]
        atomic_csv(out / 'a2_input_features.csv', a2.to_dict('records'))
        plot_results(results.to_dict('records'), out)
        print('Completed 18 conditions. Outputs:', out.resolve(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default='./assets/ETTm2.csv')
    p.add_argument('--grid-dir', default='results/grid_search_ot')
    p.add_argument('--out', default='results/ablation_ot')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    run(p.parse_args())
