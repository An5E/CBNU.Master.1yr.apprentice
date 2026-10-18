"""Fixed-feature RNN/LSTM/GRU grid search; compare validation performance changes."""
import argparse
import copy
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from main import (FEATURE_COLS, TARGET_COL, SEQ_LEN, HORIZON, SEED, DEVICE,
                  RecurrentModel, WindowDataset, load_data, save_correlation_heatmap,
                  seed_everything)

MODELS = ('RNN', 'LSTM', 'GRU')
HIDDEN = (32, 64, 128)
LAYERS = (1, 2)
LRS = (1e-3, 3e-4)
DROPOUTS = (0.0, 0.1, 0.2)
WEIGHT_DECAY = 1e-4


def configs():
    for hidden, layers, lr in itertools.product(HIDDEN, LAYERS, LRS):
        for dropout in ((0.0,) if layers == 1 else DROPOUTS):
            yield {'hidden_size': hidden, 'num_layers': layers,
                   'learning_rate': lr, 'dropout': dropout}


def mse(model, loader):
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(DEVICE, non_blocking=True))
            err = (pred - y.to(DEVICE, non_blocking=True)).square()
            total += err.sum().item()
            count += err.numel()
    return total / count


def train(model, train_loader, val_loader, lr, epochs, patience, target_scale):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best, best_epoch, best_state, remaining = float('inf'), 0, None, patience
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0., 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * y.numel()
            count += y.numel()
        val_mse = mse(model, val_loader)
        history.append({'epoch': epoch, 'train_RMSE_OT': (total/count)**.5*target_scale,
                        'val_RMSE_OT': val_mse**.5*target_scale})
        if val_mse < best - 1e-8:
            best, best_epoch, best_state, remaining = val_mse, epoch, copy.deepcopy(model.state_dict()), patience
        else:
            remaining -= 1
            if remaining <= 0:
                break
    model.load_state_dict(best_state)
    return best**.5*target_scale, best_epoch, history


def test_metrics(model, loader, mean, scale):
    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for x, y in loader:
            predicted.append(model(x.to(DEVICE)).cpu().numpy())
            actual.append(y.numpy())
    y = np.concatenate(actual) * scale + mean
    p = np.concatenate(predicted) * scale + mean
    err = p - y
    return {'test_MAE': float(np.abs(err).mean()),
            'test_RMSE': float(np.square(err).mean()**.5),
            'test_sMAPE_pct': float((200*np.abs(err)/np.maximum(np.abs(y)+np.abs(p), 1e-8)).mean())}


def make_loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=DEVICE.type == 'cuda')


def write_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def plot_results(df, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Model comparison across hyperparameter choices; axes intentionally in original OT units.
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODELS:
        group = df[df.model == name].sort_values('val_RMSE_OT')
        ax.plot(range(1, len(group)+1), group.val_RMSE_OT.to_numpy(), marker='o', ms=3, label=name)
    ax.set(xlabel='Configuration rank within each model (best to worst)',
           ylabel='Validation RMSE (OT units)', title='Grid search validation performance')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / 'validation_rank_comparison.png', dpi=150); plt.close(fig)
    for param in ('hidden_size', 'num_layers', 'learning_rate', 'dropout'):
        fig, ax = plt.subplots(figsize=(8, 4))
        # Descriptive marginal means: other hyperparameters are mixed, not causal effects.
        grouped = df.groupby(['model', param])['val_RMSE_OT'].mean().unstack('model')
        grouped.plot(ax=ax, marker='o')
        ax.set(xlabel=param, ylabel='Mean validation RMSE (OT units)',
               title=f'Validation RMSE by {param} (descriptive)')
        ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(out / f'impact_{param}.png', dpi=150); plt.close(fig)


def run(args):
    seed_everything(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, z, scaler, target_idx, train_end, val_end = load_data(args.csv)
    # save_correlation_heatmap(df, train_end, out)
    manifest = {'features': FEATURE_COLS, 'target': TARGET_COL, 'seq_len': SEQ_LEN,
                'horizon': HORIZON, 'train_end': train_end, 'val_end': val_end,
                'rows': len(df), 'batch': args.batch, 'epochs': args.epochs,
                'patience': args.patience, 'seed': SEED, 'grid': list(configs()),
                'weight_decay': WEIGHT_DECAY}
    manifest_file = out / 'experiment_config.json'
    if manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing output has a different experiment configuration. Use a new --out directory.')
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    train_ds = WindowDataset(z, SEQ_LEN, train_end, SEQ_LEN, HORIZON, target_idx)
    val_ds = WindowDataset(z, train_end, val_end, SEQ_LEN, HORIZON, target_idx)
    test_ds = WindowDataset(z, val_end, len(z), SEQ_LEN, HORIZON, target_idx)
    train_loader = make_loader(train_ds, args.batch, True)
    val_loader = make_loader(val_ds, args.batch, False)
    
    target_scale = float(scaler.scale_[target_idx])
    target_mean = float(scaler.mean_[target_idx])
    runs_file = out / 'grid_results.csv'
    rows = pd.read_csv(runs_file).to_dict('records') if runs_file.exists() else []
    checkpoints = out / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    histories = out / 'histories'; histories.mkdir(exist_ok=True)
    for name in MODELS:
        for i, params in enumerate(configs()):
            run_id = f'{name}_{i:02d}'
            ckpt = checkpoints / f'{run_id}.pt'
            if any(r['run_id'] == run_id for r in rows) and ckpt.exists():
                print(f'SKIP {run_id}', flush=True)
                continue
            seed_everything(SEED)
            model = RecurrentModel(name, z.shape[1], HORIZON, hidden=params['hidden_size'],
                                   layers=params['num_layers'], dropout=params['dropout']).to(DEVICE)
            start = time.perf_counter()
            val_rmse, best_epoch, history = train(model, train_loader, val_loader,
                                                   params['learning_rate'], args.epochs,
                                                   args.patience, target_scale)
            seconds = time.perf_counter() - start
            # Save checkpoint before row; interruption may cause a safe re-run of this configuration.
            torch.save({'state_dict': model.cpu().state_dict(), 'params': params,
                        'model': name, 'run_id': run_id}, ckpt)
            write_csv(histories / f'{run_id}.csv', history)
            rows = [r for r in rows if r['run_id'] != run_id]
            rows.append({'run_id': run_id, 'model': name, **params,
                         'val_RMSE_OT': val_rmse, 'best_epoch': best_epoch,
                         'train_seconds': seconds,
                         'parameters': sum(p.numel() for p in model.parameters())})
            write_csv(runs_file, rows)
            print(f'{run_id} RMSE={val_rmse:.4f} epoch={best_epoch} seconds={seconds:.1f}', flush=True)
    results = pd.DataFrame(rows).sort_values(['model', 'val_RMSE_OT'])
    plot_results(results, out)
    test_loader = make_loader(test_ds, args.batch, False)
    best_rows = []
    for name in MODELS:
        chosen = results[results.model == name].iloc[0]
        params = {k: chosen[k] for k in ('hidden_size', 'num_layers', 'dropout')}
        model = RecurrentModel(name, z.shape[1], HORIZON, hidden=int(params['hidden_size']),
                               layers=int(params['num_layers']), dropout=float(params['dropout']))
        ckpt = torch.load(checkpoints / f"{chosen['run_id']}.pt", map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
        model.to(DEVICE)
        metrics = test_metrics(model, test_loader, target_mean, target_scale)
        best_rows.append({**chosen.to_dict(), **metrics})
    write_csv(out / 'best_models_test_results.csv', best_rows)
    print(pd.DataFrame(best_rows)[['model', 'val_RMSE_OT', 'test_RMSE', 'test_MAE']].to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='./assets/ETTm2.csv')
    p.add_argument('--out', default='results/grid_search_ot')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    run(p.parse_args())
