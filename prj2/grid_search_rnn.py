
# grid_search_rnn.py
# 기존 benchmark.py의 모델 및 전처리 결과 재활용

import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from main import (
    RecurrentModel,
    WindowDataset,
    z,
    train_end,
    val_end,
    SEQ_LEN,
    HORIZON,
    TARGET_COL,
    FEATURE_COLS,
    scaler,
    DEVICE,
)

# =====================================
# 1. Configuration
# =====================================
OUT = Path("results/grid_search")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_EPOCHS = 30
PATIENCE = 5

# 1차 Grid Search: 36 configurations per model
PARAM_GRID = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "learning_rate": [1e-3, 3e-4],
    "dropout": [0.0, 0.1, 0.2],
}

BATCH_SIZE_CANDIDATES = [64]
WEIGHT_DECAY_CANDIDATES = [1e-4]

MODEL_NAMES = ["RNN", "LSTM", "GRU"]


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================
# 2. DataLoaders
# =====================================
def make_grid_loaders(seq_len, horizon, batch_size):
    train_ds = WindowDataset(
        z, seq_len, train_end, seq_len, horizon
    )
    val_ds = WindowDataset(
        z, train_end, val_end, seq_len, horizon
    )
    test_ds = WindowDataset(
        z, val_end, len(z), seq_len, horizon
    )

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise ValueError("Empty dataset.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


# =====================================
# 3. Train / Validation
# =====================================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    count = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()

        # Prevent exploding gradients in recurrent networks.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        count += x.size(0)

    return total_loss / count


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0
    count = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)

        total_loss += loss.item() * x.size(0)
        count += x.size(0)

    return total_loss / count


def fit_with_early_stopping(
    model,
    train_loader,
    val_loader,
    learning_rate,
    weight_decay,
):
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = PATIENCE

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer
        )
        val_loss = evaluate_loss(model, val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = PATIENCE
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"Train MSE={train_loss:.6f} | "
                f"Val MSE={val_loss:.6f}"
            )

        if patience_left == 0:
            break

    elapsed = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")

    model.load_state_dict(best_state)

    return model, best_val_loss, best_epoch, elapsed


# =====================================
# 4. Test metrics
# =====================================
@torch.no_grad()
def predict(model, loader):
    model.eval()
    actual_list = []
    pred_list = []

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for x, y in loader:
        pred = model(x.to(DEVICE)).cpu().numpy()
        actual_list.append(y.numpy())
        pred_list.append(pred)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (
        np.concatenate(actual_list),
        np.concatenate(pred_list),
        elapsed,
    )


def inverse_transform_3d(arr):
    """[N,H,C] standardized values -> original units."""
    n, h, c = arr.shape
    return scaler.inverse_transform(
        arr.reshape(-1, c)
    ).reshape(n, h, c)


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = np.mean(
        200.0 * np.abs(err) / np.maximum(denom, 1e-8)
    )

    return {
        "test_MAE": float(mae),
        "test_RMSE": float(rmse),
        "test_sMAPE_pct": float(smape),
    }


# =====================================
# 5. Grid Search
# =====================================
def run_grid_search():
    grid_keys = list(PARAM_GRID.keys())
    grid_values = list(PARAM_GRID.values())

    all_results = []
    best_by_model = {}

    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 60}")
        print(f"GRID SEARCH: {model_name}")
        print(f"{'=' * 60}")

        best_val_loss = float("inf")
        best_state = None
        best_params = None
        best_epoch = None

        configurations = list(itertools.product(*grid_values))

        for batch_size in BATCH_SIZE_CANDIDATES:
            for weight_decay in WEIGHT_DECAY_CANDIDATES:
                for config_values in configurations:
                    params = dict(zip(grid_keys, config_values))
                    params["batch_size"] = batch_size
                    params["weight_decay"] = weight_decay

                    set_seed(SEED)

                    train_loader, val_loader, _ = make_grid_loaders(
                        SEQ_LEN, HORIZON, batch_size
                    )

                    model = RecurrentModel(
                        kind=model_name,
                        n_features=z.shape[1],
                        horizon=HORIZON,
                        hidden=params["hidden_size"],
                        layers=params["num_layers"],
                        dropout=params["dropout"],
                    )

                    print(
                        f"\n{model_name} | "
                        f"{json.dumps(params)}"
                    )

                    model, val_loss, epoch, train_seconds = (
                        fit_with_early_stopping(
                            model,
                            train_loader,
                            val_loader,
                            params["learning_rate"],
                            params["weight_decay"],
                        )
                    )

                    row = {
                        "model": model_name,
                        **params,
                        "best_val_MSE_scaled": val_loss,
                        "best_epoch": epoch,
                        "train_seconds": train_seconds,
                    }
                    all_results.append(row)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = params.copy()
                        best_epoch = epoch
                        best_state = copy.deepcopy(
                            model.state_dict()
                        )

                    del model

        # Save the best validation checkpoint for this model.
        checkpoint_path = OUT / f"{model_name}_best.pt"
        torch.save(
            {
                "model": model_name,
                "params": best_params,
                "best_val_MSE_scaled": best_val_loss,
                "best_epoch": best_epoch,
                "state_dict": best_state,
                "seq_len": SEQ_LEN,
                "horizon": HORIZON,
            },
            checkpoint_path,
        )

        best_by_model[model_name] = {
            "params": best_params,
            "val_loss": best_val_loss,
            "epoch": best_epoch,
            "checkpoint": str(checkpoint_path),
        }

        print(f"\nBest {model_name}:")
        print(best_by_model[model_name])

    # Save all configurations before final test evaluation.
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        ["model", "best_val_MSE_scaled"]
    )
    results_df.to_csv(
        OUT / "grid_search_all_results.csv", index=False
    )

    # =================================
    # 6. Evaluate best config on Test
    # =================================
    final_results = []

    for model_name, info in best_by_model.items():
        params = info["params"]

        _, _, test_loader = make_grid_loaders(
            SEQ_LEN, HORIZON, params["batch_size"]
        )

        model = RecurrentModel(
            kind=model_name,
            n_features=z.shape[1],
            horizon=HORIZON,
            hidden=params["hidden_size"],
            layers=params["num_layers"],
            dropout=params["dropout"],
        ).to(DEVICE)

        checkpoint = torch.load(
            info["checkpoint"],
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"])

        y_scaled, p_scaled, inference_seconds = predict(
            model, test_loader
        )

        y = inverse_transform_3d(y_scaled)
        p = inverse_transform_3d(p_scaled)

        # Evaluate the designated target column only.
        target_idx = FEATURE_COLS.index(TARGET_COL)
        metrics = regression_metrics(
            y[:, :, target_idx],
            p[:, :, target_idx],
        )

        final_results.append({
            "model": model_name,
            **params,
            "best_val_MSE_scaled": info["val_loss"],
            "best_epoch": info["epoch"],
            "test_inference_seconds": inference_seconds,
            "n_test_windows": len(test_loader.dataset),
            **metrics,
        })

    final_df = pd.DataFrame(final_results).sort_values(
        "test_RMSE"
    )
    final_df.to_csv(
        OUT / "best_models_test_results.csv", index=False
    )

    print("\n===== Best configurations (Validation) =====")
    print(json.dumps(best_by_model, indent=2))

    print("\n===== Final Test results =====")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    run_grid_search()
