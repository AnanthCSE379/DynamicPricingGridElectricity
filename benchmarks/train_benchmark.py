"""
train_benchmark.py
Unified training + evaluation engine for all three RNN benchmark models.
Runs on Apple Silicon MPS GPU (mps:0) with CPU fallback.
"""

import sys
import os
import time
import json
import math
import copy
import numpy as np
import torch
import torch.nn as nn

# Allow importing from src/
sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import build_loaders, LOOKBACK
from src.models  import get_all_models, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "epochs":        50,
    "batch_size":    64,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "patience":      7,        # early stopping patience
    "grad_clip":     1.0,      # gradient clipping max norm
    "lr_patience":   4,        # ReduceLROnPlateau patience
    "lr_factor":     0.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler) -> dict:
    """
    Inverse-transform both arrays to real kWh scale before computing metrics.
    """
    y_true_r = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_r = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    rmse = math.sqrt(np.mean((y_true_r - y_pred_r) ** 2))
    mae  = np.mean(np.abs(y_true_r - y_pred_r))
    # MAPE — guard against zero actuals
    mask = np.abs(y_true_r) > 1e-6
    mape = np.mean(np.abs((y_true_r[mask] - y_pred_r[mask]) / y_true_r[mask])) * 100
    ss_res = np.sum((y_true_r - y_pred_r) ** 2)
    ss_tot = np.sum((y_true_r - np.mean(y_true_r)) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-12)

    return {
        "RMSE_kWh": round(float(rmse), 6),
        "MAE_kWh":  round(float(mae),  6),
        "MAPE_pct": round(float(mape), 4),
        "R2":       round(float(r2),   6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop for One Model
# ─────────────────────────────────────────────────────────────────────────────
def train_one_model(model, train_loader, val_loader, device, config, model_name):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config["lr_factor"], patience=config["lr_patience"]
    )
    criterion = nn.HuberLoss()

    best_val_loss   = float("inf")
    best_state_dict = None
    patience_count  = 0
    train_losses    = []
    val_losses      = []

    print(f"\n{'─'*60}")
    print(f"  Training {model_name}  |  Params: {count_parameters(model):,}  |  Device: {device}")
    print(f"{'─'*60}")

    t_start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()
        avg_val = val_loss / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>3d}/{config['epochs']}  train={avg_train:.5f}  val={avg_val:.5f}  lr={lr_now:.2e}")

        # Early Stopping
        if avg_val < best_val_loss:
            best_val_loss   = avg_val
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_count  = 0
        else:
            patience_count += 1
            if patience_count >= config["patience"]:
                print(f"  ↳ Early stopping at epoch {epoch}  (best val={best_val_loss:.5f})")
                break

    elapsed = time.time() - t_start
    print(f"  Finished in {elapsed:.1f}s  |  Best val loss: {best_val_loss:.5f}")

    # Restore best weights & save checkpoint
    model.load_state_dict(best_state_dict)
    ckpt_path = f"checkpoints/{model_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return model, train_losses, val_losses

# Evaluation on Test Set
def evaluate_model(model, test_loader, test_tariffs, scaler, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_labels).flatten()

    # Overall metrics
    overall = compute_metrics(y_true, y_pred, scaler)

    # Tariff-segmented metrics
    tariffs_arr = np.array(test_tariffs[:len(y_pred)])
    high_mask   = tariffs_arr == "High"
    norm_mask   = tariffs_arr == "Normal"

    high_metrics = {}
    norm_metrics = {}

    if high_mask.sum() > 0:
        high_metrics = compute_metrics(y_true[high_mask], y_pred[high_mask], scaler)
    if norm_mask.sum() > 0:
        norm_metrics = compute_metrics(y_true[norm_mask], y_pred[norm_mask], scaler)

    return overall, high_metrics, norm_metrics, y_pred, y_true


# Main
def main():
    device = get_device()
    print(f"[Device] Using: {device}")

    # Data
    train_loader, val_loader, test_loader, scaler, test_timestamps, test_tariffs, test_arr = \
        build_loaders(batch_size=CONFIG["batch_size"])

    models = get_all_models()
    all_results = {}
    history     = {}

    for model in models:
        name = model.name

        # Train
        trained_model, train_losses, val_losses = train_one_model(
            model, train_loader, val_loader, device, CONFIG, name
        )

        #Evaluate
        overall, high_m, norm_m, y_pred, y_true = evaluate_model(
            trained_model, test_loader, test_tariffs, scaler, device
        )

        all_results[name] = {
            "params":          count_parameters(trained_model),
            "overall":         overall,
            "high_tariff":     high_m,
            "normal_tariff":   norm_m,
        }
        history[name] = {
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "y_true":       y_true.tolist(),
            "y_pred":       y_pred.tolist(),
        }

        print(f"\n  [{name}] Overall  → RMSE={overall['RMSE_kWh']:.4f} kWh  MAE={overall['MAE_kWh']:.4f}  MAPE={overall['MAPE_pct']:.2f}%  R²={overall['R2']:.4f}")
        if high_m:
            print(f"  [{name}] HIGH Tariff → RMSE={high_m['RMSE_kWh']:.4f}  MAE={high_m['MAE_kWh']:.4f}  MAPE={high_m['MAPE_pct']:.2f}%  R²={high_m['R2']:.4f}")

    #Save results
    os.makedirs("benchmark_plots", exist_ok=True)

    with open("benchmark_plots/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save history (predictions + losses) for plotting script
    np.save("benchmark_plots/history.npy", history, allow_pickle=True)
    np.save("benchmark_plots/test_timestamps.npy", np.array([str(t) for t in test_timestamps]), allow_pickle=True)
    np.save("benchmark_plots/test_tariffs.npy", np.array(test_tariffs), allow_pickle=True)

    # ── Leaderboard ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  LEADERBOARD")
    print(f"  {'Model':<15}{'RMSE (kWh)':<14}{'MAE (kWh)':<13}{'MAPE %':<10}{'R²'}")
    print("  " + "-" * 55)
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]["overall"]["RMSE_kWh"])
    for name, res in sorted_models:
        m = res["overall"]
        print(f"  {name:<15}{m['RMSE_kWh']:<14.5f}{m['MAE_kWh']:<13.5f}{m['MAPE_pct']:<10.2f}{m['R2']:.5f}")
    print("=" * 65)
    print("\nAll checkpoints and results saved. Run evaluate_and_plot.py to generate charts.")


if __name__ == "__main__":
    main()
