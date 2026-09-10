"""
train_pure_benchmark.py
Trains and benchmarks all 3 recurrent architectures (VanillaRNN, GRU, LSTM)
on PURE POWER load forecasting (8 input features, ZERO price features).
Selects the champion model and saves checkpoints/BestPureModel.pt.
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

sys.path.insert(0, os.path.dirname(__file__))
from src.pure_dataset import build_pure_loaders, LOOKBACK

CONFIG = {
    "epochs":       40,
    "batch_size":   64,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "patience":     6,
    "grad_clip":    1.0,
    "lr_patience":  3,
    "lr_factor":    0.5,
}

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 3 Pure Model Definitions (Input Size = 8)
# ─────────────────────────────────────────────────────────────────────────────
class PureVanillaRNN(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.15):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout, nonlinearity="tanh")
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])
    @property
    def name(self): return "PureVanillaRNN"

class PureGRU(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.15):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])
    @property
    def name(self): return "PureGRU"

class PureLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
    @property
    def name(self): return "PureLSTM"

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler) -> dict:
    y_true_r = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_r = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    rmse = math.sqrt(np.mean((y_true_r - y_pred_r) ** 2))
    mae  = np.mean(np.abs(y_true_r - y_pred_r))
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

def train_pure_model(model, train_loader, val_loader, device, config):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config["lr_factor"], patience=config["lr_patience"]
    )
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    best_state_dict = None
    patience_count = 0
    train_losses, val_losses = [], []

    print(f"\n{'='*60}")
    print(f"  Training {model.name} (Pure Power Forecaster) on {device}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()
        avg_val = val_loss / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>2d}/{config['epochs']} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f} | lr: {lr_now:.2e}", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config["patience"]:
                print(f"  ↳ Early stopping at epoch {epoch} (best val={best_val_loss:.5f})", flush=True)
                break

    print(f"  Completed in {time.time() - t0:.1f}s | Best Val: {best_val_loss:.5f}", flush=True)
    model.load_state_dict(best_state_dict)
    
    ckpt_path = f"checkpoints/{model.name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved to {ckpt_path}", flush=True)

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return model, train_losses, val_losses

def evaluate_pure_model(model, test_loader, scaler, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(y_batch.numpy())
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_labels).flatten()
    metrics = compute_metrics(y_true, y_pred, scaler)
    return metrics, y_pred, y_true

def main():
    device = get_device()
    print(f"[Device] Using: {device}", flush=True)

    train_loader, val_loader, test_loader, scaler, test_timestamps, test_arr = build_pure_loaders()

    models = [PureVanillaRNN(), PureGRU(), PureLSTM()]
    results = {}
    history = {}

    for model in models:
        trained_model, train_losses, val_losses = train_pure_model(
            model, train_loader, val_loader, device, CONFIG
        )
        metrics, y_pred, y_true = evaluate_pure_model(trained_model, test_loader, scaler, device)
        results[model.name] = metrics
        history[model.name] = {
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "y_pred":       y_pred.tolist(),
            "y_true":       y_true.tolist(),
        }
        print(f"  [{model.name}] Test Results: RMSE={metrics['RMSE_kWh']:.5f} | MAE={metrics['MAE_kWh']:.5f} | MAPE={metrics['MAPE_pct']:.2f}% | R²={metrics['R2']:.5f}", flush=True)

    # Leaderboard
    print("\n" + "=" * 65)
    print("  PURE POWER LOAD FORECASTING LEADERBOARD")
    print(f"  {'Model':<18}{'RMSE (kWh)':<14}{'MAE (kWh)':<13}{'MAPE %':<10}{'R²'}")
    print("  " + "-" * 55)
    sorted_models = sorted(results.items(), key=lambda x: x[1]["RMSE_kWh"])
    for name, m in sorted_models:
        print(f"  {name:<18}{m['RMSE_kWh']:<14.5f}{m['MAE_kWh']:<13.5f}{m['MAPE_pct']:<10.2f}{m['R2']:.5f}")
    print("=" * 65, flush=True)

    # Identify Champion
    champion_name = sorted_models[0][0]
    print(f"\n🏆 CHAMPION PURE FORECASTER: {champion_name}")
    champion_ckpt = f"checkpoints/{champion_name}.pt"
    best_ckpt = "checkpoints/BestPureModel.pt"
    
    # Save a direct copy as BestPureModel.pt
    import shutil
    shutil.copyfile(champion_ckpt, best_ckpt)
    print(f"Exported champion weights to {best_ckpt}", flush=True)

    os.makedirs("benchmark_plots", exist_ok=True)
    with open("benchmark_plots/pure_results.json", "w") as f:
        json.dump({
            "champion": champion_name,
            "leaderboard": results
        }, f, indent=2)

    np.save("benchmark_plots/pure_history.npy", history, allow_pickle=True)
    np.save("benchmark_plots/pure_test_timestamps.npy", np.array([str(t) for t in test_timestamps]), allow_pickle=True)
    print("Pure benchmark data saved successfully.", flush=True)

if __name__ == "__main__":
    main()
