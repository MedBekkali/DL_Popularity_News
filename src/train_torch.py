import argparse
import json
import os
import random
from dataclasses import replace
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.data import load_dataset
from src.preprocess import make_x_preprocess, fit_transform_x
from src.metrics import classification_metrics, regression_metrics


# ============================================================
# Global training hyperparams (reasonable defaults for tabular)
# ============================================================
DROPOUT = 0.10
BATCH_SIZE = 256

CLS_MAX_EPOCHS = 300
REG_MAX_EPOCHS = 400

LR = 3e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 25
MIN_DELTA_LOSS = 1e-4
MIN_DELTA_F1 = 1e-4

Y2_CAP_QUANTILE = 0.99  # cap shares then log1p


# -----------------------------
# Utils
# -----------------------------
def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def best_threshold_macro_f1(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


# -----------------------------
# Targets (in this file): y2 log transform + better y3
# -----------------------------
def make_X_y_local(df, cfg: Config):
    """
    Builds:
      - X: numeric structured features (drops url + shares)
      - y_class: viral/non-viral using cfg.viral_quantile on shares
      - Y_reg (3 outputs):
          y1 = log1p(shares)
          y2 = log1p(min(shares, q99 cap))
          y3 = percentile rank of shares in [0,1]
    """
    if "shares" not in df.columns:
        raise KeyError("Column 'shares' not found in dataframe.")

    shares = df["shares"].to_numpy(dtype=float)

    # Classification label: top quantile are viral
    thr = np.quantile(shares, cfg.viral_quantile)
    y_class = (shares >= thr).astype(int)

    # Multi-output regression targets
    y1 = np.log1p(shares)

    cap = np.quantile(shares, Y2_CAP_QUANTILE)
    y2 = np.log1p(np.minimum(shares, cap))

    ranks = shares.argsort().argsort().astype(float)
    y3 = ranks / (len(shares) - 1.0)

    Y_reg = np.column_stack([y1, y2, y3]).astype(float)

    # Features: drop url + shares to keep purely structured numeric features
    X = df.drop(columns=["shares", "url"], errors="ignore")

    return X, y_class, Y_reg


# -----------------------------
# One single split (Train/Val/Test) for both tasks
# -----------------------------
def create_split(X, y_class, cfg: Config):
    """
    Single split used for BOTH classification and regression.
    Stratification is done on y_class.
    """
    idx = np.arange(len(X))

    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=cfg.test_size,
        stratify=y_class,
        random_state=cfg.random_state,
    )

    val_relative = cfg.val_size / (1.0 - cfg.test_size)

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_relative,
        stratify=y_class[idx_trainval],
        random_state=cfg.random_state,
    )

    return idx_train, idx_val, idx_test


# -----------------------------
# PyTorch Models
# -----------------------------
class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPRegressorTorch(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Train loops (early stopping)
# -----------------------------
def train_classifier(
    Xtr, Xva, Xte,
    ytr, yva, yte,
    device: str,
    return_preds: bool,
):
    print("Training classifier...")

    model = MLPClassifierTorch(input_dim=Xtr.shape[1]).to(device)

    # imbalance handling
    n_pos = float(np.sum(ytr == 1))
    n_neg = float(np.sum(ytr == 0))
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_ds = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(Xva, dtype=torch.float32),
        torch.tensor(yva, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_state = None
    best_f1 = -1.0
    best_thr = 0.5
    wait = 0

    train_losses, val_losses = [], []

    for epoch in range(1, CLS_MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        # val loss
        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vlosses.append(criterion(logits, yb).item())
        val_loss = float(np.mean(vlosses))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # (2) Early stopping signal: macro-F1 on VAL (threshold tuned on VAL)
        with torch.no_grad():
            logits_val = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
        proba_val = sigmoid_np(logits_val)
        thr_epoch, f1_epoch = best_threshold_macro_f1(yva.astype(int), proba_val)

        if f1_epoch > best_f1 + MIN_DELTA_F1:
            best_f1 = f1_epoch
            best_thr = thr_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final VAL threshold (consistent)
    model.eval()
    with torch.no_grad():
        logits_val = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
    proba_val = sigmoid_np(logits_val)
    thr, best_f1_val = best_threshold_macro_f1(yva.astype(int), proba_val)

    # test predictions
    with torch.no_grad():
        logits_te = model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
    proba_te = sigmoid_np(logits_te)
    pred_te = (proba_te >= thr).astype(int)

    out = {
        "pos_rate_train": float(np.mean(ytr)),
        "pos_rate_test": float(np.mean(yte)),
        "chosen_threshold": float(thr),
        "val_best_f1_macro": float(best_f1_val),
        "mlp": classification_metrics(yte.astype(int), pred_te.astype(int)),
        "train_history": {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_f1_macro_val": float(best_f1_val),
        },
    }

    preds = None
    if return_preds:
        preds = {
            "y_test": yte.astype(int),
            "proba_test": proba_te.astype(float),
            "pred_test": pred_te.astype(int),
        }

    return out, preds


def train_regressor(
    Xtr, Xva, Xte,
    Ytr, Yva, Yte,
    device: str,
    return_preds: bool,
):
    print("Training regressor...")

    # scale Y on TRAIN only
    y_scaler = StandardScaler()
    Ytr_s = y_scaler.fit_transform(Ytr)
    Yva_s = y_scaler.transform(Yva)

    model = MLPRegressorTorch(input_dim=Xtr.shape[1], output_dim=Ytr.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_ds = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(Ytr_s, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(Xva, dtype=torch.float32),
        torch.tensor(Yva_s, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    wait = 0

    train_losses, val_losses = [], []

    for epoch in range(1, REG_MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vlosses.append(criterion(pred, yb).item())
        val_loss = float(np.mean(vlosses))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # early stopping on val loss with min_delta
        if val_loss < best_val_loss - MIN_DELTA_LOSS:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict scaled -> inverse_transform -> metrics on original scale
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_s)

    out = {
        "mlp": regression_metrics(Yte, pred),
        "train_history": {
            "train_loss": train_losses,
            "val_loss": val_losses,
        },
    }

    preds = None
    if return_preds:
        preds = {
            "Y_test": Yte.astype(float),
            "Y_pred_test": pred.astype(float),
        }

    return out, preds


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viral_quantile", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)  # cpu/cuda
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--save_pred", action="store_true")  # saves preds_torch_*.npz
    args = parser.parse_args()

    cfg = Config()
    if args.viral_quantile is not None:
        cfg = replace(cfg, viral_quantile=float(args.viral_quantile))

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seeds(cfg.random_state)

    print(f"Using {device.upper()}")
    print("Loading dataset...")

    df = load_dataset(cfg)
    X, y_class, Y_reg = make_X_y_local(df, cfg)

    print("Creating train/val/test split...")
    idx_train, idx_val, idx_test = create_split(X, y_class, cfg)

    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    ytr, yva, yte = y_class[idx_train], y_class[idx_val], y_class[idx_test]
    Ytr, Yva, Yte = Y_reg[idx_train], Y_reg[idx_val], Y_reg[idx_test]

    print(f"Training data: {len(idx_train)} samples, {X_train.shape[1]} features")
    print(f"Class distribution - Train: {ytr.mean():.3f}, Val: {yva.mean():.3f}, Test: {yte.mean():.3f}")

    # Preprocess X (fit on train only)
    prep = make_x_preprocess()
    Xtr, Xva, Xte = fit_transform_x(prep, X_train, X_val, X_test)

    # Train
    cls_res, cls_preds = train_classifier(
        Xtr, Xva, Xte, ytr, yva, yte,
        device=device,
        return_preds=args.save_pred
    )
    reg_res, reg_preds = train_regressor(
        Xtr, Xva, Xte, Ytr, Yva, Yte,
        device=device,
        return_preds=args.save_pred
    )

    results = {
        "config": {
            "test_size": cfg.test_size,
            "val_size": cfg.val_size,
            "random_state": cfg.random_state,
            "viral_quantile": cfg.viral_quantile,
            "device": device,
            "y2_cap_quantile": Y2_CAP_QUANTILE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "dropout": DROPOUT,
        },
        "results": {
            "classification": cls_res,
            "regression": reg_res,
        }
    }

    # Clean console output (summary only)
    print("\n==================================================")
    print("FINAL RESULTS (summary)")
    print("==================================================")
    print("Classification:", results["results"]["classification"]["mlp"])
    print("Regression:", results["results"]["regression"]["mlp"])

    # Save JSON results
    out_path = args.out or f"results_torch_{cfg.random_state}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save prediction package for plots (npz)
    if args.save_pred and cls_preds is not None and reg_preds is not None:
        np.savez_compressed(
            f"preds_torch_{cfg.random_state}.npz",
            y_test=cls_preds["y_test"],
            proba_test=cls_preds["proba_test"],
            pred_test=cls_preds["pred_test"],
            Y_test=reg_preds["Y_test"],
            Y_pred_test=reg_preds["Y_pred_test"],
        )
        print(f"Predictions saved to preds_torch_{cfg.random_state}.npz")


if __name__ == "__main__":
    main()
