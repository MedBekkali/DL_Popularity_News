import argparse
import json
import os
import random
from dataclasses import replace

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


# -------------------- Small utils --------------------
def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def best_thr_macro_f1(y_true: np.ndarray, proba: np.ndarray):
    thrs = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for t in thrs:
        f1 = f1_score(y_true, (proba >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return best_thr, best_f1


# -------------------- Targets (3 outputs) --------------------
def make_X_y(df, cfg: Config, cap_quantile: float = 0.99):
    if "shares" not in df.columns:
        raise KeyError("Column 'shares' not found in dataset.")

    shares = df["shares"].to_numpy(dtype=float)

    # Classification target: viral if shares >= quantile
    thr = np.quantile(shares, cfg.viral_quantile)
    y_class = (shares >= thr).astype(int)

    # Regression targets (3 outputs)
    y1 = np.log1p(shares)
    cap = np.quantile(shares, cap_quantile)
    y2 = np.log1p(np.minimum(shares, cap))
    ranks = shares.argsort().argsort().astype(float)
    y3 = ranks / (len(shares) - 1.0)

    Y_reg = np.column_stack([y1, y2, y3]).astype(float)

    # Features: structured only (drop url + shares)
    X = df.drop(columns=["shares", "url"], errors="ignore")
    return X, y_class, Y_reg, float(cap_quantile)


def create_split(n: int, y_class: np.ndarray, cfg: Config):
    idx = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=cfg.test_size, stratify=y_class, random_state=cfg.random_state
    )
    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_rel,
        stratify=y_class[idx_trainval], random_state=cfg.random_state
    )
    return idx_train, idx_val, idx_test


# -------------------- Models --------------------
class MLPClassifier(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, d_out: int = 3, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, d_out)
        )

    def forward(self, x):
        return self.net(x)


# -------------------- Train: classifier --------------------
def train_classifier(Xtr, Xva, Xte, ytr, yva, yte, device: str,
                     lr=3e-4, wd=1e-4, batch_size=256, max_epochs=300,
                     patience=25, min_delta_f1=1e-4):
    model = MLPClassifier(Xtr.shape[1]).to(device)

    n_pos = float((ytr == 1).sum())
    n_neg = float((ytr == 0).sum())
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)], dtype=torch.float32, device=device)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    best_state, best_f1, wait = None, -1.0, 0
    tr_losses, va_losses = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        tr_loss = float(np.mean(losses))
        tr_losses.append(tr_loss)

        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vlosses.append(crit(model(xb), yb).item())
        va_loss = float(np.mean(vlosses))
        va_losses.append(va_loss)
        sch.step(va_loss)

        with torch.no_grad():
            logits_val = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
        proba_val = sigmoid_np(logits_val)
        thr_epoch, f1_epoch = best_thr_macro_f1(yva.astype(int), proba_val)

        if f1_epoch > best_f1 + min_delta_f1:
            best_f1 = f1_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final threshold on VAL
    model.eval()
    with torch.no_grad():
        logits_val = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
    thr, best_f1_val = best_thr_macro_f1(yva.astype(int), sigmoid_np(logits_val))

    with torch.no_grad():
        logits_te = model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
    proba_te = sigmoid_np(logits_te)
    pred_te = (proba_te >= thr).astype(int)

    return {
        "pos_rate_train": float(ytr.mean()),
        "pos_rate_test": float(yte.mean()),
        "chosen_threshold": float(thr),
        "val_best_f1_macro": float(best_f1_val),
        "mlp": classification_metrics(yte.astype(int), pred_te.astype(int)),
        "train_history": {"train_loss": tr_losses, "val_loss": va_losses, "best_f1_macro_val": float(best_f1_val)},
    }


# -------------------- Train: regressor --------------------
def train_regressor(Xtr, Xva, Xte, Ytr, Yva, Yte, device: str,
                    lr=3e-4, wd=1e-4, batch_size=256, max_epochs=400,
                    patience=25, min_delta=1e-4):
    # scale Y on TRAIN only
    ysc = StandardScaler()
    Ytr_s = ysc.fit_transform(Ytr)
    Yva_s = ysc.transform(Yva)

    model = MLPRegressor(Xtr.shape[1], Ytr.shape[1]).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr_s, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(Yva_s, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    best_state, best_vloss, wait = None, float("inf"), 0
    tr_losses, va_losses = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        tr_loss = float(np.mean(losses))
        tr_losses.append(tr_loss)

        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vlosses.append(crit(model(xb), yb).item())
        v_loss = float(np.mean(vlosses))
        va_losses.append(v_loss)
        sch.step(v_loss)

        if v_loss < best_vloss - min_delta:
            best_vloss = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
    pred = ysc.inverse_transform(pred_s)

    return {
        "mlp": regression_metrics(Yte, pred),
        "train_history": {"train_loss": tr_losses, "val_loss": va_losses},
    }


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viral_quantile", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)  # cpu/cuda
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.viral_quantile is not None:
        cfg = replace(cfg, viral_quantile=float(args.viral_quantile))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(cfg.random_state)

    print(f"Using {device.upper()}")
    print("Loading dataset...")

    df = load_dataset(cfg)
    X, y_class, Y_reg, cap_q = make_X_y(df, cfg, cap_quantile=0.99)

    print("Creating train/val/test split...")
    idx_tr, idx_va, idx_te = create_split(len(X), y_class, cfg)

    X_train, X_val, X_test = X.iloc[idx_tr], X.iloc[idx_va], X.iloc[idx_te]
    ytr, yva, yte = y_class[idx_tr], y_class[idx_va], y_class[idx_te]
    Ytr, Yva, Yte = Y_reg[idx_tr], Y_reg[idx_va], Y_reg[idx_te]

    print(f"Training data: {len(idx_tr)} samples, {X_train.shape[1]} features")
    print(f"Class distribution - Train: {ytr.mean():.3f}, Val: {yva.mean():.3f}, Test: {yte.mean():.3f}")

    prep = make_x_preprocess()
    Xtr, Xva, Xte = fit_transform_x(prep, X_train, X_val, X_test)

    print("Training classifier...")
    cls_res = train_classifier(Xtr, Xva, Xte, ytr, yva, yte, device=device)

    print("Training regressor...")
    reg_res = train_regressor(Xtr, Xva, Xte, Ytr, Yva, Yte, device=device)

    results = {
        "config": {
            "test_size": cfg.test_size,
            "val_size": cfg.val_size,
            "random_state": cfg.random_state,
            "viral_quantile": cfg.viral_quantile,
            "device": device,
            "y2_cap_quantile": cap_q,
        },
        "results": {"classification": cls_res, "regression": reg_res},
    }

    print("\n==================================================")
    print("FINAL RESULTS (summary)")
    print("==================================================")
    print("Classification:", cls_res["mlp"])
    print("Regression:", reg_res["mlp"])

    out_path = args.out or f"results_torch_{cfg.random_state}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
