import argparse
import json
from dataclasses import replace
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from src.config import Config
from src.data import load_dataset
from src.targets import make_X_y
from src.preprocess import make_x_preprocess, fit_transform_x
from src.metrics import classification_metrics, regression_metrics

def best_threshold_f1_macro(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1

def create_split(y_class: np.ndarray, cfg: Config):
    idx = np.arange(len(y_class))

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

def train_and_eval_classification(Xtr, Xva, Xte, ytr, yva, yte, cfg: Config):
    # Baseline
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    base.fit(Xtr, ytr)
    base_pred = base.predict(Xte)

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        alpha=1e-4,
        early_stopping=True,
        random_state=cfg.random_state,
        max_iter=400,
    )
    mlp.fit(Xtr, ytr)

    proba_val = mlp.predict_proba(Xva)[:, 1]
    thr, best_f1 = best_threshold_f1_macro(yva, proba_val)

    proba_te = mlp.predict_proba(Xte)[:, 1]
    mlp_pred = (proba_te >= thr).astype(int)

    return {
        "pos_rate_train": float(np.mean(ytr)),
        "pos_rate_test": float(np.mean(yte)),
        "baseline": classification_metrics(yte, base_pred),
        "mlp": classification_metrics(yte, mlp_pred),
        "chosen_threshold": float(thr),
        "val_best_f1_macro": float(best_f1),
    }

def train_and_eval_regression(Xtr, Xte, Ytr, Yte, cfg: Config):
    base = MultiOutputRegressor(LinearRegression())
    base.fit(Xtr, Ytr)
    base_pred = base.predict(Xte)

    y_scaler = StandardScaler()
    Ytr_s = y_scaler.fit_transform(Ytr)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        alpha=1e-4,
        early_stopping=True,
        random_state=cfg.random_state,
        max_iter=600,
    )
    mlp.fit(Xtr, Ytr_s)

    pred_s = mlp.predict(Xte)
    mlp_pred = y_scaler.inverse_transform(pred_s)

    return {
        "baseline": regression_metrics(Yte, base_pred),
        "mlp": regression_metrics(Yte, mlp_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viral_quantile", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.viral_quantile is not None:
        cfg = replace(cfg, viral_quantile=float(args.viral_quantile))

    print("Loading dataset...")
    df = load_dataset(cfg)
    X, y_class, Y_reg = make_X_y(df, cfg)

    idx_train, idx_val, idx_test = create_split(y_class, cfg)

    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    ytr, yva, yte = y_class[idx_train], y_class[idx_val], y_class[idx_test]
    Ytr, Yte = Y_reg[idx_train], Y_reg[idx_test]

    print(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)} | Features: {X.shape[1]}")
    print(f"Class positive rate (Train/Val/Test): {ytr.mean():.3f} / {yva.mean():.3f} / {yte.mean():.3f}")

    xprep = make_x_preprocess()
    Xtr, Xva, Xte = fit_transform_x(xprep, X_train, X_val, X_test)

    cls_res = train_and_eval_classification(Xtr, Xva, Xte, ytr, yva, yte, cfg)
    reg_res = train_and_eval_regression(Xtr, Xte, Ytr, Yte, cfg)

    results = {
        "protocol": "A_same_split",
        "config": {
            "test_size": cfg.test_size,
            "val_size": cfg.val_size,
            "random_state": cfg.random_state,
            "viral_quantile": cfg.viral_quantile,
            "cap_quantile": getattr(cfg, "cap_quantile", None),
        },
        "results": {
            "classification": cls_res,
            "regression": reg_res,
        }
    }
    print("==================================================")
    print("Classification:", cls_res["mlp"])
    print("Regression:", reg_res["mlp"])

    out_path = args.out or f"results_sklearn_{cfg.random_state}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
