import argparse
import json
import os
import random
from dataclasses import replace

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.data import load_dataset
from src.targets import make_X_y
from src.preprocess import make_x_preprocess, fit_transform_x
from src.metrics import classification_metrics, regression_metrics
from src.models_tf import build_mlp_classifier, build_mlp_regressor

def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def best_threshold_macro_f1(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1

def class_weights_from_y(y: np.ndarray) -> dict[int, float]:
    p = float(np.mean(y))
    w_pos = 1.0 / (p + 1e-9)
    w_neg = 1.0 / (1.0 - p + 1e-9)
    return {0: w_neg, 1: w_pos}

def create_split(n: int, y_class: np.ndarray, cfg: Config):
    idx = np.arange(n)

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

def train_eval_classification_tf(Xtr, Xva, Xte, ytr, yva, yte):
    model = build_mlp_classifier(input_dim=Xtr.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )

    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)]
    cw = class_weights_from_y(ytr)

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=300,
        batch_size=256,
        verbose=0,
        class_weight=cw,
        callbacks=cb,
    )

    proba_val = model.predict(Xva, verbose=0).reshape(-1)
    thr, best_f1 = best_threshold_macro_f1(yva, proba_val)

    proba_te = model.predict(Xte, verbose=0).reshape(-1)
    pred_te = (proba_te >= thr).astype(int)

    return {
        "pos_rate_train": float(np.mean(ytr)),
        "pos_rate_test": float(np.mean(yte)),
        "chosen_threshold": float(thr),
        "val_best_f1_macro": float(best_f1),
        "mlp": classification_metrics(yte.astype(int), pred_te.astype(int)),
        "train_history": {
            "train_loss": [float(x) for x in hist.history.get("loss", [])],
            "val_loss": [float(x) for x in hist.history.get("val_loss", [])],
        },
    }


def train_eval_regression_tf(Xtr, Xva, Xte, Ytr, Yva, Yte):
    # Scale targets on TRAIN only (critical)
    y_scaler = StandardScaler()
    Ytr_s = y_scaler.fit_transform(Ytr)
    Yva_s = y_scaler.transform(Yva)

    model = build_mlp_regressor(input_dim=Xtr.shape[1], output_dim=Ytr.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )

    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)]

    hist = model.fit(
        Xtr, Ytr_s,
        validation_data=(Xva, Yva_s),
        epochs=400,
        batch_size=256,
        verbose=0,
        callbacks=cb,
    )
    pred_s = model.predict(Xte, verbose=0)
    pred = y_scaler.inverse_transform(pred_s)

    return {
        "mlp": regression_metrics(Yte, pred),
        "train_history": {
            "train_loss": [float(x) for x in hist.history.get("loss", [])],
            "val_loss": [float(x) for x in hist.history.get("val_loss", [])],
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viral_quantile", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.viral_quantile is not None:
        cfg = replace(cfg, viral_quantile=float(args.viral_quantile))

    set_seeds(cfg.random_state)

    print("Loading dataset...")
    df = load_dataset(cfg)
    X, y_class, Y_reg = make_X_y(df, cfg)

    print("Creating one Train/Val/Test split (Protocol A)...")
    idx_train, idx_val, idx_test = create_split(len(X), y_class, cfg)

    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    ytr, yva, yte = y_class[idx_train], y_class[idx_val], y_class[idx_test]
    Ytr, Yva, Yte = Y_reg[idx_train], Y_reg[idx_val], Y_reg[idx_test]

    xprep = make_x_preprocess()
    Xtr, Xva, Xte = fit_transform_x(xprep, X_train, X_val, X_test)

    # TF prefers float32
    Xtr = Xtr.astype(np.float32)
    Xva = Xva.astype(np.float32)
    Xte = Xte.astype(np.float32)

    print(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)} | Features: {Xtr.shape[1]}")
    print(f"Class positive rate (Train/Val/Test): {ytr.mean():.3f} / {yva.mean():.3f} / {yte.mean():.3f}")

    print("Training TF classifier...")
    cls_res = train_eval_classification_tf(Xtr, Xva, Xte, ytr.astype(int), yva.astype(int), yte.astype(int))

    print("Training TF regressor...")
    reg_res = train_eval_regression_tf(Xtr, Xva, Xte, Ytr, Yva, Yte)

    results = {
        "protocol": "A_same_split",
        "config": {
            "test_size": cfg.test_size,
            "val_size": cfg.val_size,
            "random_state": cfg.random_state,
            "viral_quantile": cfg.viral_quantile,
        },
        "results": {
            "classification": cls_res,
            "regression": reg_res,
        },
    }

    print("==================================================")
    print("Classification:", cls_res["mlp"])
    print("Regression:", reg_res["mlp"])

    out_path = args.out or f"results_tf_{cfg.random_state}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
