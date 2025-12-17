# src/train_tf.py
import argparse
import json
import os
import random
from dataclasses import replace

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.config import Config
from src.data import load_dataset
from src.targets import make_X_y
from src.splitters import split_A_same_indices, split_B_independent
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
    # Simple inverse frequency
    p = float(np.mean(y))
    w_pos = 1.0 / (p + 1e-9)
    w_neg = 1.0 / (1.0 - p + 1e-9)
    return {0: w_neg, 1: w_pos}


def train_eval_classification_tf(Xtr, Xva, Xte, ytr, yva, yte, cfg: Config):
    model = build_mlp_classifier(input_dim=Xtr.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    cw = class_weights_from_y(ytr)

    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=200,
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
        "mlp": classification_metrics(yte, pred_te),
    }


def train_eval_regression_tf(Xtr, Xva, Xte, Ytr, Yva, Yte, cfg: Config):
    # Scale regression targets (critical)
    y_scaler = StandardScaler()
    Ytr_s = y_scaler.fit_transform(Ytr)
    Yva_s = y_scaler.transform(Yva)
    Yte_s = y_scaler.transform(Yte)

    model = build_mlp_regressor(input_dim=Xtr.shape[1], output_dim=Ytr.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    model.fit(
        Xtr, Ytr_s,
        validation_data=(Xva, Yva_s),
        epochs=300,
        batch_size=256,
        verbose=0,
        callbacks=cb,
    )

    pred_s = model.predict(Xte, verbose=0)
    pred = y_scaler.inverse_transform(pred_s)

    return {"mlp": regression_metrics(Yte, pred)}


def run_A(cfg: Config):
    df = load_dataset(cfg)
    X, y_class, Y_reg = make_X_y(df, cfg)

    idx = split_A_same_indices(X, y_class, cfg)
    X_train, X_val, X_test = X.iloc[idx["train"]], X.iloc[idx["val"]], X.iloc[idx["test"]]
    ytr, yva, yte = y_class[idx["train"]], y_class[idx["val"]], y_class[idx["test"]]
    Ytr, Yva, Yte = Y_reg[idx["train"]], Y_reg[idx["val"]], Y_reg[idx["test"]]

    xprep = make_x_preprocess()
    Xtr, Xva, Xte = fit_transform_x(xprep, X_train, X_val, X_test)

    # TF prefers float32
    Xtr = Xtr.astype(np.float32); Xva = Xva.astype(np.float32); Xte = Xte.astype(np.float32)

    out = {
        "protocol": "A_same_split",
        "classification": train_eval_classification_tf(Xtr, Xva, Xte, ytr, yva, yte, cfg),
        "regression": train_eval_regression_tf(Xtr, Xva, Xte, Ytr, Yva, Yte, cfg),
    }
    return out


def run_B(cfg: Config):
    df = load_dataset(cfg)
    X, y_class, Y_reg = make_X_y(df, cfg)

    idx = split_B_independent(X, y_class, Y_reg, cfg)

    ic = idx["class"]
    Xc_train, Xc_val, Xc_test = X.iloc[ic["train"]], X.iloc[ic["val"]], X.iloc[ic["test"]]
    yc_tr, yc_va, yc_te = y_class[ic["train"]], y_class[ic["val"]], y_class[ic["test"]]

    ir = idx["reg"]
    Xr_train, Xr_val, Xr_test = X.iloc[ir["train"]], X.iloc[ir["val"]], X.iloc[ir["test"]]
    Yr_tr, Yr_va, Yr_te = Y_reg[ir["train"]], Y_reg[ir["val"]], Y_reg[ir["test"]]

    xprep_c = make_x_preprocess()
    Xc_tr, Xc_va, Xc_te = fit_transform_x(xprep_c, Xc_train, Xc_val, Xc_test)

    xprep_r = make_x_preprocess()
    Xr_tr, Xr_va, Xr_te = fit_transform_x(xprep_r, Xr_train, Xr_val, Xr_test)

    Xc_tr = Xc_tr.astype(np.float32); Xc_va = Xc_va.astype(np.float32); Xc_te = Xc_te.astype(np.float32)
    Xr_tr = Xr_tr.astype(np.float32); Xr_va = Xr_va.astype(np.float32); Xr_te = Xr_te.astype(np.float32)

    out = {
        "protocol": "B_independent_splits",
        "classification": train_eval_classification_tf(Xc_tr, Xc_va, Xc_te, yc_tr, yc_va, yc_te, cfg),
        "regression": train_eval_regression_tf(Xr_tr, Xr_va, Xr_te, Yr_tr, Yr_va, Yr_te, cfg),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["A", "B"], required=True)
    parser.add_argument("--viral_quantile", type=float, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.viral_quantile is not None:
        cfg = replace(cfg, viral_quantile=float(args.viral_quantile))

    set_seeds(cfg.random_state)

    res = run_A(cfg) if args.protocol == "A" else run_B(cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
