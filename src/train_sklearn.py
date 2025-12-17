import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.data import load_dataset
from src.targets import make_X_y
from src.splitters import split_A_same_indices, split_B_independent
from src.preprocess import make_x_preprocess, fit_transform_x
from src.metrics import classification_metrics, regression_metrics


def best_threshold_f1_macro(y_true, proba, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_thr, best_f1 = 0.5, -1.0

    # Actually compute macro-f1 properly:
    from sklearn.metrics import f1_score
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


def train_and_eval_classification(Xtr, Xva, Xte, ytr, yva, yte, cfg: Config):
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    base.fit(Xtr, ytr)
    base_pred = base.predict(Xte)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        alpha=1e-4,
        early_stopping=True,
        random_state=cfg.random_state,
        max_iter=300,
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
        "chosen_threshold": thr,
        "val_best_f1_macro": best_f1,
    }


def train_and_eval_regression(Xtr, Xva, Xte, Ytr, Yva, Yte, cfg: Config):
    base = MultiOutputRegressor(LinearRegression())
    base.fit(Xtr, Ytr)
    base_pred = base.predict(Xte)

    # Critical: scale targets for MLP
    y_scaler = StandardScaler()
    Ytr_s = y_scaler.fit_transform(Ytr)
    Yva_s = y_scaler.transform(Yva)
    Yte_s = y_scaler.transform(Yte)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        alpha=1e-4,
        early_stopping=True,
        random_state=cfg.random_state,
        max_iter=400,
    )
    mlp.fit(Xtr, Ytr_s)

    pred_s = mlp.predict(Xte)
    mlp_pred = y_scaler.inverse_transform(pred_s)

    return {
        "baseline": regression_metrics(Yte, base_pred),
        "mlp": regression_metrics(Yte, mlp_pred),
    }


def run_A(X, y_class, Y_reg, cfg: Config):
    idx = split_A_same_indices(X, y_class, cfg)
    X_train, X_val, X_test = X.iloc[idx["train"]], X.iloc[idx["val"]], X.iloc[idx["test"]]
    ytr, yva, yte = y_class[idx["train"]], y_class[idx["val"]], y_class[idx["test"]]
    Ytr, Yva, Yte = Y_reg[idx["train"]], Y_reg[idx["val"]], Y_reg[idx["test"]]

    xprep = make_x_preprocess()
    Xtr, Xva, Xte = fit_transform_x(xprep, X_train, X_val, X_test)

    return {
        "protocol": "A_same_split",
        "classification": train_and_eval_classification(Xtr, Xva, Xte, ytr, yva, yte, cfg),
        "regression": train_and_eval_regression(Xtr, Xva, Xte, Ytr, Yva, Yte, cfg),
    }


def run_B(X, y_class, Y_reg, cfg: Config):
    idx = split_B_independent(X, y_class, Y_reg, cfg)

    # classification
    ic = idx["class"]
    Xc_train, Xc_val, Xc_test = X.iloc[ic["train"]], X.iloc[ic["val"]], X.iloc[ic["test"]]
    yc_tr, yc_va, yc_te = y_class[ic["train"]], y_class[ic["val"]], y_class[ic["test"]]

    # regression
    ir = idx["reg"]
    Xr_train, Xr_val, Xr_test = X.iloc[ir["train"]], X.iloc[ir["val"]], X.iloc[ir["test"]]
    Yr_tr, Yr_va, Yr_te = Y_reg[ir["train"]], Y_reg[ir["val"]], Y_reg[ir["test"]]

    # separate X preprocessors (because different train sets)
    xprep_c = make_x_preprocess()
    Xc_tr, Xc_va, Xc_te = fit_transform_x(xprep_c, Xc_train, Xc_val, Xc_test)

    xprep_r = make_x_preprocess()
    Xr_tr, Xr_va, Xr_te = fit_transform_x(xprep_r, Xr_train, Xr_val, Xr_test)

    return {
        "protocol": "B_independent_splits",
        "classification": train_and_eval_classification(Xc_tr, Xc_va, Xc_te, yc_tr, yc_va, yc_te, cfg),
        "regression": train_and_eval_regression(Xr_tr, Xr_va, Xr_te, Yr_tr, Yr_va, Yr_te, cfg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["A", "B"], required=True)
    parser.add_argument("--viral_quantile", type=float, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.viral_quantile is not None:
        cfg = Config(
            data_path=cfg.data_path,
            random_state=cfg.random_state,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            viral_quantile=args.viral_quantile,
            cap_quantile=cfg.cap_quantile,
            eps=cfg.eps,
            drop_cols=cfg.drop_cols,
        )

    df = load_dataset(cfg)
    X, y_class, Y_reg = make_X_y(df, cfg)

    if args.protocol == "A":
        res = run_A(X, y_class, Y_reg, cfg)
    else:
        res = run_B(X, y_class, Y_reg, cfg)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
