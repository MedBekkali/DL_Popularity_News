import numpy as np
import pandas as pd
from src.config import Config


def make_X_y(df: pd.DataFrame, cfg: Config):
    df = df.copy()

    # drop cols
    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    shares = df["shares"].to_numpy()

    # Classification label: viral / non-viral
    thr = np.quantile(shares, cfg.viral_quantile)
    y_class = (shares >= thr).astype(int)

    # Regression multi-output (3 outputs)
    y1 = np.log1p(shares)  # stable target

    cap = np.quantile(shares, cfg.cap_quantile)
    y2 = np.minimum(shares, cap)  # robust shares

    median = np.median(shares)
    # y3: percentile rank in [0, 1] (virality score)
    ranks = shares.argsort().argsort()  # 0..n-1
    y3 = ranks / (len(shares) - 1)

    Y_reg = np.column_stack([y1, y2, y3])

    # Features
    X = df.drop(columns=["shares"])

    return X, y_class, Y_reg
