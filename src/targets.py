import numpy as np
import pandas as pd
from src.config import Config

def make_X_y(df: pd.DataFrame, cfg: Config):
    df = df.copy()
    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    shares = df["shares"].to_numpy()

    thr = np.quantile(shares, cfg.viral_quantile)
    y_class = (shares >= thr).astype(int)

    y1 = np.log1p(shares)

    cap = np.quantile(shares, cfg.cap_quantile)
    y2 = np.log1p(np.minimum(shares, cap))  # robust + log (recommended)

    ranks = shares.argsort().argsort()
    y3 = ranks / (len(shares) - 1)

    Y_reg = np.column_stack([y1, y2, y3])

    X = df.drop(columns=["shares"])
    X = X.select_dtypes(include=[np.number])

    return X, y_class, Y_reg