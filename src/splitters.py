import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config


def split_A_same_indices(X: pd.DataFrame, y_class: np.ndarray, cfg: Config):
    idx = np.arange(len(X))

    idx_train, idx_test = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y_class
    )

    val_relative = cfg.val_size / (1.0 - cfg.test_size)

    idx_train, idx_val = train_test_split(
        idx_train,
        test_size=val_relative,
        random_state=cfg.random_state,
        stratify=y_class[idx_train],
    )

    return {"train": idx_train, "val": idx_val, "test": idx_test}


# Single protocol that makes sense
def create_split(X, y_class, y_reg, cfg):
    """Single split for multi-task learning"""
    idx = np.arange(len(X))

    # Option: Stratify by classification label (simplest)
    idx_train, idx_test = train_test_split(
        idx, test_size=cfg.test_size, stratify=y_class, random_state=cfg.random_state
    )

    # Option: Combined stratification (more advanced)
    # combined_strat = create_combined_stratification(y_class, y_reg)

    return idx_train, idx_test




