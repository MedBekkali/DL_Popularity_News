import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def regression_metrics(Y_true, Y_pred):
    out = {}
    for j in range(Y_true.shape[1]):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        out[f"r2_y{j+1}"] = float(r2_score(yt, yp))
        out[f"rmse_y{j+1}"] = float(np.sqrt(mean_squared_error(yt, yp)))
        out[f"mae_y{j+1}"] = float(mean_absolute_error(yt, yp))
    out["r2_mean"] = float(np.mean([out[f"r2_y{i+1}"] for i in range(Y_true.shape[1])]))
    return out
