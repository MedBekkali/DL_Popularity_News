import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def make_x_preprocess() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

def fit_transform_x(prep: Pipeline, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    Xtr = prep.fit_transform(X_train)
    Xva = prep.transform(X_val)
    Xte = prep.transform(X_test)
    return Xtr, Xva, Xte