import pandas as pd
from src.config import Config


def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)
    df.columns = df.columns.str.strip()
    return df
