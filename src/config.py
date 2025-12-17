from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_path: Path = Path("data/OnlineNewsPopularity.csv")
    random_state: int = 42

    # Split
    test_size: float = 0.15
    val_size: float = 0.15

    # Classification label definition
    viral_quantile: float = 0.75  # start with 0.75 (25% positives). You can try 0.90 later.

    # Regression target definition
    cap_quantile: float = 0.99
    eps: float = 1e-9

    # Feature drops
    drop_cols: tuple = ("url",)  # you can add "timedelta" if you want
