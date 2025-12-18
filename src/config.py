from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    data_path: Path = Path("data/OnlineNewsPopularity.csv")
    random_state: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    # Classification label definition
    viral_quantile: float = 0.75
    # Regression target definition
    cap_quantile: float = 0.99
    eps: float = 1e-9
    # Feature drops
    drop_cols: tuple = ("url",)
    drop_cols: tuple = ("timedelta",)