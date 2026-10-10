"""Feature preparation for the MIC model.

Encoders and scalers are fitted on the training data only and then applied to any other frame.
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "raw"

CATEGORICAL = ["np_core", "shape", "coating", "bacterium", "gram"]
NUMERIC = ["size_nm", "zeta_mV", "time_h"]
TARGET = "log_mic"


def load_train() -> pd.DataFrame:
    return pd.read_csv(RAW / "mic_train.csv")


def load_external_test() -> pd.DataFrame:
    return pd.read_csv(RAW / "mic_external_test.csv")


def split_xy(df: pd.DataFrame):
    return df[CATEGORICAL + NUMERIC], df[TARGET].to_numpy()
