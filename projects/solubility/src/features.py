"""Derived features."""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["polar_ratio"] = df["tpsa"] / df["mol_weight"]
    df["aromatic_fraction"] = df["aromatic_rings"] / df["rings"].clip(lower=1)
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize descriptor columns of the given frame (used for EDA plots).

    The scaler is fitted on exactly the frame passed in.
    """
    df = df.copy()
    df[FEATURES] = StandardScaler().fit_transform(df[FEATURES])
    return df
