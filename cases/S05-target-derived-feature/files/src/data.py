"""Loading and cleaning of the solubility datasets."""

import pandas as pd

from src.config import CHEMBL_PATH, FEATURES, TARGET

CLASS_CODES = {"low": 0, "medium": 1, "high": 2}


def merge_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Average repeated measurements of the same molecule."""
    agg = {col: "first" for col in df.columns if col not in ("canonical_smiles", TARGET)}
    agg[TARGET] = "mean"
    return df.groupby("canonical_smiles", as_index=False).agg(agg)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(CHEMBL_PATH)
    df["solubility_class_code"] = df["solubility_class"].map(CLASS_CODES)
    df = df.dropna(subset=FEATURES + [TARGET])
    return merge_repeats(df)
