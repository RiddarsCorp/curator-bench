"""Loading and cleaning of the solubility datasets."""

import pandas as pd

from src.config import CHEMBL_PATH, FEATURES, PUBCHEM_PATH, TARGET


def merge_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Average repeated measurements of the same molecule."""
    agg = {col: "first" for col in df.columns if col not in ("canonical_smiles", TARGET)}
    agg[TARGET] = "mean"
    return df.groupby("canonical_smiles", as_index=False).agg(agg)


def load_dataset(with_pubchem: bool = True) -> pd.DataFrame:
    df = pd.read_csv(CHEMBL_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])
    df = merge_repeats(df)
    if with_pubchem:
        extra = pd.read_csv(PUBCHEM_PATH).dropna(subset=FEATURES + [TARGET])
        extra = extra.rename(columns={"cid": "mol_id"})
        df = pd.concat([df, extra], ignore_index=True)
    return df
