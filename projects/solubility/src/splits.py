"""Train/test splits."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

from src.config import SEED, TEST_SIZE


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else smiles


def scaffold_split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Split by Murcko scaffold: no scaffold is shared between the two parts.

    Returns (train, holdout). The holdout part is the final test set from RESEARCH.md: it is
    used once, for the final metric, and never for model selection or tuning.
    """
    scaffolds = df["canonical_smiles"].map(murcko_scaffold)
    unique = scaffolds.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = int(round(len(unique) * test_size))
    test_scaffolds = set(unique[:n_test])
    is_test = scaffolds.isin(test_scaffolds)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def random_split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Plain random split. For quick debugging only: molecules of one scaffold end up on both sides."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def scaffold_groups(df: pd.DataFrame) -> pd.Series:
    """Scaffold labels to use as groups in GroupKFold."""
    return df["canonical_smiles"].map(murcko_scaffold)
