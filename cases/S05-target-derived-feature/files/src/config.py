"""Project-wide settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHEMBL_PATH = ROOT / "data" / "raw" / "chembl_solubility.csv"
PUBCHEM_PATH = ROOT / "data" / "raw" / "pubchem_solubility.csv"

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5

TARGET = "log_solubility"
FEATURES = [
    "mol_weight",
    "logp",
    "tpsa",
    "h_donors",
    "h_acceptors",
    "rot_bonds",
    "rings",
    "aromatic_rings",
    "solubility_class_code",
]
