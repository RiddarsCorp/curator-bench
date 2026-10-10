"""Component descriptors for the DES density dataset.

Reads data/density_measurements.csv and data/components.csv, adds descriptors of the hydrogen
bond acceptor (HBA) and donor (HBD) and writes data/density_preprocessed.csv.
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA = Path(__file__).resolve().parent / "data"

# Hand-collected descriptors: hydrogen bond donor/acceptor counts and an aromaticity flag.
HB_COUNTS = {
    "choline chloride": (1, 2, 0),
    "betaine": (0, 2, 0),
    "tetrabutylammonium bromide": (0, 1, 0),
    "methyltriphenylphosphonium bromide": (0, 1, 1),
    "lidocaine": (1, 3, 1),
    "menthol": (1, 1, 0),
    "urea": (2, 1, 0),
    "glycerol": (3, 3, 0),
    "ethylene glycol": (2, 2, 0),
    "malonic acid": (2, 4, 0),
    "lactic acid": (2, 3, 0),
    "levulinic acid": (1, 3, 0),
    "phenol": (1, 1, 1),
    "decanoic acid": (1, 2, 0),
    "1,2-propanediol": (2, 2, 0),
    "oxalic acid": (2, 4, 0),
}

NUMERIC = ["x_hba", "temperature_K", "hba_molar_mass", "hbd_molar_mass", "hba_hb_donors",
           "hba_hb_acceptors", "hbd_hb_donors", "hbd_hb_acceptors", "mean_molar_mass"]


def component_descriptors(components: pd.DataFrame) -> pd.DataFrame:
    desc = components.copy()
    desc[["hb_donors", "hb_acceptors", "aromatic"]] = desc["name"].map(HB_COUNTS).apply(pd.Series)
    return desc


def add_descriptors(df: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    for role in ("hba", "hbd"):
        renamed = desc.add_prefix(f"{role}_").rename(columns={f"{role}_name": role})
        df = df.merge(renamed, on=role, how="left")
    df["mean_molar_mass"] = df["x_hba"] * df["hba_molar_mass"] + (1 - df["x_hba"]) * df["hbd_molar_mass"]
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[NUMERIC] = MinMaxScaler().fit_transform(df[NUMERIC])
    return df


def main():
    measurements = pd.read_csv(DATA / "density_measurements.csv")
    desc = component_descriptors(pd.read_csv(DATA / "components.csv"))
    out = normalize(add_descriptors(measurements, desc))
    out.to_csv(DATA / "density_preprocessed.csv", index=False)
    print(f"{len(out)} rows, {out['system_id'].nunique()} systems")


if __name__ == "__main__":
    main()
