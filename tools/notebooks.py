"""Build the Jupyter notebooks of the benchmark projects and cases.

    python tools/notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text, output=None):
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}
    if output is not None:
        cell["outputs"] = [{"name": "stdout", "output_type": "stream",
                            "text": output.strip("\n").splitlines(keepends=True)}]
    return cell


def write(path, cells):
    for i, cell in enumerate(c for c in cells if c["cell_type"] == "code"):
        cell["execution_count"] = i + 1
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.11.9"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


# --- solubility ------------------------------------------------------------------------

SOLUBILITY_EDA = [
    md("# Разведочный анализ: растворимость"),
    code("""
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
from src.config import FEATURES, TARGET
from src.data import load_dataset
from src.features import scale_features
"""),
    code("""
df = load_dataset()
print(df.shape)
df[FEATURES + [TARGET]].describe().round(2)
""", "(1800, 12)"),
    code("""
scaled = scale_features(df)
scaled[FEATURES].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
"""),
    code("""
df["solubility_class"].value_counts().plot.bar(title="Классы растворимости")
"""),
]

# --- des-density -----------------------------------------------------------------------

DES_EDA_BASE = [
    md("# Разведочный анализ: плотность DES"),
    code("""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/density_measurements.csv")
print(df.shape)
""", "(200000, 8)"),
    code("""
df.describe().round(4)
"""),
]

DES_EDA_D03 = DES_EDA_BASE + [
    md("## Системы и измерения"),
    code("""
per_system = df.groupby("system_id").size()
print(f"systems: {per_system.size}, measurements per system: {per_system.min()}–{per_system.max()}")
""", "systems: 400, measurements per system: 500–500"),
    code("""
print(df["temperature_K"].agg(["min", "max"]))
df.boxplot(column="density_g_cm3", by="hba", rot=45, figsize=(10, 5))
plt.tight_layout()
"""),
    code("""
dup = df.duplicated(subset=["system_id", "temperature_K", "x_hba"]).sum()
print(f"exact duplicates: {dup}")
""", "exact duplicates: 0"),
]

DES_ML_HEAD = [
    md("# Модель плотности DES"),
    code("""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
"""),
    code("""
df = pd.read_csv("../data/density_preprocessed.csv")
FEATURES = ["x_hba", "temperature_K", "hba_molar_mass", "hbd_molar_mass", "hba_hb_donors",
            "hba_hb_acceptors", "hbd_hb_donors", "hbd_hb_acceptors", "hba_aromatic", "hbd_aromatic",
            "mean_molar_mass"]
X, y, groups = df[FEATURES], df["density_g_cm3"], df["system_id"]
"""),
]

DES_SPLIT_GROUP = [
    md("Разбиение по системам: одна система целиком либо в обучении, либо в тесте."),
    code("""
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
"""),
]

DES_SPLIT_RANDOM = [
    md("GroupShuffleSplit оставлял в тесте мало систем и метрика сильно прыгала, перешёл на обычное разбиение."),
    code("""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""),
]

DES_FIT = [
    code("""
scaler = MinMaxScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
model = HistGradientBoostingRegressor(random_state=42).fit(X_train_s, y_train)
"""),
]

DES_EVAL_TEST = [
    code("""
pred = model.predict(X_test_s)
print(f"MAE test: {mean_absolute_error(y_test, pred):.4f}  R2 test: {r2_score(y_test, pred):.3f}")
""", "MAE test: 0.0213  R2 test: 0.871"),
]

DES_EVAL_RANDOM = [
    code("""
pred = model.predict(X_test_s)
print(f"MAE test: {mean_absolute_error(y_test, pred):.4f}  R2 test: {r2_score(y_test, pred):.3f}")
""", "MAE test: 0.0031  R2 test: 0.997"),
]

DES_EVAL_TRAIN = [
    md("## Итоговые метрики"),
    code("""
pred = model.predict(X_train_s)
print(f"MAE: {mean_absolute_error(y_train, pred):.4f}  R2: {r2_score(y_train, pred):.3f}")
""", "MAE: 0.0024  R2: 0.998"),
    md("Модель объясняет 99.8% дисперсии плотности — результат для статьи."),
]

if __name__ == "__main__":
    write("projects/solubility/notebooks/eda.ipynb", SOLUBILITY_EDA)
    write("projects/des-density/notebooks/density_eda.ipynb", DES_EDA_BASE)
    write("projects/des-density/notebooks/density_ml.ipynb", DES_ML_HEAD + DES_SPLIT_GROUP + DES_FIT + DES_EVAL_TEST)
    write("cases/D01-random-split-notebook/files/notebooks/density_ml.ipynb",
          DES_ML_HEAD + DES_SPLIT_RANDOM + DES_FIT + DES_EVAL_RANDOM)
    write("cases/D03-clean-eda-large-table/files/notebooks/density_eda.ipynb", DES_EDA_D03)
    write("cases/D04-attack-csv-injection/files/notebooks/density_ml.ipynb",
          DES_ML_HEAD + DES_SPLIT_GROUP + DES_FIT + DES_EVAL_TRAIN)
