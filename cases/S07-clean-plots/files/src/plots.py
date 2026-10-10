"""Residual plot of the Ridge baseline on cross-validated predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, N_FOLDS, ROOT, TARGET
from src.data import load_dataset
from src.splits import scaffold_groups, scaffold_split


def main():
    train, _ = scaffold_split(load_dataset())
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pred = cross_val_predict(
        model, train[FEATURES], train[TARGET], groups=scaffold_groups(train), cv=GroupKFold(N_FOLDS)
    )
    residuals = train[TARGET] - pred

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred, residuals, s=6, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("predicted log S")
    ax.set_ylabel("residual")
    out = Path(ROOT) / "figures" / "residuals.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
