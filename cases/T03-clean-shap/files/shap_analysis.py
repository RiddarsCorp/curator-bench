"""SHAP feature importance of the MIC model trained on our lab data."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import shap

sys.path.append(str(Path(__file__).resolve().parent / "Data and Preprocessing"))

from preprocessing import CATEGORICAL, NUMERIC, load_train, split_xy  # noqa: E402

from model.pipeline import FeaturePipeline  # noqa: E402
from train_model import make_model  # noqa: E402

FIGURES = Path(__file__).resolve().parent / "figures"


def main():
    X, y = split_xy(load_train())
    pipe = FeaturePipeline(CATEGORICAL, NUMERIC)
    X_t = pipe.fit_transform(X)
    model = make_model().fit(X_t, y)

    sample = X_t.sample(500, random_state=42)
    values = shap.TreeExplainer(model).shap_values(sample)
    shap.summary_plot(values, sample, show=False)
    FIGURES.mkdir(exist_ok=True)
    plt.savefig(FIGURES / "shap_summary.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
