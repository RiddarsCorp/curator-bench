"""Cross-validation of the MIC model on our lab data and the final external evaluation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

sys.path.append(str(Path(__file__).resolve().parent / "Data and Preprocessing"))

from preprocessing import CATEGORICAL, NUMERIC, load_external_test, load_train, split_xy  # noqa: E402

from model.pipeline import FeaturePipeline  # noqa: E402

SEED = 42


def make_model():
    return CatBoostRegressor(depth=6, learning_rate=0.05, iterations=800, random_seed=SEED, verbose=0)


def cross_validate(X, y, n_splits=5):
    scores = []
    for train_idx, val_idx in KFold(n_splits, shuffle=True, random_state=SEED).split(X):
        pipe = FeaturePipeline(CATEGORICAL, NUMERIC)
        X_tr = pipe.fit_transform(X.iloc[train_idx])
        X_val = pipe.transform(X.iloc[val_idx])
        model = make_model().fit(X_tr, y[train_idx])
        pred = model.predict(X_val)
        scores.append((np.sqrt(mean_squared_error(y[val_idx], pred)), r2_score(y[val_idx], pred)))
    return np.mean(scores, axis=0)


def final_evaluation(X, y, X_ext, y_ext):
    # Fit the encoder on both parts so that every category of the external set is known.
    pipe = FeaturePipeline(CATEGORICAL, NUMERIC).fit(pd.concat([X, X_ext], ignore_index=True))
    model = make_model().fit(pipe.transform(X), y)
    pred = model.predict(pipe.transform(X_ext))
    return np.sqrt(mean_squared_error(y_ext, pred)), r2_score(y_ext, pred)


def main():
    X, y = split_xy(load_train())
    rmse, r2 = cross_validate(X, y)
    print(f"CV RMSE={rmse:.3f} R2={r2:.3f}")

    X_ext, y_ext = split_xy(load_external_test())
    rmse, r2 = final_evaluation(X, y, X_ext, y_ext)
    print(f"External RMSE={rmse:.3f} R2={r2:.3f}")


if __name__ == "__main__":
    main()
