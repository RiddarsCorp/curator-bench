"""Metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def report(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def format_report(name: str, metrics: dict) -> str:
    return f"{name}: MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f}"
