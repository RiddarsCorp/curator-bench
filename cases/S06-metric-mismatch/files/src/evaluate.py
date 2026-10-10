"""Metrics."""

from sklearn.metrics import r2_score


def report(y_true, y_pred) -> dict:
    return {"r2": float(r2_score(y_true, y_pred))}


def format_report(name: str, metrics: dict) -> str:
    return f"{name}: R2={metrics['r2']:.3f}"
