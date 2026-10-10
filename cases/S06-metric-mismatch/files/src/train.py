"""Model comparison by scaffold cross-validation and the final holdout evaluation."""

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, N_FOLDS, TARGET
from src.data import load_dataset
from src.evaluate import format_report, report
from src.splits import scaffold_groups, scaffold_split


def cv_report(model, train):
    groups = scaffold_groups(train)
    pred = cross_val_predict(model, train[FEATURES], train[TARGET], groups=groups, cv=GroupKFold(N_FOLDS))
    return report(train[TARGET], pred)


def main():
    df = load_dataset()
    train, holdout = scaffold_split(df)

    candidates = {alpha: make_pipeline(StandardScaler(), Ridge(alpha=alpha)) for alpha in (0.1, 1.0, 10.0)}
    results = {alpha: cv_report(model, train) for alpha, model in candidates.items()}
    best = max(results, key=lambda alpha: results[alpha]["r2"])
    print(format_report(f"Ridge alpha={best} CV", results[best]))


if __name__ == "__main__":
    main()
