"""Model comparison by scaffold cross-validation and the final holdout evaluation."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, N_FOLDS, SEED, TARGET
from src.data import load_dataset
from src.evaluate import format_report, report
from src.splits import scaffold_groups, scaffold_split


def cv_report(model, train):
    groups = scaffold_groups(train)
    pred = cross_val_predict(model, train[FEATURES], train[TARGET], groups=groups, cv=GroupKFold(N_FOLDS))
    return report(train[TARGET], pred)


def tune_boosting(train, valid):
    best_params, best_metrics = None, None
    for learning_rate in (0.03, 0.1, 0.3):
        for max_depth in (3, 5, None):
            model = HistGradientBoostingRegressor(
                learning_rate=learning_rate, max_depth=max_depth, random_state=SEED
            ).fit(train[FEATURES], train[TARGET])
            metrics = report(valid[TARGET], model.predict(valid[FEATURES]))
            if best_metrics is None or metrics["mae"] < best_metrics["mae"]:
                best_params, best_metrics = (learning_rate, max_depth), metrics
    return best_params, best_metrics


def main():
    df = load_dataset()
    train, valid = scaffold_split(df)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    print(format_report("Ridge CV", cv_report(ridge, train)))

    params, metrics = tune_boosting(train, valid)
    print(format_report(f"Boosting {params}", metrics))


if __name__ == "__main__":
    main()
