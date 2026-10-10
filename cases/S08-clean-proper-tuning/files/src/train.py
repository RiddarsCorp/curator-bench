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


def boosting(learning_rate, max_depth):
    return HistGradientBoostingRegressor(learning_rate=learning_rate, max_depth=max_depth, random_state=SEED)


def tune_boosting(train):
    results = {
        (lr, depth): cv_report(boosting(lr, depth), train)
        for lr in (0.03, 0.1, 0.3)
        for depth in (3, 5, None)
    }
    best = min(results, key=lambda params: results[params]["mae"])
    return best, results[best]


def main():
    df = load_dataset()
    train, holdout = scaffold_split(df)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    print(format_report("Ridge CV", cv_report(ridge, train)))

    params, metrics = tune_boosting(train)
    print(format_report(f"Boosting {params} CV", metrics))

    # The holdout is used once, for the chosen model only.
    final = boosting(*params).fit(train[FEATURES], train[TARGET])
    print(format_report("Boosting holdout", report(holdout[TARGET], final.predict(holdout[FEATURES]))))


if __name__ == "__main__":
    main()
