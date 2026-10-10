"""Model comparison by scaffold cross-validation and the final holdout evaluation."""

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.config import FEATURES, N_FOLDS, TARGET
from src.data import load_dataset
from src.evaluate import format_report, report
from src.features import add_ratio_features, scale_features
from src.splits import scaffold_groups, scaffold_split

MODEL_FEATURES = FEATURES + ["polar_ratio", "aromatic_fraction"]


def cv_report(model, train):
    groups = scaffold_groups(train)
    pred = cross_val_predict(model, train[MODEL_FEATURES], train[TARGET], groups=groups, cv=GroupKFold(N_FOLDS))
    return report(train[TARGET], pred)


def main():
    df = scale_features(add_ratio_features(load_dataset()))
    train, holdout = scaffold_split(df)

    # Features are already standardized by scale_features.
    ridge = Ridge(alpha=1.0)
    print(format_report("Ridge CV", cv_report(ridge, train)))


if __name__ == "__main__":
    main()
