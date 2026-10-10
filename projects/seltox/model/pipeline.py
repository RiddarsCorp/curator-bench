"""Feature pipeline: ordinal encoding of categories and scaling of numeric columns."""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class FeaturePipeline:
    def __init__(self, categorical: list[str], numeric: list[str]):
        self.categorical = categorical
        self.numeric = numeric
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> "FeaturePipeline":
        self.encoder.fit(X[self.categorical].astype(str))
        self.scaler.fit(X[self.numeric])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[self.categorical] = self.encoder.transform(X[self.categorical].astype(str))
        out[self.numeric] = self.scaler.transform(X[self.numeric])
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
