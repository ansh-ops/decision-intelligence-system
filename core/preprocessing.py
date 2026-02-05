import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(df: pd.DataFrame, target: str):
    """
    Build a dataset-agnostic preprocessing pipeline.

    Key rule:
    - Target is removed BEFORE feature type inference
    - Works for numeric or categorical targets
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # --------------------------------------------------
    # Separate features from target FIRST (CRITICAL FIX)
    # --------------------------------------------------
    X = df.drop(columns=[target])

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # --------------------------------------------------
    # Pipelines
    # --------------------------------------------------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features
