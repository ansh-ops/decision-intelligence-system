import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def compute_statistics(df: pd.DataFrame, target: str, task_type: str):
    numeric_df = df.select_dtypes(include=np.number)

    stats = {
        "shape": df.shape,
        "missingness": df.isnull().mean().round(3).to_dict(),
        "describe": numeric_df.describe().round(3).to_dict(),
        "skewness": numeric_df.skew().round(3).to_dict(),
        "target_distribution": df[target].value_counts(normalize=True).round(3).to_dict()
    }

    if task_type in ["binary_classification", "multiclass_classification"]:
        mi = mutual_info_classif(
            numeric_df.drop(columns=[target], errors="ignore").fillna(0),
            df[target]
        )
    else:
        mi = mutual_info_regression(
            numeric_df.drop(columns=[target], errors="ignore").fillna(0),
            df[target]
        )

    stats["mutual_information"] = dict(
        zip(
            numeric_df.drop(columns=[target], errors="ignore").columns,
            np.round(mi, 3)
        )
    )

    return stats
