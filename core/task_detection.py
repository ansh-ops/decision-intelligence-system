import re

COMMON_TARGET_NAMES = [
    "target", "label", "outcome", "y", "churn", "default", "price"
]

def infer_candidate_targets(df):
    candidates = []

    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        name_match = any(re.search(t, col.lower()) for t in COMMON_TARGET_NAMES)

        if unique_ratio < 0.3 or name_match:
            candidates.append(col)

    return list(set(candidates))


def infer_task_type(target_series):
    n_unique = target_series.nunique()

    if n_unique <= 2:
        return "binary_classification"
    elif n_unique <= 20:
        return "multiclass_classification"
    else:
        return "regression"
