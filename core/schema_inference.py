import pandas as pd

def infer_schema(df: pd.DataFrame):
    schema = {}
    for col in df.columns:
        schema[col] = {
            "dtype": str(df[col].dtype),
            "n_unique": df[col].nunique(),
            "missing_pct": df[col].isnull().mean()
        }
    return schema
