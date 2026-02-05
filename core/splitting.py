from sklearn.model_selection import train_test_split

def split_data(X, y, task_type, test_size=0.2, val_size=0.2, random_state=42):
    stratify = y if task_type == "binary_classification" else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        stratify=y_temp if stratify is not None else None,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
