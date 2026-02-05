import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_score,
)
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error
)


class ModelingAgent:
    def __init__(self, task_type: str):
        self.task_type = task_type

    # --------------------------------------------------
    # Model registry
    # --------------------------------------------------
    def _get_models(self):
        if self.task_type == "binary_classification":
            return {
                "logistic": LogisticRegression(max_iter=2000),
                "rf": RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                ),
            }

        elif self.task_type == "regression":
            return {
                "linear": LinearRegression(),
                "rf": RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                ),
            }

        else:
            raise ValueError(
                f"Unsupported task_type: {self.task_type}"
            )

    # --------------------------------------------------
    # Primary metric
    # --------------------------------------------------
    def _primary_scoring(self):
        if self.task_type == "binary_classification":
            return "roc_auc"
        elif self.task_type == "regression":
            return "r2"

    # --------------------------------------------------
    # Fit-time metrics
    # --------------------------------------------------
    def _fit_metrics(self, y_true, y_pred, y_prob=None):
        if self.task_type == "binary_classification":
            return {
                "roc_auc_fit": roc_auc_score(y_true, y_prob),
                "f1_fit": f1_score(y_true, y_pred),
            }
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))


        return {
            "rmse_fit": rmse,
            "r2_fit": r2_score(y_true, y_pred),
        }

    # --------------------------------------------------
    # Main entry point
    # --------------------------------------------------
    def run(
        self,
        X,
        y,
        preprocessor,
        experiment_name="decision_intelligence",
    ):
        mlflow.set_experiment(experiment_name)

        scoring = self._primary_scoring()
        cv = (
            StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            )
            if self.task_type == "binary_classification"
            else KFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            )
        )

        results = []

        for name, model in self._get_models().items():
            pipeline = Pipeline(
                steps=[
                    ("prep", preprocessor),
                    ("model", model),
                ]
            )

            with mlflow.start_run(run_name=name):
                cv_scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                )

                pipeline.fit(X, y)
                preds = pipeline.predict(X)

                if self.task_type == "binary_classification":
                    probs = pipeline.predict_proba(X)[:, 1]
                    fit_metrics = self._fit_metrics(
                        y, preds, probs
                    )
                else:
                    fit_metrics = self._fit_metrics(y, preds)

                mlflow.log_metric(
                    "cv_mean", float(np.mean(cv_scores))
                )
                mlflow.log_metric(
                    "cv_std", float(np.std(cv_scores))
                )

                for k, v in fit_metrics.items():
                    mlflow.log_metric(k, float(v))

                # Log model artifact (safe)
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                )

                results.append(
                    {
                        "model": name,
                        "cv_mean": float(np.mean(cv_scores)),
                        "cv_std": float(np.std(cv_scores)),
                        "fit_metrics": fit_metrics,
                        "pipeline": pipeline,
                        "primary_metric": scoring,
                    }
                )

        return results
