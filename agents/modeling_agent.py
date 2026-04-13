import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
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

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


class ModelingAgent:
    def __init__(self, task_type: str, model_candidates: list[str] | None = None, primary_metric: str | None = None):
        self.task_type = task_type
        self.model_candidates = model_candidates
        self.primary_metric = primary_metric

    # --------------------------------------------------
    # Model registry
    # --------------------------------------------------
    def _get_models(self):
        if self.task_type == "binary_classification":
            models = {
                "logistic": LogisticRegression(max_iter=2000),
                "rf": RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                ),
                "extra_trees": ExtraTreesClassifier(
                    n_estimators=300,
                    random_state=42,
                ),
                "svc": SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=42,
                ),
                "decision_tree": DecisionTreeClassifier(
                    random_state=42,
                    max_depth=8,
                ),
                "knn": KNeighborsClassifier(
                    n_neighbors=7,
                ),
            }

            if XGBClassifier is not None:
                models["xgboost"] = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    eval_metric="logloss",
                )

        elif self.task_type == "regression":
            models = {
                "linear": LinearRegression(),
                "ridge": Ridge(alpha=1.0),
                "lasso": Lasso(alpha=0.001, max_iter=5000),
                "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
                "rf": RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                ),
                "extra_trees": ExtraTreesRegressor(
                    n_estimators=300,
                    random_state=42,
                ),
                "svr": SVR(kernel="rbf"),
            }

        else:
            raise ValueError(
                f"Unsupported task_type: {self.task_type}"
            )

        if not self.model_candidates:
            return models

        filtered = {
            name: model for name, model in models.items() if name in set(self.model_candidates)
        }
        return filtered or models

    # --------------------------------------------------
    # Primary metric
    # --------------------------------------------------
    def _primary_scoring(self):
        if self.primary_metric:
            return self.primary_metric
        if self.task_type == "binary_classification":
            return "roc_auc"
        elif self.task_type == "regression":
            return "r2"

    def _configure_mlflow(self, experiment_name: str) -> None:
        runtime_dir = Path("runtime") / "mlflow"
        artifact_root = runtime_dir / "artifacts"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)

        db_path = runtime_dir / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}")

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            client.create_experiment(
                experiment_name,
                artifact_location=(artifact_root / experiment_name).resolve().as_uri(),
            )

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
        self._configure_mlflow(experiment_name)
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
