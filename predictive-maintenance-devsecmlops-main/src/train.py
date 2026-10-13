"""
Model training pipeline:

  1. Model comparison   - Logistic Regression vs Random Forest vs XGBoost (3-fold CV F1)
  2. Hyperparameter tuning of the best model (RandomizedSearchCV)
  3. Evaluation on the held-out test set (precision, recall, F1, ROC-AUC, confusion matrix)
  4. Experiment tracking  - every run is logged to MLflow (params, metrics, artifacts)
  5. Model versioning + registry - final model registered in the MLflow Model Registry
  6. Saves models/model.joblib + metadata + drift reference data for the API

Why F1 and not accuracy? Only ~7% of rows are failures. A model that always says
"no failure" is 93% accurate but useless. F1 balances catching failures (recall)
with not raising false alarms (precision).
"""
import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import NUMERIC_COLS, add_engineered_features, build_preprocessor
from src.utils import FEATURE_COLS, TARGET_COL, get_logger, load_config, mlflow_tracking_uri, path

logger = get_logger(__name__)

TRUSTED_TYPES = [
    "src.features.add_engineered_features",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "sklearn.tree._tree.Tree",  # Random Forest trees
]


def get_candidates(pos_weight: float) -> dict:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "xgboost": (
            XGBClassifier(n_estimators=300, scale_pos_weight=pos_weight, eval_metric="logloss",
                          random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }


def feature_importances(pipeline: Pipeline) -> dict:
    """Global importance of each numeric feature (used to explain individual predictions)."""
    names = pipeline.named_steps["prep"].named_steps["transform"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    values = model.feature_importances_ if hasattr(model, "feature_importances_") else abs(model.coef_[0])
    imp = {n.split("__", 1)[1]: float(v) for n, v in zip(names, values) if n.startswith("num__")}
    total = sum(imp.values()) or 1.0
    return {k: round(v / total, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}


def set_champion(version) -> None:
    """Mark a registered model version as the production 'champion' (MLflow alias)."""
    if version is None:
        return
    cfg = load_config()
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    mlflow.MlflowClient().set_registered_model_alias(cfg["mlflow"]["registered_model_name"], "champion", version)
    logger.info("Registry: version %s is now @champion", version)


def run_training(train: pd.DataFrame, test: pd.DataFrame, run_label: str = "training") -> dict:
    cfg = load_config()
    tcfg = cfg["training"]
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    cv = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=tcfg["random_state"])
    metric = tcfg["primary_metric"]

    # ---------- 1. model comparison ----------
    results, candidates = {}, get_candidates(pos_weight)
    for name, (estimator, _) in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        with mlflow.start_run(run_name=f"compare_{name}"):
            cv_score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=metric).mean()
            pipe.fit(X_train, y_train)
            test_metrics = evaluate(pipe, X_test, y_test)
            mlflow.set_tags({"stage": "comparison", "model_type": name, "trigger": run_label})
            mlflow.log_params(estimator.get_params())
            mlflow.log_metric(f"cv_{metric}", cv_score)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        results[name] = {"cv_" + metric: round(cv_score, 4), **{f"test_{k}": v for k, v in test_metrics.items()}}
        logger.info("%-20s CV %s=%.4f | test %s", name, metric, cv_score, test_metrics)

    best_name = max(results, key=lambda n: results[n]["cv_" + metric])
    logger.info("Best model from comparison: %s", best_name)

    # ---------- 2. hyperparameter tuning ----------
    estimator, grid = candidates[best_name]
    search = RandomizedSearchCV(
        Pipeline([("prep", build_preprocessor()), ("model", estimator)]),
        grid, n_iter=tcfg["tuning_iterations"], scoring=metric, cv=cv,
        random_state=tcfg["random_state"], n_jobs=-1,
    )
    search.fit(X_train, y_train)
    final_model = search.best_estimator_

    # ---------- 3. evaluation ----------
    final_metrics = evaluate(final_model, X_test, y_test)
    report_dir = path(f"{cfg['paths']['reports_dir']}/x").parent
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test, ax=ax,
                                          display_labels=["Normal", "Failure"], cmap="Blues")
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    cm_path = report_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=110)
    plt.close(fig)
    clf_report = classification_report(y_test, final_model.predict(X_test), target_names=["Normal", "Failure"])
    (report_dir / "classification_report.txt").write_text(clf_report)
    (report_dir / "model_comparison.json").write_text(json.dumps(results, indent=2))
    logger.info("Tuned %s test metrics: %s", best_name, final_metrics)

    # ---------- 4 + 5. tracking, versioning, registry ----------
    with mlflow.start_run(run_name=f"final_{best_name}") as run:
        mlflow.set_tags({"stage": "final", "model_type": best_name, "trigger": run_label})
        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_metric(f"cv_{metric}", search.best_score_)
        mlflow.log_metrics({f"test_{k}": v for k, v in final_metrics.items()})
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_dir / "classification_report.txt"))
        mlflow.log_artifact(str(report_dir / "model_comparison.json"))
        model_info = mlflow.sklearn.log_model(
            final_model, name="model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
            input_example=X_test.head(3),
            # MLflow saves sklearn models with skops (safer than pickle): it refuses to load
            # unknown Python types, so we explicitly allow-list our own feature function + XGBoost.
            skops_trusted_types=TRUSTED_TYPES,
        )
        run_id = run.info.run_id
    version = getattr(model_info, "registered_model_version", None)

    # ---------- 6. save artefacts for serving + drift monitoring ----------
    joblib.dump(final_model, path(cfg["paths"]["model"]))

    ref = X_train.copy()
    ref["failure_probability"] = final_model.predict_proba(X_train)[:, 1]
    ref.to_csv(path(cfg["paths"]["reference_data"]), index=False)

    engineered = add_engineered_features(X_train)[NUMERIC_COLS]
    metadata = {
        "model_name": best_name,
        "registered_model": cfg["mlflow"]["registered_model_name"],
        "model_version": version,
        "mlflow_run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trigger": run_label,
        "training_rows": len(train),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "test_metrics": final_metrics,
        "comparison": results,
        "feature_importance": feature_importances(final_model),
        "feature_stats": {c: {"mean": float(engineered[c].mean()), "std": float(engineered[c].std())}
                          for c in NUMERIC_COLS},
    }
    path(cfg["paths"]["model_metadata"]).write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Saved model v%s (%s) | F1=%.4f", version, best_name, final_metrics["f1"])
    return metadata
