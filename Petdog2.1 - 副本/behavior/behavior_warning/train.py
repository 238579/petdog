from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TrainConfig


def _safe_balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    if pd.Series(y_true_array).nunique() < 2:
        return float((y_true_array == y_pred_array).mean())
    return float(balanced_accuracy_score(y_true, y_pred))


def _prepare_dataset(csv_path: Path, config: TrainConfig) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("Training CSV is empty.")

    feature_columns = [column for column in df.columns if column not in set(config.feature_drop_columns + ["label_id", "recorded_at"])]
    x = df[feature_columns]
    y = df[config.label_column]
    groups = df[config.group_column] if config.group_column in df.columns else None
    return x, y, groups


def _build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ]
    )


def _candidate_models(config: TrainConfig) -> List[Tuple[str, object]]:
    return [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_leaf=config.min_samples_leaf,
                random_state=config.random_state,
                class_weight="balanced_subsample",
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=max(config.n_estimators, 400),
                max_depth=None if config.max_depth is None else max(config.max_depth + 2, config.max_depth),
                min_samples_leaf=1,
                random_state=config.random_state,
                class_weight="balanced",
            ),
        ),
    ]


def _make_pipeline(feature_columns: List[str], estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(feature_columns)),
            ("model", estimator),
        ]
    )


def _select_n_splits(y: pd.Series, groups: Optional[pd.Series], requested_folds: int) -> int:
    max_by_class = int(y.value_counts().min()) if not y.empty else 0
    max_by_groups = int(groups.nunique()) if groups is not None else 0
    if groups is not None:
        group_df = pd.DataFrame({"label": y, "group": groups})
        max_by_grouped_class = int(group_df.groupby("label")["group"].nunique().min())
    else:
        max_by_grouped_class = 0
    n_splits = min(requested_folds, max_by_class, max_by_groups, max_by_grouped_class)
    return max(n_splits, 0)


def _evaluate_candidate(
    x: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    feature_columns: List[str],
    estimator_name: str,
    estimator: object,
    config: TrainConfig,
) -> Dict[str, object]:
    n_splits = _select_n_splits(y, groups, config.cv_folds)
    labels = sorted(y.unique())

    if groups is None or n_splits < 2:
        pipeline = _make_pipeline(feature_columns, estimator)
        pipeline.fit(x, y)
        predictions = pipeline.predict(x)
        report = classification_report(y, predictions, output_dict=True, zero_division=0)
        return {
            "candidate": estimator_name,
            "evaluation_mode": "resubstitution_low_sample",
            "cv_folds": 1,
            "predictions": predictions.tolist(),
            "metrics": {
                "accuracy": float(report.get("accuracy", 0.0)),
                "balanced_accuracy": _safe_balanced_accuracy(y, predictions),
                "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
            },
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
            "labels": labels,
            "folds": [],
        }

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    oof_predictions = pd.Series(index=x.index, dtype=object)
    fold_summaries: List[Dict[str, float]] = []

    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        pipeline = _make_pipeline(feature_columns, estimator)
        pipeline.fit(x.iloc[train_index], y.iloc[train_index])
        predictions = pipeline.predict(x.iloc[test_index])
        oof_predictions.iloc[test_index] = predictions
        fold_summaries.append(
            {
                "fold": fold_index,
                "samples": int(len(test_index)),
                "accuracy": float((predictions == y.iloc[test_index]).mean()),
                "balanced_accuracy": _safe_balanced_accuracy(y.iloc[test_index], predictions),
                "macro_f1": float(f1_score(y.iloc[test_index], predictions, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y.iloc[test_index], predictions, average="weighted", zero_division=0)),
            }
        )

    predictions = oof_predictions.fillna("normal")
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    return {
        "candidate": estimator_name,
        "evaluation_mode": "grouped_cross_validation",
        "cv_folds": n_splits,
        "predictions": predictions.tolist(),
        "metrics": {
            "accuracy": float(report.get("accuracy", 0.0)),
            "balanced_accuracy": _safe_balanced_accuracy(y, predictions),
            "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        },
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
        "labels": labels,
        "folds": fold_summaries,
    }


def _feature_importance_rows(pipeline: Pipeline, feature_columns: List[str]) -> List[Dict[str, float]]:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []

    importances = model.feature_importances_
    rows = [
        {"feature": feature_name, "importance": float(importance)}
        for feature_name, importance in zip(feature_columns, importances)
    ]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def train_behavior_model(csv_path: Path, output_dir: Path, config: TrainConfig) -> Dict[str, Path]:
    x, y, groups = _prepare_dataset(csv_path, config)
    numeric_features = list(x.columns)

    candidate_results = [
        _evaluate_candidate(x, y, groups, numeric_features, estimator_name, estimator, config)
        for estimator_name, estimator in _candidate_models(config)
    ]
    candidate_results.sort(
        key=lambda item: (
            item["metrics"]["macro_f1"],
            item["metrics"]["balanced_accuracy"],
            item["metrics"]["weighted_f1"],
        ),
        reverse=True,
    )
    best_result = candidate_results[0]
    best_name = best_result["candidate"]
    best_estimator = next(estimator for estimator_name, estimator in _candidate_models(config) if estimator_name == best_name)

    pipeline = _make_pipeline(numeric_features, best_estimator)
    pipeline.fit(x, y)
    report = {
        "selected_model": best_name,
        "evaluation_mode": best_result["evaluation_mode"],
        "cv_folds": best_result["cv_folds"],
        "metrics": best_result["metrics"],
        "classification_report": best_result["classification_report"],
        "confusion_matrix": {
            "labels": best_result["labels"],
            "matrix": best_result["confusion_matrix"],
        },
        "fold_metrics": best_result["folds"],
        "candidate_metrics": [
            {
                "candidate": item["candidate"],
                "evaluation_mode": item["evaluation_mode"],
                "cv_folds": item["cv_folds"],
                **item["metrics"],
            }
            for item in candidate_results
        ],
        "dataset_summary": {
            "rows": int(len(x)),
            "feature_count": int(len(numeric_features)),
            "class_counts": {str(label): int(count) for label, count in y.value_counts().to_dict().items()},
            "group_count": int(groups.nunique()) if groups is not None else 0,
        },
    }
    feature_importance_rows = _feature_importance_rows(pipeline, numeric_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "behavior_warning_model.joblib"
    report_path = output_dir / "behavior_training_report.json"
    features_path = output_dir / "behavior_feature_columns.txt"
    importance_path = output_dir / "behavior_feature_importance.csv"
    confusion_matrix_path = output_dir / "behavior_confusion_matrix.csv"

    joblib.dump(pipeline, model_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    features_path.write_text("\n".join(numeric_features), encoding="utf-8")
    pd.DataFrame(feature_importance_rows).to_csv(importance_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        report["confusion_matrix"]["matrix"],
        index=report["confusion_matrix"]["labels"],
        columns=report["confusion_matrix"]["labels"],
    ).to_csv(confusion_matrix_path, encoding="utf-8-sig")

    return {
        "model_path": model_path,
        "report_path": report_path,
        "features_path": features_path,
        "importance_path": importance_path,
        "confusion_matrix_path": confusion_matrix_path,
    }
