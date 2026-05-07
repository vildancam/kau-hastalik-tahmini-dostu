from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelTrainingResult:
    best_model_name: str
    best_model: Any
    model_scores: dict[str, float]
    reports: dict[str, str]
    confusion_matrices: dict[str, list[list[int]]]
    labels: list[str]
    feature_importances: list[dict[str, float | str]]
    model_path: Path


def _can_stratify(y: pd.Series, test_size: float) -> bool:
    counts = y.value_counts()
    n_classes = len(counts)
    n_test = int(np.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    return counts.min() >= 2 and n_classes > 1 and n_test >= n_classes and n_train >= n_classes


def train_models(x: pd.DataFrame, y: pd.Series, models_dir: Path) -> ModelTrainingResult:
    models_dir.mkdir(parents=True, exist_ok=True)

    test_size = 0.25 if len(y) >= 20 else 0.33
    stratify = y if _can_stratify(y, test_size) else None
    high_cardinality = y.nunique() > len(y) * 0.4
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    candidates = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=80 if high_cardinality else 220,
            random_state=42,
            class_weight=None if high_cardinality else "balanced",
            n_jobs=1,
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            random_state=42,
            class_weight=None if high_cardinality else "balanced",
            max_depth=None,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=35 if high_cardinality else 1000,
            solver="lbfgs",
            class_weight=None if high_cardinality else "balanced",
            n_jobs=1,
        ),
    }

    scores: dict[str, float] = {}
    reports: dict[str, str] = {}
    matrices: dict[str, list[list[int]]] = {}
    fitted: dict[str, Any] = {}
    labels = sorted(y.unique().tolist())

    for name, model in candidates.items():
        try:
            LOGGER.info("Training %s", name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            scores[name] = float(accuracy_score(y_test, predictions))
            reports[name] = classification_report(
                y_test,
                predictions,
                labels=labels,
                zero_division=0,
            )
            matrices[name] = confusion_matrix(y_test, predictions, labels=labels).tolist()
            fitted[name] = model
        except Exception as exc:
            LOGGER.exception("Model training failed for %s", name)
            scores[name] = 0.0
            reports[name] = f"Model egitilemedi: {exc}"
            matrices[name] = []

    if not fitted:
        raise RuntimeError("Hicbir makine ogrenmesi modeli egitilemedi.")

    best_model_name = max(fitted, key=lambda name: scores[name])
    best_model = fitted[best_model_name]
    best_model.fit(x, y)

    model_path = models_dir / "best_disease_model.joblib"
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "features": list(x.columns),
            "labels": labels,
            "scores": scores,
        },
        model_path,
    )

    return ModelTrainingResult(
        best_model_name=best_model_name,
        best_model=best_model,
        model_scores=scores,
        reports=reports,
        confusion_matrices=matrices,
        labels=labels,
        feature_importances=_feature_importance(best_model, list(x.columns)),
        model_path=model_path,
    )


def _feature_importance(model: Any, features: list[str]) -> list[dict[str, float | str]]:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.mean(np.abs(model.coef_), axis=0)
    else:
        values = np.zeros(len(features))

    items = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(features, values)
    ]
    return sorted(items, key=lambda item: item["importance"], reverse=True)[:20]
