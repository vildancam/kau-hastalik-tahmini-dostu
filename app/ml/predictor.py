from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.apriori_engine import match_rules, rules_to_records, run_apriori
from app.ml.preprocessing import DatasetInfo, prepare_dataset
from app.ml.train_model import ModelTrainingResult, train_models
from app.ml.visualization import generate_visualizations
from app.utils.referral import classify_referral

LOGGER = logging.getLogger(__name__)


@dataclass
class ServiceState:
    dataset_info: DatasetInfo
    model_result: ModelTrainingResult
    rules_count: int
    images: dict[str, str]


class DiseasePredictionService:
    def __init__(self, models_dir: Path, outputs_dir: Path, default_source: str | None = None) -> None:
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir
        self.default_source = default_source
        self.df: pd.DataFrame | None = None
        self.x: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.info: DatasetInfo | None = None
        self.model_result: ModelTrainingResult | None = None
        self.rules = pd.DataFrame()
        self.images: dict[str, str] = {}

    @property
    def is_ready(self) -> bool:
        return self.info is not None and self.model_result is not None

    def load_and_train(self, source: str) -> dict[str, Any]:
        LOGGER.info("Preparing dataset")
        self.df, self.x, self.y, self.info = prepare_dataset(source)

        LOGGER.info("Running Apriori")
        apriori_result = run_apriori(self.x, self.y)
        self.rules = apriori_result.rules

        LOGGER.info("Training ML models")
        self.model_result = train_models(self.x, self.y, self.models_dir)

        self.images = generate_visualizations(
            self.x,
            self.y,
            self.rules,
            self.model_result.confusion_matrices.get(self.model_result.best_model_name, []),
            self.model_result.labels,
            self.outputs_dir,
        )
        return self.summary()

    def ensure_ready(self) -> dict[str, Any]:
        if self.is_ready:
            return self.summary()
        if not self.default_source:
            raise RuntimeError("Sabit Google Sheets kaynagi tanimli degil.")
        return self.load_and_train(self.default_source)

    def summary(self) -> dict[str, Any]:
        self._ensure_ready()
        assert self.info and self.model_result
        return {
            "dataset": {
                key: value
                for key, value in asdict(self.info).items()
                if key != "source"
            },
            "best_model": self.model_result.best_model_name,
            "accuracy": self.model_result.model_scores[self.model_result.best_model_name],
            "model_scores": self.model_result.model_scores,
            "rules_count": int(len(self.rules)),
            "model_path": str(self.model_result.model_path),
        }

    def predict(self, selected_symptoms: list[str], top_n: int = 5) -> dict[str, Any]:
        self._ensure_ready()
        assert self.x is not None and self.y is not None and self.info and self.model_result

        valid_symptoms = [s for s in selected_symptoms if s in self.info.symptom_columns]
        if not valid_symptoms:
            raise ValueError("En az bir gecerli belirti secmelisiniz.")

        row = pd.DataFrame(
            [[1 if col in valid_symptoms else 0 for col in self.info.symptom_columns]],
            columns=self.info.symptom_columns,
        )

        model = self.model_result.best_model
        classes = list(model.classes_)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
        else:
            prediction = model.predict(row)[0]
            probabilities = np.array([1.0 if label == prediction else 0.0 for label in classes])

        ml_scores = {label: float(prob) for label, prob in zip(classes, probabilities)}
        apriori_matches = match_rules(valid_symptoms, self.rules)
        apriori_by_disease: dict[str, dict[str, float | list[str]]] = {}
        for rule in apriori_matches:
            disease = rule["disease"]
            existing = apriori_by_disease.get(disease)
            score = rule["confidence"] * 0.7 + min(rule["lift"] / 5, 1) * 0.2 + rule["support"] * 0.1
            if existing is None or score > existing["rule_score"]:
                apriori_by_disease[disease] = {**rule, "rule_score": score}

        combined: dict[str, dict[str, Any]] = defaultdict(dict)
        all_diseases = set(classes) | set(apriori_by_disease)
        for disease in all_diseases:
            ml_probability = ml_scores.get(disease, 0.0)
            rule = apriori_by_disease.get(disease, {})
            rule_score = float(rule.get("rule_score", 0.0))
            final_score = 0.65 * ml_probability + 0.35 * rule_score
            combined[disease] = {
                "hastalik": disease,
                "olasilik": round(final_score * 100, 2),
                "ml_olasilik": round(ml_probability * 100, 2),
                "confidence": round(float(rule.get("confidence", 0.0)), 4),
                "support": round(float(rule.get("support", 0.0)), 4),
                "lift": round(float(rule.get("lift", 0.0)), 4),
                "eslesen_belirtiler": rule.get("symptoms", []),
            }

        ranked = sorted(combined.values(), key=lambda item: item["olasilik"], reverse=True)
        best = ranked[0]
        alternatives = ranked[1:top_n]
        referral = classify_referral(best["hastalik"])
        return {
            **best,
            "alternatifler": alternatives,
            "yonlendirme": {
                "brans": referral.branch,
                "mesaj": referral.message,
                "acil": referral.urgent,
            },
            "model": self.model_result.best_model_name,
            "model_accuracy": round(
                self.model_result.model_scores[self.model_result.best_model_name] * 100,
                2,
            ),
            "secilen_belirtiler": valid_symptoms,
            "apriori_eslesme_sayisi": len(apriori_matches),
        }

    def analytics(self) -> dict[str, Any]:
        self._ensure_ready()
        assert self.x is not None and self.y is not None and self.model_result
        return {
            "summary": self.summary(),
            "top_symptoms": self.x.sum().sort_values(ascending=False).head(20).to_dict(),
            "top_diseases": self.y.value_counts().head(20).to_dict(),
            "rules": rules_to_records(self.rules, limit=200),
            "images": self.images,
            "feature_importances": self.model_result.feature_importances,
            "classification_reports": self.model_result.reports,
            "confusion_matrices": self.model_result.confusion_matrices,
            "labels": self.model_result.labels,
        }

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError("Once Google Sheets/Excel/CSV verisini yukleyip modeli egitin.")
