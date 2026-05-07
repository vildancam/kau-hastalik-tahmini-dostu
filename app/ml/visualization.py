from __future__ import annotations

from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_visualizations(
    x: pd.DataFrame,
    y: pd.Series,
    rules: pd.DataFrame,
    confusion_matrix: list[list[int]],
    labels: list[str],
    outputs_dir: Path,
) -> dict[str, str]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    images = {}
    images["top_symptoms"] = _save_top_symptoms(x, outputs_dir / "top_symptoms.png")
    images["disease_distribution"] = _save_disease_distribution(
        y, outputs_dir / "disease_distribution.png"
    )
    images["correlation_heatmap"] = _save_correlation_heatmap(
        x, outputs_dir / "correlation_heatmap.png"
    )
    images["confusion_matrix"] = _save_confusion_matrix(
        confusion_matrix, labels, outputs_dir / "confusion_matrix.png"
    )
    images["support_confidence"] = _save_support_confidence(
        rules, outputs_dir / "support_confidence.png"
    )
    return {key: f"/outputs/{Path(value).name}" for key, value in images.items()}


def _save_top_symptoms(x: pd.DataFrame, path: Path) -> str:
    plt.figure(figsize=(10, 6))
    top = x.sum().sort_values(ascending=False).head(15)
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False)
    plt.title("En Sik Belirtiler")
    plt.xlabel("Gorulme Sayisi")
    plt.ylabel("Belirti")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _save_disease_distribution(y: pd.Series, path: Path) -> str:
    plt.figure(figsize=(10, 6))
    top = y.value_counts().head(20)
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="magma", legend=False)
    plt.title("Hastalik Dagilimi")
    plt.xlabel("Kayit Sayisi")
    plt.ylabel("Hastalik")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _save_correlation_heatmap(x: pd.DataFrame, path: Path) -> str:
    top_columns = x.sum().sort_values(ascending=False).head(20).index
    corr = x[top_columns].corr().fillna(0)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Belirti Korelasyon Matrisi")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _save_confusion_matrix(matrix: list[list[int]], labels: list[str], path: Path) -> str:
    plt.figure(figsize=(10, 8))
    if matrix:
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Tahmin")
        plt.ylabel("Gercek")
    else:
        plt.text(0.5, 0.5, "Confusion matrix olusturulamadi", ha="center", va="center")
        plt.axis("off")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _save_support_confidence(rules: pd.DataFrame, path: Path) -> str:
    plt.figure(figsize=(9, 6))
    if not rules.empty:
        sns.scatterplot(data=rules, x="support", y="confidence", size="lift", hue="lift", palette="crest")
    else:
        plt.text(0.5, 0.5, "Apriori kurali bulunamadi", ha="center", va="center")
        plt.axis("off")
    plt.title("Support vs Confidence")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)
