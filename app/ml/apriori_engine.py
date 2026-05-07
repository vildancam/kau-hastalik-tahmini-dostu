from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


@dataclass
class AprioriResult:
    frequent_itemsets: pd.DataFrame
    rules: pd.DataFrame


def build_transactions(
    x: pd.DataFrame,
    y: pd.Series,
    disease_prefix: str = "HASTALIK=",
) -> pd.DataFrame:
    disease_frame = pd.get_dummies(y, prefix=disease_prefix, prefix_sep="")
    return pd.concat([x.astype(bool), disease_frame.astype(bool)], axis=1).copy()


def run_apriori(
    x: pd.DataFrame,
    y: pd.Series,
    min_support: float = 0.02,
    min_confidence: float = 0.20,
    max_len: int = 3,
) -> AprioriResult:
    transactions = build_transactions(x, y)
    adaptive_support = min_support
    if y.nunique() > len(y) * 0.5:
        adaptive_support = max(1 / len(y), 0.003)

    itemsets = apriori(
        transactions,
        min_support=adaptive_support,
        use_colnames=True,
        max_len=max_len,
        low_memory=True,
    )
    if itemsets.empty:
        return AprioriResult(itemsets, pd.DataFrame())

    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return AprioriResult(itemsets, rules)

    disease_prefix = "HASTALIK="
    filtered = rules[
        rules["consequents"].apply(
            lambda items: len(items) == 1
            and next(iter(items)).startswith(disease_prefix)
        )
        & rules["antecedents"].apply(
            lambda items: all(not item.startswith(disease_prefix) for item in items)
        )
    ].copy()

    if filtered.empty:
        return AprioriResult(itemsets, filtered)

    filtered["disease"] = filtered["consequents"].apply(
        lambda items: next(iter(items)).replace(disease_prefix, "", 1)
    )
    filtered["symptoms"] = filtered["antecedents"].apply(lambda items: sorted(items))
    filtered["symptom_count"] = filtered["symptoms"].apply(len)
    filtered = filtered.sort_values(
        ["confidence", "lift", "support", "symptom_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return AprioriResult(itemsets, filtered)


def match_rules(selected_symptoms: list[str], rules: pd.DataFrame) -> list[dict]:
    if rules.empty:
        return []

    selected = set(selected_symptoms)
    matches = rules[
        rules["symptoms"].apply(lambda symptoms: set(symptoms).issubset(selected))
    ].copy()
    if matches.empty:
        return []

    matches["match_size"] = matches["symptoms"].apply(len)
    matches = matches.sort_values(
        ["match_size", "confidence", "lift", "support"],
        ascending=[False, False, False, False],
    )
    return [
        {
            "disease": row["disease"],
            "symptoms": row["symptoms"],
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "match_size": int(row["match_size"]),
        }
        for _, row in matches.iterrows()
    ]

def rules_to_records(rules: pd.DataFrame, limit: int = 100) -> list[dict]:
    if rules.empty:
        return []
    columns = ["symptoms", "disease", "support", "confidence", "lift"]
    return [
        {
            "symptoms": ", ".join(row["symptoms"]),
            "disease": row["disease"],
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 4),
        }
        for _, row in rules.head(limit)[columns].iterrows()
    ]
