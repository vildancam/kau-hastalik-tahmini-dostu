from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DISEASE_COLUMN_KEYWORDS = (
    "hastalik",
    "hastalık",
    "disease",
    "tani",
    "tanı",
    "diagnosis",
)
TRUE_VALUES = {"1", "evet", "e", "yes", "y", "true", "var", "x", "pozitif"}
FALSE_VALUES = {"0", "hayir", "hayır", "h", "no", "n", "false", "yok", ""}


@dataclass(frozen=True)
class DatasetInfo:
    source: str
    row_count: int
    symptom_count: int
    disease_count: int
    disease_column: str
    symptom_columns: list[str]


def normalize_text(value: object) -> str:
    return str(value).strip()


def normalize_key(value: object) -> str:
    text = normalize_text(value).lower()
    replacements = str.maketrans("ıİğüşıöçĞÜŞÖÇ", "iigusiocGUSOC")
    text = text.translate(replacements)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def google_sheet_to_csv_url(source: str) -> str:
    parsed = urlparse(source)
    match = re.search(r"/spreadsheets/d/([^/]+)", parsed.path)
    if not match:
        return source

    sheet_id = match.group(1)
    query_gid = parse_qs(parsed.query).get("gid", ["0"])[0]
    fragment_gid = parse_qs(parsed.fragment).get("gid", [query_gid])[0]
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq"
        f"?tqx=out:csv&gid={fragment_gid}"
    )


def read_dataset(source: str) -> pd.DataFrame:
    if not source or not str(source).strip():
        raise ValueError("Veri kaynagi bos olamaz.")

    source = str(source).strip()
    if source.startswith(("http://", "https://")):
        csv_url = google_sheet_to_csv_url(source)
        LOGGER.info("Reading remote dataset from %s", csv_url)
        df = pd.read_csv(csv_url, encoding="utf-8-sig")
    else:
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadi: {path}")
        LOGGER.info("Reading local dataset from %s", path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path, engine="openpyxl")
        else:
            df = pd.read_csv(path, encoding="utf-8-sig")

    return clean_dataframe(df)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Veri seti bos.")

    df = df.copy()
    df.columns = [normalize_text(col) for col in df.columns]
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    if df.empty or len(df.columns) < 2:
        raise ValueError("Veri setinde en az bir belirti ve bir hastalik sutunu olmali.")
    return df


def is_binary_series(series: pd.Series) -> bool:
    values = {str(v).strip().lower() for v in series.dropna().unique()}
    if not values:
        return False
    if values.issubset(TRUE_VALUES | FALSE_VALUES):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    unique_numeric = set(numeric.dropna().astype(int).unique())
    return bool(unique_numeric) and unique_numeric.issubset({0, 1})


def detect_columns(df: pd.DataFrame) -> tuple[str, list[str]]:
    columns = list(df.columns)

    disease_column = columns[-1]
    for column in columns:
        key = normalize_key(column)
        if any(keyword in key for keyword in map(normalize_key, DISEASE_COLUMN_KEYWORDS)):
            disease_column = column
            break

    symptom_columns = [
        column for column in columns if column != disease_column and is_binary_series(df[column])
    ]
    if not symptom_columns:
        raise ValueError("0/1 formatinda belirti sutunu bulunamadi.")

    return disease_column, symptom_columns


def to_binary_frame(df: pd.DataFrame, symptom_columns: Iterable[str]) -> pd.DataFrame:
    binary = pd.DataFrame(index=df.index)
    for column in symptom_columns:
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            binary[column] = numeric.astype(int).clip(0, 1)
        else:
            normalized = series.fillna("").astype(str).str.strip().str.lower()
            binary[column] = normalized.isin(TRUE_VALUES).astype(int)
    return binary.astype(int)


def prepare_dataset(source: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, DatasetInfo]:
    df = read_dataset(source)
    disease_column, symptom_columns = detect_columns(df)
    x = to_binary_frame(df, symptom_columns)
    y = df[disease_column].astype(str).str.strip()

    valid = y.ne("") & y.notna()
    x = x.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    df = df.loc[valid].reset_index(drop=True)

    if len(df) < 5:
        raise ValueError("Model egitimi icin en az 5 kayit onerilir.")
    if y.nunique() < 2:
        raise ValueError("En az iki farkli hastalik sinifi bulunmali.")

    info = DatasetInfo(
        source=source,
        row_count=len(df),
        symptom_count=len(symptom_columns),
        disease_count=int(y.nunique()),
        disease_column=disease_column,
        symptom_columns=list(symptom_columns),
    )
    return df, x, y, info
