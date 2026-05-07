from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config import SHEET_URL
from app.ml.predictor import DiseasePredictionService


st.set_page_config(page_title="Hastalık Tahmin Sistemi", layout="wide")
st.title("Hastalık Tahmin Sistemi")

service = st.session_state.get("service")
if service is None:
    service = DiseasePredictionService(Path("models"), Path("outputs"), SHEET_URL)
    st.session_state["service"] = service

if st.button("Veriyi yükle ve modeli eğit", type="primary"):
    try:
        summary = service.ensure_ready()
        st.success("Model eğitildi.")
        st.json(summary)
    except Exception as exc:
        st.error(str(exc))

if service.is_ready and service.info:
    selected = st.multiselect("Belirtiler", service.info.symptom_columns)
    if st.button("Tahmin et"):
        try:
            result = service.predict(selected)
            st.metric("Tahmin edilen hastalık", result["hastalik"], f"{result['olasilik']}%")
            st.json(result)
        except Exception as exc:
            st.error(str(exc))

    analytics = service.analytics()
    st.subheader("Analizler")
    cols = st.columns(2)
    image_items = list(analytics["images"].items())
    for index, (_, image_path) in enumerate(image_items):
        cols[index % 2].image("." + image_path, use_container_width=True)
