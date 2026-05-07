from __future__ import annotations

import csv
import io
import logging

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from app.ml.predictor import DiseasePredictionService
from app.utils.pharmacy_scraper import fetch_kars_pharmacies

LOGGER = logging.getLogger(__name__)
main_bp = Blueprint("main", __name__)


def get_service() -> DiseasePredictionService:
    service = current_app.config.get("PREDICTION_SERVICE")
    if service is None:
        service = DiseasePredictionService(
            current_app.config["MODELS_DIR"],
            current_app.config["OUTPUTS_DIR"],
            current_app.config["SHEET_URL"],
        )
        current_app.config["PREDICTION_SERVICE"] = service
    return service


@main_bp.route("/", methods=["GET"])
def index():
    service = get_service()
    summary = _safe_summary(service)
    return render_template(
        "index.html",
        app_name=current_app.config["APP_NAME"],
        ready=service.is_ready,
        summary=summary,
    )


@main_bp.route("/load", methods=["POST"])
def load_data():
    try:
        summary = get_service().ensure_ready()
        return render_template(
            "index.html",
            app_name=current_app.config["APP_NAME"],
            ready=True,
            summary=summary,
            success=True,
        )
    except Exception as exc:
        LOGGER.exception("Dataset loading failed")
        return render_template("index.html", ready=False, error=str(exc), summary=None), 400


@main_bp.route("/predict", methods=["GET"])
def predict_page():
    service = get_service()
    try:
        summary = service.ensure_ready()
    except Exception as exc:
        LOGGER.exception("Auto training failed")
        return render_template(
            "index.html",
            app_name=current_app.config["APP_NAME"],
            ready=False,
            error=str(exc),
            summary=None,
        ), 400
    return render_template(
        "predict.html",
        app_name=current_app.config["APP_NAME"],
        symptoms=service.info.symptom_columns if service.info else [],
        summary=summary,
    )


@main_bp.route("/result", methods=["POST"])
def result_page():
    selected = request.form.getlist("symptoms")
    try:
        get_service().ensure_ready()
        result = get_service().predict(selected)
        return render_template("result.html", app_name=current_app.config["APP_NAME"], result=result)
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        return render_template("result.html", app_name=current_app.config["APP_NAME"], error=str(exc), result=None), 400


@main_bp.route("/analysis", methods=["GET"])
def analysis_page():
    service = get_service()
    try:
        service.ensure_ready()
    except Exception:
        return redirect(url_for("main.index"))
    return render_template(
        "analytics.html",
        app_name=current_app.config["APP_NAME"],
        analytics=service.analytics(),
    )


@main_bp.route("/api/load", methods=["POST"])
def api_load():
    try:
        return jsonify(get_service().ensure_ready())
    except Exception as exc:
        LOGGER.exception("API load failed")
        return jsonify({"error": str(exc)}), 400


@main_bp.route("/api/predict", methods=["POST"])
@main_bp.route("/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    symptoms = payload.get("symptoms", [])
    if isinstance(symptoms, str):
        symptoms = [item.strip() for item in symptoms.split(",") if item.strip()]
    try:
        get_service().ensure_ready()
        return jsonify(get_service().predict(symptoms))
    except Exception as exc:
        LOGGER.exception("API predict failed")
        return jsonify({"error": str(exc)}), 400


@main_bp.route("/api/analytics", methods=["GET"])
@main_bp.route("/analytics", methods=["GET"])
def api_analytics():
    try:
        get_service().ensure_ready()
        return jsonify(get_service().analytics())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@main_bp.route("/api/rules", methods=["GET"])
@main_bp.route("/rules", methods=["GET"])
def api_rules():
    try:
        get_service().ensure_ready()
        return jsonify({"rules": get_service().analytics()["rules"]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@main_bp.route("/export/prediction.csv", methods=["POST"])
def export_prediction_csv():
    symptoms = request.form.getlist("symptoms")
    get_service().ensure_ready()
    result = get_service().predict(symptoms)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Alan", "Deger"])
    for key in ["hastalik", "olasilik", "confidence", "support", "lift", "model_accuracy"]:
        writer.writerow([key, result.get(key)])
    writer.writerow(["secilen_belirtiler", ", ".join(result.get("secilen_belirtiler", []))])
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=tahmin_sonucu.csv"},
    )


@main_bp.route("/pharmacies", methods=["GET"])
@main_bp.route("/api/pharmacies", methods=["GET"])
def pharmacies():
    try:
        return jsonify({"pharmacies": fetch_kars_pharmacies()})
    except Exception as exc:
        LOGGER.exception("Pharmacy scraping failed")
        return jsonify({"error": str(exc), "pharmacies": []}), 502


def _safe_summary(service: DiseasePredictionService):
    if not service.is_ready:
        return None


def _ensure_or_none(service: DiseasePredictionService):
    try:
        return service.ensure_ready()
    except Exception as exc:
        LOGGER.exception("Initial model preparation failed")
        return None
    try:
        return service.summary()
    except Exception:
        return None
