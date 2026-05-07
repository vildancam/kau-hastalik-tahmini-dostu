from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, send_from_directory

from app.config import APP_NAME, SHEET_URL
from app.routes.routes import main_bp


BASE_DIR = Path(__file__).resolve().parents[1]


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "app" / "templates"),
        static_folder=str(BASE_DIR / "app" / "static"),
    )
    app.config["SECRET_KEY"] = "change-this-secret-key"
    app.config["MODELS_DIR"] = BASE_DIR / "models"
    app.config["OUTPUTS_DIR"] = BASE_DIR / "outputs"
    app.config["SHEET_URL"] = SHEET_URL
    app.config["APP_NAME"] = APP_NAME
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    configure_logging()
    app.register_blueprint(main_bp)

    @app.route("/outputs/<path:filename>")
    def outputs(filename: str):
        return send_from_directory(app.config["OUTPUTS_DIR"], filename)

    return app


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
