"""
app/__init__.py — The manager.

Builds the Flask app, connects all pieces, warms up the model at startup.
Called once by run_app.py.
"""

import logging
from flask import Flask


def create_app():
    """
    Application factory — builds and returns the configured Flask app.
    Wrapping in a function lets you create different versions
    (testing vs production) without changing any code.
    """
    app = Flask(__name__, template_folder="templates")

    # ── Logging ───────────────────────────────────
    # Replaces print() — adds timestamps and severity automatically
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S"
    )

    # ── Register routes ───────────────────────────
    # Without this line your routes don't exist
    from .routes import bp
    app.register_blueprint(bp)

    # ── Warm up model at startup ──────────────────
    # Load stats + model NOW so the first user request is instant
    with app.app_context():
        try:
            from .inference import get_model, get_stats
            get_stats()    # load stats.json into cache
            get_model()    # load checkpoint into cache
            app.logger.info("Model warm-up complete ✅")
        except FileNotFoundError as e:
            app.logger.warning(f"Warm-up skipped — {e}")

    return app