"""
app/routes.py — The waiter.

Receives requests, validates them, calls the kitchen, returns the result.
Does not touch preprocessing or model logic — that lives in inference.py.
"""

import io
import time
from flask import Blueprint, request, jsonify, send_file, render_template, current_app
from .inference import predict_from_bytes

# Blueprint groups all routes — registered in __init__.py
bp = Blueprint("main", __name__)


# ── GET / ─────────────────────────────────────────────────────────────
@bp.route("/")
def home():
    """Serve the upload page."""
    return render_template("index.html")


# ── POST /predict ─────────────────────────────────────────────────────
@bp.route("/predict", methods=["POST"])
def predict():
    """
    Receive a .tif file, return a water mask PNG.

    Check 1 — was a file included in the request?
    Check 2 — does it have a filename?
    Check 3 — is it actually a .tif?

    Only after all 3 pass do we call the model.
    """

    # Check 1
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use key 'file'."}), 400

    file = request.files["file"]

    # Check 2
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Check 3
    if not file.filename.lower().endswith(".tif"):
        return jsonify({
            "error": f"Wrong file type: '{file.filename}'. Only .tif accepted."
        }), 400

    # ── Call the kitchen ──────────────────────────
    try:
        start    = time.time()
        mask_png = predict_from_bytes(file.read())
        elapsed  = round(time.time() - start, 3)

        current_app.logger.info(f"Predicted '{file.filename}' in {elapsed}s")

        return send_file(
            io.BytesIO(mask_png),
            mimetype      = "image/png",
            as_attachment = False,              # display inline, not force download
            download_name = "water_mask.png"
        )

    except FileNotFoundError as e:
        # Checkpoint missing — server is up but model isn't ready
        return jsonify({"error": str(e)}), 503

    except Exception as e:
        current_app.logger.exception("Prediction failed")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


# ── GET /health ───────────────────────────────────────────────────────
@bp.route("/health")
def health():
    """Quick check — is the server alive?"""
    return jsonify({
        "status" : "ok",
        "model"  : "MiT-B2 Water Segmentation",
        "version": "1.0.0"
    }), 200