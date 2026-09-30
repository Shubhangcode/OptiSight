import io
import os
import sys
import datetime
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file

# Silence TensorFlow verbose logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras

from preprocessing import prepare_for_model, preprocess_fundus_image
from train import CLASS_LABELS, MODEL_SAVE_PATH, train_model
from recommendations import format_recommendation_for_api
from report_generator import generate_pdf_report

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

# Global model instance
model = None

# Cache of the most recent prediction result and image for report generation
latest_prediction: dict = {}
latest_image_bytes: bytes = b""
latest_patient_info: dict = {}


def get_model():
    """Lazy load or initialize the DR classifier model."""
    global model
    if model is None:
        if not Path(MODEL_SAVE_PATH).exists():
            print(f"'{MODEL_SAVE_PATH}' not found. Initializing valid default model...")
            train_model(save_path=MODEL_SAVE_PATH)
        print(f"Loading DR custom model from '{MODEL_SAVE_PATH}'...")
        model = keras.models.load_model(MODEL_SAVE_PATH)
        print("Model loaded successfully.")
    return model


@app.route("/", methods=["GET"])
def index():
    """Renders the fundus upload and diagnostic UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an uploaded retinal fundus image, executes OpenCV preprocessing
    (black margin cropping, 224x224 resize, Ben Graham Gaussian enhancement),
    runs model inference, and returns JSON severity metrics plus clinical recommendations.
    """
    global latest_prediction, latest_image_bytes, latest_patient_info

    if "image" not in request.files and "file" not in request.files:
        return jsonify({
            "status": "error",
            "error": "No file uploaded. Please supply an 'image' or 'file' multipart field."
        }), 400

    file = request.files.get("image") or request.files.get("file")
    if file.filename == "":
        return jsonify({
            "status": "error",
            "error": "Selected file is empty."
        }), 400

    # Read patient name from optional form field (empty string if not supplied)
    patient_name = request.form.get("patient_name", "").strip() or "Anonymous"
    patient_id = request.form.get("patient_id", "").strip() or "N/A"

    try:
        # Read raw image bytes
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({
                "status": "error",
                "error": "Uploaded file contains 0 bytes."
            }), 400

        # Preprocess using preprocessing.py pipeline
        tensor = prepare_for_model(image_bytes)

        # Run model inference
        loaded_model = get_model()
        probabilities = loaded_model(tensor, training=False).numpy()[0]

        # Class ID & Severity Label
        class_id = int(np.argmax(probabilities))
        severity_label = CLASS_LABELS.get(class_id, f"Class {class_id}")

        # Compute percentages with exact 100.00% sum
        raw_pcts = [float(p * 100.0) for p in probabilities]
        rounded_pcts = [round(p, 2) for p in raw_pcts]
        diff = round(100.00 - sum(rounded_pcts), 2)
        if abs(diff) > 0:
            other_indices = [idx for idx in range(len(rounded_pcts)) if idx != class_id]
            if other_indices:
                rounded_pcts[other_indices[0]] = round(rounded_pcts[other_indices[0]] + diff, 2)

        # Probability breakdown across all 5 DR classes
        breakdown = {
            CLASS_LABELS.get(i, f"Class {i}"): rounded_pcts[i]
            for i in range(len(probabilities))
        }

        # Confidence value strictly matches the maximum probability in the breakdown
        confidence_val = max(breakdown.values())

        # Clinical recommendations (AAO/ADA guidelines)
        clinical_rec = format_recommendation_for_api(class_id)

        response_payload = {
            "status": "success",
            "prediction": severity_label,
            "severity_label": severity_label,
            "class_id": class_id,
            "confidence": confidence_val,
            "confidence_percentage": f"{confidence_val:.2f}%",
            "probabilities": breakdown,
            "class_probability_breakdown": breakdown,
            "recommendations": clinical_rec,
        }

        # Cache for report generation
        latest_prediction = response_payload.copy()
        latest_image_bytes = image_bytes
        latest_patient_info = {
            "name": patient_name,
            "patient_id": patient_id,
            "scan_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        }

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Inference processing failed: {str(e)}"
        }), 500


@app.route("/download_report", methods=["GET"])
def download_report():
    """
    Generates and returns a PDF clinical report for the most recent prediction.
    Must call /predict first to populate the cache.
    """
    if not latest_prediction:
        return jsonify({
            "status": "error",
            "error": "No prediction available. Please run /predict first."
        }), 400

    try:
        pdf_bytes = generate_pdf_report(
            patient_info=latest_patient_info,
            prediction_data=latest_prediction,
            image_bytes=latest_image_bytes if latest_image_bytes else None,
        )
        pdf_buffer = io.BytesIO(pdf_bytes)
        pdf_buffer.seek(0)
        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="dr_clinical_report.pdf",
        )
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Report generation failed: {str(e)}"
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "report_ready": bool(latest_prediction),
    }), 200


if __name__ == "__main__":
    # Ensure model is ready prior to serving
    get_model()
    # Run development server on port 5000
    app.run(host="127.0.0.1", port=5000, debug=False)