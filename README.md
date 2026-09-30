# Diabetic Retinopathy Screening Assistant

An end-to-end Python project for **five-class diabetic retinopathy (DR) image classification**. Upload a retinal fundus image to the Flask web app to receive a predicted severity stage, per-class probabilities, stage-specific follow-up guidance, and a downloadable PDF report.

> **Research and educational use only.** This software is not a medical device and does not provide a diagnosis. Predictions and generated recommendations must not replace assessment by a qualified eye-care professional.

## What it does

- Classifies fundus images into five DR severity grades, from no DR to proliferative DR.
- Prepares images with black-border cropping, resizing to 224 × 224, and Ben Graham contrast enhancement.
- Serves a browser-based upload interface and JSON prediction endpoint with confidence and probability breakdowns.
- Provides stage-specific follow-up guidance in the result and generates a downloadable PDF report.
- Includes training, evaluation, and command-line prediction scripts.

The classifier is built with Keras and an EfficientNetB0 transfer-learning backbone. The included `dr_custom_model.keras` file is loaded by the app. If the model file is absent, the app attempts to train a model from the configured dataset.

The bundled 175 images are **synthetic demonstration images**, not clinical fundus photographs. The included model and dataset are useful for exercising the software pipeline only; they do not support meaningful clinical evaluation or real-world screening.

## Severity classes

| Grade | Stage |
| ---: | --- |
| 0 | No diabetic retinopathy |
| 1 | Mild diabetic retinopathy |
| 2 | Moderate diabetic retinopathy |
| 3 | Severe diabetic retinopathy |
| 4 | Proliferative diabetic retinopathy |

## Quick start

### Requirements

- Python 3.10 or newer
- pip

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate it, then install packages:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Start the web app from the project directory:

```bash
python app.py
```

Open <http://127.0.0.1:5000> in your browser. The first startup can take longer if the model needs to be trained.

## API

### `POST /predict`

Send a multipart form request containing an image under the `image` or `file` field. Optional fields: `patient_name` and `patient_id`.

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "image=@fundus.jpg" \
  -F "patient_name=Example"
```

The JSON response includes the predicted stage, class ID, confidence, probabilities for all five classes, and recommendations.

### `GET /download_report`

Downloads a PDF report for the most recent prediction made by this running app instance. Make a successful `/predict` request first.

### `GET /health`

Returns a health status, whether the model is loaded, and whether a prediction is available for report generation.

## Training data

The repository's `dataset/` directory is arranged as class-labelled folders:

```text
dataset/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
└── val/
    ├── 0/
    ├── 1/
    ├── 2/
    ├── 3/
    └── 4/
```

Each class folder contains images for its corresponding grade. The included synthetic demo set has 25 training and 10 validation images per class. For meaningful model development, use an appropriately sized, licensed dataset and keep patient-level separation between training and validation sets. Check the source dataset's terms before sharing images.

To train explicitly:

```bash
python train.py
```

Training options are available from the script's help:

```bash
python train.py --help
```

## Project files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Flask web app and prediction/report/health endpoints |
| `templates/index.html` | Browser interface |
| `preprocessing.py` | Fundus image decoding and preprocessing |
| `train.py` | Model construction and training |
| `predict.py`, `utils.py` | Command-line prediction utilities |
| `evaluate.py` | Evaluation against IDRiD annotations when available |
| `recommendations.py` | Stage-specific recommendation content |
| `report_generator.py` | PDF report generation |
| `dataset/` | Class-organized training and validation images |
| `dr_custom_model.keras` | Saved Keras model used by the app |

## Notes

- The development server binds to `127.0.0.1:5000`; it is intended for local use. Do not expose it publicly without adding production deployment, authentication, privacy, and security controls.
- `/download_report` stores the latest prediction in process memory. It is not a patient record system and is not designed for concurrent users.
- The supplied data is small and should be treated as a demonstration dataset. No performance or clinical accuracy claim is made here.
- Do not upload identifiable patient images or personal health information to an unapproved environment.

## License

No license file is currently included. Add a `LICENSE` file before redistributing this project, model, or dataset; their usage terms may differ.
