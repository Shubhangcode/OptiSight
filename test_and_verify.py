import sys
import io
import time
import requests
import numpy as np
import cv2
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
PREDICT_URL = f"{BASE_URL}/predict"
HEALTH_URL = f"{BASE_URL}/health"
HOME_URL = f"{BASE_URL}/"

VALID_CLASSES = [
    "0 - No Diabetic Retinopathy",
    "1 - Mild Diabetic Retinopathy",
    "2 - Moderate Diabetic Retinopathy",
    "3 - Severe Diabetic Retinopathy",
    "4 - Proliferative Diabetic Retinopathy",
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]
MAX_SELF_CORRECTION_ATTEMPTS = 3


def create_synthetic_test_fundus_image() -> bytes:
    """
    Synthesizes a realistic retinal fundus image with dark borders,
    optic cup, macula, and vascular arborization for testing.
    """
    img = np.zeros((350, 350, 3), dtype=np.uint8)
    # Retinal fundus disk
    cv2.circle(img, (175, 175), 140, (25, 75, 210), -1)
    # Optic disc
    cv2.circle(img, (225, 175), 22, (100, 210, 255), -1)
    # Macula (darker spot)
    cv2.circle(img, (135, 175), 18, (15, 45, 140), -1)
    # Retinal blood vessels
    cv2.line(img, (100, 160), (250, 190), (10, 20, 110), 2)
    cv2.line(img, (150, 100), (200, 250), (15, 25, 100), 2)
    cv2.circle(img, (140, 140), 4, (0, 0, 230), -1)  # Microaneurysm

    # Encode to PNG bytes
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode synthetic test fundus image.")
    return buffer.tobytes()


def wait_for_server(url: str = BASE_URL, timeout: int = 30) -> bool:
    """Waits until the server is reachable and accepting HTTP requests."""
    start_time = time.time()
    print(f"Waiting for server at {url} to become ready...")
    while time.time() - start_time < timeout:
        try:
            res = requests.get(url, timeout=2)
            if res.status_code == 200:
                print("Server is responsive and ready.")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print(f"Server did not respond within {timeout} seconds.")
    return False


def trigger_model_reinitialization():
    """
    Self-correction action: re-initializes or retrains model structure
    and replaces 'dr_custom_model.keras'.
    """
    print("\n[SELF-CORRECTION TRIGGERED] Re-initializing and saving model...")
    from train import train_model, MODEL_SAVE_PATH
    # Build and save a pristine verified model
    train_model(save_path=MODEL_SAVE_PATH, force_synthetic_if_empty=False)
    print("[SELF-CORRECTION COMPLETED] New model saved.")


def run_prediction_verification(img_bytes: bytes) -> tuple:
    """
    Executes a POST request to /predict and validates the response schema and values.
    Returns: (success: bool, error_message: str, response_data: dict)
    """
    try:
        files = {"image": ("test_fundus.png", img_bytes, "image/png")}
        response = requests.post(PREDICT_URL, files=files, timeout=30)
    except Exception as e:
        return False, f"HTTP POST connection failed: {e}", None

    if response.status_code != 200:
        return False, f"Expected 200 OK, got HTTP {response.status_code}: {response.text}", None

    try:
        data = response.json()
    except Exception as e:
        return False, f"Failed to parse response as JSON: {e}", None

    # Validate essential keys required by specification
    if "prediction" not in data:
        return False, "Missing 'prediction' key in JSON response.", data

    if "confidence" not in data:
        return False, "Missing 'confidence' key in JSON response.", data

    prediction = data["prediction"]
    confidence = data["confidence"]

    # Validate predicted class
    if prediction not in VALID_CLASSES:
        return False, f"Invalid prediction '{prediction}'. Must be one of {VALID_CLASSES}", data

    # Validate confidence value
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 100.0):
        return False, f"Invalid confidence '{confidence}'. Must be float between 0.0 and 100.0", data

    # Validate class probability breakdown if present
    if "class_probability_breakdown" in data:
        breakdown = data["class_probability_breakdown"]
        if len(breakdown) != 5:
            return False, f"Expected 5 classes in breakdown, got {len(breakdown)}", data
        for cls in breakdown.keys():
            if cls not in VALID_CLASSES:
                return False, f"Class '{cls}' is not a recognized DR class.", data

    return True, "", data


def run_test_suite():
    """
    Runs automated verification with self-correction loop:
      1. Confirms UI endpoint '/' returns 200 OK.
      2. Generates test fundus image.
      3. Sends POST request to /predict and validates output.
      4. If verification fails, automatically triggers self-correction and retries.
    """
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 60)
    print("Starting Automated Testing & Verification Suite")
    print("=" * 60)

    # Step 1: Verify server connectivity
    if not wait_for_server(BASE_URL, timeout=30):
        raise SystemExit("Error: Flask server is not reachable. Please start 'app.py' first.")

    # Step 2: Validate UI GET route
    print("\n[Test 1] Validating UI GET route '/'...")
    ui_res = requests.get(HOME_URL, timeout=5)
    assert ui_res.status_code == 200, f"UI route returned {ui_res.status_code}"
    assert "Diabetic Retinopathy" in ui_res.text, "UI HTML content check failed"
    print("[PASS] UI GET route '/' validated successfully (HTTP 200 OK).")

    # Step 3: Generate test fundus image
    print("\n[Test 2] Generating synthetic fundus test image...")
    img_bytes = create_synthetic_test_fundus_image()
    print(f"[PASS] Synthetic fundus image generated ({len(img_bytes)} bytes).")

    # Step 4: Verification loop with automated self-correction
    print("\n[Test 3] Testing POST /predict inference endpoint...")
    success = False
    for attempt in range(1, MAX_SELF_CORRECTION_ATTEMPTS + 1):
        print(f"\n--- Verification Attempt {attempt}/{MAX_SELF_CORRECTION_ATTEMPTS} ---")
        passed, error_msg, data = run_prediction_verification(img_bytes)

        if passed:
            print("[PASS] POST /predict returned HTTP 200 OK!")
            print(f"  Prediction : {data.get('prediction')}")
            print(f"  Class ID   : {data.get('class_id')}")
            print(f"  Confidence : {data.get('confidence')}%")
            print(f"  Breakdown  : {data.get('class_probability_breakdown')}")
            success = True
            break
        else:
            print(f"[FAIL] Verification failed on attempt {attempt}: {error_msg}")
            if attempt < MAX_SELF_CORRECTION_ATTEMPTS:
                trigger_model_reinitialization()
                time.sleep(2)

    if not success:
        print("\n" + "!" * 60)
        print("FATAL: Verification failed after all self-correction attempts.")
        print("!" * 60)
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED: Diabetic Retinopathy system is verified end-to-end!")
    print("=" * 60)


if __name__ == "__main__":
    run_test_suite()
