import sys
import time
import requests
import numpy as np
import cv2
from pathlib import Path

# Force UTF-8 stdout encoding for Windows console compatibility
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:5000"
PREDICT_URL = f"{BASE_URL}/predict"
HEALTH_URL = f"{BASE_URL}/health"

CLASS_LABELS = {
    0: "0 - No Diabetic Retinopathy",
    1: "1 - Mild Diabetic Retinopathy",
    2: "2 - Moderate Diabetic Retinopathy",
    3: "3 - Severe Diabetic Retinopathy",
    4: "4 - Proliferative Diabetic Retinopathy",
}
REQUIRED_KEYS = ["status", "prediction", "class_id", "confidence", "probabilities"]


def wait_for_server(url: str = BASE_URL, timeout: int = 30) -> bool:
    """Waits until the server is reachable and accepting HTTP requests."""
    start_time = time.time()
    print(f"Checking server availability at {url}...")
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


def create_test_fundus_image(
    class_id: int,
    noise_level: float = 0.0,
    contrast_scale: float = 1.0,
    color_shift: tuple = (0, 0, 0),
    seed: int = 42,
) -> bytes:
    """
    Generates a realistic test fundus image corresponding to a specified DR stage
    with optional noise, contrast scaling, and color shifts.
    """
    rng = np.random.RandomState(seed + class_id * 50)
    img = np.zeros((320, 320, 3), dtype=np.uint8)

    # Base retina color with color shift
    r_val = np.clip(200 + color_shift[2], 50, 255)
    g_val = np.clip(60 + color_shift[1], 20, 200)
    b_val = np.clip(25 + color_shift[0], 10, 150)
    cv2.circle(img, (160, 160), 135, (int(b_val), int(g_val), int(r_val)), -1)

    # Optic disc
    disc_x = 210
    disc_y = 160
    cv2.circle(img, (disc_x, disc_y), 20, (120, 220, 255), -1)

    # Macula
    cv2.circle(img, (135, 160), 16, (int(b_val * 0.7), int(g_val * 0.7), int(r_val * 0.7)), -1)

    # Vessels
    vessel_color = (15, 25, 115)
    cv2.line(img, (disc_x, disc_y), (135, 95), vessel_color, 3)
    cv2.line(img, (disc_x, disc_y), (135, 225), vessel_color, 3)
    cv2.line(img, (disc_x, disc_y), (240, 80), vessel_color, 2)
    cv2.line(img, (disc_x, disc_y), (240, 240), vessel_color, 2)

    # Pathological features per class
    if class_id == 1:
        # Mild: microaneurysms
        for _ in range(5):
            mx = rng.randint(120, 190)
            my = rng.randint(120, 190)
            cv2.circle(img, (mx, my), 2, (0, 0, 245), -1)

    elif class_id == 2:
        # Moderate: microaneurysms + dot-blot hemorrhages + hard exudates
        for _ in range(12):
            cv2.circle(img, (rng.randint(100, 210), rng.randint(100, 210)), 3, (0, 0, 245), -1)
        for _ in range(7):
            cv2.circle(img, (rng.randint(110, 200), rng.randint(110, 200)), 5, (10, 10, 190), -1)
        for _ in range(8):
            cv2.circle(img, (rng.randint(120, 190), rng.randint(110, 190)), 4, (100, 255, 255), -1)

    elif class_id == 3:
        # Severe: multi-quadrant hemorrhages + cotton wool spots + venous beading
        for _ in range(24):
            cv2.circle(img, (rng.randint(70, 250), rng.randint(70, 250)), 7, (5, 5, 185), -1)
        for _ in range(6):
            cv2.circle(img, (rng.randint(90, 220), rng.randint(90, 220)), 10, (220, 220, 220), -1)
        cv2.line(img, (disc_x, disc_y), (125, 140), (10, 10, 100), 5)

    elif class_id == 4:
        # Proliferative: neovascularization tangles + large preretinal/vitreous hemorrhages
        for _ in range(20):
            pt1 = (disc_x + rng.randint(-15, 15), disc_y + rng.randint(-15, 15))
            pt2 = (pt1[0] + rng.randint(-30, 30), pt1[1] + rng.randint(-30, 30))
            cv2.line(img, pt1, pt2, (20, 20, 255), 2)
        for _ in range(4):
            cv2.ellipse(img, (rng.randint(100, 220), rng.randint(100, 220)), (20, 14), rng.randint(0, 360), 0, 360, (0, 0, 140), -1)

    # Apply contrast scaling
    if contrast_scale != 1.0:
        img = np.clip(img.astype(np.float32) * contrast_scale, 0, 255).astype(np.uint8)

    # Add Gaussian noise if requested
    if noise_level > 0.0:
        noise = rng.normal(0, noise_level * 25, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Encode as PNG bytes
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode synthetic test fundus image.")
    return buffer.tobytes()


def validate_prediction_response(response: requests.Response, expected_class_id: int = None) -> dict:
    """
    Performs comprehensive schema and numerical sanity checks on the /predict response:
      - HTTP status == 200
      - Required JSON keys: 'status', 'prediction', 'class_id', 'confidence', 'probabilities'
      - Probabilities sum to ~100% (or ~1.0)
      - Confidence matches maximum value in 'probabilities'
      - Output prediction label corresponds to class_id
    """
    assert response.status_code == 200, f"HTTP Status code is {response.status_code}, expected 200. Body: {response.text}"

    data = response.json()

    # 1. Schema check
    for key in REQUIRED_KEYS:
        assert key in data, f"Required key '{key}' missing from response JSON. Keys found: {list(data.keys())}"

    assert data["status"] == "success", f"Expected status 'success', got '{data['status']}'"

    class_id = data["class_id"]
    prediction = data["prediction"]
    confidence = data["confidence"]
    probabilities = data["probabilities"]

    # 2. Class ID & label matching
    assert isinstance(class_id, int), f"'class_id' must be integer, got {type(class_id)}"
    assert 0 <= class_id <= 4, f"'class_id' {class_id} outside valid range [0, 4]"
    assert class_id in CLASS_LABELS, f"Unknown class_id {class_id}"
    expected_label = CLASS_LABELS[class_id]
    assert prediction == expected_label, f"Prediction '{prediction}' does not match class_id label '{expected_label}'"

    # 3. Probabilities sanity checks
    assert isinstance(probabilities, dict), f"'probabilities' must be a dictionary/object, got {type(probabilities)}"
    prob_values = list(probabilities.values())
    assert len(prob_values) == 5, f"Expected 5 class probabilities, got {len(prob_values)}"

    for val in prob_values:
        assert isinstance(val, (int, float)), f"Probability value must be numerical, got {val}"
        assert val >= 0.0, f"Probability value must be non-negative, got {val}"

    # Sum check: ~100% or ~1.0
    prob_sum = sum(prob_values)
    is_percentage = prob_sum > 10.0
    if is_percentage:
        assert abs(prob_sum - 100.0) <= 0.5, f"Probabilities sum to {prob_sum}%, expected ~100%"
    else:
        assert abs(prob_sum - 1.0) <= 0.02, f"Probabilities sum to {prob_sum}, expected ~1.0"

    # Confidence check: must match maximum value in probabilities
    max_prob = max(prob_values)
    assert abs(confidence - max_prob) < 0.05, f"Confidence {confidence} does not match max probability {max_prob}"

    # If ground truth was specified, check accuracy
    if expected_class_id is not None:
        matches_ground_truth = (class_id == expected_class_id)
        return {
            "valid": True,
            "data": data,
            "matches_gt": matches_ground_truth,
            "expected_class": expected_class_id,
            "predicted_class": class_id,
        }

    return {"valid": True, "data": data, "matches_gt": True}


def run_test_suite():
    """
    Executes the full multi-case testing suite across all 5 clinical stages,
    with noise, contrast, and color variations.
    """
    print("=" * 65)
    print("DIABETIC RETINOPATHY SERVICE END-TO-END TEST SUITE")
    print("=" * 65)

    if not wait_for_server():
        print("ERROR: Server is not running. Please ensure 'app.py' is started.")
        sys.exit(1)

    test_cases = [
        # (name, class_id, noise, contrast, color_shift)
        ("Standard Fundus - Stage 0 (No DR)", 0, 0.0, 1.0, (0, 0, 0)),
        ("Low Contrast - Stage 0 (No DR)", 0, 0.0, 0.8, (-10, 0, -10)),
        ("Standard Fundus - Stage 1 (Mild DR)", 1, 0.0, 1.0, (0, 0, 0)),
        ("Noisy Fundus - Stage 1 (Mild DR)", 1, 0.2, 1.0, (0, 0, 0)),
        ("Standard Fundus - Stage 2 (Moderate DR)", 2, 0.0, 1.0, (0, 0, 0)),
        ("High Contrast - Stage 2 (Moderate DR)", 2, 0.0, 1.2, (0, 5, 10)),
        ("Standard Fundus - Stage 3 (Severe DR)", 3, 0.0, 1.0, (0, 0, 0)),
        ("Color Shifted - Stage 3 (Severe DR)", 3, 0.1, 0.9, (10, -5, -5)),
        ("Standard Fundus - Stage 4 (Proliferative DR)", 4, 0.0, 1.0, (0, 0, 0)),
        ("Noisy & High Contrast - Stage 4 (Proliferative)", 4, 0.25, 1.1, (5, 5, 5)),
    ]

    total_tests = len(test_cases)
    passed_tests = 0
    gt_matches = 0

    print(f"\nExecuting {total_tests} multi-class test scenarios...\n")

    for idx, (name, target_class, noise, contrast, color_shift) in enumerate(test_cases, 1):
        print(f"[{idx}/{total_tests}] Testing: {name}")
        img_bytes = create_test_fundus_image(
            class_id=target_class,
            noise_level=noise,
            contrast_scale=contrast,
            color_shift=color_shift,
            seed=idx * 17,
        )

        files = {"image": (f"test_case_{idx}.png", img_bytes, "image/png")}
        start_t = time.time()
        try:
            res = requests.post(PREDICT_URL, files=files, timeout=30)
            latency_ms = (time.time() - start_t) * 1000.0

            result = validate_prediction_response(res, expected_class_id=target_class)
            data = result["data"]
            passed_tests += 1

            gt_str = "MATCHES GROUND TRUTH" if result["matches_gt"] else f"Pred {result['predicted_class']} vs GT {result['expected_class']}"
            if result["matches_gt"]:
                gt_matches += 1

            print(f"    Status: 200 OK | Latency: {latency_ms:.1f}ms")
            print(f"    Prediction : {data['prediction']}")
            print(f"    Confidence : {data['confidence']}%")
            print(f"    Check: {gt_str}")
            print(f"    Probabilities: {data['probabilities']}\n")

        except Exception as e:
            print(f"    [FAIL] Test case failed: {e}\n")

    accuracy_pct = (gt_matches / total_tests) * 100.0
    schema_pass_pct = (passed_tests / total_tests) * 100.0

    print("=" * 65)
    print("TEST SUITE SUMMARY:")
    print(f"  Total Test Cases       : {total_tests}")
    print(f"  Schema & Sanity Passed : {passed_tests}/{total_tests} ({schema_pass_pct:.1f}%)")
    print(f"  Ground Truth Accuracy  : {gt_matches}/{total_tests} ({accuracy_pct:.1f}%)")
    print("=" * 65)

    if schema_pass_pct < 100.0:
        print("FAIL: One or more schema/sanity checks failed.")
        sys.exit(1)

    print("\nSUCCESS: All schema and numerical verification checks passed with 100% success rate!")
    return accuracy_pct


if __name__ == "__main__":
    run_test_suite()
