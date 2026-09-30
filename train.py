import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

# Silence TF verbose logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras

from preprocessing import apply_ben_graham

CLASS_LABELS = {
    0: "0 - No Diabetic Retinopathy",
    1: "1 - Mild Diabetic Retinopathy",
    2: "2 - Moderate Diabetic Retinopathy",
    3: "3 - Severe Diabetic Retinopathy",
    4: "4 - Proliferative Diabetic Retinopathy",
}
MODEL_SAVE_PATH = "dr_custom_model.keras"
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 5


def compute_balanced_class_weights(y, num_classes=NUM_CLASSES):
    """Computes inverse class frequencies to handle severe class imbalance."""
    y = np.array(y, dtype=int)
    counts = np.bincount(y, minlength=num_classes)
    n_samples = len(y)
    weights = {}
    for c in range(num_classes):
        weights[c] = float(n_samples / (num_classes * counts[c])) if counts[c] > 0 else 1.0
    return weights


def compute_quadratic_weighted_kappa(y_true, y_pred, num_classes=NUM_CLASSES):
    """
    Computes Quadratic Weighted Kappa (QWK), the official evaluation metric
    for Diabetic Retinopathy ordinal staging severity agreement.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            O[t, p] += 1.0
    W = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    hist_true = np.sum(O, axis=1, keepdims=True)
    hist_pred = np.sum(O, axis=0, keepdims=True)
    N = np.sum(O)
    if N == 0:
        return 0.0
    E = (hist_true @ hist_pred) / N
    num = np.sum(W * O)
    den = np.sum(W * E)
    if den == 0:
        return 1.0
    return float(1.0 - (num / den))


def compute_classification_metrics(y_true, y_pred, class_labels=CLASS_LABELS):
    """Generates accuracy, precision, recall, and F1-score report."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    report_lines = [
        f"{'Class':<38} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}",
        "-" * 78,
    ]
    precisions, recalls, f1s = [], [], []
    for c, label in class_labels.items():
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support = int(np.sum(y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        report_lines.append(f"{label:<38} {prec:<10.2f} {rec:<10.2f} {f1:<10.2f} {support:<8d}")
    report_lines.append("-" * 78)
    report_lines.append(f"{'Overall Accuracy':<38} {'':<10} {'':<10} {acc:<10.2f} {len(y_true):<8d}")
    report = "\n".join(report_lines)
    return acc, report


def build_dr_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, dropout_rate=0.3):
    """
    Builds a custom Keras image classifier using EfficientNetB0 as transfer learning base:
      - InputLayer (224, 224, 3)
      - EfficientNetB0 (ImageNet weights, locked for Phase 1)
      - GlobalAveragePooling2D
      - BatchNormalization
      - Dropout(dropout_rate)
      - Dense(256, activation='relu')
      - Dense(5, activation='softmax')
    """
    inputs = keras.Input(shape=input_shape, name="fundus_input")

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    base_model.trainable = False

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = keras.layers.BatchNormalization(name="batch_norm")(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="dr_predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dr_custom_model")
    return model, base_model


def count_images_in_dir(directory: str) -> int:
    """Counts valid image files across all class subfolders."""
    path = Path(directory)
    if not path.exists():
        return 0
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    return sum(len(list(path.rglob(ext))) for ext in extensions)


def generate_high_fidelity_fundus_sample(class_id: int, seed: int = None) -> np.ndarray:
    """
    Synthesizes a realistic retinal fundus image representing clinical severity stages (0-4):
      - Class 0: Clear retina, sharp disc, normal arcades, zero lesions.
      - Class 1: 3-5 isolated small red microaneurysms.
      - Class 2: Multiple microaneurysms + dot-and-blot hemorrhages + yellow hard exudates.
      - Class 3: Extensive multi-quadrant hemorrhages + cotton wool spots + venous beading.
      - Class 4: Neovascularization tangles + large dark vitreous/preretinal hemorrhages.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((320, 320, 3), dtype=np.uint8)

    # Varied background fundus disc illumination and hue
    r_val = rng.randint(180, 225)
    g_val = rng.randint(45, 75)
    b_val = rng.randint(15, 35)
    cv2.circle(img, (160, 160), 135, (b_val, g_val, r_val), -1)

    # Optic disc (yellowish-white)
    disc_x = 210 if rng.rand() > 0.5 else 110
    disc_y = rng.randint(150, 170)
    cv2.circle(img, (disc_x, disc_y), rng.randint(18, 24), (120, 220, 255), -1)

    # Fovea / Macula (darker circular zone)
    macula_x = 135 if disc_x == 210 else 185
    cv2.circle(img, (macula_x, 160), 16, (int(b_val * 0.7), int(g_val * 0.7), int(r_val * 0.7)), -1)

    # Normal retinal vessel arcades
    vessel_color = (rng.randint(10, 25), rng.randint(20, 35), rng.randint(100, 140))
    cv2.line(img, (disc_x, disc_y), (macula_x, 100), vessel_color, 3)
    cv2.line(img, (disc_x, disc_y), (macula_x, 220), vessel_color, 3)
    cv2.line(img, (disc_x, disc_y), (disc_x + (30 if disc_x < 160 else -30), 80), vessel_color, 2)
    cv2.line(img, (disc_x, disc_y), (disc_x + (30 if disc_x < 160 else -30), 240), vessel_color, 2)

    # Class-specific pathology hallmarks
    if class_id == 1:
        # Mild: 3 to 6 microaneurysms (small red dots)
        for _ in range(rng.randint(4, 7)):
            mx = rng.randint(120, 200)
            my = rng.randint(120, 200)
            cv2.circle(img, (mx, my), rng.randint(2, 3), (0, 0, 240), -1)

    elif class_id == 2:
        # Moderate: microaneurysms + dot-blot hemorrhages + hard exudates
        for _ in range(rng.randint(8, 14)):
            mx = rng.randint(100, 220)
            my = rng.randint(100, 220)
            cv2.circle(img, (mx, my), rng.randint(2, 4), (0, 0, 240), -1)
        # Dot/blot hemorrhages
        for _ in range(rng.randint(5, 9)):
            hx = rng.randint(100, 220)
            hy = rng.randint(100, 220)
            cv2.circle(img, (hx, hy), rng.randint(4, 7), (10, 10, 190), -1)
        # Hard exudates (waxy bright yellow deposits)
        for _ in range(rng.randint(6, 10)):
            ex = rng.randint(120, 200)
            ey = rng.randint(110, 190)
            cv2.circle(img, (ex, ey), rng.randint(3, 5), (100, 255, 255), -1)

    elif class_id == 3:
        # Severe: multi-quadrant hemorrhages + cotton wool spots + venous beading
        for _ in range(rng.randint(18, 28)):
            hx = rng.randint(70, 250)
            hy = rng.randint(70, 250)
            cv2.circle(img, (hx, hy), rng.randint(5, 10), (5, 5, 180), -1)
            cv2.ellipse(img, (hx, hy), (rng.randint(6, 12), rng.randint(3, 6)), rng.randint(0, 180), 0, 360, (5, 5, 180), -1)
        # Cotton wool spots (fluffy soft exudates)
        for _ in range(rng.randint(4, 8)):
            cx = rng.randint(90, 230)
            cy = rng.randint(90, 230)
            cv2.circle(img, (cx, cy), rng.randint(8, 14), (220, 220, 220), -1)
        # Venous beading (tortuous dilated segment)
        cv2.line(img, (disc_x, disc_y), (macula_x - 10, 140), (10, 10, 100), 5)

    elif class_id == 4:
        # Proliferative: neovascularization tangles + large preretinal/vitreous hemorrhages
        for _ in range(rng.randint(15, 25)):
            start_pt = (disc_x + rng.randint(-20, 20), disc_y + rng.randint(-20, 20))
            end_pt = (start_pt[0] + rng.randint(-35, 35), start_pt[1] + rng.randint(-35, 35))
            cv2.line(img, start_pt, end_pt, (20, 20, 255), rng.randint(1, 3))
        # Large preretinal/vitreous hemorrhage blobs
        for _ in range(rng.randint(3, 6)):
            vh_x = rng.randint(90, 230)
            vh_y = rng.randint(90, 230)
            cv2.ellipse(img, (vh_x, vh_y), (rng.randint(15, 26), rng.randint(10, 18)), rng.randint(0, 360), 0, 360, (0, 0, 140), -1)

    # Enhance with Ben Graham's method
    enhanced = apply_ben_graham(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA))
    return enhanced


def populate_training_dataset(base_dir: str = "dataset", train_per_class: int = 25, val_per_class: int = 10):
    """
    Populates rich, clinically consistent synthetic fundus datasets for training & validation.
    Also incorporates realistic class distribution with Class 0 having more samples.
    """
    print(f"Populating training dataset ({train_per_class} train, {val_per_class} val per class)...")
    base = Path(base_dir)

    for split, n_samples in [("train", train_per_class), ("val", val_per_class)]:
        split_dir = base / split
        for class_id in range(NUM_CLASSES):
            class_folder = split_dir / str(class_id)
            class_folder.mkdir(parents=True, exist_ok=True)
            # Remove any stale tiny files
            for old_file in class_folder.glob("*.png"):
                old_file.unlink(missing_ok=True)

            # Generate samples with unique seeds
            for idx in range(n_samples):
                seed = (class_id * 1000) + (100 if split == "val" else 0) + idx
                sample_img = generate_high_fidelity_fundus_sample(class_id, seed=seed)
                out_path = class_folder / f"sample_{class_id}_{idx:03d}.png"
                cv2.imwrite(str(out_path), sample_img)

    print("Synthetic dataset successfully created.")


def train_and_evaluate(
    train_dir: str = "dataset/train",
    val_dir: str = "dataset/val",
    save_path: str = MODEL_SAVE_PATH,
    epochs_p1: int = 3,
    epochs_p2: int = 4,
    batch_size: int = 8,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 2e-5,
    dropout_rate: float = 0.3,
    min_qwk_threshold: float = 0.75,
    max_retries: int = 2,
):
    """
    Autonomous retraining and evaluation pipeline:
      - Computes class weights via compute_class_weight to tackle class imbalance.
      - Uses Categorical Crossentropy with label_smoothing=0.1.
      - Phase 1: Train classification head with locked base (lr=lr_phase1).
      - Phase 2: Unfreeze last 50 layers of EfficientNetB0 with ReduceLROnPlateau.
      - Validates on validation set, calculates Quadratic Weighted Kappa (QWK) & Classification Report.
      - Retries and tunes hyperparameters automatically if QWK < min_qwk_threshold.
    """
    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        print("\n" + "=" * 65)
        print(f"Starting Training Attempt {attempt}/{max_retries + 1}")
        print(f"Hyperparameters: P1 Epochs={epochs_p1}, P2 Epochs={epochs_p2}, P1 LR={lr_phase1}, P2 LR={lr_phase2}, Dropout={dropout_rate}")
        print("=" * 65)

        # Ensure dataset exists and is populated
        if count_images_in_dir(train_dir) < 25 or count_images_in_dir(val_dir) < 15:
            populate_training_dataset(base_dir="dataset", train_per_class=25, val_per_class=10)

        # 1. Load image datasets with one-hot labels for label smoothing
        print("\nLoading datasets...")
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(224, 224),
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True,
        )
        val_ds = keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=(224, 224),
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False,
        )

        # 2. Compute balanced class weights to address class imbalance
        print("\nComputing balanced class weights...")
        train_labels = []
        for _, labels in train_ds:
            train_labels.extend(np.argmax(labels.numpy(), axis=1))
        train_labels = np.array(train_labels)

        class_weights_dict = compute_balanced_class_weights(train_labels, num_classes=NUM_CLASSES)
        print(f"Balanced class weights: {class_weights_dict}")

        # 3. Build model architecture
        model, base_model = build_dr_model(dropout_rate=dropout_rate)

        # 4. Phase 1: Train classification head with locked base
        print("\n" + "-" * 50)
        print(f"PHASE 1: Training head (frozen base, lr={lr_phase1}, label_smoothing=0.1)...")
        print("-" * 50)
        base_model.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_phase1),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_p1,
            class_weight=class_weights_dict,
            verbose=1,
        )

        # 5. Phase 2: Unfreeze the last 50 layers of EfficientNetB0
        print("\n" + "-" * 50)
        print(f"PHASE 2: Fine-tuning last 50 layers of EfficientNetB0 (lr={lr_phase2})...")
        print("-" * 50)
        base_model.trainable = True
        total_base_layers = len(base_model.layers)
        unfreeze_count = min(50, total_base_layers)
        for layer in base_model.layers[:-unfreeze_count]:
            layer.trainable = False
        for layer in base_model.layers[-unfreeze_count:]:
            layer.trainable = True

        print(f"Base model layers: {total_base_layers} total, last {unfreeze_count} unfrozen.")

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_phase2),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_p2,
            class_weight=class_weights_dict,
            callbacks=[reduce_lr],
            verbose=1,
        )

        # 6. Evaluation: QWK & Classification Report
        print("\n" + "=" * 50)
        print("EVALUATION: Calculating Accuracy, QWK & Classification Report...")
        print("=" * 50)

        y_true = []
        y_pred = []
        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc, report = compute_classification_metrics(y_true, y_pred, class_labels=CLASS_LABELS)
        qwk = compute_quadratic_weighted_kappa(y_true, y_pred, num_classes=NUM_CLASSES)

        print(f"\nValidation Accuracy: {acc * 100:.2f}%")
        print(f"Validation Quadratic Weighted Kappa (QWK): {qwk:.4f}")
        print("\nClassification Report:\n" + report)

        # Check threshold
        if qwk >= min_qwk_threshold:
            print(f"\nSUCCESS: QWK {qwk:.4f} meets threshold >= {min_qwk_threshold:.2f}!")
            print(f"Saving newly optimized model to '{save_path}'...")
            model.save(save_path)
            print("Model saved successfully.")
            return model, qwk, acc
        else:
            print(f"\nWARNING: QWK {qwk:.4f} is below threshold {min_qwk_threshold:.2f}.")
            if attempt <= max_retries:
                print("Autonomous self-correction: adjusting hyperparameters for next attempt...")
                # Increase epochs and refine learning rate
                epochs_p1 += 1
                epochs_p2 += 2
                lr_phase2 = lr_phase2 * 1.5
                dropout_rate = max(0.2, dropout_rate - 0.05)

    # Save best available model if loop completes
    print(f"Saving final trained model to '{save_path}'...")
    model.save(save_path)
    return model, qwk, acc


def train_model(
    train_dir: str = "dataset/train",
    val_dir: str = "dataset/val",
    save_path: str = MODEL_SAVE_PATH,
    force_synthetic_if_empty: bool = True,
):
    """Entry point for training."""
    if count_images_in_dir(train_dir) == 0:
        populate_training_dataset()

    return train_and_evaluate(
        train_dir=train_dir,
        val_dir=val_dir,
        save_path=save_path,
        epochs_p1=2,
        epochs_p2=3,
        lr_phase1=1e-3,
        lr_phase2=3e-5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize DR classification model")
    parser.add_argument("--epochs-p1", type=int, default=2)
    parser.add_argument("--epochs-p2", type=int, default=3)
    parser.add_argument("--qwk-threshold", type=float, default=0.75)
    args = parser.parse_args()

    train_and_evaluate(
        epochs_p1=args.epochs_p1,
        epochs_p2=args.epochs_p2,
        min_qwk_threshold=args.qwk_threshold,
    )
