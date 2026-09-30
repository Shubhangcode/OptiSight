
import os
import glob
import pandas as pd
from sklearn.metrics import classification_report
from utils import load_model, preprocess_image, GRADE_NAMES
import numpy as np

model = load_model()

image_paths = (
    glob.glob("IDRiD/**/*.jpg", recursive=True) +
    glob.glob("IDRiD/**/*.png", recursive=True) +
    glob.glob("IDRiD/**/*.jpeg", recursive=True)
)
print(f"Total images found: {len(image_paths)}")

train_labels_path = "IDRiD/IDRiD/Train/annotations.csv"
test_labels_path  = "IDRiD/IDRiD/Test/annotations.csv"

all_labels_df = pd.DataFrame()
for path in [train_labels_path, test_labels_path]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_labels_df = pd.concat([all_labels_df, df], ignore_index=True)
        print(f"Loaded labels from: {path}")

if all_labels_df.empty:
    raise FileNotFoundError("No labels CSV found. Download the IDRiD dataset first.")

image_to_grade = dict(zip(all_labels_df["Image name"], all_labels_df["Retinopathy grade"]))

true_labels = []
predicted_labels = []
correct = 0
total = 0

print("\nEvaluating...")

for img_path in image_paths:
    filename = os.path.basename(img_path)
    if filename in image_to_grade:
        true_grade = image_to_grade[filename]
        
        img = preprocess_image(img_path)
        batch = np.expand_dims(img, axis=0)
        preds = model.predict(batch, verbose=0)[0]
        pred_grade = int(np.argmax(preds))

        true_labels.append(true_grade)
        predicted_labels.append(pred_grade)

        if pred_grade == true_grade:
            correct += 1
        total += 1

if total > 0:
    accuracy = (correct / total) * 100
    print(f"\n{'='*50}")
    print(f"Accuracy on {total} images: {accuracy:.2f}%")
    print(f"Correct: {correct} / {total}")
    print(f"{'='*50}\n")

    target_names = [GRADE_NAMES[i] for i in sorted(GRADE_NAMES.keys())]
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=target_names, zero_division=0))
else:
    print("No images with matching labels were found.")