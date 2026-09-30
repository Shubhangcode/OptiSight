import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from huggingface_hub import hf_hub_download
import cv2
import numpy as np

GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def load_model():
    print("Downloading model from Hugging Face... (first time may take a few minutes)")
    model_path = hf_hub_download(
        repo_id="Arko007/diabetic-retinopathy-v1",
        filename="best_model.h5"
    )
    model = keras.saving.load_model(model_path)
    print("✅ Model loaded successfully!")
    print("Input shape :", model.input_shape)
    print("Output shape:", model.output_shape)
    return model

def preprocess_image(image_path, target_size=(384, 384)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

def predict(model, image_path):
    img = preprocess_image(image_path)
    batch = np.expand_dims(img, axis=0)
    preds = model.predict(batch, verbose=0)[0]
    
    predicted_class = int(np.argmax(preds))
    confidence = float(preds[predicted_class])
    
    return {
        "grade": predicted_class,
        "label": GRADE_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": preds.tolist()
    }