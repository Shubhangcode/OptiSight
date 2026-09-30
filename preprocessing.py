import cv2
import numpy as np
from PIL import Image
import io
from typing import Union

IMAGE_SIZE = (224, 224)


def crop_black_borders(image: np.ndarray, tol: int = 10, padding: int = 4) -> np.ndarray:
    """
    Crops uninformative black border margins from retinal fundus images using
    grayscale thresholding while preserving all peripheral retinal lesions.
    Includes safety padding to prevent clipping of marginal microaneurysms or hemorrhages.
    """
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = (gray > tol).astype(np.uint8)
    if not np.any(mask):
        return image

    x, y, w, h = cv2.boundingRect(mask)
    if w > 10 and h > 10:
        h_img, w_img = image.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        return image[y1:y2, x1:x2]
    return image


def apply_ben_graham(image: np.ndarray, sigma_x: float = 10.0) -> np.ndarray:
    """
    Applies Ben Graham's method (Gaussian blur weighted subtraction) to enhance
    retinal features like lesions, microaneurysms, and blood vessels while normalizing
    illumination across varied imaging modalities.
    
    Formula: addWeighted(image, 4, GaussianBlur(image, (0,0), sigma_x), -4, 128)
    Applies a smooth circular mask to eliminate boundary rim artifacts and
    maintain clear retinal boundaries without dark background distortion.
    """
    if image is None or image.size == 0:
        return image

    # 1. Ben Graham local contrast enhancement
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_x)
    enhanced = cv2.addWeighted(image, 4, blurred, -4, 128)

    # 2. Circular mask to prevent artificial grey halos on the non-retinal perimeter
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w / 2), int(h / 2))
    radius = int(min(w, h) * 0.485)
    cv2.circle(mask, center, radius, 255, -1)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Retain enhanced features within retinal disc, keep exterior clean
    result = np.where(mask_3d > 0, enhanced, 0).astype(np.uint8)
    return result


def preprocess_fundus_image(
    image_input: Union[str, bytes, np.ndarray, Image.Image],
    target_size: tuple = IMAGE_SIZE,
    return_rgb: bool = True,
) -> np.ndarray:
    """
    Complete clinical preprocessing pipeline:
      1. Load / decode image into BGR/RGB numpy array.
      2. Crop uninformative black borders with lesion margin preservation.
      3. Resize to target size (224x224) using area-based interpolation.
      4. Apply Ben Graham's enhancement with perimeter artifact suppression.
      5. Return as uint8 array (RGB or BGR).
    """
    # 1. Convert input to BGR numpy array
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"Could not read image from path: {image_input}")
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from bytes.")
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # 2. Crop uninformative margins
    cropped = crop_black_borders(img_bgr, tol=10, padding=4)

    # 3. Resize to target size with anti-aliasing
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    # 4. Apply Ben Graham enhancement
    enhanced = apply_ben_graham(resized, sigma_x=10.0)

    # 5. Format output color space
    if return_rgb:
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        return enhanced_rgb
    return enhanced


def prepare_for_model(
    image_input: Union[str, bytes, np.ndarray, Image.Image],
    target_size: tuple = IMAGE_SIZE,
) -> np.ndarray:
    """
    Preprocesses fundus image and prepares batch tensor for Keras EfficientNet:
    Shape: (1, 224, 224, 3), float32 in [0, 255].
    
    IMPORTANT DATA NORMALIZATION NOTICE:
    EfficientNetB0 architecture includes an internal Rescaling(scale=1./255) and
    Normalization layer. Passing float32 values in [0, 255] ensures the internal
    pipeline handles scaling without double-normalization.
    """
    processed_rgb = preprocess_fundus_image(image_input, target_size, return_rgb=True)
    batch_tensor = np.expand_dims(processed_rgb.astype(np.float32), axis=0)
    return batch_tensor


if __name__ == "__main__":
    print("Verifying enhanced preprocessing module...")
    sample = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(sample, (200, 200), 150, (30, 80, 200), -1)
    cv2.circle(sample, (250, 200), 20, (100, 200, 255), -1)
    cv2.line(sample, (120, 180), (280, 220), (20, 30, 100), 3)

    processed = preprocess_fundus_image(sample)
    tensor = prepare_for_model(sample)
    assert processed.shape == (224, 224, 3)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.max() <= 255.0 and tensor.min() >= 0.0
    print("Enhanced preprocessing verified successfully!")
