from utils import load_model, predict, GRADE_NAMES
import matplotlib.pyplot as plt
import cv2

model = load_model()

def predict_and_show(image_path):
    result = predict(model, image_path)
    
    print("\n" + "=" * 55)
    print(f"Predicted Grade : {result['grade']}  →  {result['label']}")
    print(f"Confidence      : {result['confidence']*100:.2f}%")
    print("-" * 55)
    print("Class probabilities:")
    for i, p in enumerate(result["probabilities"]):
        bar = "█" * int(p * 30)
        print(f"  {i} ({GRADE_NAMES[i]:18s}): {p*100:6.2f}%  {bar}")
    print("=" * 55)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"{result['label']} ({result['confidence']*100:.1f}%)")
    plt.axis("off")
    plt.show()

    return result


if __name__ == "__main__":
    image_path = "test_image.jpg"   # ← change this
    predict_and_show(image_path)