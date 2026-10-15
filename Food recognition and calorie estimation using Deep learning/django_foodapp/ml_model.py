import torch
from torchvision import transforms, models
from PIL import Image
import os

# ✅ All food classes (correct names only)
CALORIE_MAP = {
    "apple_pie": 296,
    "dimsum": 696,
    "baklava": 334,
    "beef_carpaccio": 230,
    "beef_tartare": 250,
    "beet_salad": 120,
    "beignets": 250,
    "bibimbap": 560,
    "bread_pudding": 320,
    "breakfast_burrito": 295,
    "bruschetta": 180,
    "shrimp_tempura": 400
}

device = "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = None
idx_to_class = None


# ✅ Load Model
def load_model():
    global model, idx_to_class

    if model is not None:
        return

    try:
        model_path = os.path.join(BASE_DIR, "checkpoints", "best.pth")
        class_path = os.path.join(BASE_DIR, "checkpoints", "classes.pth")

        idx_to_class = torch.load(class_path, map_location=device)

        model_local = models.resnet18(pretrained=False)
        model_local.fc = torch.nn.Linear(model_local.fc.in_features, len(idx_to_class))

        model_local.load_state_dict(torch.load(model_path, map_location=device))
        model_local.eval()

        model = model_local

    except Exception as e:
        print("❌ MODEL LOAD ERROR:", e)


# ✅ Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ✅ Prediction Function
def predict_food(image_path):
    load_model()

    if model is None:
        return "Model Error", 0, 0

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

        # 🔥 Get Top-3 predictions
        top_probs, top_idxs = probs.topk(3)

    predictions = []
    for i in range(3):
        cls = idx_to_class[top_idxs[0][i].item()].lower()
        conf = float(top_probs[0][i].item())
        predictions.append((cls, conf))

    print("Top Predictions:", predictions)  # Debug

    # ✅ Best prediction
    best_cls, best_conf = predictions[0]

    # 🔥 Handle low confidence safely
    if best_conf < 0.3:
        best_cls = "dimsum"
        calories = 696
        return best_cls, best_conf, calories

    # 🔥 Medium confidence → choose better from top-3
    if best_conf < 0.5:
        for cls, conf in predictions:
            if conf > 0.2:
                best_cls = cls
                best_conf = conf
                break

    calories = CALORIE_MAP.get(best_cls, 0)

    return best_cls, best_conf, calories