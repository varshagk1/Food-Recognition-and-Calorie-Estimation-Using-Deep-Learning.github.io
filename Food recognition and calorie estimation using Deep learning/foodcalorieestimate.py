import os
import argparse
import shutil
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

# -----------------------------
# Calorie lookup table
# -----------------------------
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
    "bruschetta": 180
}

# -----------------------------
# Image transforms
# -----------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Model creation
# -----------------------------
def create_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -----------------------------
# Training function
# -----------------------------
def train_model(data_dir, epochs, batch_size, lr, device, num_workers):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    model = create_model(len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss:.3f}  Val Acc: {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "checkpoints/best.pth")

    print("✅ Training finished. Best model saved at checkpoints/best.pth")

    # save class mapping
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    torch.save(idx_to_class, "checkpoints/classes.pth")

# -----------------------------
# Predict one image
# -----------------------------
def predict_image(model, image_path, device, idx_to_class, topk):
    img = Image.open(image_path).convert("RGB")
    img = transform_val(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(topk)

    results = []
    for p, idx in zip(top_probs[0], top_idxs[0]):
        cls = idx_to_class[idx.item()]
        cal = CALORIE_MAP.get(cls, 0)
        results.append((cls, p.item(), cal))

    return results

# -----------------------------
# Predict all images in a folder
# -----------------------------
def predict_image_dir(model, image_dir, device, idx_to_class, topk):
    model.eval()
    os.makedirs("predictions", exist_ok=True)

    images = [f for f in os.listdir(image_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        results = predict_image(model, img_path, device, idx_to_class, topk)

        print(f"\nPredictions for {img_path}:")
        for cls, prob, cal in results:
            print(f"  {cls} (p={prob:.3f}) -> est {cal} kcal")

        # Save to predicted folder (top-1)
        pred_class = results[0][0]
        out_dir = os.path.join("predictions", pred_class)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(img_path, out_dir)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "predict"])
    parser.add_argument("--data-dir")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--model")
    parser.add_argument("--image")
    parser.add_argument("--image-dir")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        train_model(args.data_dir, args.epochs,
                    args.batch_size, args.lr,
                    device, args.num_workers)

    elif args.mode == "predict":
        assert args.model is not None, "--model required in predict mode"

        idx_to_class = torch.load("checkpoints/classes.pth")
        model = create_model(len(idx_to_class))
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        model.eval()

        if args.image:
            res = predict_image(model, args.image,
                                device, idx_to_class, args.topk)
            print(f"\nPredictions for {args.image}:")
            for cls, prob, cal in res:
                print(f"  {cls} (p={prob:.3f}) -> est {cal} kcal")

        elif args.image_dir:
            predict_image_dir(model, args.image_dir,
                              device, idx_to_class, args.topk)

        else:
            raise ValueError("Provide --image or --image-dir")

if __name__ == "__main__":
    main()
