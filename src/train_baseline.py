"""
Baseline experiment: train a standard ResNet18 transfer-learning classifier on ONE
glaucoma dataset, then measure how much accuracy/AUC drops when the same model is
evaluated on the OTHER datasets without any fine-tuning. That gap is the empirical
evidence for the whole project (domain shift across datasets).

Usage (defaults are sane for a first run):
    python src/train_baseline.py
    python src/train_baseline.py --train-dataset G1020 --epochs 10
    python src/train_baseline.py --max-samples 40 --epochs 1   # quick smoke test

Run this from the project root (so data/ and checkpoints/ resolve correctly).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import models, transforms

from datasets import FundusDataset, load_unified_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # binary classification as 2-way softmax
    return model


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_labels, all_probs_pos = [], []
    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(glaucoma)
        all_probs_pos.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs_pos = np.array(all_probs_pos)
    all_preds = (all_probs_pos >= 0.5).astype(int)

    accuracy = (all_preds == all_labels).mean()
    try:
        auc = roc_auc_score(all_labels, all_probs_pos)
    except ValueError:
        auc = float("nan")  # only one class present in this split

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # recall on glaucoma
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # recall on normal

    return {
        "n": len(all_labels),
        "accuracy": round(float(accuracy), 4),
        "auc": round(float(auc), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
    }


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    unified = load_unified_labels()

    train_dataset_df = unified[unified["dataset"] == args.train_dataset].copy()
    if args.max_samples:
        train_dataset_df = train_dataset_df.sample(
            n=min(args.max_samples, len(train_dataset_df)), random_state=42
        )

    # args.train_dataset has no official split (G1020) -> make our own stratified 80/20 split.
    # Stratifying keeps the glaucoma/normal ratio consistent between train and val.
    train_df, val_df = train_test_split(
        train_dataset_df,
        test_size=0.2,
        stratify=train_dataset_df["label"],
        random_state=42,
    )
    print(f"{args.train_dataset}: {len(train_df)} train / {len(val_df)} val")

    train_ds = FundusDataset(train_df, transform=train_transform)
    val_ds = FundusDataset(val_df, transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Class weights to counter G1020's ~2.4:1 normal:glaucoma imbalance.
    class_counts = train_df["label"].value_counts().sort_index()
    class_weights = torch.tensor(
        [1.0 / class_counts.get(0, 1), 1.0 / class_counts.get(1, 1)], dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * 2
    print(f"Class weights (normal, glaucoma): {class_weights.tolist()}")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -1.0
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_path = CHECKPOINT_DIR / f"resnet18_{args.train_dataset}_best.pt"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - train_loss={train_loss:.4f} "
            f"- val_acc={val_metrics['accuracy']:.4f} - val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best (AUC={best_auc:.4f}), saved to {best_path}")

    # --- Reload best checkpoint and run the cross-dataset generalization check ---
    model.load_state_dict(torch.load(best_path, map_location=device))

    results = {"train_dataset": args.train_dataset, "in_domain_val": val_metrics}
    results["in_domain_val"] = evaluate(model, val_loader, device)

    other_datasets = [d for d in unified["dataset"].unique() if d != args.train_dataset]
    results["cross_dataset"] = {}
    for other in other_datasets:
        other_df = unified[unified["dataset"] == other]
        other_ds = FundusDataset(other_df, transform=eval_transform)
        other_loader = DataLoader(other_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        results["cross_dataset"][other] = evaluate(model, other_loader, device)

    print("\n=== Summary: in-domain vs cross-dataset ===")
    print(f"Trained on {args.train_dataset}:")
    print(f"  In-domain val   : {results['in_domain_val']}")
    for other, metrics in results["cross_dataset"].items():
        print(f"  Cross -> {other:10s}: {metrics}")

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"baseline_{args.train_dataset}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", default="G1020", choices=["G1020", "ORIGA", "REFUGE"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="cap dataset size, for quick smoke tests")
    args = parser.parse_args()
    train(args)
