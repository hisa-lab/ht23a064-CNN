import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from utils import set_seed, accuracy, compute_confusion_matrix


def get_loaders(data_root, img_size, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, 'val'),   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, 'test'),  transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes=2, freeze_until_layer2=False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_until_layer2:
        # conv1, bn1, layer1, layer2 を凍結
        for name, param in model.named_parameters():
            if not (name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False
    return model


def evaluate(model, loader, device):
    model.eval()
    total_acc = 0.0
    n = 0
    ys, yps = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            total_acc += accuracy(outputs, targets) * targets.size(0)
            n += targets.size(0)
            yps.extend(outputs.argmax(dim=1).cpu().tolist())
            ys.extend(targets.cpu().tolist())
    cm = compute_confusion_matrix(ys, yps, labels=[0, 1])
    return total_acc / max(n, 1), cm


def train(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_loaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers
    )

    model = build_model(num_classes=2, freeze_until_layer2=True).to(device)

    for n, p in model.named_parameters():
        print("[TRAIN]" if p.requires_grad else "[FROZEN]", n)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_ckpt = out_dir / 'best.pt'

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy(outputs, targets) * bs
            n += bs

            pbar.set_postfix({
                'loss': f"{running_loss / n:.4f}",
                'acc': f"{running_acc / n:.4f}"
            })

        # validation
        val_acc, val_cm = evaluate(model, val_loader, device)
        print(f"\n[Val] Acc: {val_acc:.4f}\nConfusion Matrix:\n{val_cm}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': model.state_dict(), 'classes': class_names}, best_ckpt)
            print(f"Saved best checkpoint to {best_ckpt}")

    # test with best model
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded best checkpoint: {best_ckpt}")

    test_acc, test_cm = evaluate(model, test_loader, device)
    print(f"\n[Test] Acc: {test_acc:.4f}\nConfusion Matrix:\n{test_cm}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True, help='path to data split root (contains train/val/test)')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default='runs')
    args = parser.parse_args()
    train(args)
