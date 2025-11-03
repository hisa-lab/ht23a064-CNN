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


# =========================
# データローダー
# 変更点：
#  - 学習：RandomResizedCrop のみに一本化（アスペクト比保持、二重リサイズ撤廃）
#  - 評価：Resize(短辺合わせ) → CenterCrop（歪み防止）
#  - scale=(0.85, 1.0)
# =========================
def get_loaders(data_root, img_size, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),  # 念のため3ch統一
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),  # 念のため3ch統一
        transforms.Resize(img_size),          # 短辺=img_size（アスペクト比保持）
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, 'val'),   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, 'test'),  transform=eval_tfms)

    # 再現性を少し上げたい場合の Generator
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


# =========================
# モデル構築
# =========================
def build_model(num_classes=2, tune_mode="l34", pretrained=True):
    """
    tune_mode:
      - "l34":    layer3, layer4, fc を学習（デフォルト）
      - "l2_4":   layer2, layer3, layer4, fc を学習
      - "l1_4":   layer1〜fc（ほぼ全層）
      - "all":    conv1〜fc まで完全全層
    pretrained:
      - True  : ImageNet 事前学習重みで初期化（転移学習）
      - False : ランダム初期化（転移学習なし）
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # いったん全凍結（後で必要層のみ解凍）
    for p in model.parameters():
        p.requires_grad = False

    # 解凍ヘルパー
    def unfreeze(prefixes):
        for n, p in model.named_parameters():
            if any(n.startswith(pref) for pref in prefixes):
                p.requires_grad = True

    # 転移学習なしの場合、tune_mode を all にする
    if (not pretrained) and (tune_mode != "all"):
        print(f"[INFO] pretrained=False なので tune_mode='{tune_mode}' -> 'all' に変更します。", flush=True)
        tune_mode = "all"

    if tune_mode == "l34":
        unfreeze(["layer3", "layer4", "fc"])
    elif tune_mode == "l2_4":
        unfreeze(["layer2", "layer3", "layer4", "fc"])
    elif tune_mode == "l1_4":
        unfreeze(["layer1", "layer2", "layer3", "layer4", "fc"])
    elif tune_mode == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown tune_mode: {tune_mode}")

    return model


# =========================
# 評価関数
# =========================
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


# =========================
# 学習関数
# =========================
def train(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_loaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers
    )

    # Faceは real/fake の2クラス前提
    model = build_model(num_classes=2, tune_mode=args.tune_mode, pretrained=args.pretrained).to(device)

    # 確認ログ
    print(f"[INFO] pretrained={args.pretrained}  tune_mode={args.tune_mode}", flush=True)
    print(f"[INFO] classes={class_names}", flush=True)
    for n, p in model.named_parameters():
        print(("[TRAIN] " if p.requires_grad else "[FROZEN] ") + n, flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable params: {trainable:,} / frozen params: {frozen:,}", flush=True)

    # 層別LR
    def lr_mult(name):
        if   name.startswith(("conv1","bn1","layer1")): return 0.1
        elif name.startswith("layer2"):                 return 0.3
        elif name.startswith(("layer3","layer4")):      return 1.0
        elif name.startswith("fc"):                     return 3.0
        else:                                           return 1.0

    param_groups = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            m = lr_mult(n)
            param_groups.setdefault(m, []).append(p)

    optimizer = optim.Adam(
        [{"params": ps, "lr": args.lr * mult} for mult, ps in param_groups.items()],
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    # === 保存ファイル名を分ける
    best_name = "best_transfer.pt" if args.pretrained else "best_scratch.pt"
    best_ckpt = out_dir / best_name
    common_best = out_dir / "best.pt"  # 互換用

    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"\n=== Epoch {epoch}/{args.epochs} start ===", flush=True)
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
            payload = {'model': model.state_dict(), 'classes': class_names}
            torch.save(payload, best_ckpt)
            torch.save(payload, common_best)  # 既存パイプラインへの互換
            print(f"Saved best checkpoint to {best_ckpt} (and {common_best})")

    # test with best model
    target_best = best_ckpt if best_ckpt.exists() else common_best
    if target_best.exists():
        ckpt = torch.load(target_best, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded best checkpoint: {target_best}")

    test_acc, test_cm = evaluate(model, test_loader, device)
    print(f"\n[Test] Acc: {test_acc:.4f}\nConfusion Matrix:\n{test_cm}")


# =========================
# エントリポイント
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True, help='path to data split root (train/val/test)')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default='runs')
    parser.add_argument('--tune-mode', type=str, default='l34',
                        choices=['l34', 'l2_4', 'l1_4', 'all'],
                        help='which layers to fine-tune')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use ImageNet pretrained weights (default)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='train from scratch (no transfer)')
    parser.set_defaults(pretrained=True)

    args = parser.parse_args()
    train(args)