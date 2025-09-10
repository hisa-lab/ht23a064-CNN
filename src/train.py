import argparse
train_loader, val_loader, test_loader, class_names = get_loaders(
args.data_root, args.img_size, args.batch_size, args.num_workers
)


model = build_model(num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
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