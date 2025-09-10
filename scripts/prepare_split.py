import argparse
import random
import shutil
from pathlib import Path




def gather_images(class_dir):
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [p for p in Path(class_dir).rglob('*') if p.suffix.lower() in img_exts]




def split_list(items, val_ratio, test_ratio, seed=42):
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val = items[:n_val]
    test = items[n_val:n_val+n_test]
    train = items[n_val+n_test:]
    return train, val, test




def place(files, out_dir, class_name, mode='symlink'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = out / class_name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if mode == 'copy':
            shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src.resolve())
            except Exception:
                shutil.copy2(src, dst)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-dir', required=True, help='path to REAL images root')
    ap.add_argument('--fake-dir', required=True, help='path to FAKE images root')
    ap.add_argument('--out', required=True, help='output split root (will create train/val/test)')
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--test-ratio', type=float, default=0.1)
    ap.add_argument('--mode', choices=['symlink', 'copy'], default='symlink')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()


    real = gather_images(args.real_dir)
    fake = gather_images(args.fake_dir)


    for cls_name, files in [('real', real), ('fake', fake)]:
        tr, va, te = split_list(files, args.val_ratio, args.test_ratio, args.seed)
        place(tr, Path(args.out) / 'train', cls_name, args.mode)
        place(va, Path(args.out) / 'val', cls_name, args.mode)
        place(te, Path(args.out) / 'test', cls_name, args.mode)

    print('Done. Split created at:', args.out)

if __name__ == '__main__':
    main()