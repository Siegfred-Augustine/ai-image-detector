"""
scripts/prepare_class_folder.py

Builds a data/raw/real/ or data/raw/fake/ folder from one or more
source directories of downloaded images, with optional random
subsampling to a target count. Handles:

  - FFHQ (Kaggle): one flat folder of ~52,000 face images -> subsample
    down to e.g. 2,500 for data/raw/real/.
  - AI-Face (or any dataset split across multiple downloads/agents):
    pass --src once per downloaded chunk -> recursively collects every
    image found under each, pools them, then optionally subsamples the
    pooled set down to a target count for data/raw/fake/.

Usage:
    # FFHQ -> 2500 real images
    python scripts/prepare_class_folder.py \
        --src /path/to/ffhq_extracted \
        --dst data/raw/real \
        --n 2500 --seed 42

    # AI-Face, downloaded in 3 separate chunks -> matched fake count
    python scripts/prepare_class_folder.py \
        --src /path/to/ai_face_part1 \
        --src /path/to/ai_face_part2 \
        --src /path/to/ai_face_part3 \
        --dst data/raw/fake \
        --n 2500 --seed 42 --prefix aiface_

Copies files by default (originals stay untouched). Use --symlink
instead if you don't want to duplicate disk space (Linux/macOS only).
"""

import argparse
import random
import shutil
from pathlib import Path

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_images(src_dirs):
    paths = []
    for src in src_dirs:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Source path not found: {src}")
        for p in src_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Build a data/raw/<class> folder from one or more source dirs, with optional random subsampling."
    )
    parser.add_argument(
        "--src", action="append", required=True,
        help="Source directory to scan (repeatable -- e.g. --src dir1 --src dir2 for multi-part downloads like AI-Face).",
    )
    parser.add_argument("--dst", required=True, help="Destination folder, e.g. data/raw/real or data/raw/fake")
    parser.add_argument("--n", type=int, default=None, help="Target number of images. Omit to use every image found.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling (reproducible -- keep this fixed if you re-run it).")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copying (saves disk space; Linux/macOS only).")
    parser.add_argument("--prefix", default="", help="Filename prefix to avoid collisions across multiple sources, e.g. 'ffhq_' or 'aiface_'.")
    args = parser.parse_args()

    all_images = collect_images(args.src)
    print(f"Found {len(all_images)} images across {len(args.src)} source folder(s).")

    if not all_images:
        raise RuntimeError("No images found -- check your --src paths.")

    if args.n is not None:
        if args.n < len(all_images):
            rng = random.Random(args.seed)
            all_images = rng.sample(all_images, args.n)
            print(f"Subsampled down to {len(all_images)} images (seed={args.seed}).")
        elif args.n > len(all_images):
            print(f"WARNING: requested {args.n} but only found {len(all_images)}; using all {len(all_images)}.")

    dst_path = Path(args.dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(all_images):
        dst_file = dst_path / f"{args.prefix}{i:06d}{img_path.suffix.lower()}"
        if args.symlink:
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            dst_file.symlink_to(img_path.resolve())
        else:
            shutil.copy2(img_path, dst_file)

    print(f"Done -- {len(all_images)} images written to {dst_path}/")


if __name__ == "__main__":
    main()