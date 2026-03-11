"""
filter_vizwiz.py
================
Reads the VizWiz-Classification annotations.json, scans every image
with yolov8n.pt, and exports only the ones containing objects that
are relevant to the blind-locator prototype.

Output structure:
  dataset_tools/
    filtered_dataset/
      images/   <- Filtered images
      labels/   <- Auto-generated YOLO .txt labels (ready for augment.py)
      classes.txt
      summary.json  <- Stats on what was found

Usage:
  python filter_vizwiz.py \\
      --ann   path/to/dataset/annotations.json \\
      --imgs  path/to/dataset/images \\
      --out   filtered_dataset
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ── TARGET CLASSES ────────────────────────────────────────────────────────────
# COCO class ID → label name (subset of YOLO's 80 classes)
# Chosen because a blind person would realistically search for these
TARGET_CLASSES = {
    39: "bottle",       # Water bottle, any bottle
    41: "cup",          # Cup, mug
    67: "cell phone",   # Smartphone
    63: "laptop",       # Laptop computer
    65: "remote",       # Remote control
    73: "book",         # Book, notebook
    64: "mouse",        # Computer mouse
    66: "keyboard",     # Computer keyboard
    56: "chair",        # Chair, seat
    62: "tv",           # TV, monitor
    74: "clock",        # Clock, alarm clock
    24: "backpack",     # Backpack, school bag
    25: "umbrella",     # Umbrella
    26: "handbag",      # Handbag, purse
    44: "spoon",        # Spoon
    43: "knife",        # Knife
    42: "fork",         # Fork
    45: "bowl",         # Bowl
    76: "scissors",     # Scissors
    79: "toothbrush",   # Toothbrush
    46: "banana",       # Banana
    47: "apple",        # Apple
    49: "orange",       # Orange
    75: "vase",         # Vase
    77: "teddy bear",   # Stuffed animal
}

# Build local class index (0-based) from the dict above
COCO_TO_LOCAL = {coco_id: idx for idx, coco_id in enumerate(TARGET_CLASSES.keys())}
CLASS_NAMES   = list(TARGET_CLASSES.values())

# Confidence threshold — lower = more images kept (review more), higher = fewer but cleaner
CONFIDENCE = 0.25
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Filter VizWiz images for blind-locator training")
    parser.add_argument("--ann",  default="dataset/annotations.json",
                        help="Path to VizWiz annotations.json")
    parser.add_argument("--imgs", default="dataset/images",
                        help="Path to VizWiz images directory")
    parser.add_argument("--out",  default="filtered_dataset",
                        help="Output directory (inside dataset_tools/)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help=f"YOLO detection confidence (default {CONFIDENCE})")
    parser.add_argument("--model", default=None,
                        help="Path to YOLO model .pt file (default: auto-find yolov8n.pt)")
    return parser.parse_args()


def find_model():
    """Auto-find yolov8n.pt relative to this script."""
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / "ai_module" / "models" / "yolov8n.pt"
    if model_path.exists():
        return str(model_path)
    raise FileNotFoundError(
        "Could not find yolov8n.pt. Run the main pipeline once first, "
        "or pass --model path/to/yolov8n.pt"
    )


def main():
    args = parse_args()

    ann_path  = Path(args.ann)
    imgs_dir  = Path(args.imgs)
    out_dir   = Path(__file__).parent / args.out

    # Validate inputs
    if not ann_path.exists():
        print(f"[ERROR] annotations.json not found at: {ann_path}")
        print("  Download VizWiz dataset and pass --ann <path>")
        return
    if not imgs_dir.exists():
        print(f"[ERROR] Images directory not found at: {imgs_dir}")
        return

    # Setup output folders
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Write classes.txt
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES))

    # Load annotations
    print(f"[Filter] Loading annotations from {ann_path}...")
    with open(ann_path) as f:
        annotations = json.load(f)

    # Get image filenames - handle both list-of-strings and list-of-dicts
    raw_images = annotations.get("images", [])
    if isinstance(raw_images[0], dict):
        image_filenames = [img["file_name"] for img in raw_images]
    else:
        image_filenames = [str(img) for img in raw_images]

    print(f"[Filter] Total images in VizWiz: {len(image_filenames)}")
    print(f"[Filter] Target classes: {', '.join(CLASS_NAMES)}")

    # Load YOLO model
    model_path = args.model or find_model()
    print(f"[Filter] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # Stats tracking
    class_counts = {name: 0 for name in CLASS_NAMES}
    kept   = 0
    skipped = 0
    errors  = 0

    print(f"\n[Filter] Scanning {len(image_filenames)} images...\n")

    for filename in tqdm(image_filenames, unit="img"):
        img_path = imgs_dir / filename
        if not img_path.exists():
            errors += 1
            continue

        try:
            results = model(str(img_path), conf=args.conf, verbose=False)
            r = results[0]
        except Exception as e:
            errors += 1
            continue

        label_lines = []
        for box in r.boxes:
            coco_cls = int(box.cls[0])
            if coco_cls not in TARGET_CLASSES:
                continue
            local_cls = COCO_TO_LOCAL[coco_cls]
            cx, cy, bw, bh = box.xywhn[0].tolist()
            label_lines.append(f"{local_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_counts[CLASS_NAMES[local_cls]] += 1

        if not label_lines:
            skipped += 1
            continue

        # Copy image + write label
        shutil.copy2(img_path, out_images / filename)
        label_file = out_labels / (Path(filename).stem + ".txt")
        label_file.write_text("\n".join(label_lines))
        kept += 1

    # Summary
    summary = {
        "total_input"  : len(image_filenames),
        "kept"         : kept,
        "skipped_no_target": skipped,
        "errors"       : errors,
        "class_counts" : class_counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print report
    print(f"\n{'='*55}")
    print(f"  VizWiz Filter Complete")
    print(f"{'='*55}")
    print(f"  Total input images : {len(image_filenames)}")
    print(f"  Kept (relevant) : {kept}  ({kept*100//len(image_filenames)}%)")
    print(f"  Skipped         : {skipped}")
    print(f"  Errors          : {errors}")
    print(f"\n  Objects found per class:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {cls_name:<20} {count}")
    print(f"\n  Output → {out_dir}")
    print(f"\n  Next step: Run augment.py on this filtered_dataset/")
    print(f"  Update augment.py INPUT_DIR to: dataset_tools/filtered_dataset")


if __name__ == "__main__":
    main()
