from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Resolve sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from inference import (
    process_image,
    process_folder,
    process_video,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    CONF_DEFAULT,
)



# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Safety Monitor — Quick Demo")
    p.add_argument(
        "--source", "-s",
        default=None,
        help="Image, folder, or video to process.  "
             "Defaults to data/test_images/ if it exists.",
    )
    p.add_argument(
        "--model", "-m",
        default=str(DEFAULT_MODEL),
    )
    p.add_argument(
        "--show", action="store_true", default=False,
        help="Display annotated image/video in a window.",
    )
    args = p.parse_args()

    # -----------------------------------------------------------------------
    # Locate model
    # -----------------------------------------------------------------------
    model_path = Path(args.model)
    if not model_path.exists():
        print("=" * 60)
        print("  ⚠️  Model weights not found!")
        print(f"  Expected: {model_path}")
        print()
        print("  To fix this:")
        print("  1. Open your Google Colab notebook.")
        print("  2. Run the download cell:")
        print("       from google.colab import files")
        print("       files.download('/content/runs/safety_monitor_v1/weights/best.pt')")
        print("  3. Move the downloaded best.pt into your project:")
        print("       models/best.pt")
        print("=" * 60)
        sys.exit(1)

    print(f"[INFO] Loading models...")
    model_ppe = YOLO(str(model_path))
    model_person = YOLO("yolov8n.pt") # Base COCO model

    # -----------------------------------------------------------------------
    # Locate source
    # -----------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    default_test_dir = project_root / "data" / "test_images"

    if args.source:
        source = Path(args.source)
    elif default_test_dir.exists():
        source = default_test_dir
        print(f"[INFO] Using default test folder: {source}")
    else:
        print()
        print("  No --source given and data/test_images/ not found.")
        print("  Please supply a source:  python src/run_demo.py --source <path>")
        print()
        sys.exit(0)

    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Run inference
    # -----------------------------------------------------------------------
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if source.is_dir():
        process_folder(
            model_ppe, model_person, source, output_dir,
            conf=CONF_DEFAULT,
            save_json=True,
            show=args.show,
        )
    elif source.suffix.lower() in video_exts:
        process_video(
            model_ppe, model_person, source, output_dir,
            conf=CONF_DEFAULT,
            save_json=True,
            show=args.show,
        )
    else:
        process_image(
            model_ppe, model_person, source, output_dir,
            conf=CONF_DEFAULT,
            save_json=True,
            show=args.show,
        )

    print()
    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
