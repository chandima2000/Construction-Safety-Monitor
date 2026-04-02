from __future__ import annotations

import argparse
import sys
import json
import cv2
import datetime
from pathlib import Path
from typing import List, Tuple

# Make sure sibling modules resolve when running src/inference.py directly
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

from rules import Detection, apply_rules, CONF_DEFAULT
from alerts import print_report, save_json_report, to_dict


# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
DEFAULT_MODEL  = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "inference"

# Colours for bounding-box overlay (BGR)
COLOUR_VIOLATION = {
    "no-helmet":  (0,   0,   255),   # red
    "no-vest":    (0,   0,   255),   # red
    "no-goggles": (0, 165,   255),   # orange
    "no-boots":   (0, 165,   255),   # orange
    "no-gloves":  (0, 255,   255),   # yellow
}
COLOUR_COMPLIANT  = (0, 200, 0)      # green
COLOUR_PERSON     = (200, 200, 200)  # light grey

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _draw_detections(
    frame: "cv2.Mat",
    detections: List[Detection],
    scene_result,
) -> "cv2.Mat":
    """
    Draw bounding boxes and labels on a frame.

    Green  = compliant PPE
    Red    = critical violation
    Orange = moderate violation
    Yellow = low-severity violation
    Grey   = person
    """
    frame = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        label = f"{det.class_name} {det.confidence:.2f}"

        if det.class_name in COLOUR_VIOLATION:
            colour = COLOUR_VIOLATION[det.class_name]
        elif det.class_name == "person":
            colour = COLOUR_PERSON
        else:
            colour = COLOUR_COMPLIANT

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Scene status overlay (top-left banner)
    sev   = scene_result.scene_severity
    text  = f"{sev.emoji()} {sev.value}  violations: {scene_result.violation_count}"
    bg_colour = (0, 0, 200) if sev.value == "CRITICAL" else \
                (0, 140, 200) if sev.value == "MODERATE" else \
                (0, 170, 0)   if sev.value == "SAFE" else \
                (0, 200, 230)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), bg_colour, -1)
    cv2.putText(frame, text, (10, 22),
                FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference_on_frame(
    model_ppe: YOLO,
    model_person: YOLO,
    frame: "cv2.Mat",
    conf_threshold: float = CONF_DEFAULT,
) -> Tuple[List[Detection], object]:
    """
    Run YOLOv8 inference on a single frame using two models:
    1. model_person (COCO) to reliably detect workers
    2. model_ppe (Custom) to detect PPE and violations

    Returns
    -------
    (detections, scene_result)
    """
    # 1. Custom model for PPE and violations
    results_ppe = model_ppe.predict(
        source=frame,
        conf=conf_threshold,
        verbose=False,
    )

    detections: List[Detection] = []
    
    for r in results_ppe:
        boxes = r.boxes
        for box in boxes:
            cls_id  = int(box.cls[0])
            cls_name = model_ppe.names[cls_id]
            # Ignore the unlearned 'person' class from the custom model
            if cls_name == "person":
                continue
                
            conf    = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                class_name=cls_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
            ))

    # 2. Standard COCO model for reliable person detection (class 0)
    results_person = model_person.predict(
        source=frame,
        conf=conf_threshold,
        classes=[0], # only detect 'person'
        verbose=False,
    )
    
    for r in results_person:
        boxes = r.boxes
        for box in boxes:
            conf    = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                class_name="person",
                confidence=conf,
                bbox=(x1, y1, x2, y2),
            ))

    scene_result = apply_rules(detections, conf_threshold)
    return detections, scene_result


# ---------------------------------------------------------------------------
# Image / video / webcam runners
# ---------------------------------------------------------------------------

def process_image(
    model_ppe: YOLO,
    model_person: YOLO,
    image_path: str | Path,
    output_dir: Path,
    conf: float,
    save_json: bool,
    show: bool,
) -> None:
    """Process a single image file."""
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[WARN] Cannot read image: {image_path}")
        return

    detections, result = run_inference_on_frame(model_ppe, model_person, frame, conf)
    print_report(result, image_path=str(image_path.name))

    annotated = _draw_detections(frame, detections, result)

    # Save annotated image
    out_img = output_dir / f"annotated_{image_path.name}"
    cv2.imwrite(str(out_img), annotated)
    print(f"[INFO] Annotated image saved → {out_img}")

    # Optional JSON
    if save_json:
        out_json = output_dir / f"{image_path.stem}_report.json"
        save_json_report(result, str(out_json), str(image_path))

    if show:
        cv2.imshow("Safety Monitor", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_folder(
    model_ppe: YOLO,
    model_person: YOLO,
    folder: str | Path,
    output_dir: Path,
    conf: float,
    save_json: bool,
    show: bool,
) -> None:
    """Process all images in a folder."""
    folder = Path(folder)
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in folder.iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"[WARN] No images found in {folder}")
        return

    print(f"[INFO] Found {len(images)} image(s) in {folder}")

    summary = []
    for img_path in sorted(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        detections, result = run_inference_on_frame(model_ppe, model_person, frame, conf)
        print_report(result, image_path=str(img_path.name))

        annotated = _draw_detections(frame, detections, result)
        out_img   = output_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(out_img), annotated)

        if save_json:
            summary.append(to_dict(result, str(img_path)))

        if show:
            cv2.imshow("Safety Monitor", annotated)
            if cv2.waitKey(500) & 0xFF == ord("q"):
                break

    if save_json and summary:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = output_dir / f"batch_report_{ts}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Batch JSON report saved → {out_json}")

    if show:
        cv2.destroyAllWindows()


def process_video(
    model_ppe: YOLO,
    model_person: YOLO,
    video_path: str | Path,
    output_dir: Path,
    conf: float,
    save_json: bool,
    show: bool,
    skip_frames: int = 5,
) -> None:
    """
    Process a video file.

    Parameters
    ----------
    skip_frames : Run inference every N frames (default=5) to keep it real-time.
                  Set to 1 to analyse every frame.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = output_dir / f"annotated_{video_path.stem}.mp4"
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx     = 0
    last_result   = None
    last_dets     = []
    json_log      = []

    print(f"[INFO] Processing video: {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            last_dets, last_result = run_inference_on_frame(model_ppe, model_person, frame, conf)
            if save_json:
                json_log.append(to_dict(last_result, str(video_path), frame_idx))

        if last_result:
            annotated = _draw_detections(frame, last_dets, last_result)
        else:
            annotated = frame

        writer.write(annotated)

        if show:
            cv2.imshow("Safety Monitor — Video", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Annotated video saved → {out_video_path}")

    if save_json and json_log:
        out_json = output_dir / f"{video_path.stem}_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(json_log, f, indent=2)
        print(f"[INFO] Video JSON report saved → {out_json}")


def process_webcam(
    model_ppe: YOLO,
    model_person: YOLO,
    cam_id: int,
    output_dir: Path,
    conf: float,
    save_json: bool,
    skip_frames: int = 3,
) -> None:
    """Process live webcam stream. Press Q to quit."""
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}")
        return

    print("[INFO] Webcam started. Press Q to quit.")

    frame_idx   = 0
    last_result = None
    last_dets   = []
    json_log    = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            last_dets, last_result = run_inference_on_frame(model_ppe, model_person, frame, conf)
            if save_json and last_result:
                json_log.append(to_dict(last_result, "webcam", frame_idx))

        if last_result:
            annotated = _draw_detections(frame, last_dets, last_result)
        else:
            annotated = frame

        cv2.imshow("Safety Monitor — Live", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if save_json and json_log:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = output_dir / f"webcam_report_{ts}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(json_log, f, indent=2)
        print(f"[INFO] Webcam JSON report saved → {out_json}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Construction Safety Monitor — Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", "-s",
        required=True,
        help=(
            "Input source. Can be: "
            "an image file, a folder of images, "
            "a video file, or a webcam index (e.g. 0)."
        ),
    )
    p.add_argument(
        "--model", "-m",
        default=str(DEFAULT_MODEL),
        help="Path to the trained YOLOv8 weights file (best.pt).",
    )
    p.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help="Directory to save annotated outputs.",
    )
    p.add_argument(
        "--conf", "-c",
        type=float,
        default=CONF_DEFAULT,
        help="Confidence threshold for violation detection (0–1).",
    )
    p.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        help="Save a JSON report alongside each output.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display annotated output in a window (requires display).",
    )
    p.add_argument(
        "--skip-frames",
        type=int,
        default=5,
        help="For video/webcam: run inference every N frames.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Verify model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print(
            "  → Please download best.pt from Google Colab and place it in "
            "models/best.pt"
        )
        sys.exit(1)

    print(f"[INFO] Loading target models...")
    model_ppe = YOLO(str(model_path))
    model_person = YOLO("yolov8n.pt")  # Download/Load COCO base model

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = args.source

    # Dispatch based on source type
    if source.isdigit():
        process_webcam(
            model_ppe, model_person, int(source), output_dir, args.conf,
            args.save_json, args.skip_frames,
        )
    else:
        src_path = Path(source)
        if not src_path.exists():
            print(f"[ERROR] Source not found: {src_path}")
            sys.exit(1)

        if src_path.is_dir():
            process_folder(model_ppe, model_person, src_path, output_dir, args.conf,
                           args.save_json, args.show)
        elif src_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            process_video(model_ppe, model_person, src_path, output_dir, args.conf,
                          args.save_json, args.show, args.skip_frames)
        else:
            process_image(model_ppe, model_person, src_path, output_dir, args.conf,
                          args.save_json, args.show)


if __name__ == "__main__":
    main()
