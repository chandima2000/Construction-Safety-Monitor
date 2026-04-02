# Inference Pipeline Guide

## Construction Safety Monitor

**Author:** Chandima Maduwantha  

---

## Overview

This document details the architecture and usage of the Construction Safety Monitor's inference pipeline. The system takes the trained YOLOv8 model (`best.pt`) and applies it to real-world images, video streams, or live webcam feeds to detect safety violations and generate actionable compliance reports.

The pipeline comprises four core Python modules located in the `src/` directory:

| File | Component | Description |
|---|---|---|
| `rules.py` | **Safety Engine** | Core logic mapping YOLO detections to severity levels. |
| `alerts.py` | **Reporter** | Converts rule outputs into human-readable text and JSON. |
| `inference.py` | **Main Pipeline** | Full CLI application handling media I/O and routing. |
| `run_demo.py` | **Demo Runner** | Zero-configuration script to quickly test the pipeline. |

---

## System Architecture

The pipeline processes input media frame by frame, separating object detection from the safety evaluation logic. This modular design allows safety protocols to be updated without retraining the computer vision model.

```
[Input Media: Image/Video/Live]
        │
        ▼
  ┌───────────────┐
  │  inference.py │  ← Loads optimized YOLOv8n weights
  │  YOLOv8 model │  ← Extracts bounding boxes and class scores
  └───────┬───────┘
          │  List[Detection]
          ▼
  ┌───────────────┐
  │   rules.py    │  ← Evaluates contextual safety rules
  └───────┬───────┘
          │  SceneResult (violations, compliant PPE, severity)
          ▼
  ┌───────────────┐
  │   alerts.py   │  ← Renders annotated media, JSON logs, & console alerts
  └───────────────┘
```

---

## Usage Guide

### Prerequisites

Ensure the environment is prepared and dependencies are installed:
```bash
pip install -r requirements.txt
```

Place the trained model weights in the project hierarchy:
```
models/best.pt
```

---

### Quick Demo Testing

For immediate evaluation, the demo script automatically locates test images provided within the repository (if populated inside `data/test_images/`):

```bash
# Process default test folder and save results
python src/run_demo.py

# Display the annotated output in a real-time window
python src/run_demo.py --show

# Supply a specific image or video feed
python src/run_demo.py --source path/to/sample.jpg --show
```

---

### Full Command-Line Interface (`inference.py`)

The main CLI provides fine-grained control over thresholds, outputs, and JSON logging.

#### Usage Examples

```bash
# Process a single image
python src/inference.py --source data/test_images/site_photo.jpg

# Batch process an entire directory
python src/inference.py --source data/test_images/

# Process video footage
python src/inference.py --source data/videos/site_footage.mp4

# Run on a live webcam (camera index 0)
python src/inference.py --source 0

# Enable comprehensive JSON reporting
python src/inference.py --source data/test_images/ --save-json

# Adjust confidence threshold for low-light environments
python src/inference.py --source data/test_images/ --conf 0.35
```

#### Argument Reference

| Flag | Default | Description |
|---|---|---|
| `--source` / `-s` | *(required)* | Image, directory, video file, or camera index. |
| `--model` / `-m` | `models/best.pt` | Path to trained weights. |
| `--output` / `-o` | `runs/inference/` | Output directory for logs and media. |
| `--conf` / `-c` | `0.45` | Violation confidence threshold. |
| `--save-json` | `False` | Generate structured JSON compliance reports. |
| `--show` | `False` | Display a live processing window. |
| `--skip-frames` | `5` | Video optimization: evaluate every N frames. |

---

## Evaluation Output

All generated artifacts are stored in `runs/inference/` by default.

| Artifact | Description |
|---|---|
| `annotated_<name>.jpg` | Source media overlaid with boundary boxes and status tags. |
| `annotated_<name>.mp4` | Processed video feed. |
| `<name>_report.json` | Programmatic JSON violation payload. |
| `batch_report_<ts>.json` | Aggregated report for directory batch processing. |

### Visual Identification Legend

Detected objects and violations are color-coded in the media overlay for immediate visual triage:

| Colour | Classification | Example Classes |
|---|---|---|
| 🟥 **Red** | CRITICAL Violation | `no-helmet`, `no-vest` |
| 🟧 **Orange** | MODERATE Violation | `no-goggles`, `no-boots` |
| 🟨 **Yellow** | LOW Severity | `no-gloves` |
| 🟩 **Green** | Compliant Gear | `helmet`, `vest`, `boots` |
| ⬜ **Grey** | Neutral Object | `person` |

---

## Data Integration & JSON Telemetry

By passing the `--save-json` flag, logging payloads are generated suitable for dashboard integration or long-term auditing.

**Example Payload Structure:**
```json
{
  "timestamp": "2026-04-02T14:30:00",
  "image_path": "site_photo.jpg",
  "scene_severity": "CRITICAL",
  "is_safe": false,
  "total_persons": 2,
  "compliant_ppe": ["boots", "helmet"],
  "violation_count": 1,
  "violations": [
    {
      "rule_id": 2,
      "description": "High-visibility vest missing",
      "severity": "CRITICAL",
      "confidence": 0.82,
      "bbox": [120.5, 80.3, 290.1, 450.7]
    }
  ]
}
```

---

## Known Boundaries & Operating Guidelines

Due to constraints within the custom training dataset (imbalanced representation of boots/gloves specific annotations), the following operational considerations apply to this release:

1. **Low-Visibility Contexts:** The default model confidence threshold is explicitly tuned to `0.45` for balanced daylight scenes. Reduce to `0.35` for night or poorly lit scenarios.
2. **Guidance Variables:** Output classifications for `no-boots` and `no-gloves` should be used as advisory flags rather than hard compliance failures due to minor val-set volume.
3. **Partial Wear:** The current YOLO model bounds the presence of objects. Future iteration using pose-estimation is required to detect incorrectly worn PPE (e.g., unzipped vests, helmets worn backwards).
