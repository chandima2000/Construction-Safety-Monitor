# Construction Safety Monitor

A computer vision system that automatically monitors construction site conditions and determines in real-time whether a given situation is **safe or unsafe**.

Built as part of the AI/ML Associate Software Engineer challenge.

---

## What It Does

- **Detects workers** in a scene using bounding box localization
- **Recognizes PPE** — hard hats, high-vis vests, safety glasses, boots, gloves
- **Checks compliance** per defined safety rules
- **Flags violations** with severity classification (Critical / Moderate / Low)
- **Generates alerts** describing which PPE is missing and where

---

## Quick Start

### Prerequisites

```bash
Python 3.10+
Git
pip
```

### Installation

```bash
git clone https://github.com/chandima2000/Construction-Safety-Monitor.git
cd Safty-Monitor-System

python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

> **Note on Model Weights:** 
> 1. The custom PPE model (`best.pt`) is included in the repo inside the `models/` folder.
> 2. The standard COCO `yolov8n.pt` base model (used for fallback localization) will **automatically download** from Ultralytics on your very first inference run.

### Run Inference

You can quickly test the detection engine using the provided sample images:

```bash
# Run the demo on the entire sample images folder (outputs JSON and annotated images)
python src/run_demo.py --source data/test_images/

# Run the demo and display the results in a live window
python src/run_demo.py --source data/test_images/ --show
```

Or, run inference on your own custom media:
```bash
# On a single image
python src/inference.py --source path/to/image.jpg

# On a video file
python src/inference.py --source path/to/video.mp4
```
---

## Project Structure

```
Safty-Monitor-System/
│
├── data/
│   ├── raw/custom/          # 30 custom web-sourced images
│   └── test_images/         # Sample images to quickly test run_demo.py
│
├── docs/
│   ├── dataset.md           # Dataset documentation
│   ├── inference_pipeline.md # Inference pipeline documentation
│   ├── safety_rules.md      # Safety rules definition
│   └── training_results.md  # Training metrics & analysis
│
├── models/
│   └── best.pt              # Custom trained YOLOv8 nano PPE model
│
├── research/
│   └── model_training.ipynb
│
├── src/                     # Inference pipeline codebase
│   ├── inference.py         # Main CLI — image/video/webcam processing
│   ├── rules.py             # Safety rule engine
│   ├── alerts.py            # Alert & report generation
│   └── run_demo.py          # Quick demo script
│
│
├── requirements.txt
└── README.md
```

---

## Dataset

- **5,170 source images** (5,140 base + 30 custom)
- **11 classes:** helmet, no-helmet, vest, no-vest, goggles, no-goggles, boots, no-boots, gloves, no-gloves, person
- **Base dataset:** [PPE Detection by testcasque on Roboflow Universe](https://universe.roboflow.com/testcasque/ppe-detection-qlq3d)
- **Custom data:** 30 web browsing images targeting violation scenarios
- **📂 Public dataset with annotations:** [View on Roboflow →](https://app.roboflow.com/chandimas-workspace/construction-safety-monitor-mlpd4/1)

See [`docs/dataset.md`](docs/dataset.md) for full documentation.

---

## Safety Rules

6 safety rules are defined and enforced:

| Rule | Severity | Class |
|---|---|---|
| Helmet Required | 🔴 Critical | `no-helmet` |
| High-Vis Vest Required | 🔴 Critical | `no-vest` |
| Safety Glasses Required | 🟡 Moderate | `no-goggles` |
| Safety Boots Required | 🟡 Moderate | `no-boots` |
| Gloves Required | 🟢 Low | `no-gloves` |
| Combined Violation (no helmet + no vest) | 🔴 Critical | both |

See [`docs/safety_rules.md`](docs/safety_rules.md) for full definitions, compliance logic, and edge cases.

---

## Model

- **Architecture:** YOLOv8n (nano — optimized for speed)
- **Training:** Google Colab (T4 GPU)
- **Epochs:** 50 with early stopping (patience=10)
- **Input size:** 640×640
- **Confidence threshold:** 0.45

Training notebook: `research/model_training.ipynb`

---

## Design Decisions

| Decision | Rationale |
|---|---|
| YOLOv8n over YOLOv8x | nano trains 8× faster with ~5% less accuracy |
| Confidence threshold 0.45 | Balances false positives vs false negatives for safety context |
| Extended existing dataset vs pure custom | 5,000 images was needed; 7 days wasn't enough to collect and annotate from scratch |
| Multi-Model Architecture | Base dataset lacked unannotated 'person' instances. Using standard COCO YOLOv8n alongside our custom PPE model guarantees perfect worker detection natively. |
| Both PPE-present and PPE-absent classes | Needed to distinguish "worker compliant" from "worker visible but non-compliant" |

---

## Known Limitations

- Per-worker violation attribution is not yet implemented (scene-level only)
- `no-boots` and `no-gloves` detection is less reliable due to dataset imbalance
- Night/low-light performance degrades — recommend lowering threshold to 0.35
- Partial PPE wear (open vest, tilted helmet) is not detected
- No temporal analysis — single-frame inference only

---

## Evaluation Results

Training completed: **22/50 epochs** (early stopping), **1.251 hours** on T4 GPU.

### Overall Metrics (Validation Set — 1,026 images, 4,307 instances)

| Metric | Value |
|---|---|
| Precision | 0.677 |
| Recall | 0.385 |
| **mAP50** | **0.534** |
| mAP50-95 | 0.363 |

### Per-Class mAP50

| Class | mAP50 | Notes |
|---|---|---|
| `helmet` | **0.941** | ✅ Production-quality |
| `vest` | **0.929** | ✅ Production-quality |
| `boots` | **0.907** | ✅ Production-quality |
| `no-vest` | **0.627** | 🟡 Working violation class |
| `goggles` | 0.524 | 🟡 Moderate |
| `gloves` | 0.510 | 🟠 Low recall |
| `no-helmet` | 0.505 | 🟠 Very low recall (R=0.009) |
| `no-goggles` | 0.400 | 🟠 Low |
| `no-boots` | 0 | ❌ Insufficient training data (36 val instances) |
| `no-gloves` | 0 | ❌ Insufficient training data (51 val instances) |

**Inference speed:** ~4.6ms per image on T4 (≈217 FPS)

See [`docs/training_results.md`](docs/training_results.md) for full analysis, training curves, confusion matrix interpretation, and planned improvements.

---

## Reproducing Results

1. **View the dataset:** [Roboflow Project v1](https://app.roboflow.com/chandimas-workspace/construction-safety-monitor-mlpd4/1) — all annotations are publicly visible
2. **Download dataset** via the Roboflow Python snippet in `docs/dataset.md`
3. **Open the notebook:** [`research/model_training.ipynb`](research/model_training.ipynb) in Google Colab
4. **Enable T4 GPU:** Runtime → Change runtime type → T4 GPU
5. **Run all cells** in order — training completes in ~1.25 hours
6. Download `best.pt` from `/content/runs/safety_monitor_v1/weights/`

---
