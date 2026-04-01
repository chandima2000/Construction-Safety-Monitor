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
git clone https://github.com/YOUR_USERNAME/Safty-Monitor-System.git
cd Safty-Monitor-System

python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

### Run Inference (once model is trained)

```bash
# On a single image
python src/inference.py --source path/to/image.jpg

# On a video file
python src/inference.py --source path/to/video.mp4

# On webcam
python src/inference.py --source 0
```

---

## Project Structure

```
Safty-Monitor-System/
│
├── data/
│   └── raw/custom/          # 30 custom YouTube-sourced images
│
├── docs/
│   ├── day1_guide.md        # Day 1 setup walkthrough
│   ├── dataset.md           # Dataset documentation
│   └── safety_rules.md      # Safety rules definition
│
├── roboflow/
│   └── train_yolov8_object_detection_on_custom_dataset.ipynb
│
├── src/                     # Inference pipeline (Day 3)
│   ├── inference.py
│   ├── rules.py
│   └── alerts.py
│
├── config/
│   └── secrets.yaml         # API keys (NOT committed to GitHub)
│
├── requirements.txt
└── README.md
```

---

## Dataset

- **5,170 source images** (5,140 base + 30 custom)
- **11 classes:** helmet, no-helmet, vest, no-vest, goggles, no-goggles, boots, no-boots, gloves, no-gloves, person
- **Base dataset:** [PPE Detection by testcasque on Roboflow Universe](https://universe.roboflow.com/testcasque/ppe-detection-qlq3d)
- **Custom data:** 30 YouTube construction site screenshots targeting violation scenarios

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

Training notebook: `roboflow/train_yolov8_object_detection_on_custom_dataset.ipynb`

---

## Design Decisions

| Decision | Rationale |
|---|---|
| YOLOv8n over YOLOv8x | 4-day deadline; nano trains 8× faster with ~5% less accuracy |
| Confidence threshold 0.45 | Balances false positives vs false negatives for safety context |
| Extended existing dataset vs pure custom | 5,000 images was needed; 4 days wasn't enough to collect and annotate from scratch |
| Scene-level detection (not per-worker) | Simpler to implement; `person` class added for future per-worker attribution |
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

> *(To be filled after Day 2 training — mAP50, precision, recall per class)*

---

## Reproducing Results

1. Download dataset via Roboflow (see `docs/dataset.md` for snippet)
2. Open `roboflow/train_yolov8_object_detection_on_custom_dataset.ipynb` in Google Colab
3. Enable T4 GPU runtime
4. Run all cells in order
5. Download `best.pt` from `runs/safety_monitor_v1/weights/`

---

## Development Timeline

| Day | Focus | Status |
|---|---|---|
| Day 1 | Dataset collection, annotation, documentation | ✅ Complete |
| Day 2 | Model training (YOLOv8 fine-tuning) | 🔄 In Progress |
| Day 3 | Inference pipeline + violation alert generation | ⬜ Pending |
| Day 4 | Evaluation, final documentation, submission | ⬜ Pending |
