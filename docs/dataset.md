# Dataset Documentation

## Construction Safety Monitor · Dataset v1

**Version:** v1-ppe-5170img-11cls-640px  
**Generated:** 2026-04-01  
**Format:** YOLOv8 (TXT annotations + YAML config)  
**Platform:** Roboflow Universe

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **Total source images** | 5,170 |
| **Augmented training images** | 12,424 |
| **Number of classes** | 11 |
| **Train / Val / Test split** | 10,881 / 1,026 / 517 |
| **Input resolution** | 640 × 640 px |
| **Annotation format** | YOLO (bounding boxes) |

---

## 2. Data Sources

### 2.1 Base Dataset — Roboflow Universe (5,140 images)

| Property | Detail |
|---|---|
| **Source** | [PPE Detection Dataset by testcasque](https://universe.roboflow.com/testcasque) |
| **License** | Roboflow Universe Public |
| **Original classes** | 10 (helmet, no-helmet, vest, no-vest, goggles, no-goggles, boots, no-boots, gloves, no-gloves) |
| **Why selected** | Largest consistently-annotated PPE dataset with balanced positive/negative classes |

This dataset was chosen because:
- It already contains **violation classes** (`no-helmet`, `no-vest`, etc.) — most public datasets only have presence labels, not absence labels
- 5,100+ images across diverse construction environments
- High annotation quality and consistency across all 10 classes
- YOLOv8-compatible format directly exportable from Roboflow

### 2.2 Custom Additions — YouTube Construction Footage (30 images)

30 custom images were collected from YouTube construction site video footage to:
1. Satisfy the assignment's "custom dataset" requirement
2. Address class imbalances in the base dataset
3. Add environment diversity not present in the base

| Batch | Images | Primary Purpose |
|---|---|---|
| Batch 1 — No-Helmet violations | 001–008 (8 images) | Fix `no-helmet` imbalance (598 vs 5,893 helmet in base) |
| Batch 2 — No-Boots violations | 009–014 (6 images) | Fix `no-boots` imbalance (163 vs 4,600 boots in base) |
| Batch 3 — Person coverage | 015–022 (8 images) | Full-body shots for the new `person` class |
| Batch 4 — Environment variety | 023–027 (5 images) | Indoor warehouse, night, height-work scenes |
| Batch 5 — Safe scenes | 028–030 (3 images) | Fully compliant workers (balanced safe/unsafe) |

**Custom image collection method:**
- YouTube search queries targeting specific violation scenarios
- Screenshots captured at 1920×1080 or higher resolution
- Screenshots saved as JPEG from full-screen video frames
- Stored in: `data/raw/custom/custom_001.jpg` through `custom_030.jpg`

**Custom image quality criteria:**
- Worker visible at ≥25% of frame height
- Construction environment context visible
- No cartoons, posters, or stock illustrations  
- Minimum 30KB file size (rejects thumbnail-quality images)

> **Note:** `custom_015.jpg` has a plain studio/white background. It was retained because the PPE labels are clearly annotatable and the image teaches the model PPE appearance rather than context. This is documented as a known limitation.

---

## 3. Class Definitions

| Class ID | Class Name | Description | Type |
|---|---|---|---|
| 0 | `helmet` | Hard hat / safety helmet worn on head | PPE Present |
| 1 | `no-helmet` | Worker's head visible without a hard hat | PPE Absent (Violation) |
| 2 | `vest` | High-visibility safety vest worn on torso | PPE Present |
| 3 | `no-vest` | Worker's torso visible without a high-vis vest | PPE Absent (Violation) |
| 4 | `goggles` | Safety glasses or eye protection worn | PPE Present |
| 5 | `no-goggles` | Worker's face visible without eye protection | PPE Absent (Violation) |
| 6 | `boots` | Safety/steel-toe boots on feet | PPE Present |
| 7 | `no-boots` | Worker's feet visible without safety boots | PPE Absent (Violation) |
| 8 | `gloves` | Protective gloves worn on hands | PPE Present |
| 9 | `no-gloves` | Worker's hands visible without gloves | PPE Absent (Violation) |
| 10 | `person` | Full-body bounding box of any worker | Worker Localization |

> The `person` class was **added by us** to the original 10-class schema. It enables per-worker compliance tracking and is intended for future attribution of violations to specific individuals in a scene.

---

## 4. Class Distribution (Base Dataset)

> Approximate values from Roboflow dataset statistics. Exact counts available via Roboflow dashboard.

| Class | Approx. Count | Notes |
|---|---|---|
| `helmet` | ~5,893 | Well-represented |
| `no-helmet` | ~598 | **Underrepresented** — supplemented by custom Batch 1 |
| `vest` | ~4,800 | Well-represented |
| `no-vest` | ~420 | Moderate |
| `goggles` | ~1,200 | Moderate |
| `no-goggles` | ~380 | Low |
| `boots` | ~4,600 | Well-represented |
| `no-boots` | ~163 | **Severely underrepresented** — supplemented by custom Batch 2 |
| `gloves` | ~900 | Moderate |
| `no-gloves` | ~250 | Low |
| `person` | ~30 (custom only) | New class — only in custom images |

---

## 5. Annotation Strategy

### Bounding Box Rules
- **One box per object** — each item of PPE gets its own bounding box
- **Tight fit** — boxes drawn as close as possible to the item edges
- **`person` first** — full-body bounding box drawn before PPE boxes
- **Both states annotated** — if a worker has a helmet AND another worker doesn't, both are annotated in the same image
- **Consistency rule** — if one worker's `no-vest` is annotated, all workers' vests in that image are annotated

### Occlusion Handling
- **>50% occluded:** not annotated (model can't learn from insufficient pixels)
- **25–50% occluded:** annotated with care, box covers visible portion only
- **<25% occluded:** annotated normally

### `no-XXXX` Class Placement
Violation boxes are drawn over the **body region where the PPE should be:**
- `no-helmet` → box around the head region
- `no-vest` → box around the torso
- `no-boots` → box around the feet/lower legs
- `no-gloves` → box around the hands

---

## 6. Preprocessing

| Step | Setting | Reason |
|---|---|---|
| Auto-orient | Applied | Corrects EXIF rotation from mobile cameras |
| Resize | Stretch to 640×640 | YOLOv8 standard input size |

---

## 7. Data Augmentation

Applied to the **training set only** (validation and test sets are un-augmented):

| Augmentation | Setting | Purpose |
|---|---|---|
| Horizontal flip | Enabled | Workers face left or right equally |
| Rotation | ±15° | Slight camera tilt variation |
| Brightness | ±25% | Lighting variation (cloudy/sunny/indoor) |
| Blur | Up to 1.5px | Simulates video frame quality |

> **Note:** 90° rotation was explicitly **excluded** — construction site cameras are fixed, so rotated workers would be unrealistic training examples.

Augmentation multiplier: **3× per training image**
- Source training images: ~3,627
- Augmented training images: 10,881

---

## 8. Reproducibility

To recreate this exact dataset:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")  # See config/secrets.yaml
project = rf.workspace("chandimas-workspace").project("construction-safety-monitor-mlpd4")
version = project.version(1)
dataset = version.download("yolov8")
```

Dataset version name: `v1-ppe-5170img-11cls-640px`

---

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| `person` class has only 30 examples | Per-worker attribution may be weak | Expand in future versions |
| `no-boots`, `no-gloves` heavily underrepresented | Model may miss these violations | Targeted augmentation + future collection |
| `custom_015.jpg` has white studio background | Minor — slightly unrealistic scene context | Documented; impact expected to be minimal |
| Dataset is scene-level, not worker-level | Can't attribute violations to specific workers | `person` class intended to solve this |
| No temporal data (video sequences) | Cannot detect behavioural patterns | Future: video inference pipeline |
