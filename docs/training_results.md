# Model Training Results

## Construction Safety Monitor · Training Run v1

**Model:** YOLOv8n (nano)  
**Training date:** 2026-04-01  
**Platform:** Google Colab (Tesla T4 GPU, 14,913 MiB)  
**Runtime:** 1.251 hours  
**Notebook:** `roboflow/train_yolov8_object_detection_on_custom_dataset.ipynb`

---

## 1. Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `model` | `yolov8n.pt` | Nano — fastest fine-tuning on free Colab T4 |
| `epochs` | 50 (stopped at 22) | Early stopping triggered at epoch 22 |
| `imgsz` | 640 | YOLOv8 standard input resolution |
| `batch` | 16 | Fits T4 VRAM without OOM |
| `patience` | 10 | Stop if mAP50 doesn't improve for 10 epochs |
| `conf` | 0.45 | Confidence threshold during training validation |
| `optimizer` | Auto (AdamW) | YOLOv8 default |
| `pretrained` | Yes (`yolov8n.pt` → COCO) | Transfer learning from COCO weights |

### Why YOLOv8n (nano)?

- **8× faster** to train than YOLOv8x on the same hardware
- Only ~5% accuracy difference on standard benchmarks
- 6.2MB model size vs 131MB for YOLOv8x — easier to deploy

---

## 2. Dataset Used for Training

| Split | Images | Instances |
|---|---|---|
| Training | 10,881 | ~31,000+ |
| Validation | 1,026 | 4,307 |
| Test | 517 | ~2,000+ |

Source: [v1-ppe-5170img-11cls-640px](https://app.roboflow.com/chandimas-workspace/construction-safety-monitor-mlpd4/1) on Roboflow (5,170 source images, 3× augmented training set)

---

## 3. Early Stopping

Training stopped at **epoch 22/50**.

```
EarlyStopping: Training stopped early as no improvement 
observed in last 10 epochs. Best results observed at epoch 12.
```

**What this means:**
- The model's best checkpoint was saved at epoch ~12
- Epochs 13–22 showed no improvement in mAP50 on the validation set
- The losses were still decreasing slightly, but the detection quality had plateaued
- This is expected behaviour for a well-regularised model trained on a diverse dataset

**Best weights saved to:** `/content/runs/safety_monitor_v1/weights/best.pt` (6.2 MB)

---

## 4. Validation Metrics

All metrics measured on the **validation set** (1,026 images, 4,307 instances).

### Overall Summary

| Metric | Value | Context |
|---|---|---|
| **Box Precision (P)** | **0.677** | 67.7% of predicted boxes are correct |
| **Recall (R)** | **0.385** | Model finds 38.5% of all ground-truth objects |
| **mAP50** | **0.534** | Average precision at IoU=0.50 threshold |
| **mAP50-95** | **0.363** | Strict average across IoU 0.50–0.95 |

### Per-Class Breakdown

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 | Assessment |
|---|---|---|---|---|---|---|---|
| **helmet** | 814 | 1,187 | 0.927 | 0.917 | **0.941** | 0.651 | ✅ Excellent |
| **vest** | 824 | 1,154 | 0.918 | 0.892 | **0.929** | 0.740 | ✅ Excellent |
| **boots** | 439 | 931 | 0.906 | 0.865 | **0.907** | 0.716 | ✅ Excellent |
| **no-vest** | 125 | 218 | 0.620 | 0.651 | **0.627** | 0.351 | 🟡 Good |
| **goggles** | 95 | 113 | 0.736 | 0.345 | **0.524** | 0.315 | 🟡 Moderate |
| **no-helmet** | 74 | 103 | 1.000 | 0.00971 | **0.505** | 0.303 | 🟠 Low recall |
| **gloves** | 193 | 359 | 1.000 | 0.0195 | **0.510** | 0.371 | 🟠 Low recall |
| **no-goggles** | 114 | 155 | 0.667 | 0.155 | **0.400** | 0.187 | 🟠 Low |
| **no-boots** | 20 | 36 | 0 | 0 | **0** | 0 | ❌ Zero |
| **no-gloves** | 31 | 51 | 0 | 0 | **0** | 0 | ❌ Zero |

> **Note:** `person` class showed 0 detections in validation — expected, as `person` annotations only existed in the 30 custom images (all in train split).

---

## 5. Training Curves Analysis

### Loss Curves

| Curve | Start | End | Direction | Assessment |
|---|---|---|---|---|
| `train/box_loss` | ~1.50 | ~1.10 | ↘ Decreasing | ✅ Learning object locations |
| `train/cls_loss` | ~2.00 | ~0.75 | ↘ Decreasing | ✅ Learning class identity |
| `train/dfl_loss` | ~1.40 | ~1.10 | ↘ Decreasing | ✅ Learning box distribution |
| `val/box_loss` | ~1.45 | ~1.20 | ↘ Decreasing | ✅ Not overfitting |
| `val/cls_loss` | ~1.30 | ~0.80 | ↘ Decreasing | ✅ Generalising well |
| `val/dfl_loss` | ~1.35 | ~1.22 | Slightly ↘ | ✅ Stable |

**Key observation:** Training and validation losses decrease in parallel — the model is **not overfitting**. The gap between train and val is small, indicating good generalisation on unseen data.

### Metric Curves

| Metric | Start | End | Direction | Assessment |
|---|---|---|---|---|
| `metrics/precision` | ~0.45 | ~0.65 | ↗ Noisy increase | Normal — precision fluctuates early |
| `metrics/recall` | ~0.275 | ~0.385 | ↗ Slow increase | Low recall is the main limitation |
| `metrics/mAP50` | ~0.35 | ~0.53 | ↗ Steady increase | Positive trend throughout |
| `metrics/mAP50-95` | ~0.225 | ~0.36 | ↗ Slow increase | Normal — strict metric takes longer |

---

## 6. Confusion Matrix Analysis

The confusion matrix reveals the following behavioural patterns:

### Strong Diagonal (Correctly Classified)
| Class | Correct Predictions |
|---|---|
| `helmet` | 1,093 |
| `vest` | 1,033 |
| `boots` | 808 |
| `no-vest` | 140 |
| `no-goggles` | 24 |

### Problems Identified

**1. Background absorption (most critical issue)**

Many ground-truth objects are predicted as "background" (not detected at all):
- `gloves`: 350 true instances → predicted as background
- `goggles`: 72 true instances → predicted as background  
- `no-goggles`: 92 true instances → predicted as background

This happens because the model's classification confidence for minority classes falls below the 0.45 threshold, so those detections are filtered out.

**2. `no-boots` and `no-gloves` completely undetected**

The confusion matrix shows entire rows empty for these classes. With only 36 and 51 validation instances respectively, the model never learned a confident enough representation to predict these classes.

**3. `no-helmet` near-zero recall (P=1.0, R=0.009)**

Precision=1.0 with near-zero recall means: the model predicts `no-helmet` only once (and gets it right), but misses all 102 other `no-helmet` instances. The model treats prediction as "background" rather than risk getting it wrong.

**Root cause:** The `no-helmet` supervision signal is paradoxical — the model must learn to detect the *absence* of an object (no hard hat on a head region). This is fundamentally harder than detecting presence.

---

## 7. Why Violation Classes Underperform

This is the main learning from this training run:

| Issue | Root Cause | Evidence |
|---|---|---|
| `no-boots`, `no-gloves` = 0 mAP | Too few instances (36, 51 in val) | Model never predicts these classes |
| `no-helmet` near-zero recall | Class imbalance + hard task (detecting absence) | P=1.0, R=0.009 |
| `no-goggles` low | Goggles are small objects, hard to detect absence | mAP=0.4 |

**Dataset imbalance in validation set:**
```
helmet:     1,187 instances   ← 32× more than no-helmet
vest:       1,154 instances   ← 5× more than no-vest  
boots:        931 instances   ← 26× more than no-boots
no-boots:      36 instances   ← severely underrepresented
no-gloves:     51 instances   ← severely underrepresented
```

This is a **data problem, not a model problem.** The solution is more annotated violation examples, not a different architecture.

---

## 8. Model Behaviour Summary

### What the model reliably does:

| Capability | Reliability | mAP50 |
|---|---|---|
| Detects workers wearing helmets | ✅ Very high | 0.941 |
| Detects workers wearing vests | ✅ Very high | 0.929 |
| Detects workers wearing boots | ✅ Very high | 0.907 |
| Flags workers missing a vest | ✅ Good | 0.627 |
| Detects goggles when present | 🟡 Moderate | 0.524 |
| Flags missing helmet | 🟠 Unreliable | 0.505 (low recall) |

### What the model does not reliably do:

| Limitation | Reason |
|---|---|
| Detect missing boots | Insufficient training data (36 val instances) |
| Detect missing gloves | Insufficient training data (51 val instances) |
| Flag missing helmet with high recall | Imbalanced data + detecting absence is inherently hard |
| Attribute violations to specific workers | `person` class only in training (not val/test) |

---

## 9. Speed Metrics

```
Speed: 0.2ms preprocess, 2.3ms inference, 0.0ms loss, 2.1ms postprocess per image
```

| Stage | Time |
|---|---|
| Preprocessing | 0.2ms |
| Inference (forward pass) | 2.3ms |
| Postprocessing | 2.1ms |
| **Total per image** | **~4.6ms** |

This translates to approximately **217 frames per second** on a T4 GPU — well above real-time requirements (25 fps). On a CPU, expect 30–50ms per frame (~20-30 fps) for the nano model.

---

## 10. Files Generated

| File | Size | Description |
|---|---|---|
| `weights/best.pt` | 6.2 MB | Best model weights (saved at epoch ~12) |
| `weights/last.pt` | 6.2 MB | Weights at final epoch (epoch 22) |
| `results.png` | — | Training curve plots (all metrics) |
| `confusion_matrix.png` | — | Class prediction confusion matrix |
| `results.csv` | — | Per-epoch raw metrics |
| `args.yaml` | — | Exact training configuration |
| `labels.jpg` | — | Label distribution visualisation |
| `labels_correlogram.jpg` | — | Co-occurrence of class labels |

---

## 11. Planned Improvements (v2)

Based on this training run, the following changes would improve the model:

| Improvement | Expected Impact |
|---|---|
| Collect 200+ `no-boots` images | Lift `no-boots` mAP from 0 to 0.5+ |
| Collect 200+ `no-gloves` images | Lift `no-gloves` mAP from 0 to 0.4+ |
| Use class weights in loss function | Penalise misclassification of minority classes more |
| Increase epochs to 100 with patience=20 | Allow more time for minority class learning |
| Upgrade to YOLOv8s (small) | +5–8% overall mAP at 3× training cost |
| Add mosaic augmentation at higher rate | Improve small object detection |

---

## 12. Inference Strategy Adjustments (Day 3)

Because the model's raw recall for violation classes is low, the inference pipeline (Day 3) will implement **two compensating strategies:**

**Strategy 1 — Lower per-class thresholds for violations**
```python
CLASS_THRESHOLDS = {
    'no-helmet':  0.25,   # Lower threshold — prioritise recall over precision
    'no-vest':    0.35,   # Slightly lower than default
    'no-boots':   0.20,   # Very low — model rarely predicts this
    'no-gloves':  0.20,   # Very low — model rarely predicts this
    'no-goggles': 0.30,
    'helmet':     0.45,   # Default — model is confident when it predicts
    'vest':       0.45,
    'boots':      0.45,
}
```

**Strategy 2 — Presence-absence inference rule**
```
If a 'person' is detected in the scene
  AND 'helmet' is NOT detected anywhere in the frame
  THEN flag as POTENTIAL no-helmet (with lower confidence)
```

This acts as a safety net for the model's conservative `no-helmet` prediction behaviour.
