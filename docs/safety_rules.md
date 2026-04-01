# Safety Rules Definition

## Construction Safety Monitor · Safety Violation Rules

**Version:** 1.0  
**Defined by:** Chandima Maduwantha  
**Last updated:** 2026-04-01

---

## Overview

This document defines every safety rule enforced by the Construction Safety Monitor system.  
Each rule maps to one or more detection classes and a violation condition.

The system answers the core question: **"Is this situation safe or unsafe?"**

A scene is classified as **SAFE** when all visible workers comply with all applicable rules.  
A scene is classified as **UNSAFE** when one or more violations are detected.

---

## Confidence Threshold

All violations are flagged only when the model confidence exceeds **0.45** (45%).

This threshold was chosen to:
- Reduce false positives in ambiguous cases (e.g. partially visible workers)
- Maintain acceptable recall for critical violations (no-helmet, no-vest)
- Reflect the practical trade-off between precision and safety

---

## Rule Definitions

---

### Rule 1 — Helmet Required 🪖

**Severity:** 🔴 CRITICAL

| Property | Detail |
|---|---|
| **Detection class** | `no-helmet` |
| **Applies to** | All workers on an active construction site |
| **Threshold** | Confidence ≥ 0.45 |

**What counts as a violation:**
- Worker's head is clearly visible and no hard hat is present
- Worker is wearing a cloth cap, beanie, baseball cap, or no headwear
- Hard hat is present but not on the head (e.g. held in hand)

**What does NOT count as a violation:**
- Hard hat worn but strap not visible (strap occlusion is acceptable)
- Worker's head is >50% occluded — too little information to judge

**Rationale:** Head injuries are the leading cause of construction fatalities. Hard hats protect against falling objects, electrical hazards, and impacts. OSHA standard 29 CFR 1926.100 mandates helmets in all active construction zones.

---

### Rule 2 — High-Visibility Vest Required 🦺

**Severity:** 🔴 CRITICAL

| Property | Detail |
|---|---|
| **Detection class** | `no-vest` |
| **Applies to** | All workers in active work zones, near vehicle/machinery traffic |
| **Threshold** | Confidence ≥ 0.45 |

**What counts as a violation:**
- Worker's torso is visible and no orange/yellow reflective vest is present
- Worker is wearing only a t-shirt, jacket, or dark clothing
- Vest is present but worn open/unzipped — counts as partial violation

**What does NOT count as a violation:**
- Worker is clearly in an office/control room area (not active site)
- Torso is >50% occluded

**Rationale:** High-visibility vests ensure workers can be seen by machinery operators, vehicle drivers, and other workers. The reflective strips are critical in low-light conditions. Required under ANSI/ISEA 107 for Class 2 and Class 3 work environments.

---

### Rule 3 — Safety Glasses Required 🥽

**Severity:** 🟡 MODERATE

| Property | Detail |
|---|---|
| **Detection class** | `no-goggles` |
| **Applies to** | Workers in areas with debris, dust, sparks, or chemical exposure |
| **Threshold** | Confidence ≥ 0.45 |

**What counts as a violation:**
- Worker's face is visible, worker is in an active work area, and no safety glasses/goggles are present

**Scope note:** Due to the high variability of when eye protection is strictly required, this rule is flagged as a **moderate-severity** violation. The system flags absence as a potential violation; human supervisors make the final determination.

**Rationale:** Eye injuries are among the most common construction injuries and are largely preventable with proper PPE. OSHA 29 CFR 1926.102 requires eye protection when workers are exposed to flying particles, molten metal, or chemical splashes.

---

### Rule 4 — Safety Boots Required 👢

**Severity:** 🟡 MODERATE

| Property | Detail |
|---|---|
| **Detection class** | `no-boots` |
| **Applies to** | All workers on site floors, scaffolding, or near heavy machinery |
| **Threshold** | Confidence ≥ 0.45 |

**What counts as a violation:**
- Worker's feet are visible and they are wearing sneakers, casual shoes, sandals, or no footwear
- Footwear lacks a visible steel-toe cap or protective sole profile

**Limitation:** Safety boot detection is inherently difficult — steel-toe boots can look similar to regular work boots in images. This rule has a higher false-negative rate than helmet/vest detection and should be treated as guidance rather than definitive detection.

**Rationale:** Steel-toe boots protect against crushing injuries from heavy objects and puncture injuries from nails/rebar. Required under OSHA 29 CFR 1926.96.

---

### Rule 5 — Gloves Required 🧤

**Severity:** 🟢 LOW (contextual)

| Property | Detail |
|---|---|
| **Detection class** | `no-gloves` |
| **Applies to** | Workers actively handling sharp, hot, or chemically hazardous materials |
| **Threshold** | Confidence ≥ 0.45 |

**What counts as a violation:**
- Worker's hands are clearly visible, worker is handling materials or tools, and no protective gloves are present

**Scope note:** This is flagged as **low severity** at the scene level because bare hands are acceptable in many site areas (e.g. administrative tasks, walking between zones). Context matters significantly.

**Rationale:** Hand injuries are the second most common construction injury type. Cut-resistant and heat-resistant gloves are required when handling rebar, sheet metal, or power tools.

---

### Rule 6 — Critical Combined Violation ⚠️

**Severity:** 🔴 CRITICAL

| Property | Detail |
|---|---|
| **Detection classes** | `no-helmet` + `no-vest` (both present in same scene) |
| **Applies to** | Any scene where a worker is missing both primary PPE items |
| **Threshold** | Both detections ≥ 0.45 confidence |

**Definition:** When a single worker (or scene) triggers both `no-helmet` AND `no-vest` simultaneously, the system escalates the alert to CRITICAL regardless of other compliant workers in the scene.

**Rationale:** A worker without both primary PPE items is in extreme danger and represents a systematic failure of PPE compliance — not just a momentary oversight. This warrants immediate escalation.

---

## Compliance Logic

```
FOR each detected person in scene:
    violations = []
    
    IF no-helmet detected near person (conf ≥ 0.45):
        violations.append(HELMET_MISSING)
    
    IF no-vest detected near person (conf ≥ 0.45):
        violations.append(VEST_MISSING)
    
    IF no-goggles detected near person (conf ≥ 0.45):
        violations.append(GOGGLES_MISSING)
    
    IF no-boots detected near person (conf ≥ 0.45):
        violations.append(BOOTS_MISSING)
    
    IF no-gloves detected near person (conf ≥ 0.45):
        violations.append(GLOVES_MISSING)

IF len(violations) == 0:
    scene_status = SAFE ✅
ELIF HELMET_MISSING and VEST_MISSING in violations:
    scene_status = CRITICAL UNSAFE 🔴
ELIF any CRITICAL violations present:
    scene_status = UNSAFE 🔴
ELSE:
    scene_status = UNSAFE (Moderate) 🟡
```

---

## Known Limitations & Edge Cases

| Edge Case | Behaviour | Recommended Action |
|---|---|---|
| Worker partially occluded (back turned, arms crossed) | Some PPE boxes may not be detected | Review manually for critical areas |
| Low light / night scenes | Detection confidence typically drops 10–20% | Lower threshold to 0.35 for night zones |
| Workers far from camera | Bounding boxes become very small, confidence drops | Flag if below 0.30 as uncertain/inconclusive |
| Multiple workers in frame | System detects all — violations from any worker flag the scene | Per-worker attribution uses `person` class |
| Worker in office/trailer area | May falsely flag for no-vest | Future: zone-based rule application |
| PPE partially worn (vest open, helmet tilted) | Current system detects presence, not correctness | Future enhancement — pose + state estimation |

---

## Future Rule Candidates

These rules are documented but not yet implemented in v1:

| Rule | Description | Reason Not in v1 |
|---|---|---|
| **Harness / Fall Protection** | Workers at height must wear safety harness | Requires height estimation; dataset gap |
| **Zone-Based Rules** | Strictest PPE required within 5m of active machinery | Requires site map integration |
| **Partial Wear Detection** | Helmet without chin strap, vest unzipped | Requires keypoint or segmentation model |
| **Temporal Behaviour** | Worker removes PPE after passing inspection point | Requires video sequence analysis |
