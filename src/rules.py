from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Constants — class names as emitted by YOLOv8 / Roboflow dataset
# ---------------------------------------------------------------------------

CLASS_BOOTS      = "boots"
CLASS_GLOVES     = "gloves"
CLASS_GOGGLES    = "goggles"
CLASS_HELMET     = "helmet"
CLASS_NO_VEST    = "no-vest"
CLASS_NO_BOOTS   = "no-boots"
CLASS_NO_GLOVES  = "no-gloves"
CLASS_NO_GOGGLES = "no-goggles"
CLASS_NO_HELMET  = "no-helmet"
CLASS_VEST       = "vest"
CLASS_PERSON     = "person"

# All "violation" class names
VIOLATION_CLASSES = {
    CLASS_NO_HELMET,
    CLASS_NO_VEST,
    CLASS_NO_GOGGLES,
    CLASS_NO_BOOTS,
    CLASS_NO_GLOVES,
}

# Confidence thresholds (from safety_rules.md)
CONF_DEFAULT    = 0.45   # Standard threshold for all rules
CONF_LOW_LIGHT  = 0.35   # Night / low-light scenes (future use)
CONF_UNCERTAIN  = 0.30   # Small / distant workers — flag as uncertain

# Per-class thresholds — compensating for dataset imbalance (Day 3 Strategy)
CLASS_THRESHOLDS = {
    CLASS_NO_HELMET:  0.25,
    CLASS_NO_VEST:    0.35,
    CLASS_NO_BOOTS:   0.20,
    CLASS_NO_GLOVES:  0.20,
    CLASS_NO_GOGGLES: 0.30,
    CLASS_HELMET:     0.45,
    CLASS_VEST:       0.45,
    CLASS_BOOTS:      0.45,
}

# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class Severity(Enum):
    SAFE     = "SAFE"
    LOW      = "LOW"
    MODERATE = "MODERATE"
    CRITICAL = "CRITICAL"

    def emoji(self) -> str:
        return {
            "SAFE":     "✅",
            "LOW":      "🟢",
            "MODERATE": "🟡",
            "CRITICAL": "🔴",
        }[self.value]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single raw bounding-box detection from YOLO."""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 (absolute px)


@dataclass
class Violation:
    """A single, interpreted safety violation."""
    rule_id: int
    description: str
    severity: Severity
    confidence: float
    bbox: Tuple[float, float, float, float]


@dataclass
class SceneResult:
    """Complete analysis result for one frame / image."""
    violations: List[Violation] = field(default_factory=list)
    compliant_ppe: List[str]   = field(default_factory=list)
    scene_severity: Severity   = Severity.SAFE

    # Raw detection counts (for summary)
    total_persons: int  = 0
    total_detections: int = 0

    @property
    def is_safe(self) -> bool:
        return self.scene_severity == Severity.SAFE

    @property
    def violation_count(self) -> int:
        return len(self.violations)


# ---------------------------------------------------------------------------
# Core rule engine
# ---------------------------------------------------------------------------

def _iou(a: Tuple, b: Tuple) -> float:
    """Compute Intersection-over-Union between two boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def apply_rules(
    detections: List[Detection],
    conf_threshold: float = CONF_DEFAULT,
) -> SceneResult:
    """
    Apply all safety rules to a list of detections.

    Parameters
    ----------
    detections:
        Raw Detection objects from the YOLO model.
    conf_threshold:
        Minimum confidence to flag a violation (default: 0.45).

    Returns
    -------
    SceneResult with all violations and the final scene severity.
    """

    result = SceneResult()
    result.total_detections = len(detections)

    # Separate by category using class-specific thresholds
    persons    = [d for d in detections if d.class_name == CLASS_PERSON]
    
    violations = [
        d for d in detections 
        if d.class_name in VIOLATION_CLASSES
        and d.confidence >= CLASS_THRESHOLDS.get(d.class_name, conf_threshold)
    ]
    
    compliant  = [
        d for d in detections 
        if d.class_name not in VIOLATION_CLASSES
        and d.class_name != CLASS_PERSON
        and d.confidence >= CLASS_THRESHOLDS.get(d.class_name, conf_threshold)
    ]

    result.total_persons = len(persons)

    # Track which compliant PPE was seen
    compliant_seen = set(d.class_name for d in compliant)
    result.compliant_ppe = sorted(compliant_seen)

    # -------------------------------------------------------------------
    # Rule 1 — Helmet Required  (CRITICAL)
    # -------------------------------------------------------------------
    no_helmet_dets = [d for d in violations if d.class_name == CLASS_NO_HELMET]
    
    # Compensating Strategy: Presence-Absence logic
    # If a person is present, but NO helmet is detected in the scene, add an inferred violation.
    if result.total_persons > 0 and CLASS_HELMET not in result.compliant_ppe and not no_helmet_dets:
        no_helmet_dets.append(Detection(CLASS_NO_HELMET, 1.0, (0.0, 0.0, 0.0, 0.0)))
        
    for det in no_helmet_dets:
        desc = "Helmet missing — worker's head is exposed"
        if det.confidence == 1.0 and det.bbox == (0.0, 0.0, 0.0, 0.0):
            desc = "Helmet missing — inferred via presence-absence logic"
            
        result.violations.append(Violation(
            rule_id=1,
            description=desc,
            severity=Severity.CRITICAL,
            confidence=det.confidence if det.confidence != 1.0 else 0.99,
            bbox=det.bbox,
        ))

    # -------------------------------------------------------------------
    # Rule 2 — High-Vis Vest Required  (CRITICAL)
    # -------------------------------------------------------------------
    no_vest_dets = [d for d in violations if d.class_name == CLASS_NO_VEST]
    
    # Compensating Strategy: Presence-Absence logic
    if result.total_persons > 0 and CLASS_VEST not in result.compliant_ppe and not no_vest_dets:
        no_vest_dets.append(Detection(CLASS_NO_VEST, 1.0, (0.0, 0.0, 0.0, 0.0)))

    for det in no_vest_dets:
        desc = "High-visibility vest missing"
        if det.confidence == 1.0 and det.bbox == (0.0, 0.0, 0.0, 0.0):
            desc = "High-visibility vest missing — inferred via presence-absence logic"
            
        result.violations.append(Violation(
            rule_id=2,
            description=desc,
            severity=Severity.CRITICAL,
            confidence=det.confidence if det.confidence != 1.0 else 0.99,
            bbox=det.bbox,
        ))

    # -------------------------------------------------------------------
    # Rule 3 — Safety Goggles Required  (MODERATE)
    # -------------------------------------------------------------------
    no_goggles_dets = [d for d in violations if d.class_name == CLASS_NO_GOGGLES]
    for det in no_goggles_dets:
        result.violations.append(Violation(
            rule_id=3,
            description="Safety glasses/goggles missing",
            severity=Severity.MODERATE,
            confidence=det.confidence,
            bbox=det.bbox,
        ))

    # -------------------------------------------------------------------
    # Rule 4 — Safety Boots Required  (MODERATE)
    # Note: model has low recall for no-boots; flagged as guidance
    # -------------------------------------------------------------------
    no_boots_dets = [d for d in violations if d.class_name == CLASS_NO_BOOTS]
    for det in no_boots_dets:
        result.violations.append(Violation(
            rule_id=4,
            description="Safety boots missing (low-confidence class — treat as guidance)",
            severity=Severity.MODERATE,
            confidence=det.confidence,
            bbox=det.bbox,
        ))

    # -------------------------------------------------------------------
    # Rule 5 — Gloves Required  (LOW)
    # -------------------------------------------------------------------
    no_gloves_dets = [d for d in violations if d.class_name == CLASS_NO_GLOVES]
    for det in no_gloves_dets:
        result.violations.append(Violation(
            rule_id=5,
            description="Protective gloves missing (context-dependent)",
            severity=Severity.LOW,
            confidence=det.confidence,
            bbox=det.bbox,
        ))

    # -------------------------------------------------------------------
    # Rule 6 — Combined Critical  (escalate if no-helmet + no-vest)
    # -------------------------------------------------------------------
    has_no_helmet = len(no_helmet_dets) > 0
    has_no_vest   = len(no_vest_dets)   > 0

    # -------------------------------------------------------------------
    # Determine overall scene severity
    # -------------------------------------------------------------------
    if not result.violations:
        result.scene_severity = Severity.SAFE
    elif has_no_helmet and has_no_vest:
        # Rule 6: escalate to CRITICAL Combined
        result.scene_severity = Severity.CRITICAL
    elif any(v.severity == Severity.CRITICAL for v in result.violations):
        result.scene_severity = Severity.CRITICAL
    elif any(v.severity == Severity.MODERATE for v in result.violations):
        result.scene_severity = Severity.MODERATE
    else:
        result.scene_severity = Severity.LOW

    return result
