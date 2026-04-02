from __future__ import annotations

import json
import datetime
from typing import List, Optional

from rules import SceneResult


# ---------------------------------------------------------------------------
# Human-readable rule descriptions
# ---------------------------------------------------------------------------

RULE_LABELS = {
    1: "HELMET MISSING",
    2: "VEST MISSING",
    3: "GOGGLES MISSING",
    4: "BOOTS MISSING",
    5: "GLOVES MISSING",
}

RULE_ACTIONS = {
    1: "Stop work immediately — provide a hard hat before continuing.",
    2: "Stop work immediately — provide a high-vis vest before continuing.",
    3: "Provide safety glasses before entering hazardous area.",
    4: "Provide steel-toe boots before site access.",
    5: "Provide protective gloves before handling materials.",
}


# ---------------------------------------------------------------------------
# Alert builder
# ---------------------------------------------------------------------------

def build_alerts(
    result: SceneResult,
    image_path: Optional[str] = None,
    frame_number: Optional[int] = None,
) -> List[str]:
    """
    Build a list of human-readable alert strings from a SceneResult.

    Parameters
    ----------
    result       : SceneResult from rules.apply_rules()
    image_path   : optional — path/name of the analysed image
    frame_number : optional — video frame number

    Returns
    -------
    List of alert strings, one per violation.
    """
    alerts: List[str] = []

    if result.is_safe:
        return alerts   # no alerts needed

    for v in result.violations:
        label  = RULE_LABELS.get(v.rule_id, f"RULE {v.rule_id}")
        action = RULE_ACTIONS.get(v.rule_id, "Review immediately.")
        alert  = (
            f"[{v.severity.emoji()} {v.severity.value}] "
            f"Rule {v.rule_id}: {label} "
            f"(conf={v.confidence:.2f}) — {action}"
        )
        alerts.append(alert)

    return alerts


def print_report(
    result: SceneResult,
    image_path: Optional[str] = None,
    frame_number: Optional[int] = None,
) -> None:
    """
    Print a formatted report to stdout.

    Parameters
    ----------
    result       : SceneResult from rules.apply_rules()
    image_path   : optional label printed in the header
    frame_number : optional video frame number
    """
    sep = "─" * 60

    # Header
    print(sep)
    if image_path:
        print(f"  Image : {image_path}")
    if frame_number is not None:
        print(f"  Frame : {frame_number}")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Time  : {ts}")
    print(sep)

    # Scene status
    sev = result.scene_severity
    print(f"  Scene status : {sev.emoji()} {sev.value}")
    print(f"  Persons detected : {result.total_persons}")

    # Compliant PPE
    if result.compliant_ppe:
        print(f"  Compliant PPE    : {', '.join(result.compliant_ppe)}")
    else:
        print("  Compliant PPE    : (none detected)")

    print()

    # Violations
    if result.violations:
        print(f"  Violations ({result.violation_count}):")
        for v in result.violations:
            label  = RULE_LABELS.get(v.rule_id, f"Rule {v.rule_id}")
            action = RULE_ACTIONS.get(v.rule_id, "Review immediately.")
            print(f"    {v.severity.emoji()} [{v.severity.value:8s}] "
                  f"Rule {v.rule_id}: {label} "
                  f"(conf={v.confidence:.2f})")
            print(f"               ↳ {action}")
    else:
        print("  ✅  No violations detected — scene is SAFE.")

    print(sep)
    print()


def to_dict(
    result: SceneResult,
    image_path: Optional[str] = None,
    frame_number: Optional[int] = None,
) -> dict:
    """
    Serialise a SceneResult to a plain Python dict (JSON-ready).

    Parameters
    ----------
    result       : SceneResult from rules.apply_rules()
    image_path   : optional label
    frame_number : optional video frame number

    Returns
    -------
    dict — suitable for json.dumps()
    """
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "image_path": image_path,
        "frame_number": frame_number,
        "scene_severity": result.scene_severity.value,
        "is_safe": result.is_safe,
        "total_persons": result.total_persons,
        "compliant_ppe": result.compliant_ppe,
        "violation_count": result.violation_count,
        "violations": [
            {
                "rule_id": v.rule_id,
                "description": v.description,
                "severity": v.severity.value,
                "confidence": round(v.confidence, 4),
                "bbox": list(v.bbox),
            }
            for v in result.violations
        ],
    }


def save_json_report(
    result: SceneResult,
    output_path: str,
    image_path: Optional[str] = None,
    frame_number: Optional[int] = None,
) -> None:
    """Save the SceneResult as a JSON file."""
    data = to_dict(result, image_path, frame_number)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON report saved → {output_path}")
