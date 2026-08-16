# ai_engine/detector/distance_estimator.py
from dataclasses import dataclass

from ai_engine.detector.detector import Detection


@dataclass
class ProximityWarning:
    """A detection annotated with a rough proximity level and spoken direction."""
    label: str
    proximity: str      # "close" | "medium" | "far"
    direction: str       # "left" | "center" | "right"


def _classify_proximity(box_area_ratio: float) -> str:
    """
    box_area_ratio = detection box area / total frame area.
    Thresholds are heuristic, not physically calibrated — tune based on testing.
    """
    if box_area_ratio > 0.25:
        return "close"
    elif box_area_ratio > 0.08:
        return "medium"
    else:
        return "far"


def _classify_direction(center_x: int, frame_width: int) -> str:
    """Divide the frame into three horizontal zones."""
    third = frame_width / 3
    if center_x < third:
        return "left"
    elif center_x < 2 * third:
        return "center"
    else:
        return "right"


def estimate_proximity(
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
) -> list[ProximityWarning]:
    """
    Converts raw YOLO detections into proximity + direction warnings,
    using bounding box size/position as a fast, no-extra-model proxy for distance.
    """
    frame_area = frame_width * frame_height
    warnings: list[ProximityWarning] = []

    for d in detections:
        x1, y1, x2, y2 = d.box
        box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        area_ratio = box_area / frame_area if frame_area > 0 else 0

        proximity = _classify_proximity(area_ratio)
        direction = _classify_direction(d.center[0], frame_width)

        warnings.append(
            ProximityWarning(label=d.label, proximity=proximity, direction=direction)
        )

    return warnings