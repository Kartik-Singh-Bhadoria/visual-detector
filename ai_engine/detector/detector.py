# ai_engine/detector/detector.py
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from backend.core.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """A single detected object in a frame."""
    label: str          # e.g. "person", "chair", "car"
    confidence: float   # 0.0 - 1.0
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords
    center: tuple[int, int]         # (cx, cy) pixel coords, useful for "left/right" narration


class ObjectDetector:
    """
    Thin wrapper around a YOLOv11 model.
    Any future swap (YOLO26, RT-DETR) should only require changes inside this class.
    """

    def __init__(self, weights_path: str | None = None, confidence_threshold: float = 0.5):
        settings = get_settings()
        self.weights_path = weights_path or settings.detector_weights_path
        self.confidence_threshold = confidence_threshold
        self.model: YOLO | None = None

    def load(self) -> None:
        """
        Load model weights into memory. Call once at app startup, not per-request.
        """
        logger.info(f"Loading detector weights from '{self.weights_path}'")
        try:
            self.model = YOLO(self.weights_path)
            logger.info("Detector model loaded successfully")
        except Exception as e:
            logger.warning(
                f"Could not load custom weights at '{self.weights_path}' ({e}). "
                "Falling back to pretrained YOLO11n (COCO)."
            )
            # yolo11n.pt auto-downloads from ultralytics on first use if not present locally
            self.model = YOLO("yolo11n.pt")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a single frame (numpy array, e.g. from OpenCV/webcam).
        Returns a list of Detection objects above the confidence threshold.
        """
        if self.model is None:
            raise RuntimeError("Detector model not loaded. Call .load() first.")

        results = self.model.predict(frame, verbose=False)[0]
        detections: list[Detection] = []

        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.names[int(box.cls[0])]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            detections.append(
                Detection(label=label, confidence=confidence, box=(x1, y1, x2, y2), center=(cx, cy))
            )

        logger.debug(f"Detected {len(detections)} objects above threshold")
        return detections