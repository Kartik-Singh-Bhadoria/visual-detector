# backend/services/detector_service.py
import numpy as np
import cv2

from ai_engine.detector.detector import ObjectDetector, Detection
from ai_engine.detector.distance_estimator import estimate_proximity
from backend.core.logger import get_logger

logger = get_logger(__name__)

# Loaded once at import time (module-level singleton), not per-request.
# Avoids reloading YOLO weights on every API call, which would be far too slow.
_detector = ObjectDetector()
_detector.load()


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw uploaded image bytes (e.g. from a FastAPI UploadFile) into
    a numpy array/OpenCV frame that the detector can process.
    """
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — file may be corrupted or not a valid image.")
    return frame


def detect_objects_in_image(image_bytes: bytes) -> list[dict]:
    """
    Main entrypoint the API route will call.
    Takes raw image bytes, returns a list of plain dicts (JSON-serializable)
    describing each detected object — ready to hand straight to FastAPI.
    """
    frame = _decode_image(image_bytes)
    detections: list[Detection] = _detector.detect(frame)

    frame_height, frame_width = frame.shape[:2]
    proximity_warnings = estimate_proximity(detections, frame_width, frame_height)

    logger.info(f"Detected {len(detections)} objects in uploaded image")

    return [
        {
            "label": d.label,
            "confidence": round(d.confidence, 3),
            "box": {"x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3]},
            "center": {"x": d.center[0], "y": d.center[1]},
            "proximity": p.proximity,
            "direction": p.direction,
        }
        for d, p in zip(detections, proximity_warnings)
    ]