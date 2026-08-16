# backend/services/scene_service.py
import numpy as np
import cv2

from ai_engine.scene.scene_narrator import SceneNarrator
from backend.core.logger import get_logger

logger = get_logger(__name__)

# Lazy-loaded: only initialized when first used
_narrator = None

def _get_narrator():
    global _narrator
    if _narrator is None:
        _narrator = SceneNarrator()
    return _narrator


def _decode_image(image_bytes: bytes) -> np.ndarray:
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — file may be corrupted or not a valid image.")
    return frame


def describe_scene_from_image(image_bytes: bytes) -> dict:
    """
    Main entrypoint the API route will call.
    Takes a single snapshot's image bytes, returns a natural-language
    scene description ready for text-to-speech.
    """
    frame = _decode_image(image_bytes)
    description = _get_narrator().describe(frame)

    logger.info(f"Scene description generated ({len(description)} chars)")

    return {"description": description}