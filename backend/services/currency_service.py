# backend/services/currency_service.py
import numpy as np
import cv2

from ai_engine.currency.currency_recognizer import CurrencyRecognizer
from backend.core.logger import get_logger

logger = get_logger(__name__)

# Lazy-loaded: only initialized when first used
_recognizer = None

def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = CurrencyRecognizer()
    return _recognizer


def _decode_image(image_bytes: bytes) -> np.ndarray:
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — file may be corrupted or not a valid image.")
    return frame


def identify_currency_from_image(image_bytes: bytes) -> dict:
    """
    Main entrypoint the API route will call.
    Takes a single snapshot's image bytes, returns the identified
    currency denomination as spoken-ready text.
    """
    frame = _decode_image(image_bytes)
    result = _get_recognizer().identify(frame)

    logger.info(f"Currency identification result: {result}")

    return {"result": result}