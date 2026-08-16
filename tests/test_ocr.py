# backend/services/ocr_service.py
import numpy as np
import cv2

from ai_engine.ocr.ocr_reader import OCRReader, OCRResult
from backend.core.logger import get_logger

logger = get_logger(__name__)

# Loaded once at import time — EasyOCR init is slow, must not happen per-request.
_ocr_reader = OCRReader(languages=["en"])
_ocr_reader.load()


def _decode_image(image_bytes: bytes) -> np.ndarray:
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — file may be corrupted or not a valid image.")
    return frame


def read_text_from_image(image_bytes: bytes) -> dict:
    """
    Main entrypoint the API route will call.
    Takes a single snapshot's image bytes, returns detected text blocks
    plus a combined string ready for text-to-speech.
    """
    frame = _decode_image(image_bytes)
    results: list[OCRResult] = _ocr_reader.read(frame)

    logger.info(f"OCR found {len(results)} text blocks in snapshot")

    combined_text = " ".join(r.text for r in results)

    return {
        "combined_text": combined_text,
        "blocks": [
            {
                "text": r.text,
                "confidence": r.confidence,
                "box": [{"x": x, "y": y} for x, y in r.box],
            }
            for r in results
        ],
    }