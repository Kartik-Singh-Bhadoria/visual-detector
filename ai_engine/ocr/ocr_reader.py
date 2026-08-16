# ai_engine/ocr/ocr_reader.py
from dataclasses import dataclass

import numpy as np
import easyocr

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """A single piece of detected text in an image."""
    text: str
    confidence: float          # 0.0 - 1.0
    box: list[tuple[int, int]]  # 4 corner points of the text region


class OCRReader:
    """
    Thin wrapper around EasyOCR.
    Loads the model once (call .load()) and reuses it across requests —
    EasyOCR model init is slow, so this must not happen per-request.
    """

    def __init__(self, languages: list[str] | None = None):
        self.languages = languages or ["en"]  # add "hi" here later for Hindi support
        self.reader: easyocr.Reader | None = None

    def load(self) -> None:
        logger.info(f"Loading EasyOCR reader for languages: {self.languages}")
        # gpu=False since target hardware has no GPU; flip to True later if running on Colab/GPU box
        self.reader = easyocr.Reader(self.languages, gpu=False)
        logger.info("EasyOCR reader loaded successfully")

    def read(self, frame: np.ndarray, confidence_threshold: float = 0.4) -> list[OCRResult]:
        """
        Run OCR on a single frame/image (numpy array from OpenCV).
        Returns detected text blocks above the confidence threshold.
        """
        if self.reader is None:
            raise RuntimeError("OCR reader not loaded. Call .load() first.")

        raw_results = self.reader.readtext(frame)
        results: list[OCRResult] = []

        for box, text, confidence in raw_results:
            if confidence < confidence_threshold:
                continue
            results.append(
                OCRResult(
                    text=text,
                    confidence=round(float(confidence), 3),
                    box=[(int(x), int(y)) for x, y in box],
                )
            )

        logger.debug(f"OCR found {len(results)} text blocks above threshold")
        return results