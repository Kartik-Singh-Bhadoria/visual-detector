# ai_engine/currency/currency_recognizer.py
import numpy as np
import cv2
import google.generativeai as genai

from backend.core.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class CurrencyRecognizer:
    """
    Wraps the Gemini API to identify currency notes from a photo —
    e.g. "This is a 500 rupee note."
    Reuses the same VLM as scene narration, with a specialized prompt.
    """

    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        settings = get_settings()
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your .env file before using currency recognition."
            )
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(model_name)

    def identify(self, frame: np.ndarray) -> str:
        """
        Takes a single frame (numpy array from OpenCV), returns a short
        spoken-friendly identification of the currency note shown.
        """
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Could not encode frame for currency recognition.")
        image_bytes = buffer.tobytes()

        prompt = (
            "You are identifying a currency note for a blind person. "
            "Look at the image and state ONLY the denomination and currency "
            "(e.g. 'This is a 500 rupee note.' or 'This is a 20 dollar bill.'). "
            "If no currency note is visible, say 'No currency note detected.' "
            "If the note appears damaged or folded, mention that briefly after "
            "identifying the denomination. Keep it to one short sentence."
        )

        response = self.model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        )

        result = response.text.strip()
        logger.info(f"Currency recognition: {result}")
        return result