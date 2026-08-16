# ai_engine/scene/scene_narrator.py
import base64

import numpy as np
import cv2
import google.generativeai as genai

from backend.core.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class SceneNarrator:
    """
    Wraps the Gemini API to turn a camera frame into a short spoken description
    of the scene — e.g. "You are in a hallway. A door is ahead on your right."
    """

    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        settings = get_settings()
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your .env file before using scene narration."
            )
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(model_name)

    def describe(self, frame: np.ndarray) -> str:
        """
        Takes a single frame (numpy array from OpenCV), returns a short
        natural-language scene description suitable for text-to-speech.
        """
        # Encode frame as JPEG bytes for the API call
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Could not encode frame for scene narration.")
        image_bytes = buffer.tobytes()

        prompt = (
            "You are describing this scene to a blind person in one or two short "
            "sentences. Be concise and practical: mention people, obstacles, doors, "
            "or notable objects and their approximate position (left/right/ahead). "
            "Do not describe colors or aesthetics. No preamble, just the description."
        )

        response = self.model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        )

        description = response.text.strip()
        logger.info(f"Scene narration: {description}")
        return description