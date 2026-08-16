# ai_engine/scene/scene_narrator.py
import base64

import numpy as np
import cv2
import google.generativeai as genai

from ai_engine.detector.detector import ObjectDetector
from backend.core.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class SceneNarrator:
    """
    Wraps the Gemini API to turn a camera frame into a short spoken description
    of the scene — e.g. "You are in a hallway. A door is ahead on your right."
    """

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model = None
        if not settings.google_api_key:
            logger.warning(
                "GOOGLE_API_KEY is not set. Falling back to an offline scene description."
            )
            return
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(model_name or settings.gemini_model_name)

    def _fallback_describe(self, frame: np.ndarray) -> str:
        detector = ObjectDetector()
        detector.load()
        detections = detector.detect(frame)

        if not detections:
            return "No major objects are visible right now. The area ahead looks open."

        width = frame.shape[1]
        scene_parts = []
        seen = set()

        for detection in detections[:5]:
            label = detection.label.lower()
            if label in seen:
                continue
            seen.add(label)

            if detection.center[0] < width * 0.33:
                position = "on your left"
            elif detection.center[0] > width * 0.67:
                position = "on your right"
            else:
                position = "ahead of you"

            scene_parts.append(f"{label} {position}")

        if len(scene_parts) == 1:
            return f"I can see {scene_parts[0]}."

        return f"I can see {', '.join(scene_parts[:3])}."

    def describe(self, frame: np.ndarray) -> str:
        """
        Takes a single frame (numpy array from OpenCV), returns a short
        natural-language scene description suitable for text-to-speech.
        """
        if self.model is None:
            return self._fallback_describe(frame)

        try:
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
        except Exception as exc:
            logger.warning(f"Gemini scene narration failed, using offline fallback: {exc}")
            return self._fallback_describe(frame)