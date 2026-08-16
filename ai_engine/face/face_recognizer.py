# ai_engine/face/face_recognizer.py
import os
import json

import cv2
import numpy as np

from backend.core.logger import get_logger

logger = get_logger(__name__)

FACE_DATA_DIR = "models/faces"
LABELS_PATH = os.path.join(FACE_DATA_DIR, "labels.json")
MODEL_PATH = os.path.join(FACE_DATA_DIR, "lbph_model.yml")


class FaceRecognizer:
    """
    Local, privacy-first face recognition using OpenCV's LBPH recognizer.
    No cloud calls, no raw images sent anywhere — everything stays on disk locally.
    """

    def __init__(self):
        os.makedirs(FACE_DATA_DIR, exist_ok=True)
        self.face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels: dict[int, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r") as f:
                self.labels = {int(k): v for k, v in json.load(f).items()}
        if os.path.exists(MODEL_PATH):
            self.recognizer.read(MODEL_PATH)
            logger.info(f"Loaded face recognizer with {len(self.labels)} known people")

    def _save(self):
        with open(LABELS_PATH, "w") as f:
            json.dump(self.labels, f)
        self.recognizer.write(MODEL_PATH)

    def _detect_face(self, frame: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]  # take the largest/first detected face
        return cv2.resize(gray[y:y + h, x:x + w], (200, 200))

    def recognize(self, frame: np.ndarray) -> str | None:
        """Returns the person's name if recognized, None if no face or unknown face."""
        face = self._detect_face(frame)
        if face is None or not self.labels:
            return None

        label_id, confidence = self.recognizer.predict(face)
        # Lower confidence = better match in LBPH. Threshold tuned loosely; adjust after testing.
        if confidence < 70:
            return self.labels.get(label_id)
        return None

    def save_face(self, frame: np.ndarray, name: str) -> bool:
        """Registers a new face under the given name."""
        face = self._detect_face(frame)
        if face is None:
            return False

        label_id = len(self.labels)
        self.labels[label_id] = name

        # LBPH needs at least one sample to train/update
        existing_faces = [face]
        existing_ids = [label_id]

        if os.path.exists(MODEL_PATH):
            self.recognizer.update(existing_faces, np.array(existing_ids))
        else:
            self.recognizer.train(existing_faces, np.array(existing_ids))

        self._save()
        logger.info(f"Saved new face: {name}")
        return True