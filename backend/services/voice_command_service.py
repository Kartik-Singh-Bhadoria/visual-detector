# backend/services/voice_command_service.py
import numpy as np
import cv2
import google.generativeai as genai

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.services.detector_service import detect_objects_in_image
from backend.services.ocr_service import read_text_from_image
from backend.services.scene_service import describe_scene_from_image
from backend.services.currency_service import identify_currency_from_image
from ai_engine.face.face_recognizer import FaceRecognizer

logger = get_logger(__name__)

settings = get_settings()
genai.configure(api_key=settings.google_api_key)
_intent_model = genai.GenerativeModel(settings.gemini_model_name)
_face_recognizer = FaceRecognizer()

VALID_INTENTS = [
    "describe_scene",
    "read_text",
    "identify_currency",
    "detect_obstacles",
    "remember_person",
    "general_question",
]


def _classify_intent(transcript: str) -> str:
    """
    Uses Gemini to map a spoken transcript to one of our known actions.
    Falls back to 'general_question' if classification is unclear.
    """
    prompt = (
        "Classify the user's spoken request into exactly one of these categories:\n"
        "- describe_scene: user wants to know what's around them generally\n"
        "- read_text: user wants text/labels/signs read aloud\n"
        "- identify_currency: user wants a currency note identified\n"
        "- detect_obstacles: user wants to know about obstacles/objects in their path\n"
        "- remember_person: user wants to save/remember the person currently in front of the camera\n"
        "- general_question: anything else, including specific questions about the scene\n\n"
        f"User said: \"{transcript}\"\n\n"
        "Respond with ONLY the category name, nothing else."
    )
    response = _intent_model.generate_content(prompt)
    intent = response.text.strip().lower()

    if intent not in VALID_INTENTS:
        logger.warning(f"Unrecognized intent '{intent}', defaulting to general_question")
        intent = "general_question"

    logger.info(f"Transcript: '{transcript}' -> Intent: {intent}")
    return intent


def _decode_frame(image_bytes: bytes) -> np.ndarray:
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


def _answer_general_question(image_bytes: bytes, question: str) -> str:
    """For anything that doesn't match a specific feature — pass the question + image straight to Gemini."""
    frame = _decode_frame(image_bytes)
    success, buffer = cv2.imencode(".jpg", frame)
    image_data = buffer.tobytes()

    known_name = _face_recognizer.recognize(frame)
    context = f" The person in view is recognized as {known_name}." if known_name else ""

    prompt = (
        f"You are helping a blind person. They asked: \"{question}\".{context} "
        "Answer concisely in one or two sentences based on what you see in the image."
    )
    response = _intent_model.generate_content(
        [prompt, {"mime_type": "image/jpeg", "data": image_data}]
    )
    return response.text.strip()


def handle_voice_command(image_bytes: bytes, transcript: str) -> dict:
    """
    Main entrypoint: takes a spoken transcript + current camera frame,
    routes to the correct feature, returns a spoken-ready response.
    """
    intent = _classify_intent(transcript)

    if intent == "describe_scene":
        result = describe_scene_from_image(image_bytes)
        spoken_text = result["description"]

    elif intent == "read_text":
        result = read_text_from_image(image_bytes)
        spoken_text = result["combined_text"] or "No text detected."

    elif intent == "identify_currency":
        result = identify_currency_from_image(image_bytes)
        spoken_text = result["result"]

    elif intent == "detect_obstacles":
        detections_result = detect_objects_in_image(image_bytes)
        if not detections_result:
            spoken_text = "No obstacles detected nearby."
        else:
            urgent = [d for d in detections_result if d["proximity"] != "far"]
            if not urgent:
                spoken_text = "Path looks clear nearby."
            else:
                parts = [f"{d['label']} {d['proximity']}, {d['direction']}" for d in urgent]
                spoken_text = ". ".join(parts) + "."

    elif intent == "remember_person":
        return {"intent": "remember_person", "spoken_text": "Sure, what is their name?", "awaiting_name": True}

    else:  # general_question
        spoken_text = _answer_general_question(image_bytes, transcript)

    return {"intent": intent, "spoken_text": spoken_text, "awaiting_name": False}


def save_person_name(image_bytes: bytes, name: str) -> dict:
    """
    Called as the follow-up step after 'remember_person' — saves the
    currently-visible face under the given spoken name.
    """
    frame = _decode_frame(image_bytes)
    success = _face_recognizer.save_face(frame, name)

    if success:
        spoken_text = f"Got it, I'll remember {name}."
    else:
        spoken_text = "I couldn't find a clear face to save. Please try again."

    return {"spoken_text": spoken_text}