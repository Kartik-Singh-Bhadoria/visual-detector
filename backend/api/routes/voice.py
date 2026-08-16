# backend/api/routes/voice.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.services.voice_command_service import handle_voice_command, save_person_name
from backend.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/voice-command")
async def voice_command(image: UploadFile = File(...), transcript: str = Form(...)):
    """
    Accepts a snapshot + spoken transcript, classifies intent, routes to the
    matching feature, and returns a spoken-ready response.

    If the response has "awaiting_name": true, the frontend should listen for
    the next thing the user says and send it to /save-face instead of
    routing it back through this endpoint.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    try:
        image_bytes = await image.read()
        result = handle_voice_command(image_bytes, transcript)
    except Exception as e:
        logger.error(f"Voice command failed: {e}")
        raise HTTPException(status_code=500, detail="Voice command processing failed.")

    return result


@router.post("/save-face")
async def save_face(image: UploadFile = File(...), name: str = Form(...)):
    """
    Follow-up step after 'remember_person' intent — saves the currently
    visible face under the given name, using OpenCV's local LBPH recognizer.
    No image or face data is ever sent to any cloud service.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    try:
        image_bytes = await image.read()
        result = save_person_name(image_bytes, name)
    except Exception as e:
        logger.error(f"Save face failed: {e}")
        raise HTTPException(status_code=500, detail="Saving face failed.")

    return result