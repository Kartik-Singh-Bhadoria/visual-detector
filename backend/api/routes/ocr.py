# backend/api/routes/ocr.py
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.ocr_service import read_text_from_image
from backend.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/read-text")
async def read_text(image: UploadFile = File(...)):
    """
    Accepts a single snapshot (not a video stream) and returns detected text.
    Intended to be called on-demand — e.g. user taps 'Read Text' — not continuously.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await image.read()
        result = read_text_from_image(image_bytes)
    except ValueError as e:
        logger.warning(f"Bad image upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail="Text reading failed due to an internal error.")

    return result