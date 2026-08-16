# backend/api/routes/scene.py
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.scene_service import describe_scene_from_image
from backend.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/describe-scene")
async def describe_scene(image: UploadFile = File(...)):
    """
    Accepts a single snapshot and returns a natural-language description
    of the scene, generated via Gemini. Intended to be called on-demand.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await image.read()
        result = describe_scene_from_image(image_bytes)
    except ValueError as e:
        logger.warning(f"Bad image upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Scene description failed: {e}")
        raise HTTPException(status_code=500, detail="Scene description failed due to an internal error.")

    return result