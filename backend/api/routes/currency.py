# backend/api/routes/currency.py
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.currency_service import identify_currency_from_image
from backend.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/identify-currency")
async def identify_currency(image: UploadFile = File(...)):
    """
    Accepts a single snapshot and returns the identified currency
    denomination. Intended to be called on-demand.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await image.read()
        result = identify_currency_from_image(image_bytes)
    except ValueError as e:
        logger.warning(f"Bad image upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Currency identification failed: {e}")
        raise HTTPException(status_code=500, detail="Currency identification failed due to an internal error.")

    return result