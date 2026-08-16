# backend/api/routes/detector.py
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.detector_service import detect_objects_in_image
from backend.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    """
    Accepts an image upload, runs YOLO object detection, and returns
    a list of detected objects with labels, confidence, and bounding boxes.

    Example (curl):
        curl -X POST "http://127.0.0.1:8000/api/detect" -F "image=@test.jpg"
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await image.read()
        detections = detect_objects_in_image(image_bytes)
    except ValueError as e:
        logger.warning(f"Bad image upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Detection failed due to an internal error.")

    return {"count": len(detections), "detections": detections}