# backend/api/routes/location.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.location_service import reverse_geocode
from backend.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class LocationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


@router.post("/location")
async def get_location(payload: LocationRequest):
    """
    Accepts GPS coordinates (from navigator.geolocation on the client) and
    returns a human-readable address via the Google Maps Geocoding API.
    Used to show "where am I" in the app and to make emergency alerts
    readable instead of raw lat/lng.
    """
    try:
        result = reverse_geocode(payload.lat, payload.lng)
    except Exception as e:
        logger.error(f"Location lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Location lookup failed due to an internal error.")

    return result
