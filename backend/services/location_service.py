# backend/services/location_service.py
import requests

from backend.core.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def reverse_geocode(lat: float, lng: float) -> dict:
    """
    Turns raw GPS coordinates into a human-readable address using the
    Google Maps Geocoding API.

    Requires GOOGLE_MAPS_API_KEY in .env — this is a *separate* key from
    GOOGLE_API_KEY (Gemini). Create it in Google Cloud Console with the
    "Geocoding API" enabled and billing turned on.

    Falls back to returning the raw coordinates (with a note) if no key is
    configured or the request fails, so the caller always gets something
    speakable rather than a hard error.
    """
    if not settings.google_maps_api_key:
        logger.warning("GOOGLE_MAPS_API_KEY not set — returning raw coordinates only.")
        return {
            "lat": lat,
            "lng": lng,
            "address": f"{lat:.5f}, {lng:.5f}",
            "resolved": False,
        }

    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"latlng": f"{lat},{lng}", "key": settings.google_maps_api_key},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK" or not data.get("results"):
            logger.warning(f"Geocoding API returned status={data.get('status')}")
            return {
                "lat": lat,
                "lng": lng,
                "address": f"{lat:.5f}, {lng:.5f}",
                "resolved": False,
            }

        best = data["results"][0]
        address = best.get("formatted_address", f"{lat:.5f}, {lng:.5f}")

        # Pull out a shorter "nearby place" style label when available
        # (e.g. neighborhood/locality) for a friendlier spoken summary.
        locality = None
        for component in best.get("address_components", []):
            if "sublocality" in component["types"] or "locality" in component["types"]:
                locality = component["long_name"]
                break

        logger.info(f"Reverse geocoded ({lat}, {lng}) -> {address}")

        return {
            "lat": lat,
            "lng": lng,
            "address": address,
            "locality": locality,
            "resolved": True,
        }
    except Exception as e:
        logger.error(f"Reverse geocoding failed: {e}")
        return {
            "lat": lat,
            "lng": lng,
            "address": f"{lat:.5f}, {lng:.5f}",
            "resolved": False,
        }
