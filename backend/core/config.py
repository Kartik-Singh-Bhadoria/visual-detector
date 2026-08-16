# backend/core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central app configuration.
    Values are loaded from environment variables or a .env file at project root.
    Add new fields here as new features (OCR, currency, traffic, etc.) need config.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "VisionGuide AI"
    environment: str = "development"  # development | staging | production
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS — comma-separated origins in .env, e.g. CORS_ORIGINS=http://localhost:5173
    cors_origins: str = "*"

    # Model / weights paths (filled in as each ai_engine module comes online)
    models_dir: str = "models"
    detector_weights_path: str = "models/detector/yolov12.pt"
    depth_weights_path: str = "models/depth/depth_anything_v2.pt"
    ocr_engine: str = "easyocr"  # paddleocr | easyocr | tesseract

    # Feature flags — lets you disable a module without ripping out code
    enable_scene_narration: bool = True
    enable_obstacle_detection: bool = True
    enable_ocr: bool = True
    enable_currency_recognition: bool = True
    enable_traffic_detection: bool = True
    enable_face_recognition: bool = False  # off by default: privacy-sensitive

    # Third-party keys (populate later via .env, never hardcode)
    openai_api_key: str | None = None
    google_maps_api_key: str | None = None
    google_maps_api_key: str | None = None
    google_api_key: str | None = None

    app_user_email: str = "test@example.com"
    app_user_password: str = "changeme"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance — import and call this wherever config is needed,
    instead of instantiating Settings() repeatedly.
    """
    return Settings()