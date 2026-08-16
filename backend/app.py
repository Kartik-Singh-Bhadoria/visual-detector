# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import ocr
from backend.core.config import get_settings
from backend.core.logger import setup_logging, get_logger
from backend.api.routes import currency
from backend.api.routes import detector
from backend.api.routes import voice        
from backend.api.routes import scene
from backend.api.routes import auth

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Real-time AI visual companion for visually impaired users",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scene.router, prefix="/api", tags=["scene"])
app.include_router(detector.router, prefix="/api", tags=["detector"])
app.include_router(ocr.router, prefix="/api", tags=["ocr"])
app.include_router(currency.router, prefix="/api", tags=["currency"])
app.include_router(voice.router, prefix="/api", tags=["voice"])
app.include_router(auth.router, prefix="/api", tags=["auth"])

@app.on_event("startup")
async def on_startup():
    logger.info(f"Starting {settings.app_name} in '{settings.environment}' mode")

@app.get("/health")
async def health_check():
    logger.info("Health check hit")
    return {"status": "ok", "service": settings.app_name}