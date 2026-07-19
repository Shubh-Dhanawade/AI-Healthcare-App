"""
Application Configuration Settings
Supports both local SQLite (dev) and PostgreSQL (production)
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Healthcare AI System"
    DEBUG: bool = True

    # Database — defaults to SQLite for local dev
    DATABASE_URL: str = "sqlite+aiosqlite:///./healthcare_ai.db"

    # JWT Authentication
    SECRET_KEY: str = "dev-secret-key-change-in-production-min-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Ollama AI
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_NUM_THREAD: int = 8
    OLLAMA_NUM_GPU: int = 999
    OLLAMA_KEEP_ALIVE: str = "-1"

    # File Uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50

    # CORS — allow frontend dev server
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost",
        "http://127.0.0.1:3000",
    ]
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }


settings = Settings()
