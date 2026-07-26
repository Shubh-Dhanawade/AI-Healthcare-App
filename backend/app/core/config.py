"""
Application Configuration Settings
Supports both local SQLite (dev) and PostgreSQL (production)
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union
import os
import json


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

    # Qdrant Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = ""

    # Ollama AI
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_NUM_THREAD: int = 8
    # num_gpu=-1: Ollama auto-fits as many layers as possible into VRAM without OOM.
    # DO NOT use 999 (force all layers) on a 4GB GPU — the 3.88B Q8_0 model weights
    # alone need ~3443 MiB, leaving insufficient room for the KV cache (500 Internal Server Error).
    OLLAMA_NUM_GPU: int = -1
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

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }


settings = Settings()
