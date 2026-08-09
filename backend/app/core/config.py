"""
Application Configuration Settings
Supports both local SQLite (dev) and PostgreSQL (production)
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union, Optional
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
    OLLAMA_MODEL: str = "gemma3:4b-it-q4_K_M"
    # 4 CPU threads: reduces RAM pressure (currently at 95%) when some layers fall back to CPU.
    # With Q4_K_M all layers on GPU, this becomes irrelevant but safe to keep.
    OLLAMA_NUM_THREAD: int = 4
    # Omit to let Ollama auto-detect GPU layer offloading based on free VRAM.
    OLLAMA_NUM_GPU: Optional[int] = None
    OLLAMA_KEEP_ALIVE: str = "-1"
    # Context window size: 4096 is the sweet spot for 4GB VRAM GPUs (fits model + KV cache fully in VRAM)
    OLLAMA_NUM_CTX: int = 4096

    # File Uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50

    # SMTP Email Configuration
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = "krushnakharat.official@gmail.com"
    SMTP_PASSWORD: str = "pbcgjleqmvcjsoho"

    # CORS — allow frontend dev server
    ALLOWED_ORIGINS: List[str] = [
        "*",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
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
