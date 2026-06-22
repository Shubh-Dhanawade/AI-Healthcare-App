"""
Healthcare Insurance Document Intelligence System
Main FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import asyncio
from loguru import logger

from app.core.config import settings
from app.core.database import create_tables
from app.api.v1 import auth, documents, ai_service, claims
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    setup_logging()
    logger.info("🏥 Healthcare AI Application Starting...")
    await create_tables()
    # Ensure upload and log directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Warm up the local model asynchronously
    from app.services.ollama_client import warmup_model
    asyncio.create_task(warmup_model())
    
    logger.info(f"✅ Upload directory: {settings.UPLOAD_DIR}")
    logger.info("✅ Database tables verified")
    logger.info("🚀 Application ready!")
    yield
    # Shutdown
    logger.info("👋 Application shutting down...")
    from app.services.ollama_client import close_httpx_client
    await close_httpx_client()


# Initialize FastAPI App
app = FastAPI(
    title="Healthcare Insurance Document Intelligence System",
    description="AI-powered platform for analyzing healthcare insurance documents",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploads
if os.path.exists(settings.UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Static files for data science analysis plots
if os.path.exists("../data_science_analysis"):
    app.mount("/data_science_analysis", StaticFiles(directory="../data_science_analysis"), name="data_science_analysis")

# API Routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(ai_service.router, prefix="/api/v1/ai", tags=["AI Services"])
app.include_router(claims.router, prefix="/api/v1/claims", tags=["Claims Analytics"])


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Healthcare Insurance Document Intelligence System",
        "version": "1.0.0",
    }


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "database": "connected",
        },
    }
