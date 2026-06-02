# 🏥 Healthcare Insurance Document Intelligence System

> **AI-powered platform for analyzing healthcare insurance documents using Phi-3 Mini, PaddleOCR, and PyMuPDF.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black)](https://nextjs.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Phi--3%20Mini-blue)](https://ollama.ai/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docker.com/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Environment Variables](#environment-variables)
- [API Documentation](#api-documentation)
- [AI Prompts](#ai-prompts)
- [Deployment](#deployment)

---

## 🎯 Overview

Upload any healthcare insurance PDF or image document and get:

| Feature | Description |
|---|---|
| 🔍 **OCR Extraction** | PyMuPDF for digital PDFs, PaddleOCR for scanned docs |
| 🤖 **AI Summary** | Plain-language policy summaries via Phi-3 Mini |
| 📋 **Field Extraction** | Premiums, coverage, deductibles, exclusions |
| ⚠️ **Risk Detection** | Identifies risky clauses with severity levels |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                   Nginx (Port 80)               │
│         Reverse Proxy + Load Balancer           │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────┴────────────┐
        │                        │
┌───────▼──────┐      ┌─────────▼────────┐
│  Next.js     │      │   FastAPI        │
│  Frontend    │      │   Backend        │
│  Port 3000   │      │   Port 8000      │
└──────────────┘      └─────┬────────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
    ┌────────▼───┐  ┌───────▼────┐  ┌─────▼──────┐
    │ PostgreSQL │  │   Ollama   │  │  Uploads   │
    │  Port 5432 │  │ Port 11434 │  │  Volume    │
    └────────────┘  └────────────┘  └────────────┘
```

---

## ✨ Features

### 🔐 Authentication
- JWT-based auth with bcrypt password hashing
- User roles: Admin / User
- Secure token management

### 📁 Document Upload
- Drag-and-drop file upload
- Supports PDF, JPG, PNG, TIFF, WEBP
- Up to 50MB files
- Upload progress indicator

### 🔍 OCR & Text Extraction
- **PyMuPDF** for digital PDF text extraction
- **PaddleOCR** for scanned documents and images
- Auto-detection of document type
- Text cleaning and normalization

### 🤖 AI Summarization (Phi-3 Mini)
- Plain-language policy summaries
- Coverage overview
- Exclusions and restrictions
- Waiting periods
- Premium details

### 📋 Key Field Extraction
- Policy name, insurer, policy number
- Sum insured, premium, deductible
- Co-payment, waiting periods
- Coverage type, network hospitals

### ⚠️ Risk Detection
- Long waiting periods
- High deductibles
- Broad exclusions
- Hidden conditions
- Co-payment traps
- Severity: Low / Medium / High

---

## 📁 Project Structure

```
healthcare-ai-project/
├── frontend/                   # Next.js 15 App
│   ├── src/
│   │   ├── app/
│   │   │   ├── (dashboard)/    # Protected dashboard routes
│   │   │   │   ├── dashboard/  # Overview page
│   │   │   │   ├── upload/     # File upload page
│   │   │   │   └── documents/  # Documents list + detail
│   │   │   ├── login/          # Auth pages
│   │   │   └── register/
│   │   ├── components/
│   │   │   ├── layout/         # Sidebar, Navbar
│   │   │   ├── documents/      # Status badge
│   │   │   └── providers/      # React Query
│   │   ├── contexts/           # Auth context
│   │   ├── lib/                # API client
│   │   └── types/              # TypeScript types
│   └── Dockerfile
│
├── backend/                    # FastAPI Python
│   ├── app/
│   │   ├── main.py             # App entry point
│   │   ├── core/               # Config, DB, Security, Logging
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── schemas/            # Pydantic schemas
│   │   ├── services/           # Business logic
│   │   │   ├── ocr_service.py  # PyMuPDF + PaddleOCR
│   │   │   ├── ai_service.py   # Ollama AI integration
│   │   │   └── user_service.py # User operations
│   │   └── api/v1/             # REST endpoints
│   │       ├── auth.py         # /auth/*
│   │       ├── documents.py    # /documents/*
│   │       └── ai_service.py   # /ai/*
│   ├── sql/init.sql            # DB initialization
│   ├── requirements.txt
│   └── Dockerfile
│
├── nginx/                      # Reverse proxy
│   ├── nginx.conf
│   └── Dockerfile
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 Quick Start (Docker)

### Prerequisites
- Docker Desktop installed
- 4GB+ RAM available

### 1. Clone and Configure

```bash
git clone <your-repo-url>
cd healthcare-ai-project
cp .env.example .env
# Edit .env with your values
```

### 2. Start All Services

```bash
docker-compose up -d
```

### 3. Pull the Llama 3.2 AI Model

```bash
docker exec -it healthcare_ollama ollama pull llama3.2
```

### 4. Access the Application

| Service | URL |
|---|---|
| Web App | http://localhost |
| API | http://localhost/api/v1 |
| API Docs | http://localhost/api/docs |
| Ollama | http://localhost:11434 |

---

## 💻 Local Development

### Backend (FastAPI)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env → set DATABASE_URL to localhost

# Start Ollama locally
ollama serve
ollama pull llama3.2

# Run backend
uvicorn app.main:app --reload --port 8000
```

### Frontend (Next.js)

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
echo "NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1" > .env.local

# Start development server
npm run dev
```

Access: http://localhost:3000

---

## 🔧 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `POSTGRES_DB` | Database name | `healthcare_ai` |
| `POSTGRES_USER` | DB username | `healthcare_user` |
| `POSTGRES_PASSWORD` | DB password | ⚠️ Change this! |
| `SECRET_KEY` | JWT signing key | ⚠️ Change this! |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://ollama:11434` |
| `OLLAMA_MODEL` | AI model name | `llama3.2` |
| `UPLOAD_DIR` | File upload directory | `/app/uploads` |
| `MAX_FILE_SIZE_MB` | Max upload size | `50` |

---

## 📡 API Documentation

### Authentication

```http
POST /api/v1/auth/register
{
  "email": "user@example.com",
  "full_name": "John Doe",
  "password": "securepassword"
}

POST /api/v1/auth/login
{
  "email": "user@example.com",
  "password": "securepassword"
}

GET /api/v1/auth/me
Authorization: Bearer <token>
```

### Documents

```http
POST   /api/v1/documents/upload     # Upload file (multipart/form-data)
GET    /api/v1/documents             # List all user documents
GET    /api/v1/documents/{id}        # Get document with AI results
DELETE /api/v1/documents/{id}        # Delete document
```

### AI Services

```http
POST /api/v1/ai/summarize           { "document_id": "uuid" }
POST /api/v1/ai/extract-fields      { "document_id": "uuid" }
POST /api/v1/ai/risk-analysis       { "document_id": "uuid" }
```

---

## 🤖 AI Prompts

### Summarization Prompt
```
You are a healthcare insurance expert. Analyze the following insurance 
policy document and provide a comprehensive summary in plain language.
Return JSON with: summary_text, coverage_summary, exclusions_summary,
waiting_period_summary, premium_summary.
```

### Field Extraction Prompt
```
Extract key policy fields: policy_name, insurer_name, sum_insured,
premium_amount, deductible, co_payment, waiting_period, coverage_type,
network_hospitals, etc. Return as structured JSON.
```

### Risk Analysis Prompt
```
Identify risky or unfavorable clauses: long waiting periods, high 
deductibles, broad exclusions, hidden conditions. Rate each as 
low/medium/high severity with explanation and recommendation.
```

---

## 🚀 VPS Deployment

### System Requirements
- Ubuntu 20.04+
- 4GB RAM minimum
- 20GB disk space
- Docker + Docker Compose

### Deploy Steps

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 2. Clone project
git clone <your-repo>
cd healthcare-ai-project

# 3. Set production environment
cp .env.example .env
nano .env  # Set strong passwords and SECRET_KEY

# 4. Deploy
docker-compose up -d

# 5. Pull AI model (first time only, may take a while)
docker exec healthcare_ollama ollama pull llama3.2

# 6. Check status
docker-compose ps
docker-compose logs -f backend
```

### Memory Optimization for 4GB VPS

The `docker-compose.yml` limits Ollama to 3GB RAM. Llama 3.2 runs within this limit.

To use an even lighter model:
```bash
docker exec healthcare_ollama ollama pull llama3.2:1b  # Smaller 1B parameter variant
```

---

## 🔒 Security Notes

- Change `SECRET_KEY` and `POSTGRES_PASSWORD` in production
- Enable HTTPS with Let's Encrypt via Certbot
- The `ALLOWED_ORIGINS` env var controls CORS
- Uploaded files are stored in isolated user directories

---

## 📝 License

MIT License — built for Master's project research.
