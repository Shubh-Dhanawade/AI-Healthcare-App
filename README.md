# 🏥 Healthcare Insurance Document Intelligence System

> **AI-powered platform for analyzing healthcare insurance documents using Gemma 3 (4B), custom SQLite Vector RAG, FAISS local indices, PaddleOCR, and Machine Learning Underwriting Benchmarks with SHAP & LIME XAI attributions.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black)](https://nextjs.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Gemma%203%20(4B)-blue)](https://ollama.ai/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docker.com/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#%EF%B8%8F-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start-docker)
- [Local Development](#-local-development)
- [Environment Variables](#-environment-variables)
- [API Documentation](#-api-documentation)
- [AI Prompts](#-ai-prompts)
- [Deployment](#-deployment)
- [Security Notes](#-security-notes)

---

## 🎯 Overview

Upload any healthcare insurance PDF or image document and instantly get:

| Feature | Description |
|---|---|
| 🔍 **OCR Extraction** | PyMuPDF for digital PDFs, cascading fallback to PaddleOCR, EasyOCR, or PyTesseract for scanned documents |
| 🤖 **AI Summary** | Plain-language policy summaries via Gemma 3 (4B) optimized under 80 words |
| 📋 **Field Extraction** | Premiums, coverage, deductibles, co-payments, waiting periods, network hospitals |
| ⚠️ **Risk Detection** | Identifies risky clauses (co-pays, hidden deductibles, exclusions) with severity levels |
| 💬 **SSE Streaming RAG** | Chat with your policies via a custom local vector search engine (FAISS + SQLite) with token streaming |
| 📊 **Underwriting Analytics** | Review claims denial model benchmarks and Explainable AI (SHAP/LIME) attribution plots |
| ⚖️ **Policy Comparison** | Side-by-side comparison of up to 5 policies with synthesized verdicts |
| ⏰ **Renewal Alerts** | Schedule renewal and premium reminders with automatic early warnings |

---

## 🏗️ Architecture

```
                  ┌─────────────────────────────────────────────────┐
                  │                 Nginx (Port 80)                 │
                  │         Reverse Proxy + Load Balancer           │
                  └───────────────────────┬─────────────────────────┘
                                          │
                            ┌─────────────┴─────────────┐
                            │                           │
                  ┌─────────▼────────┐        ┌─────────▼────────┐
                  │  Next.js Frontend│        │  FastAPI Backend │
                  │    (Port 3000)   │        │    (Port 8000)   │
                  └──────────────────┘        └─────────┬────────┘
                                                        │
                                   ┌────────────────────┼────────────────────┐
                                   │                    │                    │
                          ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
                          │ SQLite/Postgres │  │     Ollama      │  │  Uploads Vol    │
                          │   Database DB   │  │  (Port 11434)   │  │  FAISS Indexes  │
                          └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## ✨ Features

### 🔐 Authentication
- JWT-based auth using the HS256 algorithm.
- Secure token management with password hashing via `bcrypt`.
- Role-based access control (`Admin` or `User`).

### 📁 Intelligent Document Upload
- Drag-and-drop file uploader with dynamic progress metrics.
- Support for PDF, JPG, PNG, TIFF, and WEBP formats up to 50MB.
- **SHA-256 Duplicate Check**: Prevents identical uploads by checking hashes in the DB.

### 🔍 Optimized Cascading OCR Pipeline
- **PyMuPDF** for digital PDFs (character extraction in ~1-2s).
- **Threshold fallback**: Triggers OCR when character count falls under 50 chars/page.
- **Cascading OCR Engine**: Sequentially attempts **PaddleOCR**, falling back to **EasyOCR**, and then **PyTesseract** depending on system configurations.
- Cleans double-spacing, strips headers/footers, and ignores blank pages.

### 🤖 Asynchronous Background Auto-Processing
- The system automatically triggers the analysis sequence immediately after text extraction.
- Status changes: `uploaded` ➔ `processing` ➔ `text_extracted` ➔ `completed`.
- Summarization, key-field extraction, and risk analysis run concurrently.
- Manual reprocessing triggers become available only after initial completion.

### 📋 Key Field Extraction
- Extracts policy meta-parameters: Insurer, Policy Name, Policy Number, Sum Insured, Premium, Deductibles, Co-payments, Waiting Periods, Coverage Type, and Network Hospitals.
- Saves results as structured attributes in the database for queries and comparisons.

### ⚠️ Risk Detection
- Scans policy text to evaluate risk clauses.
- Rates severity level (`Low`, `Medium`, `High`) with clear 30-word summaries and recommendations.

### 💬 Conversational SSE Streaming RAG
- Chat window supporting **instant token streaming** using FastAPI Server-Sent Events (SSE).
- **Vector Database**: Zero-dependency SQLite BLOB serialization storing 768-dimensional embeddings generated via `nomic-embed-text`.
- **FAISS Disk Caching**: Local FAISS IndexFlatIP files saved on disk per-document (`{doc_id}.faiss`) for ultra-fast top-k search.
- **Typo Tolerance**: Implements Levenshtein edit distance logic to correct search terms (e.g. "deducable" -> "deductible") prior to retrieval.
- **No-LLM Fallback Engine**: Dynamic keyword-based QA fallback if Ollama services are offline.
- **Persistent Sessions**: Supports creating, switching, and deleting chat sessions.

### 📊 Side-by-Side Policy Comparator
- Compares up to 5 policies simultaneously.
- Formulates comparison matrices and calls Gemma 3 to generate recommendations based on user profiles.

### ⏰ Notification Reminders
- Schedule renewal and premium due alerts.
- Automatic alerts trigger **7 days prior** for renewal and **5 days prior** for premium deadlines.

### 📈 Claims Underwriting Analytics & XAI
- Evaluates classifiers predicting claims denial (`claim_denied`).
- Addresses dataset imbalance via SMOTE resampling (50/50 balance).
- Compares **Logistic Regression, Decision Tree, Random Forest,** and **XGBoost** benchmarks.
- Integrates **Explainable AI (XAI)** attributions: ROC Curves, SHAP Summary Plots, and LIME Explainer plots.

### 🌏 Translation & Claims Checklists
- Translates summaries and analyses into multiple target languages.
- Generates required steps and document checklists based on medical treatment types.

### 📊 Metric Analytics
- Analyzes processing latency and model metrics.
- Asynchronous LLM-as-a-judge logs evaluating RAG **Faithfulness** and **Relevance**.

---

## 📁 Project Structure

```
AI-Healthcare-App/
├── frontend/                   # Next.js 15 App
│   ├── src/
│   │   ├── app/
│   │   │   ├── (dashboard)/    # Protected dashboard routes
│   │   │   │   ├── analytics/  # Claims ML Underwriting & XAI plots
│   │   │   │   ├── chat/       # Chat sessions & streaming RAG
│   │   │   │   ├── compare/    # Side-by-side comparator
│   │   │   │   ├── dashboard/  # Policy details & reminders
│   │   │   │   ├── documents/  # Library list & detail view
│   │   │   │   ├── model-metrics/ # LLM-as-a-judge latency & RAG logs
│   │   │   │   └── upload/     # Drag & drop upload
│   │   │   ├── login/          # Auth
│   │   │   └── register/
│   │   ├── components/         # Layout & shared elements
│   │   ├── contexts/           # Auth context
│   │   ├── lib/                # API client helpers
│   │   └── types/              # TypeScript typings
│   └── Dockerfile
│
├── backend/                    # FastAPI Python
│   ├── app/
│   │   ├── main.py             # Entrypoint
│   │   ├── api/v1/             # Endpoints (auth, documents, ai_service, claims)
│   │   ├── core/               # Database, security config, logging
│   │   ├── models/             # SQLAlchemy schemas
│   │   ├── schemas/            # Pydantic validation structures
│   │   └── services/           # OCR, Vector store, FAISS, RAG, Ollama clients
│   ├── sql/init.sql            # SQL schema
│   ├── requirements.txt        # Backend dependencies
│   └── Dockerfile
│
├── nginx/                      # Reverse Proxy configuration
│   ├── nginx.conf
│   └── Dockerfile
│
├── data_science_analysis/      # Underwriting Machine Learning code
│   ├── train_and_analyze.py    # Training, SMOTE, evaluation and XAI plots
│   ├── mock_claims_dataset.csv # Mock underwriting dataset
│   └── *.png                   # ROC, SHAP, and LIME output charts
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 Quick Start (Docker)

### Prerequisites
- Docker Desktop installed
- 8GB+ RAM available

### 1. Clone and Configure

```bash
git clone <your-repo-url>
cd AI-Healthcare-App
cp .env.example .env
# Edit .env with your variables
```

### 2. Start Services

```bash
docker-compose up -d --build
```

### 3. Pull Models

```bash
docker exec -it healthcare_ollama ollama pull hf.co/kkross/gemma-3-4b-cord19-finetuned-new:latest
docker exec -it healthcare_ollama ollama pull nomic-embed-text
```

### 4. Port Gateways

- Web App: `http://localhost`
- API Gateway: `http://localhost/api/v1`
- Swagger Docs: `http://localhost/api/docs`

---

## 💻 Local Development

### Backend (FastAPI)

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate.bat
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

# Ensure Ollama is running locally and pull models
ollama serve
ollama pull hf.co/kkross/gemma-3-4b-cord19-finetuned-new:latest
ollama pull nomic-embed-text

# Run FastAPI dev server
uvicorn app.main:app --reload --port 8000
```

### Frontend (Next.js)

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1" > .env.local
npm run dev
```
Access: `http://localhost:3000`

---

## 🔧 Environment Variables

### Backend Configuration (`backend/.env`)
| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | Database connection URL | `sqlite+aiosqlite:///./healthcare_ai.db` |
| `SECRET_KEY` | JWT signing security key | `dev-secret-key-change-in-production...` |
| `OLLAMA_BASE_URL` | Local Ollama address | `http://localhost:11434` |
| `OLLAMA_MODEL` | Text analysis model name | `hf.co/kkross/gemma-3-4b-cord19-finetuned-new:latest` |
| `OLLAMA_NUM_THREAD` | CPU cores assigned to Ollama | `8` |
| `OLLAMA_NUM_GPU` | GPU blocks assigned to Ollama | `999` |
| `OLLAMA_KEEP_ALIVE` | Period model is kept in memory | `10m` |
| `UPLOAD_DIR` | Upload storage directory path | `./uploads` |
| `MAX_FILE_SIZE_MB` | File size limit | `50` |

---

## 📡 API Documentation

### Authentication
- `POST /api/v1/auth/register` - Create user profile.
- `POST /api/v1/auth/login` - Authenticate user & return JWT token.
- `GET /api/v1/auth/me` - Retrieve authenticated user profile.

### Documents
- `POST /api/v1/documents/upload` - Upload file (multipart).
- `GET /api/v1/documents` - List files.
- `GET /api/v1/documents/{id}` - Details with summaries/fields/risks.
- `DELETE /api/v1/documents/{id}` - Delete document and indices.
- `POST /api/v1/documents/compare` - Compare multiple policies.
- `POST /api/v1/documents/{id}/run-summary` - Asynchronously rerun summary.
- `POST /api/v1/documents/{id}/run-fields` - Asynchronously rerun field extraction.
- `POST /api/v1/documents/{id}/run-risks` - Asynchronously rerun risk analysis.
- `GET /api/v1/documents/{id}/export` - Export policy details to styled HTML/PDF.
- `POST /api/v1/documents/{id}/email` - Send generated PDF report via email.
- `GET /api/v1/documents/reminders` - Get scheduled notifications.
- `POST /api/v1/documents/reminders` - Schedule dates for alerts.
- `PATCH /api/v1/documents/reminders/{id}/dismiss` - Mark reminder dismissed.

### AI & Chat
- `POST /api/v1/ai/summarize` - Request Gemma summary.
- `POST /api/v1/ai/extract-fields` - Extract key metadata.
- `POST /api/v1/ai/risk-analysis` - Identify risky policy details.
- `POST /api/v1/ai/chat` - Synchronous RAG chatbot search.
- `POST /api/v1/ai/chat/stream` - SSE streaming RAG chatbot search.
- `POST /api/v1/ai/translate` - Translate analysis content.
- `POST /api/v1/ai/claims-checklist` - Generate required steps for a treatment.
- `GET /api/v1/ai/model-metrics` - Latency and RAG accuracy logs.
- `GET /api/v1/ai/chat/sessions` - List chat sessions.
- `POST /api/v1/ai/chat/sessions` - Create chat session.
- `GET /api/v1/ai/chat/sessions/{id}/messages` - Messages list.
- `GET /api/v1/ai/chat/history/{document_id}` - Chat history filtered by document.
- `DELETE /api/v1/ai/chat/sessions/{id}` - Delete session.

### Underwriting Analytics
- `GET /api/v1/claims/stats` - Underwriting sample splits and benchmark scores.

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

## 🔒 Security Notes

- Change `SECRET_KEY` and postgres passwords before launching in production.
- Enable HTTPS with SSL certificates.
- Configured CORS origins through `ALLOWED_ORIGINS` inside backend environment variables.
- User directories isolate file access paths securely.

---

## 📝 License

MIT License — built for Master's project research.
