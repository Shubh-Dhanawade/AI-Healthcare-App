# 🏥 Healthcare Insurance Document Intelligence System: Project Status & Features

This document provides a comprehensive technical status and structural reference of the **Healthcare Insurance Document Intelligence System**. It outlines all currently implemented features, database schemas, processing mechanics, and configurations. 

Use this file as a reference when designing, planning, or adding new features to ensure consistency with the existing system architecture.

---

## 🏗️ System Architecture & Services

The system is fully containerized using **Docker Compose** and structured as follows:

```
                  ┌─────────────────────────────────────────────────┐
                  │                 Nginx (Port 80)                 │
                  │        Web Server / Reverse Proxy & SSE         │
                  └───────────────────────┬─────────────────────────┘
                                          │
                            ┌─────────────┴─────────────┐
                            │                           │
                  ┌─────────▼────────┐        ┌─────────▼────────┐
                  │ Next.js Frontend │        │ FastAPI Backend  │
                  │    (Port 3000)   │        │    (Port 8000)   │
                  └──────────────────┘        └─────────┬────────┘
                                                        │
                                   ┌────────────────────┼────────────────────┐
                                   │                    │                    │
                          ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
                          │ SQLite/Postgres │  │  Ollama Engine  │  │ User Uploads    │
                          │   Database DB   │  │  (Port 11434)   │  │ Storage Volume  │
                          └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Stack Components

1. **Reverse Proxy (Nginx)**: 
   - Acts as the primary entry point on port 80/443.
   - Forwards `/api/*` requests to the FastAPI backend.
   - Forwards all other requests to the Next.js frontend.
   - Disables buffer caching on `/api/v1/ai/chat/stream` to support SSE (Server-Sent Events) streaming.
   - Limits client request size to `55M` for document uploads.

2. **Frontend (Next.js 15)**:
   - Uses Tailwind CSS and vanilla styling for UI design.
   - Handles client-side routing, state management (React Query), and authorization via JWT.
   - Features real-time background polling for document processing status.

3. **Backend (FastAPI - Python 3.11)**:
   - Orchestrates OCR, database models, RAG services, and data science endpoints.
   - Interacts with database sessions and local file storage.
   - Manages asynchronous analysis pipelines to prevent API request blocks.

4. **AI Engine (Ollama)**:
   - Local LLM execution provider.
   - Model for text tasks: `hf.co/kkross/gemma-3-4b-cord19-finetuned-new:latest` (or `gemma3:4b`).
   - Model for embeddings: `nomic-embed-text` (generates 768-dimensional semantic vectors).

5. **Storage & Database**:
   - **Database**: SQLite locally (`sqlite+aiosqlite:///./healthcare_ai.db`), PostgreSQL in Docker production.
   - **Uploads**: User-specific physical uploads stored at `/app/uploads/{user_id}/`.
   - **FAISS Indexes**: High-speed similarity indices saved as binary flat files in the uploads folder.

---

## 📋 Comprehensive Implemented Feature Checklist

### 1. 🔐 User Authentication & Authorization
- **Implementation Status**: Fully Complete
- **Backend File**: [auth.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/api/v1/auth.py)
- **Frontend Directory**: [login/](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/frontend/src/app/login) & [register/](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/frontend/src/app/register)
- **Details**:
  - JWT-based authentication using HS256 algorithm.
  - Hashing passwords using `bcrypt` (minimum 8 characters).
  - Standard user roles (`Admin` or `User`) to restrict API access.
  - Axios interceptors auto-append Bearer Token to outgoing requests and redirect 401 Unauthorized responses to the login page.

### 2. 📁 Document Library & Upload
- **Implementation Status**: Fully Complete
- **Backend File**: [documents.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/api/v1/documents.py)
- **Frontend Page**: [upload/page.tsx](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/frontend/src/app/%28dashboard%29/upload/page.tsx)
- **Details**:
  - Drag-and-drop file uploader with upload percentage indicator.
  - Supports PDF, JPG, PNG, TIFF, and WEBP formats up to 50MB.
  - Automatically calculates a `SHA-256` hash of file content upon upload. If a file with the same hash exists, it rejects/links to prevent duplicated resource allocation.

### 3. 🔍 Cascade OCR & Text Extraction Pipeline
- **Implementation Status**: Fully Complete
- **Backend Files**: [pdf_processor.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/services/pdf_processor.py) & [ocr_service.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/services/ocr_service.py)
- **Mechanics**:
  - **PyMuPDF**: Fast digital extraction (`page.get_text("text")`) in < 2 seconds.
  - **Threshold Check**: If character count is less than 50 per page, the pipeline assumes it is a scanned page or image, and triggers OCR.
  - **OCR Cascade Fallback**: Attempts **PaddleOCR** first. If not installed/fails, attempts **EasyOCR**. If that also fails, attempts **PyTesseract** (requires system tesseract).
  - Skips blank pages, removes duplicate whitespace, and strips running headers/footers via heuristic checks.

### 4. 🤖 Asynchronous Background Auto-Processing
- **Implementation Status**: Fully Complete
- **Backend Service**: [documents.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/api/v1/documents.py#L520-L650)
- **Frontend Page**: [[id]/page.tsx](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/frontend/src/app/%28dashboard%29/documents/%5Bid%5D/page.tsx)
- **Timeline**:
  - Document status updates: `uploaded` ➔ `processing` (Extracting Text) ➔ `text_extracted` (Launches background worker) ➔ `completed`.
  - Once status changes to `text_extracted`, the frontend auto-triggers three concurrent backend tasks:
    1. **Summary generation** — structured 5-field AI summary (400–500 word flowing narrative + 4 bullet-point sub-sections).
    2. **Field extraction** — policy details (insurer, premium, deductibles, etc.).
    3. **Risk Analysis** — clause warnings with severity ratings.
  - Progress messages and spinners are displayed in real-time. Manual rerun buttons are hidden until status is `completed` or `failed`.

### 5. 📝 Structured AI Summary Generation
- **Implementation Status**: Fully Complete
- **Backend Service**: [ai_service.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/services/ai_service.py)
- **Frontend Page**: [[id]/page.tsx](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/frontend/src/app/%28dashboard%29/documents/%5Bid%5D/page.tsx)
- **Summary Format** (5-Field Structured Output):
  - **`summary_text`** — A comprehensive **400–500 word flowing narrative** written in a warm, professional tone addressed directly to the user (e.g., "Your policy details", "You are covered for"). Structured into 6 clear prose paragraphs: *(1) Introduction — insured name and plan, (2) Coverage Overview, (3) Family & Premium Details, (4) Key Benefits, (5) Waiting Periods & Restrictions, (6) Closing Advisory.* **No bullet points** — clean flowing paragraphs only.
  - **`coverage_summary`** — Key coverages and benefits in **bullet points** (each starting with `•`). One complete benefit per point. Max 80 words.
  - **`exclusions_summary`** — Key policy exclusions in **bullet points** (each starting with `•`). One complete exclusion per point. Max 80 words.
  - **`waiting_period_summary`** — Waiting period rules in **bullet points** (each starting with `•`). One complete rule per point. Max 80 words.
  - **`premium_summary`** — Premium, deductible, and co-payment details in **bullet points** (each starting with `•`). One complete item per point. Max 80 words.
- **LLM Prompting**: Uses a strict JSON-only prompt (`SUMMARIZATION_PROMPT`) with Gemma 3 (4B). The model is instructed to return a valid JSON object with exactly these 5 keys — no markdown wrappers, no preamble, no extra keys.
- **Storage**: All 5 fields are saved into the `summaries` database table (one record per document, enforced by `UNIQUE` constraint on `document_id`).
- **Translation Support**: Each of the 5 fields can be independently translated to a target language via the Google Translate API fallback → Ollama translation chain.

### 6. 🤖 Structured Key Field Extraction
- **Implementation Status**: Fully Complete
- **Backend Service**: [ai_service.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/services/ai_service.py)
- **Details**:
  - Prompts Gemma 3 (4B) to extract key parameters from policy text as structured JSON.
  - Extracted fields: `Insurer Name`, `Policy Name`, `Policy Number`, `Sum Insured`, `Premium Amount`, `Deductible`, `Co-payment`, `Waiting Periods`, `Coverage Type`, `Policy Term`, `Network Hospitals`, `Pre-existing Coverage`, `Maternity Coverage`, `Room Rent Limit`, and `Claim Process`.
  - Saves records directly into the `extracted_fields` database table for programmatic listing and comparison.

### 7. ⚠️ Policy Clause Risk Detection
- **Implementation Status**: Fully Complete
- **Backend Service**: [ai_service.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/services/ai_service.py)
- **Details**:
  - Scans document text for hidden deductibles, long pre-existing disease waiting periods, copay traps, and broad exclusions.
  - Rates risk severity (`Low`, `Medium`, `High`).
  - Limits explanation and recommendation text to 30 words per item for faster execution. Saves data to the `risk_analyses` table.

### 8. 💬 Conversational Chat & Streaming RAG
- **Implementation Status**: Fully Complete
- **Backend Routes**: `/ai/chat`, `/ai/chat/stream`, `/chat/sessions`
- **Backend Service**: [rag_service.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/services/rag_service.py)
- **Frontend Page**: [chat/page.tsx](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/frontend/src/app/%28dashboard%29/chat/page.tsx)
- **Core Mechanics**:
  - **Embeddings**: Text chunks embedded via Ollama `nomic-embed-text` (768 dimensions).
  - **Local FAISS Index**: Generates flat inner-product index files (`{doc_id}.faiss`) stored directly on disk for high-speed local cosine similarity search.
  - **SSE Streaming**: Fast token streaming via Event Stream response (< 200ms TTFT).
  - **Typo Tolerance**: Matches user's query against document vocabulary using Levenshtein edit distance logic. Corrects typos (e.g., "deducable" -> "deductible") prior to executing query.
  - **Identity Grounding**: Seamlessly binds logged-in user names (e.g. `krushna`) directly into the LLM system prompts.
  - **No-LLM Fallback Engine**: If Ollama or the embedding service is unreachable, a smart fallback system tokenizes queries, parses keyword triggers, and extracts relevant sentences from the policy text or returns realistic fallback definitions (deductibles, co-pays).
  - **Persistent Session Storage**: Stores chat history in `chat_sessions` and `chat_messages` tables. Supports listing, deleting, creating, and switching active chat threads.

### 9. 📊 Side-by-Side Policy Comparator
- **Implementation Status**: Fully Complete
- **Backend Endpoint**: `/api/v1/documents/compare`
- **Frontend Page**: [compare/page.tsx](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/frontend/src/app/%28dashboard%29/compare/page.tsx)
- **Details**:
  - Allows selecting and comparing up to 5 uploaded documents simultaneously.
  - Displays a tabular matrix comparing sum insured, premium, co-pay, deductibles, waiting periods, and exclusions.
  - Uses Gemma 3 (4B) to generate a synthesized summary recommending which policy is "best for" specific situations and a final comparison verdict.

### 10. ⏰ Policy Renewal & Premium Payment Reminders
- **Implementation Status**: Fully Complete
- **Backend Endpoint**: `/api/v1/documents/reminders`
- **Details**:
  - Allows scheduling a Policy Renewal Date and Premium Due Date.
  - Dynamically calculates alerts: renewal notifications trigger **7 days prior**; premium notifications trigger **5 days prior**.
  - Notifications are listed in the user dashboard and can be dismissed via API PATCH.

### 11. 📈 Claims Underwriting Analytics & Explainable AI (XAI)
- **Implementation Status**: Fully Complete
- **Pipeline Script**: [train_and_analyze.py](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/data_science_analysis/train_and_analyze.py)
- **Backend Route**: `/api/v1/claims/stats`
- **Frontend Page**: [analytics/page.tsx](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/frontend/src/app/%28dashboard%29/analytics/page.tsx)
- **Details**:
  - Models training script (`train_and_analyze.py`) trains a classification model to predict `claim_denied` using mock claims data.
  - Solves severe class imbalance using SMOTE over-sampling.
  - Compares benchmark statistics (Precision, Recall, F1, and AUC-ROC) of **Logistic Regression, Decision Tree, Random Forest,** and **XGBoost**. Selected Random Forest as best-performing.
  - Generates and serves static plots for Explainable AI (XAI):
    - **ROC Curves**: Performance comparison across models.
    - **SHAP Summary Plot**: Global feature impact attribution.
    - **LIME Explainer**: Local prediction feature attribution for individual claims.

### 12. 🌏 Document Translation
- **Implementation Status**: Fully Complete
- **Backend Route**: `/api/v1/ai/translate`
- **Details**: Translates each of the 5 summary fields independently into a target language. Uses the free Google Translate web API as the primary engine for fast, highly accurate, keyless translation, falling back to local Ollama (Gemma 3) if offline. Each field is translated in parallel using `Promise.all` on the frontend.


### 13. 📋 Claims Checklist Generator
- **Implementation Status**: Fully Complete
- **Backend Route**: `/api/v1/ai/claims-checklist`
- **Details**: Generates custom required documents and steps for a specific treatment type based on the terms and exceptions extracted from the policy text.

### 14. 📊 RAG Evaluation & System Metrics
- **Implementation Status**: Fully Complete
- **Backend Route**: `/api/v1/ai/model-metrics`
- **Details**:
  - Implements an asynchronous LLM-as-a-judge metric evaluator.
  - Evaluates every RAG search query on three indexes: **Faithfulness** (is the answer grounded in the context?), **Answer Relevance** (does it address the query?), and **Context Relevance** (does retrieval match the question?).
  - Logs latency and accuracy scores into the database `rag_query_logs` for visualization.

### 15. 📤 PDF Report Export & Email
- **Implementation Status**: Fully Complete
- **Backend Endpoint**: `/api/v1/documents/{id}/export` & `/api/v1/documents/{id}/email`
- **Details**:
  - Generates clean, CSS-styled HTML representations of document summaries, fields, and risks.
  - Converts HTML page into downloadable PDF reports.
  - Automatically sends reports as PDF attachments to a specified recipient email.

---

## 🗄️ Database Models & SQLite/Postgres Schema

Models are declared in [backend/app/models/](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/backend/app/models) using SQLAlchemy Declarative Mapping.

### 1. `User` (Table: `users`)
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | Primary Key (UUID) | Unique user ID |
| `email` | VARCHAR(255) | Unique, Indexed, NOT NULL | Account login email |
| `full_name` | VARCHAR(255) | NOT NULL | User name |
| `hashed_password` | VARCHAR(255) | NOT NULL | Bcrypt hash |
| `role` | VARCHAR(50) | Default: "user" | Admin / User role |
| `is_active` | BOOLEAN | Default: True | Account status |
| `created_at` | TIMESTAMP | Default: UTC Now | Account signup timestamp |

### 2. `Document` (Table: `documents`)
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | Primary Key (UUID) | Unique document ID |
| `user_id` | VARCHAR(36) | Foreign Key (`users.id`) | Owner ID |
| `original_filename` | VARCHAR(255)| NOT NULL | Name of uploaded file |
| `file_path` | VARCHAR(512) | NOT NULL | System path to file |
| `file_hash` | VARCHAR(64) | Unique, Nullable | SHA-256 duplicate checker |
| `file_type` | VARCHAR(100) | NOT NULL | Mime-type extension |
| `file_size_bytes` | INTEGER | NOT NULL | Size in bytes |
| `status` | VARCHAR(50) | Default: "uploaded" | uploaded/processing/completed/failed |
| `page_count` | INTEGER | Default: 0 | Number of parsed pages |
| `extracted_text` | TEXT | Nullable | Combined OCR / raw text content |
| `extraction_method` | VARCHAR(50) | Nullable | pymupdf / paddleocr / easyocr |
| `renewal_date` | TIMESTAMP | Nullable | Scheduled policy renewal date |
| `premium_due_date`| TIMESTAMP | Nullable | Scheduled premium due date |

### 3. `ExtractedField` (Table: `extracted_fields`)
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | Primary Key (UUID) | Unique field ID |
| `document_id` | VARCHAR(36) | Foreign Key (`documents.id`) | Associated document |
| `field_name` | VARCHAR(100) | NOT NULL | Target label (e.g. "deductible") |
| `field_value` | TEXT | Nullable | Value extracted |
| `field_category` | VARCHAR(100) | Nullable | Section mapping (financial, etc.) |

### 4. `Summary` (Table: `summaries`)
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | Primary Key (UUID) | Unique summary ID |
| `document_id` | VARCHAR(36) | Foreign Key (`documents.id`), Unique | Linked document (one summary per doc) |
| `summary_text` | TEXT | NOT NULL | 400–500 word flowing narrative — no bullet points |
| `coverage_summary` | TEXT | Nullable | Coverage highlights in `•` bullet points (max 80 words) |
| `exclusions_summary`| TEXT | Nullable | Exclusions in `•` bullet points (max 80 words) |
| `waiting_period_summary`| TEXT | Nullable | Waiting period rules in `•` bullet points (max 80 words) |
| `premium_summary` | TEXT | Nullable | Premium / deductible / co-pay in `•` bullet points (max 80 words) |
| `model_used` | VARCHAR(100) | NOT NULL | Name of LLM used for extraction |
| `created_at` | TIMESTAMP | Default: UTC Now | Generation timestamp |

### 5. `RiskAnalysis` (Table: `risk_analyses`)
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | Primary Key (UUID) | Unique warning ID |
| `document_id` | VARCHAR(36) | Foreign Key (`documents.id`) | Linked document |
| `clause_text` | TEXT | NOT NULL | Matching snippet of policy |
| `risk_type` | VARCHAR(100) | NOT NULL | copay / deductible / exclusion |
| `severity` | VARCHAR(50) | NOT NULL | low / medium / high |
| `explanation` | TEXT | Nullable | Details of risk |
| `recommendation` | TEXT | Nullable | Suggestion |

### 6. `PolicyReminder` (Table: `policy_reminders`)
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | Primary Key (UUID) | Unique alert ID |
| `user_id` | VARCHAR(36) | Foreign Key (`users.id`) | Target user |
| `document_id` | VARCHAR(36) | Foreign Key (`documents.id`) | Linked document |
| `title` | VARCHAR(255) | NOT NULL | Notification message header |
| `reminder_type` | VARCHAR(50) | NOT NULL | renewal / premium |
| `reminder_date` | TIMESTAMP | NOT NULL | Calculated trigger date |
| `premium_amount` | VARCHAR(100) | Nullable | Associated cost |
| `is_dismissed` | BOOLEAN | Default: False | Dismiss status |

### 7. `ChatSession` (Table: `chat_sessions`)
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | Primary Key (UUID) | Session ID |
| `user_id` | VARCHAR(36) | Foreign Key (`users.id`) | Creator |
| `document_id` | VARCHAR(36) | Foreign Key (`documents.id`), Nullable | Scoped document filter |
| `title` | VARCHAR(255) | Default: "New Chat" | Chat header title |
| `created_at` | TIMESTAMP | Default: UTC Now | Creation |
| `updated_at` | TIMESTAMP | Default: UTC Now | Expiry sorting timestamp |

### 8. `ChatMessage` (Table: `chat_messages`)
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | Primary Key (UUID) | Message ID |
| `session_id` | VARCHAR(36) | Foreign Key (`chat_sessions.id`) | Chat Thread |
| `role` | VARCHAR(50) | NOT NULL | user / assistant / system |
| `content` | TEXT | NOT NULL | Message text |
| `sources` | TEXT | Nullable (JSON String) | List of chunk file source labels |
| `created_at` | TIMESTAMP | Default: UTC Now | Sent time |

---

## ⚙️ Environment Configurations

### 1. FastAPI Backend Environment (`backend/.env`)

```env
DATABASE_URL=sqlite+aiosqlite:///./healthcare_ai.db
SECRET_KEY=dev-secret-key-change-in-production-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=hf.co/kkross/gemma-3-4b-cord19-finetuned-new:latest
OLLAMA_NUM_THREAD=8
OLLAMA_NUM_GPU=999
OLLAMA_KEEP_ALIVE=10m
UPLOAD_DIR=./uploads
MAX_FILE_SIZE_MB=50
ALLOWED_ORIGINS=["http://localhost:3000","http://localhost"]
DEBUG=True
```

### 2. Next.js Frontend Environment (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

---

## 🛠️ Key Scripts & Run Instructions

### 1. Data Science & Claim Modeling Execution
To re-run the claims database modeling, balance sample distributions, and re-generate ROC curves, SHAP plots, and LIME explainer images:
```powershell
cd data_science_analysis
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib xgboost shap lime
python train_and_analyze.py
```
This updates files at:
- `data_science_analysis/best_model.joblib`
- `data_science_analysis/scaler.joblib`
- `data_science_analysis/*.png` (Attribution charts served via Nginx `/data_science_analysis/*` mount)

### 2. Fine-Tuning Gemma 3 on Custom Literature
To fine-tune models using instruction-response mappings or preprocessing AllenAI's CORD-19 dataset:
- Follow instruction schemas inside:
  - [TRAINING_GUIDE.md](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/TRAINING_GUIDE.md)
  - [CORD19_TRAINING_GUIDE.md](file:///d:/Imp/VScode/Projects/HealthCare/AI-Healthcare-App/CORD19_TRAINING_GUIDE.md)
