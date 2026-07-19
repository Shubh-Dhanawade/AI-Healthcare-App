# d:\Imp\VScode\Projects\HealthCare\AI-Healthcare-App\generate_doc_pdf.py
import fitz
import datetime
import os

def generate_pdf():
    pdf_path = r"d:\Imp\VScode\Projects\HealthCare\AI-Healthcare-App\Project_Documentation.pdf"
    doc = fitz.open()
    
    # Color Palette
    PRIMARY = (0.07, 0.15, 0.35)    # Navy Blue
    SECONDARY = (0.23, 0.51, 0.96)  # Accent Blue
    TEXT_DARK = (0.07, 0.07, 0.07)  # Dark Gray/Black
    TEXT_LIGHT = (0.45, 0.45, 0.45) # Muted Gray
    BG_LIGHT = (0.95, 0.96, 0.98)   # Light Gray
    WHITE = (1.0, 1.0, 1.0)
    BORDER_COLOR = (0.85, 0.85, 0.85)

    def draw_header_footer(page, title, page_num):
        # Header line
        page.draw_rect(fitz.Rect(50, 30, 545, 32), color=PRIMARY, fill=PRIMARY)
        page.insert_text(fitz.Point(50, 24), "AI HEALTHCARE ASSISTANT & LOCAL RAG", fontsize=8, fontname="hebo", color=PRIMARY)
        
        # Footer line
        page.draw_line(fitz.Point(50, 805), fitz.Point(545, 805), color=BORDER_COLOR, width=0.5)
        page.insert_text(fitz.Point(50, 818), "Confidential - Medical AI System Documentation", fontsize=8, fontname="helv", color=TEXT_LIGHT)
        page.insert_text(fitz.Point(510, 818), f"Page {page_num}", fontsize=8, fontname="helv", color=TEXT_LIGHT)

    # ================= PAGE 1: COVER PAGE =================
    page1 = doc.new_page()
    # Deep Navy Blue top banner
    page1.draw_rect(fitz.Rect(0, 0, 595, 450), color=PRIMARY, fill=PRIMARY)
    
    # Accent Blue line
    page1.draw_rect(fitz.Rect(0, 450, 595, 458), color=SECONDARY, fill=SECONDARY)
    
    # White Title text
    page1.insert_text(fitz.Point(50, 150), "HEALTHCARE AI", fontsize=40, fontname="hebo", color=WHITE)
    page1.insert_text(fitz.Point(50, 200), "COGNITIVE CORE", fontsize=40, fontname="hebo", color=WHITE)
    
    # Subtitle
    page1.insert_textbox(fitz.Rect(50, 240, 500, 300), 
                         "Comprehensive System Architecture, Custom Vector RAG Pipeline,\nand Gemma 3 (4B) Local LLM Integration Documentation", 
                         fontsize=14, fontname="helv", color=WHITE)

    # Info box at bottom
    page1.draw_rect(fitz.Rect(50, 550, 545, 700), color=BORDER_COLOR, fill=BG_LIGHT)
    page1.insert_text(fitz.Point(70, 580), "Project Info & Metadata", fontsize=12, fontname="hebo", color=PRIMARY)
    
    info_text = (
        f"Generated Date: {datetime.date.today().strftime('%B %d, %Y')}\n"
        "Engine Version: 1.0.0 (Custom SQLite Vector Store)\n"
        "AI Models: Gemma 3 (4B) Instruction, nomic-embed-text (137M)\n"
        "Stack: Next.js (Frontend) & FastAPI (Backend) & SQLite (Database)"
    )
    page1.insert_textbox(fitz.Rect(70, 600, 520, 680), info_text, fontsize=10, fontname="helv", color=TEXT_DARK)

    # ================= PAGE 2: EXEC SUMMARY & SYSTEM FLOW =================
    page2 = doc.new_page()
    draw_header_footer(page2, "Executive Summary & System Flow", 2)
    
    page2.insert_text(fitz.Point(50, 70), "1. Executive Summary", fontsize=18, fontname="hebo", color=PRIMARY)
    
    summary_p = (
        "The AI Healthcare Application is a secure, privacy-preserving client-server platform "
        "designed for medical professionals and patients to analyze healthcare insurance policies "
        "and medical documentation. At its core, the system runs a local RAG (Retrieval-Augmented "
        "Generation) engine that combines document OCR/extraction, chunking, high-dimensional "
        "vector embeddings, and a local Gemma 3 (4B) LLM to answer complex user queries, "
        "extract hidden risks (co-pays, waiting periods, room-rent limits), and set proactive policy reminders. "
        "By hosting everything locally via Ollama and SQLite, patient data privacy is strictly enforced "
        "without leakage to external cloud services."
    )
    page2.insert_textbox(fitz.Rect(50, 85, 545, 190), summary_p, fontsize=10.5, fontname="helv", color=TEXT_DARK, align=3)
    
    page2.insert_text(fitz.Point(50, 220), "2. Core System Architecture & Data Flow", fontsize=18, fontname="hebo", color=PRIMARY)
    
    # Draw Flowchart box
    # Upload -> Parse -> Chunk -> Embed -> DB
    flow_steps = [
        ("1. INGEST & EXTRACT", "User uploads policy PDF; backend extracts raw text using PyMuPDF/OCR."),
        ("2. CHUNK & CLEAN", "Text is cleaned and split into semantic chunks (500 tokens with 10% overlap)."),
        ("3. GENERATE VECTORS", "Each chunk is embedded into a 768-dimensional vector via nomic-embed-text."),
        ("4. HYBRID STORAGE", "Text and vector metadata are stored in SQLite, and a FAISS index is built on disk."),
        ("5. SEMANTIC QUERY", "User query is embedded; FAISS index matches query with document chunks in microseconds."),
        ("6. LOCAL LLM INFERENCE", "Retrieved chunks + query + system prompt are fed into Gemma 3 (4B) via Ollama."),
        ("7. UI RENDERING", "Gemma 3's response streams back to the Next.js React frontend via SSE/FastAPI.")
    ]
    
    y = 245
    for title, desc in flow_steps:
        # draw a little accent bullet
        page2.draw_rect(fitz.Rect(50, y, 160, y+20), color=PRIMARY, fill=PRIMARY)
        page2.insert_text(fitz.Point(54, y+14), title, fontsize=8, fontname="hebo", color=WHITE)
        page2.insert_textbox(fitz.Rect(170, y-2, 545, y+22), desc, fontsize=9.5, fontname="helv", color=TEXT_DARK)
        y += 38

    # Diagram box at the bottom
    page2.draw_rect(fitz.Rect(50, 530, 545, 780), color=SECONDARY, fill=BG_LIGHT, width=1)
    page2.insert_text(fitz.Point(65, 550), "Architecture Component Mapping", fontsize=12, fontname="hebo", color=SECONDARY)
    
    mapping_text = (
        "┌───────────────────────────────────────────────────────────────────────────┐\n"
        "│              Next.js 15 App Router (React, TailwindCSS)                     │\n"
        "└─────────────────────────────────────┬─────────────────────────────────────┘\n"
        "                                      │ HTTP APIs (JWT / Streaming)\n"
        "┌─────────────────────────────────────▼─────────────────────────────────────┐\n"
        "│                 FastAPI Python Server (Uvicorn, SQLAlchemy)               │\n"
        "│  ┌─────────────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │\n"
        "│  │ OCR / Ingestion Service │  │ RAG & Embeddings │  │ Ollama Local API │  │\n"
        "│  └─────────────────────────┘  └────────┬─────────┘  └────────┬─────────┘  │\n"
        "└────────────────────────────────────────┼─────────────────────┼────────────┘\n"
        "                                         │ SQL queries         │ Ollama port 11434\n"
        "┌────────────────────────────────────────▼─────────┐  ┌────────▼─────────┐\n"
        "│     SQLite (Metadata) & FAISS Index (Vectors)    │  │ Gemma 3 (4B)     │\n"
        "│  [users]  [documents]  [document_chunks (TEXT)]  │  │ nomic-embed-text │\n"
        "└──────────────────────────────────────────────────┘  └──────────────────┘"
    )
    page2.insert_textbox(fitz.Rect(65, 565, 530, 770), mapping_text, fontsize=9.5, fontname="cour", color=PRIMARY)

    # ================= PAGE 3: TECHNOLOGY STACK =================
    page3 = doc.new_page()
    draw_header_footer(page3, "Technology Stack Details", 3)
    
    page3.insert_text(fitz.Point(50, 70), "3. Complete Technology Stack", fontsize=18, fontname="hebo", color=PRIMARY)
    
    # Table header
    page3.draw_rect(fitz.Rect(50, 95, 545, 120), color=PRIMARY, fill=PRIMARY)
    page3.insert_text(fitz.Point(60, 112), "Layer", fontsize=10, fontname="hebo", color=WHITE)
    page3.insert_text(fitz.Point(150, 112), "Technology", fontsize=10, fontname="hebo", color=WHITE)
    page3.insert_text(fitz.Point(270, 112), "Purpose & Description", fontsize=10, fontname="hebo", color=WHITE)
    
    techs = [
        ("Frontend Core", "Next.js 15 (React)", "Standard application layout, routing, dynamic views, components."),
        ("Frontend Styles", "Vanilla CSS", "Theme configuration, custom glassmorphism components, layouts."),
        ("State / Context", "React Context API", "User authentication session management and theme toggles."),
        ("Backend Server", "FastAPI (Python)", "High-performance async API server with automatic OpenAPI docs."),
        ("WebServer", "Uvicorn", "ASGI web server running the FastAPI python service on port 8000."),
        ("Database ORM", "SQLAlchemy + aiosqlite", "Asynchronous SQLite interaction and Object-Relational Mapping."),
        ("Database Engine", "SQLite (Local File)", "Zero-configuration persistent file storing relational metadata and text chunks."),
        ("Vector Search", "FAISS-CPU (Meta)", "Executes ultra-fast vector indexing and cosine-similarity searches on local vectors."),
        ("Local LLM Core", "Ollama (Gemma 3 4B)", "Ultra-fast response generation using 4.7B parameter Google Gemma 3."),
        ("Embedding Core", "Ollama (nomic-embed-text)", "Transforms text chunks into dense 768-dimension vector spaces."),
        ("Document Extract", "PyMuPDF (fitz)", "Parses, cleans, and structures text from uploaded PDF policy files."),
        ("Authentication", "JWT (JSON Web Tokens)", "Secure, stateless user sessions with password hashing via bcrypt.")
    ]
    
    y = 120
    for layer, tech, desc in techs:
        # alternate row backgrounds
        bg = BG_LIGHT if (y // 25) % 2 == 0 else WHITE
        page3.draw_rect(fitz.Rect(50, y, 545, y+26), color=BORDER_COLOR, fill=bg)
        
        page3.insert_text(fitz.Point(60, y+17), layer, fontsize=8.5, fontname="hebo", color=PRIMARY)
        page3.insert_text(fitz.Point(150, y+17), tech, fontsize=8.5, fontname="helv", color=TEXT_DARK)
        page3.insert_textbox(fitz.Rect(270, y+1, 535, y+25), desc, fontsize=8, fontname="helv", color=TEXT_DARK)
        y += 26
        
    page3.insert_text(fitz.Point(50, 465), "4. Component Features", fontsize=18, fontname="hebo", color=PRIMARY)
    
    features = [
        ("Interactive Chat Interface", "Provides a conversational thread where the user can query their insurance documents. Supports streaming tokens directly for a highly responsive, zero-delay UI."),
        ("Automated Compliance & Risk Analysis", "Extracts critical elements from medical insurance paperwork (like co-payments, waiting periods, room-rent limits) and flags high-severity exceptions automatically."),
        ("Policy Reminders & Due Dates", "Analyzes documents for renewal deadlines and premium payment schedules, storing them in the SQLite DB to notify users in the dashboard."),
        ("Custom Local RAG Pipeline", "Bypasses slow cloud databases. Uses a custom FAISS integration to store vectors and calculate similarity locally, achieving context fetch times under 10ms.")
    ]
    
    y = 485
    for title, desc in features:
        page3.draw_rect(fitz.Rect(50, y, 545, y+50), color=BORDER_COLOR, fill=WHITE)
        page3.draw_rect(fitz.Rect(50, y, 55, y+50), color=SECONDARY, fill=SECONDARY)
        # Draw a little star or bullet in the colored part
        page3.insert_text(fitz.Point(68, y+28), "★", fontsize=14, fontname="hebo", color=WHITE)
        
        page3.insert_text(fitz.Point(115, y+16), title, fontsize=9.5, fontname="hebo", color=PRIMARY)
        page3.insert_textbox(fitz.Rect(115, y+20, 535, y+48), desc, fontsize=8.5, fontname="helv", color=TEXT_LIGHT)
        y += 60

    # ================= PAGE 4: DETAILED VECTOR DB & RAG =================
    page4 = doc.new_page()
    draw_header_footer(page4, "Vector DB & RAG Details", 4)
    
    page4.insert_text(fitz.Point(50, 70), "5. Custom SQLite Vector DB & RAG Pipeline", fontsize=18, fontname="hebo", color=PRIMARY)
    
    rag_p = (
        "Unlike enterprise architectures that rely on expensive external Vector Databases (like Pinecone or "
        "Milvus), this project employs a custom hybrid FAISS and SQLite storage system built for speed, simplicity, "
        "and zero external dependencies. Let's explore how it's implemented."
    )
    page4.insert_textbox(fitz.Rect(50, 85, 545, 140), rag_p, fontsize=10.5, fontname="helv", color=TEXT_DARK, align=3)
    
    page4.insert_text(fitz.Point(50, 155), "5.1 Database Schema (document_chunks)", fontsize=12, fontname="hebo", color=SECONDARY)
    
    schema_code = (
        "CREATE TABLE document_chunks (\n"
        "    id VARCHAR(36) PRIMARY KEY,\n"
        "    document_id VARCHAR(36) NOT NULL,\n"
        "    chunk_index INTEGER NOT NULL,\n"
        "    text_content TEXT NOT NULL,\n"
        "    embedding TEXT NOT NULL,          -- 768 float32 values stored as JSON text\n"
        "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
        "    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE\n"
        ");"
    )
    page4.draw_rect(fitz.Rect(50, 170, 545, 275), color=BORDER_COLOR, fill=BG_LIGHT)
    page4.insert_textbox(fitz.Rect(60, 180, 535, 265), schema_code, fontsize=9, fontname="cour", color=PRIMARY)
    
    page4.insert_text(fitz.Point(50, 295), "5.2 How the Vector Search Operations Work", fontsize=12, fontname="hebo", color=SECONDARY)
    
    ops = [
        ("Chunking", "Text is extracted from the PDF and broken into sections of 500 characters, with 10% overlap (50 characters) to ensure context is not severed at the boundaries. This handles long insurance policies effectively."),
        ("Embedding", "Each text chunk is sent to the local Ollama embedding endpoint (/api/embeddings) using the 'nomic-embed-text' model. This returns a vector of 768 floating-point values reflecting semantic meaning."),
        ("JSON Vector Storage", "The 768 float list is serialized into a standard JSON string and saved directly into the SQLite 'embedding' TEXT column. This makes DB schema management and migrations straightforward and highly portable."),
        ("FAISS Index Search", "When a document is indexed, its vectors are compiled into a FAISS index file on disk. When a user queries, the search is executed against this local FAISS index, performing ultra-fast similarity calculations in microseconds to retrieve top context chunks."),
        ("Prompt Synthesis", "The matching context chunks are combined and injected into the LLM system prompt as context. Gemma 3 uses this context to synthesize a fact-grounded response, eliminating hallucinations.")
    ]
    
    y = 310
    for step_title, step_desc in ops:
        page4.insert_text(fitz.Point(50, y+12), f"• {step_title}:", fontsize=9.5, fontname="hebo", color=PRIMARY)
        page4.insert_textbox(fitz.Rect(150, y, 545, y+35), step_desc, fontsize=9, fontname="helv", color=TEXT_DARK)
        y += 42
        
    page4.insert_text(fitz.Point(50, 530), "5.3 Gemma 3 (4B) Local LLM Advantage", fontsize=12, fontname="hebo", color=SECONDARY)
    
    gemma_p = (
        "The project integrates Gemma 3 (4B) Instruction model. Gemma 3 represents a significant upgrade "
        "over previous generation models like Llama 3.2 (3B). It offers:\n"
        "1. Faster Inference Speed: Highly optimized attention mechanisms make it generate responses at "
        "higher tokens per second on consumer hardware.\n"
        "2. Improved Context Utilization: Better multi-turn comprehension and strict adherence to structured "
        "JSON schemas, which is crucial for parsing policy details.\n"
        "3. Privacy: Completely offline execution ensures full compliance with medical privacy laws (HIPAA/GDPR)."
    )
    page4.insert_textbox(fitz.Rect(50, 545, 545, 680), gemma_p, fontsize=9.5, fontname="helv", color=TEXT_DARK, align=3)
    
    # Simple diagram at the bottom
    page4.draw_rect(fitz.Rect(50, 690, 545, 785), color=PRIMARY, fill=PRIMARY)
    page4.insert_text(fitz.Point(70, 715), "RAG CONTEXT EXTRACTION & QUERY LIFECYCLE", fontsize=11, fontname="hebo", color=WHITE)
    
    lifecycle = (
        "User Query ──► [Embed Query] ──► [High-Speed FAISS Index Search] ──┐\n"
        "                                                                    ▼\n"
        "User Screen ◄── [Stream Tokens] ◄── [Ollama: Gemma 3] ◄── [Synthesized Context + Prompt]"
    )
    page4.insert_textbox(fitz.Rect(70, 735, 530, 780), lifecycle, fontsize=8.5, fontname="cour", color=WHITE)

    # Save to file
    doc.save(pdf_path)
    doc.close()
    print(f"Project documentation PDF generated successfully at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
