# MediTrace

**MediTrace: A Multi-Agent AI System for Verified Clinical Information Management**

> A verification-first healthcare AI prototype for extracting, validating, and managing information from medical documents.

## Overview

MediTrace addresses a key problem in document-driven healthcare workflows: **extracted information should not be treated as verified information by default**.

The system processes prescriptions, scanned reports, discharge summaries, and other medical documents using OCR and NLP, converts extracted information into candidate facts, verifies those facts against source evidence, and builds a **verified patient profile** for downstream agents.

### Core Workflow

```text
Medical Documents
       ↓
Member 1: Document Agent
(OCR + Medical NER)
       ↓
Candidate Facts (Unverified)
       ↓
Member 2: Verification Agent
(Evidence Grounding & Conflict Resolution)
       ↓
Verified Patient Profile (Trusted Layer)
       ↓
┌───────────────────────┬────────────────────────────┐
│ Member 3: Safety &    │ Member 4: Emergency        │
│ Trend Agents          │ Summary Agent              │
│ (Interactions/Trends) │ (ER Triage & Handover)     │
└───────────────────────┴────────────────────────────┘
       ↓                                    ↓
┌────────────────────────────────────────────────────┐
│ Member 4: Unified Clinical Dashboard (/dashboard)  │
└────────────────────────────────────────────────────┘
```

## System Architecture & Multi-Agent Modules

| Agent | Responsibility | Implementation Details |
|---|---|---|
| **Member 1: Document Agent** | Ingestion, OCR & Extraction | Preprocesses document images with OpenCV, executes Tesseract OCR, detects text vs scanned PDFs via PyMuPDF, and extracts medical entities with spaCy and clinical regex patterns. Every entity is tagged `status="candidate"`. |
| **Member 2: Verification Agent** | Fact Verification & Profile | Evaluates candidate facts using deterministic evidence grounding. Merges duplicate records with source references and flags contradictions (e.g. differing dosages, conflicting allergy statements) without overwriting historical verified records. |
| **Member 3: Safety & Trend Agent** | Clinical Safety & Analytics | Consumes **only** verified patient profile data. Identifies drug-drug interactions, allergy contraindications, duplicate drug classes, maximum daily dosage limits, and longitudinal laboratory deterioration trends. |
| **Member 4: Emergency Agent & Dashboard** | Triage, Handover & Orchestrator | Synthesizes verified clinical records and safety alerts into an emergency triage categorization (RED/YELLOW/GREEN), generates an automated 15-second clinical handover briefing, and powers the unified dashboard (`/dashboard`) connecting all 4 modules end-to-end. |

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Backend | FastAPI, Uvicorn |
| OCR & Document Ingestion | Tesseract OCR, OpenCV, PyMuPDF (fitz) |
| Clinical NLP & NER | spaCy (`en_core_web_sm`), Regex Medical Rule Extractors |
| Database & Storage | MongoDB / SQLite fallback (`meditrace.db`) |
| Frontend | Jinja2 Templates, HTML5/CSS3 Dashboard, Responsive ER Cards |

## Project Structure

```text
MediTrace/
├── main.py                       # Core FastAPI application & pipeline orchestrator
├── requirements.txt              # Project dependencies
├── agents/
│   ├── __init__.py
│   ├── safety_engine.py          # Member 3: Clinical safety rules & interaction matrix
│   ├── safety_trend_agent.py     # Member 3: Safety & trend analysis agent
│   └── emergency_agent.py        # Member 4: Emergency summary, triage & handover agent
├── templates/
│   ├── dashboard.html            # Member 4: Unified 4-agent interactive dashboard
│   ├── index.html                # Member 1: Document processing upload test page
│   └── safety.html               # Member 3: Safety and trend report view
├── data/
│   ├── meditrace.db              # SQLite storage database
│   ├── sample_docs/              # Clinical test PDFs and sample reports
│   └── uploads/                  # Temporary document processing storage
├── migrate_sqlite_to_mongo.py    # Optional SQLite to MongoDB migration utility
└── seed_test_data.py             # Test data generation script
```

## Setup & Running

### 1. Prerequisites

- Python 3.10+
- Git
- Tesseract OCR (Windows: install 64-bit installer and ensure path is set)

### 2. Virtual Environment & Dependencies

```bash
# Clone the repository
git clone https://github.com/shiladeori0305/mediTrace.git
cd mediTrace

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies and download NLP model
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run Application

```bash
python main.py
```

Then open your browser:
- **Unified Multi-Agent Dashboard:** [http://localhost:8000](http://localhost:8000) or [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Safety & Trends Report:** [http://localhost:8000/safety](http://localhost:8000/safety)

## Configuration

Settings can be stored in a `.env` file in the root directory:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=meditrace
```

*Note: If MongoDB is not active locally, the system automatically runs with the embedded SQLite engine (`data/meditrace.db`) without requiring extra configuration.*

## Evaluation & Testing

The system supports automated and interactive evaluation:
- Run pre-loaded clinical cases directly from the dashboard:
  - `report_2024_01.pdf` (Baseline Diabetes & Metformin)
  - `report_2024_07.pdf` (Worsening HbA1c & Blood Glucose)
  - `prescription_2025_01.pdf` (Warfarin + Aspirin interaction, Penicillin allergy conflict)
  - `comprehensive_test_prescription.pdf` (Full 4-agent multi-condition validation)
- Compare verification effectiveness:
  ```text
  Pipeline without verification vs. Pipeline with verification
  ```

## Project Status

**Status: Multi-Agent Prototype Implemented**

All 4 planned agents (Document Processing, Verification, Safety & Trend, and Emergency Summary) are implemented and connected through the unified dashboard. Future scope includes fine-tuned clinical LLM integration for complex medical reasoning, EHR/FHIR interoperability, and large-scale hospital dataset benchmarking.

## Important Note

MediTrace is intended for **academic and controlled research use**. It is not a substitute for professional medical judgment or a certified clinical decision-support system. Use synthetic or appropriately de-identified data for development and testing.

## Team

- Saanvi Gupta
- Shiksha
- Shila Deori
- Shraddha Chaurasia
- Palak Mahajan
