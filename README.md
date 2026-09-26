MediTrace

MediTrace: A Multi-Agent AI System for Verified Clinical Information Management

A verification-first healthcare AI prototype for extracting, validating, and managing information from medical documents.

Overview

MediTrace addresses a key problem in document-driven healthcare workflows: extracted information should not be treated as verified information by default.

The system processes prescriptions, scanned reports, discharge summaries, and other medical documents using OCR and NLP, converts extracted information into candidate facts, verifies those facts against source evidence, and builds a verified patient profile for downstream agents.

Core Workflow

Medical Documents
       ↓
Document Agent
(OCR + Medical NER)
       ↓
Candidate Facts
       ↓
Verification Agent
       ↓
Verified Patient Profile
       ↓
┌──────────┬──────────┬────────────┐
│  Safety  │  Trend   │  Emergency │
│  Agent   │  Agent   │   Agent    │
└──────────┴──────────┴────────────┘
       ↓
Dashboard

Key Features

Document Processing: Extracts medicines, dosages, dates, conditions, and allergies from medical documents.

Fact Verification: Validates extracted facts against source documents and existing patient information.

Verified Patient Profile: Acts as the trusted information layer for downstream agents.

Medication Safety: Identifies potential interactions, duplicate medications, dosage issues, and allergy conflicts.

Trend Detection: Analyzes verified medical history for relevant changes over time.

Emergency Summary: Generates a concise summary from the latest verified patient information.

Traceability: Maintains source evidence and verification status for extracted facts.

Technology Stack

Component

Technology

Language

Python 3

OCR

Tesseract OCR

NLP

spaCy / Medical NER

Agent Framework

LangGraph / CrewAI

LLM

LLM APIs

Backend

FastAPI

Database

MongoDB / PostgreSQL

Frontend

React.js / Streamlit

Reference Data

DrugBank API / OpenFDA

Project Structure

MediTrace/
├── main.py
├── requirements.txt
├── agents/
│   ├── document_agent/
│   ├── verification_agent/
│   ├── safety_agent/
│   ├── trend_agent/
│   └── emergency_agent/
├── data/
├── models/
├── services/
└── utils/

The final folder structure may evolve during development.

Setup

Prerequisites

Python 3

Git

Tesseract OCR

pip

1. Install Tesseract

Windows: Download the 64-bit installer from the Tesseract project wiki and add Tesseract to the system PATH.

macOS:

brew install tesseract

2. Clone Repository

git clone <YOUR_REPOSITORY_URL>
cd MediTrace

3. Create Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

macOS

python3 -m venv venv
source venv/bin/activate

4. Install Dependencies

Windows

pip install -r requirements.txt
python -m spacy download en_core_web_sm

macOS

pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

5. Run

Windows

python main.py

macOS

python3 main.py

Windows Tesseract PATH Fix

If Tesseract is not detected:

import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

Configuration

If external APIs or databases are required, store credentials in environment variables or a .env file.

LLM_API_KEY=your_api_key
DATABASE_URL=your_database_url
OPENFDA_API_KEY=your_api_key

Do not commit API keys, credentials, or private patient information to GitHub.

Evaluation

The project is designed to compare:

Pipeline without verification
              vs.
Pipeline with verification

Evaluation uses sample clinical documents, ground-truth information, and deliberately introduced error cases to study whether the verification stage reduces downstream errors.

Project Status

Status: In Development

MediTrace is an academic/research prototype. The planned implementation progresses through document ingestion, verification, Safety and Trend agents, Emergency Agent integration, and evaluation/refinement.

Important Note

MediTrace is intended for academic and controlled research use. It is not a substitute for professional medical judgment or a production clinical decision-support system. Use synthetic or appropriately de-identified data for development and testing.

Team

Saanvi Gupta

Shiksha

Shila Deori

Shraddha Chaurasia

Palak Mahajan

Project Focus

The central idea of MediTrace is:

Extract → Verify → Build a trusted profile → Analyze using verified information

This verification-first architecture is the primary focus of the project and its planned evaluation.
