MediTrace --- Multi-Agent AI System for Verified Clinical Information Management

A healthcare information-management system that converts fragmented
clinical documents into structured candidate facts, verifies those
facts against source evidence, and makes only verified information
available to downstream AI agents.

Domain: AI, NLP, Multi-Agent Systems, Healthcare Informatics
Core stack: Python, FastAPI, Tesseract OCR, spaCy,
MongoDB/PostgreSQL, LangGraph/CrewAI or equivalent, LLM APIs,
DrugBank/OpenFDA
Architecture: single FastAPI application, single main.py, single
application port.

1. Problem Statement

Clinical information is distributed across prescriptions, scanned
reports, discharge summaries, laboratory reports, and other documents.
OCR and NLP extraction can introduce errors in medicine names, dosages,
dates, laboratory values, and clinical entities.

MediTrace therefore separates extraction from verification:

Clinical Documents
       ↓
Document Processing Agent
       ↓
Candidate Facts
       ↓
Verification Agent
       ↓
Verified Patient Profile
       ↓
Safety / Trend / Emergency Agents

The key design principle is:

Extract → Verify → Trace → Analyze

2. Objectives

Ingest PDF, JPG, JPEG and PNG clinical documents.

Extract selectable PDF text or use OCR for scanned documents.

Preprocess images when required.

Clean extracted text.

Extract medicines, dosage, frequency, dates, conditions, allergies
and laboratory values.

Store every extracted item as a structured candidate fact.

Preserve source document and source text for traceability.

Verify candidate facts before they enter the trusted profile.

Detect duplicates and conflicting records.

Maintain a verified patient profile.

Make only verified information available to downstream agents.

Perform medication safety and longitudinal trend analysis.

Generate an emergency-oriented summary.

Compare verified and unverified information during evaluation.

3. Core Innovation

Evidence-linked facts

Every fact retains its evidence:

{
  "fact_type": "medication",
  "value": "Amoxicillin",
  "dosage": "500 mg",
  "frequency": "twice daily",
  "source_document": "prescriptions_1.jpg",
  "source_text": "Amoxicillin 500 mg twice daily.",
  "extraction_confidence": 0.92,
  "status": "candidate"
}

Two confidence layers

extraction_confidence describes extraction quality.
verification_confidence describes the verification result.
Neither is medical certainty.

Verification boundary

Member 1 → Candidate Facts
Member 2 → Verified Profile
Member 3/4 → Verified Profile

Raw OCR should not bypass verification.

4. System Architecture

                 ┌───────────────────────┐
                 │ Clinical Documents    │
                 │ PDF / JPG / PNG       │
                 └──────────┬────────────┘
                            ↓
              ┌──────────────────────────┐
              │ Member 1                 │
              │ Document Processing      │
              │ OCR + NLP + Extraction   │
              └────────────┬─────────────┘
                           ↓
                    Candidate Facts
                    status=candidate
                           ↓
              ┌──────────────────────────┐
              │ Member 2                 │
              │ Verification Agent       │
              │ Evidence + Conflicts     │
              └────────────┬─────────────┘
                           ↓
                Verified Patient Profile
                           ↓
              ┌────────────┴─────────────┐
              ↓                          ↓
       Member 3                    Member 4
     Safety + Trend            Emergency + Dashboard

All components can remain in one FastAPI application and one main.py,
as required by the project's simplified architecture.

5. Multi-Agent Workflow

Member 1 --- Document Processing Agent

Accept PDF/image upload.

Save the original document.

Extract selectable PDF text when available.

Apply OCR to scanned PDFs/images.

Optionally preprocess images using Pillow/OpenCV.

Keep raw text.

Create cleaned text.

Extract medical entities with spaCy plus regex/controlled
vocabularies where necessary.

Build candidate facts.

Attach source evidence.

Keep every fact as status: "candidate".

Does not: verify facts, diagnose, recommend treatment, perform
safety analysis, or create emergency summaries.

Member 2 --- Verification Agent + Patient Profile

Receive candidate facts.

Check whether source evidence supports each fact.

Compare with existing verified information.

Detect duplicates.

Detect conflicts.

Produce verified, rejected, or conflict decisions.

Preserve source evidence.

Store verification confidence and verification logs.

Build the verified patient profile.

Does not: perform medication safety, diagnosis, treatment
recommendation, trend analysis, or emergency summarization.

Member 3 --- Safety + Trend Agents

Consumes only the verified profile.

Planned functions include medication safety checks, duplicate medication
detection, allergy-related checks, dosage-related checks, and
longitudinal trend detection.

Member 4 --- Emergency + Dashboard + Integration

Consumes verified information to create an emergency-oriented summary,
dashboard, and end-to-end integrated workflow.

6. End-to-End Data Flow

Upload
  ↓
Document storage
  ↓
PDF extraction / OCR
  ↓
Raw text
  ↓
Cleaned text
  ↓
Medical entity extraction
  ↓
Candidate facts
  ↓
Evidence verification
  ↓
Verified / Rejected / Conflict
  ↓
Verified Patient Profile
  ↓
Safety + Trend + Emergency

7. Candidate vs Verified Facts

A candidate means the system extracted the information but has not
yet verified it.

Example:

Allergy: Penicillin
status = candidate

After verification:

Allergy: Penicillin
status = verified

A conflict should not silently overwrite an existing fact.

Example:

Document A: Allergy = Penicillin
Document B: No known allergies
                         ↓
                    CONFLICT

Similarly, different dosages for the same medicine should be retained as
a conflict until resolved by the verification process.

8. Evidence Traceability

Every important fact should preserve:

fact_type
value
source_document
source_text
extraction_confidence
verification_confidence
status
timestamp

This lets an interviewer or evaluator ask:

Where did this patient-profile entry come from?

and trace it back to the document and extracted evidence.

9. Database Design

The submitted synopsis lists MongoDB/PostgreSQL as the database
technology.

The final project should use the selected MongoDB or PostgreSQL
implementation rather than leaving SQLite as the final production
database.

For MongoDB, a simple conceptual design is:

meditrace
├── candidate_facts
├── verified_facts
└── verification_logs

candidate_facts

Typical fields:

fact_type
value
dosage
frequency
date
lab_value
source_document
source_text
extraction_confidence
status
created_at

verified_facts

Typical fields:

fact_type
value
dosage
frequency
date
lab_value
source_document
source_text
extraction_confidence
verification_confidence
status
reason
created_at
updated_at

verification_logs

Typical fields:

candidate reference
decision
reason
compared_with
verification confidence
timestamp

Exact names should match the final main.py.

10. Technology Stack

Layer                               Technology

Language                            Python

API                                 FastAPI

Server                              Uvicorn

OCR                                 Tesseract + pytesseract

Image processing                    Pillow / optional OpenCV

PDF                                 PyMuPDF or equivalent

NLP                                 spaCy

Medical extraction                  spaCy NER + regex/controlled
vocabulary

Agent orchestration                 LangGraph / CrewAI or equivalent

LLM                                 LLM APIs where required

Medication data                     DrugBank / OpenFDA

Database                            MongoDB or PostgreSQL

11. Project Structure

MediTrace/
├── main.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
├── data/
│   └── uploads/
└── templates/
    └── index.html

The one-file backend is intentional for this academic version: it keeps
the project easy to run and demonstrate while still separating
functionality into clear sections/functions.

12. API Flow

The intended endpoints are:

POST /upload
POST /process
POST /verify
GET  /patient/profile

FastAPI documentation:

http://localhost:8000/docs

/upload accepts the document.

/process performs extraction and returns raw text, cleaned text,
candidate facts and warnings.

/verify receives candidate facts and returns verified facts, rejected
facts, conflicts and the verified profile.

/patient/profile returns the current verified profile.

The exact route names must match the final implementation.

13. Installation & Setup Guide

Prerequisites

Install:

Python 3

Git

Tesseract OCR

MongoDB locally or MongoDB Atlas, or PostgreSQL if selected

required API credentials for any external LLM/medical-data services
used by the implementation

Windows --- Tesseract

Download the 64-bit installer:

https://github.com/UB-Mannheim/tesseract/wiki

During installation, add Tesseract to PATH if the installer provides
that option.

If Python reports TesseractNotFoundError:

import pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR    esseract.exe"
)

macOS --- Tesseract

brew install tesseract

Clone repository

Windows

git clone <YOUR_REPOSITORY_URL>
cd MediTrace

macOS

git clone <YOUR_REPOSITORY_URL>
cd MediTrace

Create virtual environment

Windows

python -m venv venv
venv\Scriptsctivate

macOS

python3 -m venv venv
source venv/bin/activate

Install dependencies

Windows

pip install -r requirements.txt
python -m spacy download en_core_web_sm

macOS

pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

Configure environment variables

Do not hard-code credentials.

Example .env:

MONGODB_URI=<your_mongodb_connection_string>
DATABASE_NAME=meditrace
LLM_API_KEY=<your_key_if_required>

Use the exact variable names required by the final implementation.

Run

Windows

python main.py

macOS

python3 main.py

Open:

http://localhost:8000

API documentation:

http://localhost:8000/docs

14. Testing

Test document

Use a sample prescription containing:

Amoxicillin 500 mg twice daily.
Allergy: Penicillin.
Diagnosis: Bronchitis.
Date: 12/09/2026.
Hemoglobin: 13.5 g/dL

Expected candidate facts:

Medication → Amoxicillin
Dosage → 500 mg
Frequency → twice daily
Allergy → Penicillin
Condition → Bronchitis
Date → 12/09/2026
Lab → Hemoglobin 13.5 g/dL

All should initially be:

status = candidate

Then /verify should classify facts as appropriate and
/patient/profile should expose only the verified profile.

Robustness tests

Test:

selectable-text PDF

scanned PDF

clear image

poor-quality image

multi-page PDF

empty document

corrupted file

unsupported extension

document with no recognizable entities

duplicate facts

conflicting records

15. Expected Example

Input:

Amoxicillin 500 mg twice daily.
Allergy: Penicillin.
Diagnosis: Bronchitis.
Date: 12/09/2026.
Hemoglobin: 13.5 g/dL

Member 1:

Amoxicillin | 500 mg | twice daily | candidate
Penicillin | allergy | candidate
Bronchitis | condition | candidate
Hemoglobin | 13.5 g/dL | candidate
12/09/2026 | date | candidate

Member 2:

Candidate
   ↓
Evidence check
   ↓
Verified / Rejected / Conflict

Verified profile:

Medications:
  Amoxicillin 500 mg twice daily

Allergies:
  Penicillin

Conditions:
  Bronchitis

Lab values:
  Hemoglobin 13.5 g/dL

Dates:
  12/09/2026

Member 2 must not decide whether Amoxicillin is medically safe. That
belongs to the downstream safety component.

16. Error Handling

The application should handle:

unsupported file type

empty uploads

corrupted PDF/image

OCR failure

no extracted text

no recognized clinical entities

invalid candidate facts

database connection failure

duplicate records

conflicting records

Errors should be reported clearly rather than silently ignored.

17. Safety and Scope

MediTrace is a clinical information-management and
research/decision-support project, not an autonomous medical
decision-maker.

It should not claim to:

diagnose a patient

prescribe treatment

replace clinicians

independently determine treatment

treat model confidence as medical certainty

The verification layer exists to improve information reliability and
traceability before downstream analysis.

18. Evaluation Plan

Extraction

Measure:

precision

recall

OCR/extraction errors

performance on clean vs poor-quality documents

Verification

Measure:

verification accuracy

conflict detection

duplicate handling

evidence preservation

false acceptance/rejection

Downstream impact

Compare analysis based on:

Unverified extracted information
vs.
Verified patient profile

The purpose is to measure whether verification reduces propagation of
extraction errors.

19. Development Timeline

Weeks   Work

1--2    Document ingestion and OCR
3--4    Candidate extraction and verification
5--6    Safety and trend detection
7--8    Emergency summary and integration
9--10   Evaluation and refinement

20. Interview Highlights

Problem

Clinical data is fragmented and OCR/NLP extraction can introduce
errors that become dangerous if passed directly to downstream systems.

Solution

MediTrace introduces a verification layer between document extraction
and clinical AI analysis.

Key technical idea

Extract → Verify → Trace → Analyze

Why multi-agent?

Each stage has a focused responsibility:

Document Agent
      ↓
Verification Agent
      ↓
Safety / Trend Agents
      ↓
Emergency / Dashboard

This improves separation of concerns and makes the pipeline easier to
test and audit.

Strong differentiator

Every important fact can be traced back to:

source document
+
source text
+
extraction confidence
+
verification confidence
+
verification status

Conflict handling

The system is designed not to silently overwrite contradictory clinical
information.

21. Team Contribution

Member 1 --- Document Processing - upload - PDF/image processing -
OCR - cleaning - NER - candidate facts - evidence preservation

Member 2 --- Verification + Patient Profile - verification -
evidence checks - conflict detection - duplicate handling - verified
profile - database persistence

Member 3 --- Safety + Trend - medication safety - duplicate
medication checks - allergy-related checks - dosage-related checks -
trends

Member 4 --- Emergency + Dashboard + Integration - emergency
summary - dashboard - integration - final demonstration

22. Limitations

OCR accuracy depends on document quality.

Handwriting can be difficult to recognize.

Medical terminology may be ambiguous.

Conflicting records may require human review.

External APIs/databases can have availability and rate limits.

Confidence scores need calibration.

The system should not treat confidence as medical certainty.

Clinical deployment would require stronger validation, security,
privacy, interoperability, and regulatory controls.

23. Future Scope

medical-specific NER models

handwriting OCR

multilingual clinical documents

FHIR/HL7 integration

human-in-the-loop verification

calibrated uncertainty

richer longitudinal analysis

hospital-system integration

authentication and role-based access

secure audit trails

cloud deployment

larger clinical evaluation datasets

24. Important Development Rules

Member 1 must output candidate facts, never verified facts.

Every fact should preserve source evidence.

Member 2 owns verification and the trusted profile.

Member 3 and Member 4 should consume verified information.

Conflicts must not be silently overwritten.

Extraction confidence and verification confidence must remain
separate.

Do not commit .env, API keys, database passwords, or private
credentials.

Keep the final database aligned with the submitted synopsis: MongoDB
or PostgreSQL.

Keep the one-application/one-port architecture unless the team
deliberately changes it.

Keep clinical decision-support claims appropriately bounded.

Quick Start

git clone <YOUR_REPOSITORY_URL>
cd MediTrace

python -m venv venv

# Windows
venv\Scriptsctivate

# macOS
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python main.py

Open:

http://localhost:8000
http://localhost:8000/docs

Project Vision

MediTrace moves a clinical AI pipeline from:

Extract → Assume → Analyze

to:

Extract → Verify → Trace → Analyze

The objective is to make downstream clinical intelligence depend on
verified, evidence-linked information instead of unchecked document
extraction.
