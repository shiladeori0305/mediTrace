"""
MediTrace - Document Processing Agent
======================================
Member 1's module in the MediTrace multi-agent system.

Pipeline:
    File Upload -> PDF/Image Processing -> OCR -> Text Cleaning ->
    Medical Entity Extraction (NER) -> Candidate Facts -> Output for Verification Agent

IMPORTANT: This module NEVER marks facts as "verified". Every fact produced here
has status="candidate" and must be checked by the (future) Verification Agent.

Run with:
    python main.py
Then open:
    http://localhost:8000
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import re
import io
import uuid
import shutil
import logging
import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import spacy
import fitz  # PyMuPDF
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# =========================================================
# CONFIGURATION
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("meditrace.document_agent")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 20

# --- Tesseract path override (Windows) ---------------------------------
# If Tesseract is not on PATH, uncomment and set the correct install path:
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


# =========================================================
# DIRECTORY SETUP
# =========================================================
def setup_directories() -> None:
    """Create required folders on startup if they don't exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    logger.info(f"Upload directory ready at: {UPLOAD_DIR}")


setup_directories()


# =========================================================
# FASTAPI INITIALIZATION
# =========================================================
app = FastAPI(
    title="MediTrace - Document Processing Agent",
    description="Extracts candidate medical facts from uploaded documents. "
                 "Output is UNVERIFIED and intended for a downstream Verification Agent.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# =========================================================
# DATA MODELS (shared contract for the rest of the team)
# =========================================================
class CandidateFact(BaseModel):
    """
    A single unverified fact extracted from a document.
    Member 2 (Verification Agent) consumes a list of these.
    """
    fact_type: str                     # "medication" | "allergy" | "condition" | "lab_value" | "date"
    value: str                         # e.g. "Amoxicillin"
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    date: Optional[str] = None
    lab_value: Optional[str] = None
    source_document: str
    source_text: str                   # exact evidence snippet the fact came from
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    status: str = "candidate"          # ALWAYS "candidate" - never set to "verified" here


class DocumentProcessingResult(BaseModel):
    """Full output of processing one document. This is what /process returns."""
    document: str
    file_type: str
    used_ocr: bool
    raw_text: str
    cleaned_text: str
    candidate_facts: List[CandidateFact]
    warnings: List[str] = []
    processed_at: str


# =========================================================
# IMAGE PREPROCESSING FUNCTIONS
# =========================================================
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Light preprocessing to improve OCR accuracy:
    grayscale -> resize (if small) -> denoise -> threshold.
    Deliberately simple - avoids over-engineering for a prototype.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Upscale small images so Tesseract has more pixels to work with
    height, width = gray.shape[:2]
    if max(height, width) < 1200:
        scale = 1200 / max(height, width)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Mild denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold works better than a single global threshold
    # for photographed/scanned documents with uneven lighting
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    return thresh


# =========================================================
# OCR FUNCTIONS
# =========================================================
def perform_ocr(image: np.ndarray) -> str:
    """Run Tesseract OCR on a preprocessed image array and return raw text."""
    try:
        pil_image = Image.fromarray(image)
        text = pytesseract.image_to_string(pil_image, lang="eng")
        return text
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Tesseract OCR engine not found. Install it separately from pip packages "
                "(see setup instructions) and/or set pytesseract.pytesseract.tesseract_cmd."
            ),
        )
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=422, detail=f"OCR failed: {str(e)}")


def ocr_image_file(file_path: str) -> str:
    """Load an image file from disk, preprocess it, and OCR it."""
    image = cv2.imread(file_path)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not read image file. It may be corrupted.")
    processed = preprocess_image(image)
    text = perform_ocr(processed)
    return text


# =========================================================
# PDF EXTRACTION FUNCTIONS
# =========================================================
def is_pdf_text_based(pdf_path: str) -> bool:
    """Check whether a PDF has selectable text (vs being a scanned image)."""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            if page.get_text().strip():
                doc.close()
                return True
        doc.close()
        return False
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {str(e)}")


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, bool]:
    """
    Extract text from a PDF. Handles multiple pages.
    Returns (text, used_ocr).
    - If the PDF has selectable text, extract it directly.
    - Otherwise, rasterize each page and run OCR (scanned PDF).
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Corrupted or unreadable PDF: {str(e)}")

    if len(doc) == 0:
        doc.close()
        raise HTTPException(status_code=422, detail="PDF has no pages.")

    text_based = is_pdf_text_based(pdf_path)
    all_text = []

    if text_based:
        for page in doc:
            all_text.append(page.get_text())
        doc.close()
        return "\n".join(all_text), False

    # Scanned / image-based PDF -> OCR each page
    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap(dpi=300)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:  # RGBA -> BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB -> BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        processed = preprocess_image(img_array)
        page_text = perform_ocr(processed)
        all_text.append(page_text)

    doc.close()
    return "\n".join(all_text), True


# =========================================================
# TEXT CLEANING FUNCTIONS
# =========================================================
def clean_text(raw_text: str) -> str:
    """
    Clean OCR/extracted text WITHOUT aggressively altering medical terms.
    The original raw_text is always preserved separately as evidence.
    """
    text = raw_text

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse repeated spaces/tabs (but keep newlines intact)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove stray OCR noise characters commonly produced by Tesseract
    # (keep medical punctuation like %, /, -, ., : intact)
    text = re.sub(r"[^\S\n]*[|~`^_]+[^\S\n]*", " ", text)

    # Strip trailing whitespace on each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove fully empty leading/trailing lines
    text = text.strip()

    return text


# =========================================================
# MEDICAL ENTITY EXTRACTION FUNCTIONS
# =========================================================
# NOTE ON MODEL CHOICE:
# A generic spaCy model (en_core_web_sm) is NOT trained to recognize medicine
# names, dosages, or lab values - it only reliably catches general entities
# like DATE, ORG, PERSON. For real medical NER you would ideally install a
# clinical model such as "en_core_sci_sm" (from scispaCy) or "en_ner_bc5cdr_md"
# (trained on drugs/diseases). Installing those requires extra steps:
#
#   pip install scispacy
#   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/scispacy/20220729/en_ner_bc5cdr_md-0.5.1.tar.gz
#
# Those packages are large and version-sensitive, which is often impractical
# for a college minor project / Windows setup. So this module uses a HYBRID
# approach instead:
#   1. spaCy en_core_web_sm  -> general DATE entities
#   2. Regex patterns        -> dosage, frequency, lab values (structured, reliable)
#   3. Controlled keyword lists -> medication names, conditions, allergy triggers
#
# This keeps the project understandable while still being reasonably accurate
# for common prescription formats. Swapping in a real clinical model later is
# a drop-in replacement inside extract_medical_entities().

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    logger.warning(
        "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
    )

# --- Controlled keyword lists (extend as needed) ------------------------
MEDICATION_KEYWORDS = {
    "amoxicillin", "azithromycin", "paracetamol", "acetaminophen", "ibuprofen",
    "metformin", "atorvastatin", "amlodipine", "losartan", "omeprazole",
    "pantoprazole", "cetirizine", "aspirin", "insulin", "levothyroxine",
    "ciprofloxacin", "doxycycline", "prednisone", "salbutamol", "metronidazole",
    "clopidogrel", "warfarin", "diazepam", "hydrochlorothiazide", "furosemide",
    "montelukast", "dolo", "combiflam", "augmentin", "crocin",
}

CONDITION_KEYWORDS = {
    "diabetes", "hypertension", "asthma", "tuberculosis", "pneumonia",
    "bronchitis", "anemia", "arthritis", "migraine", "covid-19", "influenza",
    "typhoid", "malaria", "dengue", "hypothyroidism", "hyperthyroidism",
    "gastritis", "jaundice", "hepatitis", "fever", "infection",
}

ALLERGY_TRIGGER_PHRASES = {
    "allergy", "allergies", "allergic to", "known allergy", "drug allergy",
}

# --- Regex patterns -------------------------------------------------------
DOSAGE_REGEX = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu|units?)\b", re.IGNORECASE)

FREQUENCY_REGEX = re.compile(
    r"\b(?:once|twice|thrice|\d+\s*times?)\s+(?:a\s+day|daily|per\s+day)\b"
    r"|\bBID\b|\bTID\b|\bQID\b|\bOD\b|\bHS\b"
    r"|\bevery\s+\d+\s+hours?\b",
    re.IGNORECASE,
)

DATE_REGEX = re.compile(
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b"
)

# Textual month formats: "12 September 2026" / "September 12, 2026"
_MONTH_NAMES = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
MONTH_DATE_REGEX = re.compile(
    rf"\b(?:\d{{1,2}}\s+(?:{_MONTH_NAMES})\s+\d{{4}}|(?:{_MONTH_NAMES})\s+\d{{1,2}},?\s+\d{{4}})\b",
    re.IGNORECASE,
)

# Frequency words that spaCy's general DATE label sometimes mistakes for dates.
# Anything in this list, or anything matching FREQUENCY_REGEX, is NEVER a date.
FREQUENCY_DATE_BLOCKLIST = {
    "daily", "weekly", "monthly", "once daily", "twice daily", "thrice daily",
    "once a day", "twice a day", "thrice a day", "every day", "every night",
    "od", "bid", "tid", "qid", "hs",
}


def is_probable_date(candidate_text: str) -> bool:
    """
    Validate that a candidate string actually looks like a date before it is
    allowed into results["dates"]. This is what stops spaCy from letting
    frequency words like "daily" or "twice daily" through as dates.
    """
    stripped = candidate_text.strip().strip(".,")
    lower = stripped.lower()

    if not stripped:
        return False
    if lower in FREQUENCY_DATE_BLOCKLIST:
        return False
    if FREQUENCY_REGEX.search(stripped):
        return False

    return bool(DATE_REGEX.search(stripped) or MONTH_DATE_REGEX.search(stripped))


def clean_value(value: str) -> str:
    """
    Strip trailing/leading punctuation and whitespace from an extracted VALUE
    (never from source_text, which must stay verbatim as evidence).
    e.g. "Penicillin." -> "Penicillin"
    """
    return value.strip().strip(" .,;:-")


LAB_VALUE_REGEX = re.compile(
    r"\b([A-Za-z][A-Za-z0-9 ]{2,25}?)\s*[:\-]\s*(\d+(?:\.\d+)?)\s*"
    r"(mg/dl|g/dl|mmol/l|/mm3|%|meq/l|ng/ml|iu/l)\b",
    re.IGNORECASE,
)


def extract_medical_entities(text: str) -> dict:
    """
    Hybrid extraction: spaCy (dates/general) + regex (dosage/frequency/labs)
    + controlled keyword lists (medications/conditions/allergies).

    Returns raw grouped entity hits with the exact matched line as evidence,
    ready to be turned into CandidateFact objects.
    """
    results = {
        "medications": [],   # list of dicts: {value, dosage, frequency, line}
        "allergies": [],     # list of dicts: {value, line}
        "conditions": [],    # list of dicts: {value, line}
        "dates": [],         # list of dicts: {value, line}
        "lab_values": [],    # list of dicts: {parameter, value, unit, line}
    }

    lines = [ln for ln in text.split("\n") if ln.strip()]

    for line in lines:
        lower_line = line.lower()

        # --- Medications: keyword match, then look for dosage/frequency on same line ---
        for med in MEDICATION_KEYWORDS:
            if re.search(rf"\b{re.escape(med)}\b", lower_line):
                dosage_match = DOSAGE_REGEX.search(line)
                freq_match = FREQUENCY_REGEX.search(line)
                results["medications"].append({
                    "value": med.capitalize(),
                    "dosage": dosage_match.group(0) if dosage_match else None,
                    "frequency": freq_match.group(0) if freq_match else None,
                    "line": line.strip(),
                })

        # --- Allergies: trigger phrase -> capture rest of line as the value ---
        for phrase in ALLERGY_TRIGGER_PHRASES:
            if phrase in lower_line:
                after = re.split(phrase, line, flags=re.IGNORECASE)
                candidate_value = clean_value(after[-1].strip(" :-\t"))
                if candidate_value:
                    results["allergies"].append({
                        "value": candidate_value,
                        "line": line.strip(),
                    })

        # --- Conditions: keyword match ---
        for cond in CONDITION_KEYWORDS:
            if re.search(rf"\b{re.escape(cond)}\b", lower_line):
                results["conditions"].append({
                    "value": clean_value(cond.capitalize()),
                    "line": line.strip(),
                })

        # --- Lab values: regex "Parameter: number unit" ---
        for match in LAB_VALUE_REGEX.finditer(line):
            results["lab_values"].append({
                "parameter": clean_value(match.group(1)),
                "value": match.group(2),
                "unit": match.group(3),
                "line": line.strip(),
            })

        # --- Dates via regex: numeric (DD/MM/YYYY) and textual month formats ---
        # Frequency phrases ("twice daily", "every 8 hours") are matched by
        # FREQUENCY_REGEX elsewhere on this same line and are NEVER treated as dates.
        for match in DATE_REGEX.finditer(line):
            value = clean_value(match.group(1))
            if is_probable_date(value):
                results["dates"].append({"value": value, "line": line.strip()})

        for match in MONTH_DATE_REGEX.finditer(line):
            value = clean_value(match.group(0))
            if is_probable_date(value):
                results["dates"].append({"value": value, "line": line.strip()})

    # --- Dates via spaCy as a fallback/supplement only ---
    # spaCy's general DATE label frequently misfires on frequency words like
    # "daily" or "twice daily" - every spaCy candidate is validated against
    # is_probable_date() before being accepted, so those never slip through.
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ != "DATE":
                continue
            value = clean_value(ent.text)
            if not is_probable_date(value):
                continue
            if any(value == d["value"] for d in results["dates"]):
                continue  # avoid duplicates already captured by regex
            results["dates"].append({
                "value": value,
                "line": ent.sent.text.strip() if ent.sent else value,
            })

    return results


# =========================================================
# CONFIDENCE SCORING
# =========================================================
def calculate_confidence(fact_type: str, has_dosage: bool = False, has_frequency: bool = False) -> float:
    """
    Simple heuristic confidence score reflecting confidence in the EXTRACTION,
    not medical certainty. A medication with both dosage and frequency found
    nearby is more likely to be a real, correctly-parsed fact.
    """
    base = {
        "medication": 0.60,
        "allergy": 0.55,
        "condition": 0.50,
        "lab_value": 0.70,
        "date": 0.65,
    }.get(fact_type, 0.50)

    if fact_type == "medication":
        if has_dosage:
            base += 0.20
        if has_frequency:
            base += 0.12

    return round(min(base, 0.97), 2)


# =========================================================
# CANDIDATE FACT GENERATION
# =========================================================
def create_candidate_facts(entities: dict, source_document: str) -> List[CandidateFact]:
    """Convert raw extracted entities into standardized CandidateFact objects."""
    facts: List[CandidateFact] = []

    for med in entities["medications"]:
        facts.append(CandidateFact(
            fact_type="medication",
            value=med["value"],
            dosage=med["dosage"],
            frequency=med["frequency"],
            source_document=source_document,
            source_text=med["line"],
            extraction_confidence=calculate_confidence(
                "medication", has_dosage=bool(med["dosage"]), has_frequency=bool(med["frequency"])
            ),
            status="candidate",
        ))

    for allergy in entities["allergies"]:
        facts.append(CandidateFact(
            fact_type="allergy",
            value=allergy["value"],
            source_document=source_document,
            source_text=allergy["line"],
            extraction_confidence=calculate_confidence("allergy"),
            status="candidate",
        ))

    for cond in entities["conditions"]:
        facts.append(CandidateFact(
            fact_type="condition",
            value=cond["value"],
            source_document=source_document,
            source_text=cond["line"],
            extraction_confidence=calculate_confidence("condition"),
            status="candidate",
        ))

    for lab in entities["lab_values"]:
        facts.append(CandidateFact(
            fact_type="lab_value",
            value=lab["parameter"],
            lab_value=f'{lab["value"]} {lab["unit"]}',
            source_document=source_document,
            source_text=lab["line"],
            extraction_confidence=calculate_confidence("lab_value"),
            status="candidate",
        ))

    for date in entities["dates"]:
        facts.append(CandidateFact(
            fact_type="date",
            value=date["value"],
            date=date["value"],
            source_document=source_document,
            source_text=date["line"],
            extraction_confidence=calculate_confidence("date"),
            status="candidate",
        ))

    return facts


# =========================================================
# DOCUMENT PROCESSING PIPELINE
# =========================================================
def validate_file(filename: str, file_bytes: bytes) -> str:
    """Validate extension and size. Returns the lowercase extension."""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    return ext


def save_upload(filename: str, file_bytes: bytes) -> str:
    """Save uploaded bytes to data/uploads/ using a safe, unique filename."""
    ext = os.path.splitext(filename)[1].lower()
    safe_base = re.sub(r"[^A-Za-z0-9_\-]", "_", os.path.splitext(filename)[0])[:50]
    unique_name = f"{safe_base}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    return save_path


def process_document(file_path: str, original_filename: str) -> DocumentProcessingResult:
    """
    Full pipeline for one document:
    detect type -> extract/OCR text -> clean -> extract entities -> candidate facts.
    """
    warnings: List[str] = []
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        raw_text, used_ocr = extract_text_from_pdf(file_path)
        file_type = "pdf"
    elif ext in (".jpg", ".jpeg", ".png"):
        raw_text = ocr_image_file(file_path)
        used_ocr = True
        file_type = "image"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if not raw_text or not raw_text.strip():
        warnings.append("No text could be detected in this document.")
        raw_text = ""

    cleaned = clean_text(raw_text)

    entities = extract_medical_entities(cleaned) if cleaned else {
        "medications": [], "allergies": [], "conditions": [], "dates": [], "lab_values": []
    }

    facts = create_candidate_facts(entities, source_document=original_filename)

    if not facts:
        warnings.append("No medical entities detected. Document may be low quality or use unrecognized terms.")

    return DocumentProcessingResult(
        document=original_filename,
        file_type=file_type,
        used_ocr=used_ocr,
        raw_text=raw_text,
        cleaned_text=cleaned,
        candidate_facts=facts,
        warnings=warnings,
        processed_at=datetime.utcnow().isoformat() + "Z",
    )


# =========================================================
# MEMBER 2: VERIFICATION AGENT + VERIFIED PATIENT PROFILE
# =========================================================
# Consumes CandidateFact objects (produced above by Member 1) and decides
# whether each one may enter the trusted "verified patient profile" that
# Member 3 and Member 4 will read from later.
#
# Design rules (from the project brief):
#   - Deterministic, explainable rules only. No LLM medical judgement.
#   - Never silently overwrite a verified fact that conflicts with a new one.
#   - Always preserve the original evidence (source_document + source_text).
#   - Keep "extraction_confidence" (Member 1) and "verification_confidence"
#     (Member 2) as two separate numbers - never replace one with the other.
#
# Statuses:
#   candidate -> produced by Member 1, not yet looked at here.
#   verified  -> supported by its own source text AND does not conflict
#                with an existing verified record. Part of the active profile.
#   rejected  -> not supported by its own source text (or malformed/unsupported
#                fact_type). Never enters the profile.
#   conflict  -> supported by its own source text, but contradicts an existing
#                verified record. Stored for review; the existing verified
#                record is NEVER overwritten or deleted.

DB_PATH = os.path.join(BASE_DIR, "data", "meditrace.db")

ALLOWED_FACT_TYPES = {"medication", "allergy", "condition", "lab_value", "date"}

# Phrases meaning "the patient has been documented as having NO allergies".
# Needed to catch the "No known allergies" vs "Penicillin" conflict case.
ALLERGY_NEGATION_PHRASES = {
    "no known allergies", "no known allergy", "no allergies", "none known",
    "nka", "no known drug allergies", "none",
}


class VerifyRequest(BaseModel):
    """Input body for POST /verify. Reuses Member 1's CandidateFact model."""
    candidate_facts: List[CandidateFact]


# ---------------------------------------------------------
# DATABASE SETUP (SQLite - three simple tables)
# ---------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_verification_db() -> None:
    """Create the Member 2 tables if they don't already exist. Never drops data."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Every candidate Member 2 has ever looked at, with its final decision.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS candidate_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_type TEXT NOT NULL,
                value TEXT NOT NULL,
                dosage TEXT,
                frequency TEXT,
                date TEXT,
                lab_value TEXT,
                source_document TEXT NOT NULL,
                source_text TEXT NOT NULL,
                extraction_confidence REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # The trusted patient profile. Holds "verified" rows (the active
        # profile) AND "conflict" rows (kept for review, never merged in).
        # source_references is a JSON list so repeated evidence for the same
        # fact can be attached without creating a second row (see section 15
        # of the brief: avoid unnecessary duplicate entries).
        cur.execute("""
            CREATE TABLE IF NOT EXISTS verified_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_type TEXT NOT NULL,
                value TEXT NOT NULL,
                dosage TEXT,
                frequency TEXT,
                date TEXT,
                lab_value TEXT,
                source_document TEXT NOT NULL,
                source_text TEXT NOT NULL,
                extraction_confidence REAL NOT NULL,
                verification_confidence REAL NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                source_references TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Full audit trail: one row per verification decision ever made.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_type TEXT NOT NULL,
                value TEXT NOT NULL,
                source_document TEXT NOT NULL,
                source_text TEXT NOT NULL,
                extraction_confidence REAL NOT NULL,
                verification_confidence REAL NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                compared_with TEXT,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()
    finally:
        conn.close()


init_verification_db()


# ---------------------------------------------------------
# VERIFICATION HELPERS
# ---------------------------------------------------------
def is_allergy_negation(value: str) -> bool:
    """True if a value documents the ABSENCE of allergies (e.g. 'No known allergies')."""
    v = value.strip().lower().strip(".")
    if v in ALLERGY_NEGATION_PHRASES:
        return True
    return v.startswith("no known") or v.startswith("no allerg")


def evidence_supports_candidate(candidate: CandidateFact) -> bool:
    """
    Verification rule #1: does the candidate's OWN source_text actually contain
    the evidence for its value? This is a text-support check only - it says
    nothing about whether the fact is medically true or safe.
    """
    source_lower = (candidate.source_text or "").lower()
    value_lower = (candidate.value or "").strip().lower()

    if not source_lower or not value_lower:
        return False

    if candidate.fact_type == "lab_value":
        if value_lower not in source_lower:
            return False
        if candidate.lab_value:
            numbers = re.findall(r"\d+(?:\.\d+)?", candidate.lab_value)
            if numbers and numbers[0] not in source_lower:
                return False
        return True

    if candidate.fact_type == "allergy":
        if is_allergy_negation(candidate.value):
            return any(p in source_lower for p in ("no known", "no allerg", "none", "nka"))
        return value_lower in source_lower or "allerg" in source_lower

    # medication, condition, date: the value must appear in its own evidence text.
    return value_lower in source_lower


def fetch_verified_facts(cur: sqlite3.Cursor, fact_type: str) -> List[dict]:
    """All currently-active (status='verified') facts of one type, for conflict/duplicate checks."""
    cur.execute(
        "SELECT * FROM verified_facts WHERE fact_type = ? AND status = 'verified'",
        (fact_type,),
    )
    return [dict(row) for row in cur.fetchall()]


def find_duplicate(candidate: CandidateFact, existing: List[dict]) -> Optional[dict]:
    """An existing verified record that is functionally the SAME fact (not just the same name)."""
    value_lower = candidate.value.strip().lower()
    for rec in existing:
        if rec["value"].strip().lower() != value_lower:
            continue
        if candidate.fact_type == "medication":
            if (rec.get("dosage") or "").strip().lower() == (candidate.dosage or "").strip().lower() and \
               (rec.get("frequency") or "").strip().lower() == (candidate.frequency or "").strip().lower():
                return rec
        elif candidate.fact_type == "lab_value":
            if (rec.get("lab_value") or "") == (candidate.lab_value or ""):
                return rec
        elif candidate.fact_type == "date":
            if (rec.get("date") or "") == (candidate.date or ""):
                return rec
        else:  # allergy, condition: matching value alone is a duplicate
            return rec
    return None


def find_conflict(candidate: CandidateFact, existing: List[dict]) -> Optional[dict]:
    """
    An existing verified record that CONTRADICTS the candidate.
    Deliberately narrow, per the brief: same medication name with a different
    dosage/frequency, or an allergy statement that contradicts a "no known
    allergies" statement (or vice versa). Anything else is left alone rather
    than guessed at - that judgement belongs to a human or to Member 3.
    """
    if candidate.fact_type == "medication":
        value_lower = candidate.value.strip().lower()
        for rec in existing:
            if rec["value"].strip().lower() != value_lower:
                continue
            existing_dosage = (rec.get("dosage") or "").strip().lower()
            new_dosage = (candidate.dosage or "").strip().lower()
            existing_freq = (rec.get("frequency") or "").strip().lower()
            new_freq = (candidate.frequency or "").strip().lower()
            if existing_dosage and new_dosage and existing_dosage != new_dosage:
                return rec
            if existing_freq and new_freq and existing_freq != new_freq:
                return rec
        return None

    if candidate.fact_type == "allergy":
        candidate_is_negation = is_allergy_negation(candidate.value)
        for rec in existing:
            rec_is_negation = is_allergy_negation(rec["value"])
            if candidate_is_negation != rec_is_negation:
                return rec
        return None

    # condition / lab_value / date: no contradiction rule for this milestone.
    return None


def calculate_verification_confidence(candidate: CandidateFact, strong_evidence: bool) -> float:
    """
    Simple, explainable rule: start from Member 1's extraction_confidence and
    add a small bonus once the candidate's own text has been confirmed to
    support it. This is a SYSTEM confidence score, not a medical certainty score.
    """
    bonus = 0.15 if strong_evidence else 0.05
    return round(min(candidate.extraction_confidence + bonus, 0.97), 2)


# ---------------------------------------------------------
# PERSISTENCE HELPERS
# ---------------------------------------------------------
def insert_candidate_fact(cur: sqlite3.Cursor, candidate: CandidateFact, final_status: str) -> None:
    cur.execute(
        """INSERT INTO candidate_facts
           (fact_type, value, dosage, frequency, date, lab_value,
            source_document, source_text, extraction_confidence, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (candidate.fact_type, candidate.value, candidate.dosage, candidate.frequency,
         candidate.date, candidate.lab_value, candidate.source_document, candidate.source_text,
         candidate.extraction_confidence, final_status, datetime.utcnow().isoformat() + "Z"),
    )


def insert_verified_fact(cur: sqlite3.Cursor, candidate: CandidateFact,
                          verification_confidence: float, status: str, reason: str) -> int:
    now = datetime.utcnow().isoformat() + "Z"
    refs = json.dumps([{"source_document": candidate.source_document, "source_text": candidate.source_text}])
    cur.execute(
        """INSERT INTO verified_facts
           (fact_type, value, dosage, frequency, date, lab_value, source_document, source_text,
            extraction_confidence, verification_confidence, status, reason, source_references,
            created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (candidate.fact_type, candidate.value, candidate.dosage, candidate.frequency,
         candidate.date, candidate.lab_value, candidate.source_document, candidate.source_text,
         candidate.extraction_confidence, verification_confidence, status, reason, refs, now, now),
    )
    return cur.lastrowid


def merge_source_reference(cur: sqlite3.Cursor, verified_id: int, source_document: str, source_text: str) -> None:
    """Attach a new piece of evidence to an existing verified fact instead of duplicating the row."""
    cur.execute("SELECT source_references FROM verified_facts WHERE id = ?", (verified_id,))
    row = cur.fetchone()
    refs = json.loads(row["source_references"]) if row and row["source_references"] else []
    new_ref = {"source_document": source_document, "source_text": source_text}
    if new_ref not in refs:
        refs.append(new_ref)
    cur.execute(
        "UPDATE verified_facts SET source_references = ?, updated_at = ? WHERE id = ?",
        (json.dumps(refs), datetime.utcnow().isoformat() + "Z", verified_id),
    )


def insert_verification_log(cur: sqlite3.Cursor, candidate: CandidateFact, decision: str,
                             reason: str, verification_confidence: float,
                             compared_with: Optional[dict]) -> None:
    cur.execute(
        """INSERT INTO verification_logs
           (fact_type, value, source_document, source_text, extraction_confidence,
            verification_confidence, decision, reason, compared_with, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (candidate.fact_type, candidate.value, candidate.source_document, candidate.source_text,
         candidate.extraction_confidence, verification_confidence, decision, reason,
         json.dumps(compared_with) if compared_with else None, datetime.utcnow().isoformat() + "Z"),
    )


# ---------------------------------------------------------
# CORE VERIFICATION PIPELINE
# ---------------------------------------------------------
def verify_single_candidate(cur: sqlite3.Cursor, candidate: CandidateFact) -> dict:
    """
    Runs one candidate through: fact_type check -> evidence check -> duplicate
    check -> conflict check, persists the outcome, and returns the JSON-ready
    result record for the API response.
    """
    conflict_ref: Optional[dict] = None
    already_persisted_as_verified = False  # True once merged into an existing row

    if candidate.fact_type not in ALLOWED_FACT_TYPES:
        status = "rejected"
        reason = f"Unsupported fact_type '{candidate.fact_type}'."
        vconf = 0.0

    elif not candidate.source_document or not candidate.source_text:
        status = "rejected"
        reason = "Missing source_document or source_text; a fact cannot be verified without evidence."
        vconf = 0.0

    elif not evidence_supports_candidate(candidate):
        status = "rejected"
        reason = "The candidate's own source text does not clearly support this value."
        vconf = 0.2

    else:
        existing = fetch_verified_facts(cur, candidate.fact_type)
        dup = find_duplicate(candidate, existing)

        if dup:
            merge_source_reference(cur, dup["id"], candidate.source_document, candidate.source_text)
            status = "verified"
            reason = "Matches an already-verified fact; new source reference attached (no duplicate row created)."
            vconf = calculate_verification_confidence(candidate, True)
            already_persisted_as_verified = True
        else:
            conflict_ref = find_conflict(candidate, existing)
            if conflict_ref:
                status = "conflict"
                detail = f"'{conflict_ref['value']}'"
                if conflict_ref.get("dosage"):
                    detail += f" ({conflict_ref['dosage']})"
                reason = (f"Conflicts with an existing verified {candidate.fact_type} record {detail}. "
                          "Existing record was NOT overwritten; both pieces of evidence are preserved.")
                vconf = calculate_verification_confidence(candidate, True)
            else:
                status = "verified"
                reason = "Supported by its source text and does not conflict with any existing verified record."
                vconf = calculate_verification_confidence(candidate, True)

    # Always store the candidate itself + a full audit-log entry, whatever happened.
    insert_candidate_fact(cur, candidate, status)
    insert_verification_log(cur, candidate, status, reason, vconf, conflict_ref)

    # Conflicts are stored too (for review) but never merged into the active
    # profile; verified duplicates were already merged into an existing row above.
    if status == "conflict" or (status == "verified" and not already_persisted_as_verified):
        insert_verified_fact(cur, candidate, vconf, status, reason)

    record = {
        "fact_type": candidate.fact_type,
        "value": candidate.value,
        "dosage": candidate.dosage,
        "frequency": candidate.frequency,
        "date": candidate.date,
        "lab_value": candidate.lab_value,
        "source_document": candidate.source_document,
        "source_text": candidate.source_text,
        "extraction_confidence": candidate.extraction_confidence,
        "verification_confidence": vconf,
        "status": status,
        "reason": reason,
    }
    if conflict_ref:
        record["conflicting_with"] = {
            "value": conflict_ref["value"],
            "dosage": conflict_ref.get("dosage"),
            "frequency": conflict_ref.get("frequency"),
            "source_document": conflict_ref.get("source_document"),
            "source_text": conflict_ref.get("source_text"),
        }
    return record


# ---------------------------------------------------------
# VERIFIED PATIENT PROFILE
# ---------------------------------------------------------
def build_verified_profile() -> dict:
    """The trusted profile: only status='verified' facts, grouped by type."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM verified_facts WHERE status = 'verified' ORDER BY id ASC")
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    profile = {"medications": [], "allergies": [], "conditions": [], "lab_values": [], "dates": []}
    key_by_type = {
        "medication": "medications", "allergy": "allergies", "condition": "conditions",
        "lab_value": "lab_values", "date": "dates",
    }
    for row in rows:
        key = key_by_type.get(row["fact_type"])
        if not key:
            continue
        profile[key].append({
            "value": row["value"],
            "dosage": row["dosage"],
            "frequency": row["frequency"],
            "date": row["date"],
            "lab_value": row["lab_value"],
            "extraction_confidence": row["extraction_confidence"],
            "verification_confidence": row["verification_confidence"],
            "status": row["status"],
            "reason": row["reason"],
            "sources": json.loads(row["source_references"]) if row["source_references"] else [],
        })
    return profile


def fetch_unresolved_conflicts() -> List[dict]:
    """status='conflict' rows: evidence-supported facts that clash with the verified profile."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM verified_facts WHERE status = 'conflict' ORDER BY id ASC")
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    for r in rows:
        r.pop("source_references", None)
    return rows


def verify_candidate_facts(candidate_facts: List[CandidateFact]) -> dict:
    """
    Member 2's single public entry point.

    Takes the list of CandidateFact objects Member 1 produces, runs EACH one
    through verify_single_candidate() (evidence check -> duplicate check ->
    conflict check -> verified/rejected/conflict), persists every decision to
    meditrace.db, and returns the full result set.

    This function does the actual work; POST /verify is a thin HTTP wrapper
    around it. It can also be called directly (e.g. from a test, a script, or
    a future Member 3/4 integration) without going through the HTTP layer.
    """
    verified_out: List[dict] = []
    rejected_out: List[dict] = []
    conflict_out: List[dict] = []

    if candidate_facts:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            for candidate in candidate_facts:
                record = verify_single_candidate(cur, candidate)
                if record["status"] == "verified":
                    verified_out.append(record)
                elif record["status"] == "rejected":
                    rejected_out.append(record)
                else:
                    conflict_out.append(record)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    return {
        "verified_facts": verified_out,
        "rejected_facts": rejected_out,
        "conflicts": conflict_out,
        "verified_profile": build_verified_profile(),
    }


# =========================================================
# API ROUTES
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Simple upload UI for manual testing."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Save an uploaded document to data/uploads/ WITHOUT processing it.
    Useful if Member 2's pipeline wants to trigger processing separately.
    """
    try:
        file_bytes = await file.read()
        ext = validate_file(file.filename, file_bytes)
        save_path = save_upload(file.filename, file_bytes)
        return JSONResponse({
            "message": "File uploaded successfully.",
            "saved_as": os.path.basename(save_path),
            "original_filename": file.filename,
            "file_type": ext,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/process", response_model=DocumentProcessingResult)
async def process_uploaded_document(file: UploadFile = File(...)):
    """
    Combined endpoint: upload + process in one call.
    Returns extracted text and CANDIDATE (unverified) facts as JSON.
    This is the primary endpoint Member 2 should call/consume.
    """
    try:
        file_bytes = await file.read()
        validate_file(file.filename, file_bytes)
        save_path = save_upload(file.filename, file_bytes)

        result = process_document(save_path, original_filename=file.filename)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.post("/process/{saved_filename}", response_model=DocumentProcessingResult)
async def process_saved_document(saved_filename: str):
    """
    Process a document that was previously saved via /upload.
    saved_filename must match the 'saved_as' value returned by /upload.
    """
    file_path = os.path.join(UPLOAD_DIR, saved_filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{saved_filename}' not found in uploads.")

    try:
        result = process_document(file_path, original_filename=saved_filename)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.post("/verify")
async def verify_candidates(payload: VerifyRequest):
    """
    HTTP entry point for Member 2. Accepts {"candidate_facts": [...]} exactly
    as produced by Member 1's /process endpoint, and returns which ones were
    verified, rejected, or flagged as conflicts - plus the resulting verified
    profile. All the actual logic lives in verify_candidate_facts().
    """
    try:
        return verify_candidate_facts(payload.candidate_facts)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.get("/patient/profile")
async def get_patient_profile():
    """Returns the current verified patient profile plus any unresolved conflicts."""
    try:
        return {
            "verified_profile": build_verified_profile(),
            "unresolved_conflicts": fetch_unresolved_conflicts(),
        }
    except Exception as e:
        logger.error(f"Failed to build patient profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve patient profile: {str(e)}")


@app.get("/health")
async def health_check():
    """Basic health check, also reports whether the spaCy model loaded."""
    return {
        "status": "ok",
        "spacy_model_loaded": nlp is not None,
        "upload_dir": UPLOAD_DIR,
    }


# =========================================================
# APPLICATION STARTUP
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)