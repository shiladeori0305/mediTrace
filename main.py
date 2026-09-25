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

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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
                candidate_value = after[-1].strip(" :-\t")
                if candidate_value:
                    results["allergies"].append({
                        "value": candidate_value,
                        "line": line.strip(),
                    })

        # --- Conditions: keyword match ---
        for cond in CONDITION_KEYWORDS:
            if re.search(rf"\b{re.escape(cond)}\b", lower_line):
                results["conditions"].append({
                    "value": cond.capitalize(),
                    "line": line.strip(),
                })

        # --- Lab values: regex "Parameter: number unit" ---
        for match in LAB_VALUE_REGEX.finditer(line):
            results["lab_values"].append({
                "parameter": match.group(1).strip(),
                "value": match.group(2),
                "unit": match.group(3),
                "line": line.strip(),
            })

        # --- Dates via regex (fast, reliable for DD/MM/YYYY style) ---
        for match in DATE_REGEX.finditer(line):
            results["dates"].append({
                "value": match.group(1),
                "line": line.strip(),
            })

    # --- Dates via spaCy as a fallback/supplement (handles "12 Sept 2026" etc.) ---
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                # avoid exact duplicates already captured by regex
                if not any(ent.text == d["value"] for d in results["dates"]):
                    results["dates"].append({
                        "value": ent.text,
                        "line": ent.sent.text.strip() if ent.sent else ent.text,
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
