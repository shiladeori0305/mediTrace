"""
MediTrace - Member 3: Safety + Trend service
============================================
Reads Member 2's `verified_facts` collection (MongoDB), builds a patient profile,
runs the safety + trend engine (safety_trend_agents.py) and returns alerts.

Place in: agents/safety_trend_agent.py  (+ agents/safety_engine.py, agents/__init__.py)
Uses the same .env keys as main.py: MONGODB_URI, MONGODB_DB (default "meditrace")

Fact format consumed (as written by Member 1 + 2 in main.py):
  fact_type: medication | allergy | condition | lab_value | date
  value, dosage ("500 mg"), frequency ("twice daily"/"BID"), date, lab_value ("7.6 %"),
  source_document, source_text, status: verified | conflict, reason, created_at
"""
import logging
import os
import re
from datetime import date, datetime
from typing import List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

from .safety_engine import (
    SEVERITY_ORDER, Alert, Condition, LabResult, Medication,
    PatientProfile, analyze_patient, max_severity,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (where .env lives)
load_dotenv(os.path.join(BASE, ".env"))
log = logging.getLogger("meditrace.safety_agent")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "meditrace")  # lowercase, same as main.py

# Brands in Member 1's keyword list that hide more than one active ingredient
COMBOS = {"combiflam": ["ibuprofen", "paracetamol"], "augmentin": ["amoxicillin"]}

# (substring in lab name, key used by the engine's LAB_RULES)
LAB_ALIASES = [("hba1c", "hba1c"), ("glycated", "hba1c"), ("glucose", "glucose"),
               ("sugar", "glucose"), ("egfr", "egfr"), ("creatinine", "creatinine"),
               ("hemoglobin", "hemoglobin"), ("haemoglobin", "hemoglobin"),
               ("platelet", "platelets"), ("spo2", "spo2")]

_client = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    return _client[MONGODB_DB]


# ----------------------------------------------------------------------------
# Parsers for Member 1's string formats
# ----------------------------------------------------------------------------
_NUM_DATE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
_TXT_DATE = re.compile(r"\b(?:\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b")
_TXT_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y")


def parse_date(s) -> Optional[date]:
    if not s:
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    s = str(s)
    if re.match(r"\d{4}-\d{2}-\d{2}", s):  # ISO (created_at)
        try:
            return datetime.fromisoformat(s[:10]).date()
        except ValueError:
            return None
    m = _NUM_DATE.search(s)
    if m:  # Indian format: DD/MM/YYYY
        d, mo, y = map(int, m.groups())
        y += 2000 if y < 100 else 0
        for day, month in ((d, mo), (mo, d)):  # swapped as fallback
            try:
                return date(y, month, day)
            except ValueError:
                continue
        return None
    m = _TXT_DATE.search(s)
    if m:
        txt = re.sub(r"\bSept\b", "Sep", m.group(0))
        for fmt in _TXT_FORMATS:
            try:
                return datetime.strptime(txt, fmt).date()
            except ValueError:
                continue
    return None


def parse_dose_mg(dosage) -> Optional[float]:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b", dosage or "", re.I)
    if not m:
        return None  # ml / IU / units -> can't compare to a mg limit
    v, unit = float(m.group(1)), m.group(2).lower()
    return v / 1000 if unit == "mcg" else v * 1000 if unit == "g" else v


def parse_frequency(freq) -> Optional[float]:
    if not freq:
        return None
    s = freq.lower().strip()
    m = re.search(r"every\s+(\d+)\s+hours?", s)
    if m:
        return 24 / int(m.group(1))
    m = re.search(r"(\d+)\s*times?", s)
    if m:
        return float(m.group(1))
    for word, n in (("thrice", 3), ("twice", 2), ("once", 1)):
        if word in s:
            return float(n)
    return {"od": 1.0, "hs": 1.0, "bid": 2.0, "tid": 3.0, "qid": 4.0, "daily": 1.0}.get(s)


def lab_key(name: str) -> str:
    n = name.strip().lower()
    if n == "hb":
        return "hemoglobin"
    for sub, key in LAB_ALIASES:
        if sub in n:
            return key
    return n


_NEGATION = ("no known", "no allerg", "none", "nka", "nil")


def split_allergies(value: str) -> List[str]:
    v = value.strip().lower().strip(" .")
    if not v or v.startswith(_NEGATION):
        return []  # "No known allergies" -> no allergy
    parts = re.split(r"[,;/&]|\band\b", v)
    cleaned = (re.sub(r"\b(drugs?|allergy|allergies|antibiotics?)\b", "", p).strip(" .") for p in parts)
    return [p for p in cleaned if p]


def expand_medication(name: str) -> List[str]:
    return COMBOS.get(name.strip().lower(), [name.strip()])


# ----------------------------------------------------------------------------
# Loading + profile building
# ----------------------------------------------------------------------------

def _load_facts(patient_id):
    """Returns (facts, warnings). Never raises."""
    warnings = []
    try:
        col = get_db()["verified_facts"]
        query = {"status": {"$in": ["verified", "conflict"]}}
        if patient_id:
            if col.count_documents({"patient_id": patient_id}, limit=1):
                query["patient_id"] = patient_id
            else:
                warnings.append("verified_facts me patient_id field nahi mila; saare facts "
                                "ek hi patient ke maan ke use kiye gaye.")
        return list(col.find(query)), warnings
    except Exception as e:  # noqa: BLE001
        log.exception("MongoDB read failed")
        return [], [f"MongoDB unavailable ({type(e).__name__}); safety/trend history check nahi ho paya."]


def _document_dates(facts) -> dict:
    """source_document -> earliest `date` fact (lab/condition facts have an empty date)."""
    out = {}
    for f in facts:
        if f.get("fact_type") == "date":
            d = parse_date(f.get("date") or f.get("value"))
            if d:
                k = f.get("source_document")
                out[k] = min(out.get(k, d), d)
    return out


def _fact_date(f, doc_dates) -> Optional[date]:
    return (parse_date(f.get("date")) or parse_date(f.get("source_text"))
            or doc_dates.get(f.get("source_document")) or parse_date(f.get("created_at")))


def build_profile(patient_id, facts, current_medications=None, extra_allergies=None,
                  active_window_days=None):
    """verified_facts -> (PatientProfile, extra_alerts for unresolved verification conflicts)."""
    doc_dates = _document_dates(facts)
    today = date.today()
    meds, conds, labs, allergies, extra = [], [], [], list(extra_allergies or []), []

    for f in facts:
        ftype = f.get("fact_type")
        val = str(f.get("value") or "").strip()
        status = f.get("status")
        d = _fact_date(f, doc_dates)

        if status == "conflict":  # Member 2 flagged a contradiction -> surface it for human review
            extra.append(Alert(
                category="verification_conflict",
                severity="high" if ftype in ("allergy", "medication") else "moderate",
                title=f"Unresolved conflict: {ftype} '{val}'",
                message=f.get("reason") or "Verified records me contradiction hai; manual review chahiye.",
                evidence=[x for x in (f.get("source_document"), f.get("source_text")) if x]))

        if ftype == "allergy":
            # Safety first: even if in conflict, treat a real (non-negation) allergy as present
            allergies += split_allergies(val)
        elif status != "verified":
            continue  # do not use conflicting medication/condition/lab facts in checks
        elif ftype == "medication":
            stale = active_window_days is not None and d and (today - d).days > active_window_days
            names = expand_medication(val)
            for n in names:
                single = len(names) == 1
                meds.append(Medication(
                    name=n, start_date=d, status="stopped" if stale else "active",
                    dose_mg=parse_dose_mg(f.get("dosage")) if single else None,
                    frequency_per_day=parse_frequency(f.get("frequency")) if single else None))
        elif ftype == "condition":
            conds.append(Condition(name=val, diagnosed_on=d))
        elif ftype == "lab_value":
            raw = f.get("lab_value") or ""
            m = re.search(r"\d+(?:\.\d+)?", raw)
            if m and d:
                labs.append(LabResult(test=lab_key(val), value=float(m.group(0)),
                                      unit=raw[m.end():].strip(), date=d))

    if current_medications:  # the caller's list is the source of truth for currently active medications
        cur = set()
        for item in current_medications:
            name = item if isinstance(item, str) else item["name"]
            cur.update(n.lower() for n in expand_medication(name))
        known = {m.name.lower() for m in meds}
        for m in meds:
            m.status = "active" if m.name.lower() in cur else "stopped"
        for n in sorted(cur - known):
            meds.append(Medication(name=n))

    return PatientProfile(patient_id=patient_id or "unknown", medications=meds,
                          allergies=sorted(set(allergies)), conditions=conds,
                          lab_results=labs), extra


# ----------------------------------------------------------------------------
# Output formatting + public entry point
# ----------------------------------------------------------------------------

def _alert_to_dict(a: Alert) -> dict:
    out = {"type": f"{a.category.replace('_', ' ').title()} Alert", "category": a.category,
           "severity": a.severity.capitalize(), "title": a.title,
           "message": a.message, "evidence": a.evidence}
    if a.category == "interaction":
        out["conflicting_drugs"] = a.evidence
    return out


def run_safety_and_trend_checks(patient_id=None, current_medications=None, allergies=None,
                                active_window_days=None):
    """
    Main entry point for Member 3 (Safety + Trend Agents).

    patient_id          : optional (verified_facts currently has no patient_id field)
    current_medications : optional list of names / {"name":..} - override for "active" meds
    allergies           : optional extra allergies (list of str)
    active_window_days  : optional; prescriptions older than this are treated as 'stopped'
    """
    facts, warnings = _load_facts(patient_id)
    profile, conflict_alerts = build_profile(patient_id, facts, current_medications,
                                             allergies, active_window_days)
    report = analyze_patient(profile)
    all_alerts = sorted(report.alerts + conflict_alerts,
                        key=lambda a: SEVERITY_ORDER[a.severity], reverse=True)

    safety = [_alert_to_dict(a) for a in all_alerts if a.category != "trend"]
    trends = [_alert_to_dict(a) for a in all_alerts if a.category == "trend"]
    db_failed = any("MongoDB unavailable" in w for w in warnings)

    if all_alerts:
        status = "Flagged"
    elif db_failed:
        status = "Unverified"   # check could not complete -> never report "Safe"
    elif not facts and not current_medications:
        status = "No Data"      # nothing to check
    else:
        status = "Safe"

    return {
        "patient_id": patient_id,
        "safety_alerts": safety,
        "historical_trends": {
            "patient_id": patient_id,
            "total_visits": len({f.get("source_document") for f in facts if f.get("source_document")}),
            "trend_alerts": trends,
            "timeline": [{"date": e.date.isoformat() if e.date else None,
                          "type": e.type, "description": e.description}
                         for e in report.timeline],
        },
        "risk_level": max_severity(all_alerts) if all_alerts else "low",
        "status": status,
        "facts_used": len(facts),
        "warnings": warnings,
    }