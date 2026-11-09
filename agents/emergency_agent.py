"""
MediTrace - Member 4: Emergency Summary Agent
==============================================
Synthesizes Member 2's Verified Patient Profile and Member 3's Safety & Trend Report
into a rapid, 30-second Emergency Department (ED / ER) summary.

Inputs:
  - Verified Patient Profile (from Member 2)
  - Safety & Trend Report (from Member 3)
Outputs:
  - Triage Level (RED / YELLOW / GREEN)
  - Critical Red Flags / Resuscitation Cautions
  - Rapid Clinical Handover Narrative (Paramedic / ER Doctor 15-second brief)
  - Active Medications & Dosage Schedule
  - Confirmed Allergies & Severity
  - Diagnosed Chronic & Acute Conditions
  - Most Recent Lab Results with clinical status flags
  - Data Integrity & Verification Audit metrics
"""

import os
import re
import json
import sqlite3
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

log = logging.getLogger("meditrace.emergency_agent")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "meditrace.db")

# High-risk medications that require immediate resuscitation & clinical awareness in an ER
HIGH_RISK_MED_CLASSES = {
    "anticoagulant": ["warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin"],
    "antiplatelet": ["aspirin", "clopidogrel", "ticagrelor", "prasugrel"],
    "antidiabetic": ["insulin", "metformin", "glimepiride", "glipizide", "sitagliptin"],
    "cardiovascular": ["digoxin", "amiodarone", "metoprolol", "atenolol", "amlodipine", "ramipril"],
    "immunosuppressant": ["methotrexate", "cyclosporine", "tacrolimus", "prednisone"],
    "opioid": ["morphine", "fentanyl", "oxycodone", "tramadol", "codeine"]
}

# Clinical reference ranges for rapid lab categorization
LAB_REFERENCE_RANGES = {
    "hemoglobin": {"unit": "g/dL", "low": 12.0, "high": 17.5, "critical_low": 8.0, "critical_high": 20.0},
    "glucose": {"unit": "mg/dL", "low": 70.0, "high": 140.0, "critical_low": 50.0, "critical_high": 300.0},
    "fasting glucose": {"unit": "mg/dL", "low": 70.0, "high": 100.0, "critical_low": 50.0, "critical_high": 250.0},
    "hba1c": {"unit": "%", "low": 4.0, "high": 5.7, "critical_low": 3.0, "critical_high": 10.0},
    "creatinine": {"unit": "mg/dL", "low": 0.6, "high": 1.2, "critical_low": 0.3, "critical_high": 3.0},
    "egfr": {"unit": "mL/min", "low": 60.0, "high": 120.0, "critical_low": 15.0, "critical_high": 999.0},
    "platelets": {"unit": "K/uL", "low": 150.0, "high": 450.0, "critical_low": 50.0, "critical_high": 1000.0},
    "spo2": {"unit": "%", "low": 95.0, "high": 100.0, "critical_low": 90.0, "critical_high": 100.0}
}


def _evaluate_lab_value(name: str, raw_value: Any) -> Dict[str, Any]:
    """Classify a lab value into normal, low, high, critical_low, critical_high."""
    cleaned_name = name.strip().lower()
    ref_key = None
    for k in LAB_REFERENCE_RANGES:
        if k in cleaned_name:
            ref_key = k
            break

    num_val = None
    if isinstance(raw_value, (int, float)):
        num_val = float(raw_value)
    elif isinstance(raw_value, str):
        match = re.search(r"\d+(?:\.\d+)?", raw_value)
        if match:
            num_val = float(match.group(0))

    if num_val is None or not ref_key:
        return {"status": "recorded", "category": "normal", "reference": "N/A", "numeric_val": num_val}

    ref = LAB_REFERENCE_RANGES[ref_key]
    status = "normal"
    category = "normal"

    if num_val < ref["critical_low"]:
        status = "CRITICAL LOW"
        category = "critical"
    elif num_val > ref["critical_high"]:
        status = "CRITICAL HIGH"
        category = "critical"
    elif num_val < ref["low"]:
        status = "LOW"
        category = "warning"
    elif num_val > ref["high"]:
        status = "HIGH"
        category = "warning"

    return {
        "status": status,
        "category": category,
        "reference_range": f"{ref['low']} - {ref['high']} {ref['unit']}",
        "numeric_val": num_val
    }


def _load_profile_from_mongo() -> Dict[str, Any]:
    """Load verified profile directly from MongoDB."""
    profile = {"medications": [], "allergies": [], "conditions": [], "lab_values": [], "dates": []}
    try:
        from pymongo import MongoClient, ASCENDING
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "meditrace")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        rows = list(client[db_name].verified_facts.find({"status": "verified"}).sort("_id", ASCENDING))
        key_by_type = {
            "medication": "medications", "allergy": "allergies", "condition": "conditions",
            "lab_value": "lab_values", "date": "dates",
        }
        for row in rows:
            key = key_by_type.get(row.get("fact_type"))
            if not key:
                continue
            profile[key].append({
                "value": row.get("value"),
                "dosage": row.get("dosage"),
                "frequency": row.get("frequency"),
                "date": row.get("date"),
                "lab_value": row.get("lab_value"),
                "extraction_confidence": row.get("extraction_confidence", 1.0),
                "verification_confidence": row.get("verification_confidence", 1.0),
                "status": row.get("status"),
                "reason": row.get("reason"),
                "source_document": row.get("source_document")
            })
    except Exception as e:
        log.error(f"Failed to load profile from MongoDB: {e}")
    return profile


def generate_emergency_summary(
    patient_id: Optional[str] = None,
    verified_profile: Optional[Dict[str, Any]] = None,
    safety_report: Optional[Dict[str, Any]] = None,
    conflict_facts: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Main entry point for Member 4: Emergency Summary Agent.
    Generates a rapid ER triage view based on verified clinical facts.
    """
    if verified_profile is None or not any(verified_profile.values()):
        verified_profile = _load_profile_from_mongo()

    if safety_report is None:
        # Load sample report if available as fallback
        sample_path = os.path.join(BASE_DIR, "data", "sample_docs", "last_safety_report.json")
        if os.path.exists(sample_path):
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    safety_report = json.load(f)
            except Exception:
                safety_report = {}
        else:
            safety_report = {}

    safety_alerts = safety_report.get("safety_alerts", [])
    trend_alerts = safety_report.get("historical_trends", {}).get("trend_alerts", [])
    all_alerts = safety_alerts + trend_alerts

    # 1. Critical Red Flags & Resuscitation Cautions
    cautions = []
    triage_score = 0  # 0 = GREEN, 1-2 = YELLOW, >=3 = RED

    # Check confirmed allergies
    verified_allergies = []
    for item in verified_profile.get("allergies", []):
        val = item.get("value") if isinstance(item, dict) else str(item)
        if val and not val.lower().startswith(("no known", "no allerg", "none", "nka")):
            verified_allergies.append(val)
            cautions.append(f"ALLERGY CAUTION: Confirmed allergy to {val.upper()}. Avoid cross-reactive compounds.")
            triage_score += 2

    # Check high-risk medications
    active_meds = []
    has_anticoagulant = False
    has_antiplatelet = False
    for item in verified_profile.get("medications", []):
        name = (item.get("value") if isinstance(item, dict) else str(item)).strip()
        dose = item.get("dosage") if isinstance(item, dict) else ""
        freq = item.get("frequency") if isinstance(item, dict) else ""
        active_meds.append({
            "name": name,
            "dosage": dose or "As prescribed",
            "frequency": freq or "Regular",
            "confidence": item.get("verification_confidence", 0.9) if isinstance(item, dict) else 0.9,
            "source": item.get("source_document", "Verified Record") if isinstance(item, dict) else "Verified Record"
        })

        n_lower = name.lower()
        if any(drug in n_lower for drug in HIGH_RISK_MED_CLASSES["anticoagulant"]):
            has_anticoagulant = True
            cautions.append(f"BLEEDING RISK: Active anticoagulant therapy ({name}). Caution with invasive procedures or trauma.")
            triage_score += 2
        if any(drug in n_lower for drug in HIGH_RISK_MED_CLASSES["antiplatelet"]):
            has_antiplatelet = True

    if has_anticoagulant and has_antiplatelet:
        cautions.append("CRITICAL HEMORRHAGE RISK: Concurrent Anticoagulant + Antiplatelet therapy detected!")
        triage_score += 2

    # Check severe safety alerts from Member 3
    for alert in safety_alerts:
        sev = (alert.get("severity") or "").lower()
        title = alert.get("title", "")
        msg = alert.get("message", "")
        if sev in ("critical", "high"):
            cautions.append(f"SAFETY ALERT ({sev.upper()}): {title} - {msg}")
            triage_score += 2
        elif sev == "moderate":
            triage_score += 1

    # Check unresolved conflicts from Member 2
    if conflict_facts:
        for cf in conflict_facts:
            cautions.append(f"UNRESOLVED CLINICAL CONFLICT: {cf.get('fact_type', 'Fact')} '{cf.get('value')}' has conflicting source evidence. Manual review required.")
            triage_score += 1

    # Evaluate Labs
    evaluated_labs = []
    for lab in verified_profile.get("lab_values", []):
        val_name = lab.get("value", "")
        val_str = lab.get("lab_value") or lab.get("value", "")
        ev = _evaluate_lab_value(val_name, val_str)
        evaluated_labs.append({
            "test": val_name,
            "value": val_str,
            "status": ev["status"],
            "category": ev["category"],
            "reference_range": ev.get("reference_range", "N/A"),
            "source_doc": lab.get("source_document", "Lab Report")
        })
        if ev["category"] == "critical":
            cautions.append(f"CRITICAL LAB VALUE: {val_name} is {ev['status']} ({val_str}).")
            triage_score += 2
        elif ev["category"] == "warning":
            triage_score += 1

    # Conditions list
    conditions = []
    for cond in verified_profile.get("conditions", []):
        c_name = cond.get("value") if isinstance(cond, dict) else str(cond)
        if c_name:
            conditions.append(c_name)

    # Determine Triage Level
    if triage_score >= 3 or any(a.get("severity", "").lower() == "critical" for a in safety_alerts):
        triage_level = "RED"
        triage_description = "CRITICAL / HIGH RISK - Immediate medical evaluation required. Strict precautions on medications and allergies."
    elif triage_score >= 1 or any(a.get("severity", "").lower() == "high" for a in safety_alerts):
        triage_level = "YELLOW"
        triage_description = "URGENT - Significant alerts or polypharmacy detected. Review cautions prior to intervention."
    else:
        triage_level = "GREEN"
        triage_description = "STABLE - Verified records indicate no critical contraindications or life-threatening lab anomalies."

    # Generate 15-second Clinical Handover Narrative
    allergy_str = ", ".join(verified_allergies) if verified_allergies else "None documented"
    condition_str = ", ".join(conditions) if conditions else "No chronic conditions documented"
    med_count = len(active_meds)

    narrative_parts = [
        f"Patient presented with {len(conditions)} verified conditions ({condition_str}) and {med_count} active medications."
    ]
    if verified_allergies:
        narrative_parts.append(f"CRITICAL ALLERGIES: {allergy_str}.")
    if has_anticoagulant:
        narrative_parts.append("Patient is on active ANTICOAGULATION therapy (high bleeding hazard).")
    if evaluated_labs:
        abnormal_labs = [f"{l['test']} ({l['value']})" for l in evaluated_labs if l['category'] in ('warning', 'critical')]
        if abnormal_labs:
            narrative_parts.append(f"Abnormal lab findings: {', '.join(abnormal_labs)}.")
    if safety_alerts:
        high_alerts = [a.get("title") for a in safety_alerts if a.get("severity", "").lower() in ("critical", "high")]
        if high_alerts:
            narrative_parts.append(f"System safety alerts triggered: {'; '.join(high_alerts[:2])}.")

    clinical_handover = " ".join(narrative_parts)

    # Build response summary
    return {
        "patient_id": patient_id or "PT-" + datetime.utcnow().strftime("%Y%m%d-01"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "triage_level": triage_level,
        "triage_description": triage_description,
        "clinical_handover": clinical_handover,
        "resuscitation_cautions": cautions,
        "active_medications": active_meds,
        "verified_allergies": verified_allergies,
        "chronic_conditions": conditions,
        "recent_labs": evaluated_labs,
        "safety_summary": {
            "total_alerts": len(safety_alerts),
            "critical_or_high_alerts": len([a for a in safety_alerts if a.get("severity", "").lower() in ("critical", "high")]),
            "trend_alerts_count": len(trend_alerts)
        },
        "data_integrity": {
            "verified_facts_count": sum(len(v) for v in verified_profile.values() if isinstance(v, list)),
            "conflicts_flagged": len(conflict_facts or []),
            "verification_status": "Verified Profile Grounded in Source Documents"
        }
    }


if __name__ == "__main__":
    summary = generate_emergency_summary()
    print("MEMBER 4 EMERGENCY SUMMARY GENERATED:")
    print(json.dumps(summary, indent=2))
