"""
Seed test data for MediTrace (no manual uploads needed).

What it does:
  1. Creates 3 sample medical PDFs (different dates) in data/sample_docs/
  2. Runs each through Member 1 (process_document) and Member 2 (verify_candidate_facts)
  3. Prints Member 3's safety + trend report

Run from the repo root (same folder as main.py), with the venv active:
    python seed_test_data.py            # add test data and print the report
    python seed_test_data.py --reset    # first clear the 3 Mongo collections (asks for confirmation)

WARNING: this writes into the real MongoDB database used by main.py.
"""
import json
import os
import sys

import fitz  # PyMuPDF (already used by main.py)

import main  # loads the app, spaCy and MongoDB config; does not start the server
from agents.safety_trend_agent import run_safety_and_trend_checks

SAMPLE_DIR = os.path.join(main.BASE_DIR, "data", "sample_docs")

# Three reports on different dates -> gives Member 3 something to trend on.
SAMPLE_DOCS = {
    "report_2024_01.pdf": [
        "City Hospital - Lab Report",
        "Date: 15/01/2024",
        "Known case of Diabetes",
        "Metformin 500 mg BID",
        "Hemoglobin: 13.0 g/dl",
        "Fasting Glucose: 120 mg/dl",
    ],
    "report_2024_07.pdf": [
        "City Hospital - Lab Report",
        "Date: 12 July 2024",
        "Hemoglobin: 11.2 g/dl",
        "Fasting Glucose: 160 mg/dl",
    ],
    "prescription_2025_01.pdf": [
        "City Hospital - Prescription",
        "Date: 10/01/2025",
        "Allergic to Penicillin",
        "Warfarin 5 mg OD",
        "Aspirin 75 mg once daily",
        "Crocin 1000 mg thrice daily",
        "Dolo 650 mg TID",
        "Augmentin 625 mg BID",
        "Hemoglobin: 9.4 g/dl",
        "Fasting Glucose: 210 mg/dl",
    ],
}


def make_pdf(path, lines):
    """Write a simple text-based PDF (selectable text, so no OCR is needed)."""
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=12)
        y += 22
    doc.save(path)
    doc.close()


def reset_collections():
    db = main.get_db()
    answer = input(f"This DELETES candidate_facts, verified_facts and verification_logs "
                   f"in database '{main.MONGODB_DB}'. Type 'yes' to continue: ")
    if answer.strip().lower() != "yes":
        print("Reset cancelled.")
        return
    for name in ("candidate_facts", "verified_facts", "verification_logs"):
        db[name].delete_many({})
    print("Collections cleared.\n")


def main_flow():
    if "--reset" in sys.argv:
        reset_collections()

    os.makedirs(SAMPLE_DIR, exist_ok=True)

    for name, lines in SAMPLE_DOCS.items():  # dict keeps insertion (chronological) order
        path = os.path.join(SAMPLE_DIR, name)
        make_pdf(path, lines)

        result = main.process_document(path, original_filename=name)
        outcome = main.verify_candidate_facts(result.candidate_facts)

        print(f"{name}: {len(result.candidate_facts)} candidate facts -> "
              f"{len(outcome['verified_facts'])} verified, "
              f"{len(outcome['rejected_facts'])} rejected, "
              f"{len(outcome['conflicts'])} conflicts")

    print("\n" + "=" * 60)
    print("MEMBER 3 REPORT")
    print("=" * 60)
    report = run_safety_and_trend_checks()
    print(f"Status: {report['status']} | Risk: {report['risk_level']} | "
          f"Facts used: {report['facts_used']}\n")

    for a in report["safety_alerts"] + report["historical_trends"]["trend_alerts"]:
        print(f"[{a['severity']:8}] ({a['category']}) {a['title']}")
        print(f"           {a['message']}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")

    out_path = os.path.join(SAMPLE_DIR, "last_safety_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull JSON saved to: {out_path}")


if __name__ == "__main__":
    main_flow()