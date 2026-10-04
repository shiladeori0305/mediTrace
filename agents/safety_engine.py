"""
Member 3 - Safety + Trend Agents
Input : Verified Patient Profile (from Member 2)
Output: Safety & Trend Alerts (to Member 4: backend/dashboard)

Checks:
  1. Drug-drug interactions
  2. Duplicate medicines (same drug / same class)
  3. Dosage conflicts (max daily dose)
  4. Allergy conflicts (direct + class + cross-reactivity)
  5. Drug-condition conflicts
  6. Chronological medical timeline
  7. Lab/vitals trend + deterioration detection

Install: pip install pydantic
"""
from __future__ import annotations

from datetime import date
from itertools import combinations
from typing import List, Optional

from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# 1. INPUT SCHEMA
# ----------------------------------------------------------------------------


class Medication(BaseModel):
    name: str
    dose_mg: Optional[float] = None            # single dose in mg
    frequency_per_day: Optional[float] = None  # times per day
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: str = "active"                     # active / stopped


class Condition(BaseModel):
    name: str
    diagnosed_on: Optional[date] = None


class LabResult(BaseModel):
    test: str      # e.g. "hba1c", "egfr", "hemoglobin", "systolic_bp"
    value: float
    unit: str = ""
    date: date


class PatientProfile(BaseModel):
    patient_id: str
    medications: List[Medication] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    conditions: List[Condition] = Field(default_factory=list)
    lab_results: List[LabResult] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# 2. OUTPUT SCHEMA
# ----------------------------------------------------------------------------

SEVERITY_ORDER = {"low": 1, "moderate": 2, "high": 3, "critical": 4}


class Alert(BaseModel):
    category: str   # interaction | duplicate | dosage | allergy | condition | trend
    severity: str   # low | moderate | high | critical
    title: str
    message: str
    evidence: List[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    date: Optional[date]
    type: str       # medication_start | medication_stop | condition | lab
    description: str


class SafetyTrendReport(BaseModel):
    patient_id: str
    risk_level: str
    alerts: List[Alert]
    timeline: List[TimelineEvent]


# ----------------------------------------------------------------------------
# 3. KNOWLEDGE BASE  (small demo set; replace with RxNorm/DrugBank later)
# ----------------------------------------------------------------------------

BRAND_TO_GENERIC = {
    "crocin": "paracetamol", "dolo": "paracetamol", "calpol": "paracetamol",
    "acetaminophen": "paracetamol",
    "brufen": "ibuprofen", "combiflam": "ibuprofen",
    "ecosprin": "aspirin", "disprin": "aspirin",
    "glycomet": "metformin", "glucophage": "metformin",
    "voveran": "diclofenac",
    "mox": "amoxicillin", "novamox": "amoxicillin",
    "septran": "cotrimoxazole", "bactrim": "cotrimoxazole",
    "lipitor": "atorvastatin", "storvas": "atorvastatin",
    "coumadin": "warfarin",
}

DRUG_CLASS = {
    "amoxicillin": "penicillin", "ampicillin": "penicillin", "penicillin": "penicillin",
    "cefixime": "cephalosporin", "cephalexin": "cephalosporin", "ceftriaxone": "cephalosporin",
    "ibuprofen": "nsaid", "diclofenac": "nsaid", "naproxen": "nsaid", "aspirin": "nsaid",
    "cotrimoxazole": "sulfa", "sulfamethoxazole": "sulfa",
    "atorvastatin": "statin", "simvastatin": "statin", "rosuvastatin": "statin",
    "sertraline": "ssri", "fluoxetine": "ssri",
}

# (drugA, drugB) -> (severity, explanation)
INTERACTIONS = {
    frozenset({"warfarin", "aspirin"}): ("high", "Bleeding risk bahut badh jata hai."),
    frozenset({"warfarin", "ibuprofen"}): ("high", "Bleeding risk badhta hai (NSAID + anticoagulant)."),
    frozenset({"warfarin", "diclofenac"}): ("high", "Bleeding risk badhta hai (NSAID + anticoagulant)."),
    frozenset({"aspirin", "ibuprofen"}): ("moderate", "Ibuprofen aspirin ka antiplatelet effect kam kar sakta hai; GI bleeding risk."),
    frozenset({"sildenafil", "nitroglycerin"}): ("critical", "Severe hypotension - combination contraindicated."),
    frozenset({"lisinopril", "spironolactone"}): ("high", "Hyperkalemia (high potassium) ka risk."),
    frozenset({"simvastatin", "clarithromycin"}): ("high", "Statin level badhta hai -> myopathy/rhabdomyolysis risk."),
    frozenset({"atorvastatin", "clarithromycin"}): ("high", "Statin level badhta hai -> myopathy risk."),
    frozenset({"tramadol", "sertraline"}): ("high", "Serotonin syndrome + seizure risk."),
    frozenset({"clopidogrel", "omeprazole"}): ("moderate", "Omeprazole clopidogrel ka effect kam kar sakta hai."),
    frozenset({"metformin", "alcohol"}): ("moderate", "Lactic acidosis risk."),
}

# Adult max daily dose in mg
MAX_DAILY_DOSE_MG = {
    "paracetamol": 4000, "ibuprofen": 3200, "aspirin": 4000, "diclofenac": 150,
    "metformin": 3000, "amoxicillin": 3000, "atorvastatin": 80,
}

# Allergy cross-reactivity: allergy class -> related classes (lower certainty)
CROSS_REACTIVE = {"penicillin": ["cephalosporin"]}

# drug/class -> conditions where it's risky
CONDITION_CONFLICTS = {
    "nsaid": [("kidney", "high", "NSAIDs kidney disease ko worsen kar sakte hain."),
              ("ulcer", "high", "NSAIDs GI bleeding/ulcer risk badhate hain."),
              ("gastritis", "moderate", "NSAIDs gastritis worsen kar sakte hain.")],
    "metformin": [("kidney", "high", "Reduced kidney function me metformin se lactic acidosis ka risk."),
                  ("liver", "moderate", "Liver disease me metformin careful use hona chahiye."),
                  ("hepatitis", "moderate", "Hepatitis me metformin careful use hona chahiye.")],
}

# Lab trend rules: 'worse' = the direction in which a change is bad
LAB_RULES = {
    "hba1c":         {"worse": "up",   "critical": 9.0,  "label": "HbA1c"},
    "glucose":       {"worse": "up",   "critical": 250,  "label": "Blood glucose"},
    "creatinine":    {"worse": "up",   "critical": 2.0,  "label": "Creatinine"},
    "egfr":          {"worse": "down", "critical": 30,   "label": "eGFR"},
    "hemoglobin":    {"worse": "down", "critical": 7.0,  "label": "Hemoglobin"},
    "systolic_bp":   {"worse": "up",   "critical": 180,  "label": "Systolic BP"},
    "spo2":          {"worse": "down", "critical": 90,   "label": "SpO2"},
    "platelets":     {"worse": "down", "critical": 50,   "label": "Platelets"},
}

# ----------------------------------------------------------------------------
# 4. HELPERS
# ----------------------------------------------------------------------------


def normalize(name: str) -> str:
    n = name.strip().lower()
    return BRAND_TO_GENERIC.get(n, n)


def is_active(m: Medication, today: Optional[date] = None) -> bool:
    today = today or date.today()
    if m.status.lower() in ("stopped", "discontinued"):
        return False
    return m.end_date is None or m.end_date >= today


def max_severity(alerts: List[Alert]) -> str:
    if not alerts:
        return "low"
    return max(alerts, key=lambda a: SEVERITY_ORDER[a.severity]).severity


# ----------------------------------------------------------------------------
# 5. SAFETY AGENT
# ----------------------------------------------------------------------------


class SafetyAgent:
    def run(self, profile: PatientProfile) -> List[Alert]:
        active = [m for m in profile.medications if is_active(m)]
        alerts: List[Alert] = []
        alerts += self.check_interactions(active)
        alerts += self.check_duplicates(active)
        alerts += self.check_dosage(active)
        alerts += self.check_allergies(active, profile.allergies)
        alerts += self.check_conditions(active, profile.conditions)
        return alerts

    def check_interactions(self, meds: List[Medication]) -> List[Alert]:
        out = []
        names = sorted({normalize(m.name) for m in meds})
        for a, b in combinations(names, 2):
            hit = INTERACTIONS.get(frozenset({a, b}))
            if hit:
                sev, msg = hit
                out.append(Alert(category="interaction", severity=sev,
                                 title=f"Interaction: {a} + {b}", message=msg,
                                 evidence=[a, b]))
        return out

    def check_duplicates(self, meds: List[Medication]) -> List[Alert]:
        out = []
        # same generic
        seen = {}
        for m in meds:
            seen.setdefault(normalize(m.name), []).append(m.name)
        for generic, raw_names in seen.items():
            if len(raw_names) > 1:
                out.append(Alert(category="duplicate", severity="high",
                                 title=f"Duplicate medicine: {generic}",
                                 message=f"'{generic}' ek se zyada baar prescribed hai "
                                         f"(naam: {', '.join(raw_names)}).",
                                 evidence=raw_names))
        # same class, different drugs (e.g. 2 NSAIDs)
        by_class = {}
        for g in seen:
            cls = DRUG_CLASS.get(g)
            if cls:
                by_class.setdefault(cls, []).append(g)
        for cls, drugs in by_class.items():
            if len(drugs) > 1:
                out.append(Alert(category="duplicate", severity="moderate",
                                 title=f"Same-class therapy: {cls}",
                                 message=f"Ek hi class ({cls}) ki multiple dawaiyan: {', '.join(drugs)}.",
                                 evidence=drugs))
        return out

    def check_dosage(self, meds: List[Medication]) -> List[Alert]:
        out = []
        totals = {}
        for m in meds:
            if m.dose_mg and m.frequency_per_day:
                g = normalize(m.name)
                totals[g] = totals.get(g, 0) + m.dose_mg * m.frequency_per_day
        for g, total in totals.items():
            limit = MAX_DAILY_DOSE_MG.get(g)
            if limit and total > limit:
                sev = "critical" if total > 1.5 * limit else "high"
                out.append(Alert(category="dosage", severity=sev,
                                 title=f"Overdose risk: {g}",
                                 message=f"Total daily dose {total:.0f} mg hai, max safe limit {limit} mg/day.",
                                 evidence=[f"{total:.0f} mg/day", f"limit {limit} mg/day"]))
        return out

    def check_allergies(self, meds: List[Medication], allergies: List[str]) -> List[Alert]:
        out = []
        allergy_set = {normalize(a) for a in allergies}
        for m in meds:
            g = normalize(m.name)
            cls = DRUG_CLASS.get(g)
            # direct or class match
            if g in allergy_set or (cls and cls in allergy_set):
                out.append(Alert(category="allergy", severity="critical",
                                 title=f"Allergy conflict: {g}",
                                 message=f"Patient ko '{', '.join(allergies)}' se allergy hai, "
                                         f"lekin {g} prescribed hai.",
                                 evidence=[g] + ([cls] if cls else [])))
                continue
            # cross-reactivity
            for allergy_cls, related in CROSS_REACTIVE.items():
                if allergy_cls in allergy_set and cls in related:
                    out.append(Alert(category="allergy", severity="moderate",
                                     title=f"Possible cross-reactivity: {g}",
                                     message=f"{allergy_cls} allergy ke saath {cls} me cross-reaction ho sakta hai.",
                                     evidence=[g, allergy_cls]))
        return out

    def check_conditions(self, meds: List[Medication], conditions: List[Condition]) -> List[Alert]:
        out = []
        cond_names = [c.name.lower() for c in conditions]
        for m in meds:
            g = normalize(m.name)
            for key in {g, DRUG_CLASS.get(g)}:
                for keyword, sev, msg in CONDITION_CONFLICTS.get(key, []):
                    matched = [c for c in cond_names if keyword in c]
                    if matched:
                        out.append(Alert(category="condition", severity=sev,
                                         title=f"{g} vs condition: {matched[0]}",
                                         message=msg, evidence=[g, matched[0]]))
        return out


# ----------------------------------------------------------------------------
# 6. TREND AGENT
# ----------------------------------------------------------------------------


class TrendAgent:
    def run(self, profile: PatientProfile):
        return self.detect_deterioration(profile.lab_results), self.build_timeline(profile)

    def build_timeline(self, profile: PatientProfile) -> List[TimelineEvent]:
        ev: List[TimelineEvent] = []
        for m in profile.medications:
            if m.start_date:
                dose = f" {m.dose_mg:g}mg x{m.frequency_per_day:g}/day" if m.dose_mg and m.frequency_per_day else ""
                ev.append(TimelineEvent(date=m.start_date, type="medication_start",
                                        description=f"Started {m.name}{dose}"))
            if m.end_date:
                ev.append(TimelineEvent(date=m.end_date, type="medication_stop",
                                        description=f"Stopped {m.name}"))
        for c in profile.conditions:
            ev.append(TimelineEvent(date=c.diagnosed_on, type="condition",
                                    description=f"Diagnosed: {c.name}"))
        for r in profile.lab_results:
            ev.append(TimelineEvent(date=r.date, type="lab",
                                    description=f"{r.test}: {r.value} {r.unit}".strip()))
        # undated events go last
        ev.sort(key=lambda e: (e.date is None, e.date or date.max))
        return ev

    def detect_deterioration(self, labs: List[LabResult]) -> List[Alert]:
        out = []
        by_test = {}
        for r in labs:
            by_test.setdefault(r.test.strip().lower(), []).append(r)

        for test, readings in by_test.items():
            rule = LAB_RULES.get(test)
            if not rule or len(readings) < 2:
                continue
            readings.sort(key=lambda r: r.date)
            vals = [r.value for r in readings]
            first, last = vals[0], vals[-1]
            sign = 1 if rule["worse"] == "up" else -1

            # consecutive worsening steps
            worsening_steps = sum(1 for a, b in zip(vals, vals[1:]) if (b - a) * sign > 0)
            monotonic = worsening_steps == len(vals) - 1 and len(vals) >= 3
            pct = ((last - first) / first * 100) if first else 0
            worsened_pct = pct * sign  # +ve => worse

            crossed = (last >= rule["critical"]) if rule["worse"] == "up" else (last <= rule["critical"])
            label = rule["label"]
            series = " -> ".join(f"{v:g}" for v in vals)

            if crossed:
                out.append(Alert(category="trend", severity="high",
                                 title=f"{label} critical level par",
                                 message=f"{label} ab {last:g} hai (critical threshold {rule['critical']}). Trend: {series}",
                                 evidence=[f"{r.date}: {r.value:g}" for r in readings]))
            elif monotonic or worsened_pct >= 20:
                sev = "high" if worsened_pct >= 30 else "moderate"
                out.append(Alert(category="trend", severity=sev,
                                 title=f"{label} deteriorating",
                                 message=f"{label} lagataar kharab ho raha hai ({pct:+.1f}% change). Trend: {series}",
                                 evidence=[f"{r.date}: {r.value:g}" for r in readings]))
        return out


# ----------------------------------------------------------------------------
# 7. ORCHESTRATOR  <- Member 4 only needs to call this function
# ----------------------------------------------------------------------------


def analyze_patient(profile: PatientProfile) -> SafetyTrendReport:
    safety_alerts = SafetyAgent().run(profile)
    trend_alerts, timeline = TrendAgent().run(profile)
    alerts = sorted(safety_alerts + trend_alerts,
                    key=lambda a: SEVERITY_ORDER[a.severity], reverse=True)
    return SafetyTrendReport(patient_id=profile.patient_id,
                             risk_level=max_severity(alerts),
                             alerts=alerts, timeline=timeline)


# ----------------------------------------------------------------------------
# DEMO:  python -m agents.safety_engine
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    demo = PatientProfile(
        patient_id="P001",
        allergies=["penicillin"],
        conditions=[Condition(name="Chronic kidney disease", diagnosed_on=date(2022, 3, 1)),
                    Condition(name="Type 2 diabetes", diagnosed_on=date(2019, 6, 10))],
        medications=[
            Medication(name="Warfarin", dose_mg=5, frequency_per_day=1, start_date=date(2023, 1, 5)),
            Medication(name="Ecosprin", dose_mg=75, frequency_per_day=1, start_date=date(2023, 1, 5)),
            Medication(name="Crocin", dose_mg=1000, frequency_per_day=3, start_date=date(2024, 5, 1)),
            Medication(name="Dolo", dose_mg=650, frequency_per_day=3, start_date=date(2024, 5, 2)),
            Medication(name="Amoxicillin", dose_mg=500, frequency_per_day=3, start_date=date(2024, 6, 1)),
            Medication(name="Ibuprofen", dose_mg=400, frequency_per_day=3, start_date=date(2024, 6, 1)),
            Medication(name="Metformin", dose_mg=500, frequency_per_day=2, start_date=date(2019, 7, 1)),
        ],
        lab_results=[
            LabResult(test="hba1c", value=6.8, unit="%", date=date(2023, 1, 10)),
            LabResult(test="hba1c", value=7.6, unit="%", date=date(2023, 7, 12)),
            LabResult(test="hba1c", value=8.4, unit="%", date=date(2024, 1, 15)),
            LabResult(test="egfr", value=62, unit="mL/min", date=date(2023, 1, 10)),
            LabResult(test="egfr", value=48, unit="mL/min", date=date(2024, 1, 15)),
            LabResult(test="egfr", value=34, unit="mL/min", date=date(2024, 9, 20)),
        ],
    )
    report = analyze_patient(demo)
    print(f"Patient {report.patient_id} | Risk: {report.risk_level.upper()}\n")
    for a in report.alerts:
        print(f"[{a.severity.upper():8}] ({a.category}) {a.title}\n           {a.message}")
    print("\nTimeline:")
    for e in report.timeline:
        print(f"  {e.date}  {e.type:17} {e.description}")