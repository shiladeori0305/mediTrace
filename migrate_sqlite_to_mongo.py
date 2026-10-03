"""One-time: copy existing data/meditrace.db (SQLite) into MongoDB.
Run:  python migrate_sqlite_to_mongo.py
Safe to skip if you want to start with an empty database."""
import json, os, sqlite3
from dotenv import load_dotenv
from pymongo import MongoClient

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))
db = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))[os.getenv("MONGODB_DB", "meditrace")]
src = sqlite3.connect(os.path.join(BASE, "data", "meditrace.db")); src.row_factory = sqlite3.Row

for table in ("candidate_facts", "verified_facts", "verification_logs"):
    rows = []
    for r in src.execute(f"SELECT * FROM {table}"):
        d = dict(r); d.pop("id")
        if table == "verified_facts":
            d["source_references"] = json.loads(d["source_references"] or "[]")
        if table == "verification_logs" and d.get("compared_with"):
            d["compared_with"] = json.loads(d["compared_with"])
        rows.append(d)
    if rows and db[table].count_documents({}) == 0:
        db[table].insert_many(rows)
    print(f"{table}: {len(rows)} rows -> MongoDB ({db[table].count_documents({})} now in collection)")
