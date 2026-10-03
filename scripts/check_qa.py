from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "pipeline.db"

with sqlite3.connect(DB_PATH) as conn:
    rows = conn.execute("""
        SELECT r.company, r.report_date, c.qa_found
        FROM raw_transcripts r
        JOIN cleaned_transcripts c ON c.transcript_id = r.id
        ORDER BY r.company;
    """).fetchall()

for company, report_date, qa_found in rows:
    print(f"{company} {report_date}  qa_found={qa_found}")