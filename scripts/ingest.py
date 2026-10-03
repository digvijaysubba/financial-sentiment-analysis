from pathlib import Path
import sqlite3
import re

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "db" / "pipeline.db"

FILENAME_PATTERN = re.compile(r"^(?P<company>[A-Z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})\.txt$")

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS raw_transcripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT NOT NULL,
        report_date TEXT NOT NULL,
        raw_text TEXT NOT NULL,
        source_file TEXT NOT NULL,
        UNIQUE(company, report_date, source_file)
    );
    """)
    conn.commit()

def parse_filename(file_path: Path) -> tuple[str, str]:
    m = FILENAME_PATTERN.match(file_path.name)
    if not m:
        raise ValueError(
            f"Bad filename: {file_path.name}\n"
            f"Expected: COMPANY_YYYY-MM-DD.txt (e.g., MSFT_2024-01-24.txt)"
        )
    return m.group("company"), m.group("date")

def ingest_file(conn: sqlite3.Connection, file_path: Path) -> bool:
    company, report_date = parse_filename(file_path)
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

    if len(raw_text) < 200:
        print(f"Skipping (too short): {file_path.name}")
        return False

    conn.execute(
        """
        INSERT OR IGNORE INTO raw_transcripts (company, report_date, raw_text, source_file)
        VALUES (?, ?, ?, ?)
        """,
        (company, report_date, raw_text, file_path.name),
    )
    conn.commit()
    return True

def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(RAW_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {RAW_DIR}")

    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)

        inserted = 0
        for fp in txt_files:
            try:
                if ingest_file(conn, fp):
                    inserted += 1
            except ValueError as e:
                print(e)

        total = conn.execute("SELECT COUNT(*) FROM raw_transcripts;").fetchone()[0]

    print("✅ Ingestion finished")
    print(f"Files processed: {len(txt_files)}")
    print(f"Inserted this run: {inserted}")
    print(f"Total rows in raw_transcripts: {total}")
    print(f"Database: {DB_PATH}")

if __name__ == "__main__":
    main()