from pathlib import Path
import sqlite3
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "pipeline.db"

# Common Q&A markers across transcript sources
QA_MARKERS = [
     # Classic labels
    r"question-and-answer session",
    r"question and answer session",
    r"questions and answers",
    r"q\s*&\s*a",
    r"q\s*and\s*a",
    r"q\s*-\s*a",

    # Variations you found
    r"move over to q\s*and\s*a",
    r"move over to q\s*&\s*a",
    r"move over to q\s*a",
    r"open the call to questions",
    r"let'?s open the call to questions",
    r"we'?ll now move over to q\s*and\s*a",
    r"we'?ll now take questions",
    r"we will now take questions",

    # Operator transitions often used
    r"operator[,:\s].{0,80}instructions",
    r"operator[,:\s].{0,80}first question",
    r"our first question comes from",
    r"head over to investor questions",
    r"we'?re going to head over to investor questions",
    r"we are going to head over to investor questions",
    r"we'?ll now head over to investor questions",
    r"we will now head over to investor questions",
    r"investor questions",
    r"questions from say\.com",



]

def init_clean_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cleaned_transcripts (
        transcript_id INTEGER PRIMARY KEY,
        cleaned_text TEXT NOT NULL,
        prepared_text TEXT,
        qa_text TEXT,
        qa_found INTEGER NOT NULL,
        FOREIGN KEY (transcript_id) REFERENCES raw_transcripts(id)
    );
    """)
    conn.commit()

def basic_clean(text: str) -> str:
    # normalize line endings and spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_prepared_vs_qa(text: str) -> tuple[str | None, str | None, int]:
    """
    Returns (prepared_text, qa_text, qa_found)
    qa_found = 1 if marker found else 0
    """
    lower = text.lower()

    # Find earliest occurrence of any Q&A marker
    earliest_idx = None
    for pat in QA_MARKERS:
        m = re.search(pat, lower)
        if m:
            idx = m.start()
            if earliest_idx is None or idx < earliest_idx:
                earliest_idx = idx

    if earliest_idx is None:
        return text, None, 0

    prepared = text[:earliest_idx].strip()
    qa = text[earliest_idx:].strip()
    return prepared if prepared else None, qa if qa else None, 1

def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        init_clean_table(conn)

        rows = conn.execute("SELECT id, raw_text FROM raw_transcripts;").fetchall()

        upserted = 0
        qa_count = 0

        for transcript_id, raw_text in rows:
            cleaned = basic_clean(raw_text)
            prepared, qa, qa_found = split_prepared_vs_qa(cleaned)

            conn.execute(
                """
                INSERT INTO cleaned_transcripts (transcript_id, cleaned_text, prepared_text, qa_text, qa_found)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(transcript_id) DO UPDATE SET
                    cleaned_text=excluded.cleaned_text,
                    prepared_text=excluded.prepared_text,
                    qa_text=excluded.qa_text,
                    qa_found=excluded.qa_found;
                """,
                (transcript_id, cleaned, prepared, qa, qa_found),
            )
            upserted += 1
            qa_count += qa_found

        conn.commit()

        total = conn.execute("SELECT COUNT(*) FROM cleaned_transcripts;").fetchone()[0]

    print("✅ Cleaning finished")
    print(f"Transcripts processed: {upserted}")
    print(f"Total rows in cleaned_transcripts: {total}")
    print(f"Q&A detected in: {qa_count} transcripts")

if __name__ == "__main__":
    main()