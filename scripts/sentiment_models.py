from __future__ import annotations

from pathlib import Path
import sqlite3
import re
import json
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "pipeline.db"


def init_sentiment_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER NOT NULL,
            section TEXT NOT NULL,           -- 'prepared' or 'qa'
            model_name TEXT NOT NULL,        -- 'vader' or 'textblob'
            sentiment_score REAL NOT NULL,
            extra_json TEXT,                 -- JSON string with extra metrics
            created_at TEXT NOT NULL,
            FOREIGN KEY (transcript_id) REFERENCES raw_transcripts(id),
            UNIQUE(transcript_id, section, model_name)
        );
        """
    )
    conn.commit()


def _split_sentences(text: str) -> list[str]:
    """
    Simple transcript-friendly sentence splitter.
    Filters out tiny fragments (speaker tags, etc.).
    """
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return [p.strip() for p in parts if p and len(p.strip()) >= 25]


def vader_score(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[float, str]:
    """
    Document-level VADER:
    - score sentence-by-sentence
    - store mean compound as sentiment_score
    - store additional useful diagnostics in extra_json
    """
    sents = _split_sentences(text)

    # Fallback: if we couldn't split well, score whole text
    if not sents:
        scores = analyzer.polarity_scores(text)
        compound = float(scores["compound"])
        extra = {
            "mode": "whole_text",
            "compound": compound,
            "pos": float(scores["pos"]),
            "neg": float(scores["neg"]),
            "neu": float(scores["neu"]),
            "n_sentences": 0,
            "pct_negative": None,
        }
        return compound, json.dumps(extra)

    comps: list[float] = []
    pos_vals: list[float] = []
    neg_vals: list[float] = []
    neu_vals: list[float] = []

    for s in sents:
        sc = analyzer.polarity_scores(s)
        comps.append(float(sc["compound"]))
        pos_vals.append(float(sc["pos"]))
        neg_vals.append(float(sc["neg"]))
        neu_vals.append(float(sc["neu"]))

    mean_compound = sum(comps) / len(comps)
    mean_pos = sum(pos_vals) / len(pos_vals)
    mean_neg = sum(neg_vals) / len(neg_vals)
    mean_neu = sum(neu_vals) / len(neu_vals)

    pct_negative = sum(1 for x in comps if x < 0) / len(comps)

    comps_sorted = sorted(comps)
    mid = len(comps_sorted) // 2
    median_compound = (
        comps_sorted[mid]
        if len(comps_sorted) % 2 == 1
        else (comps_sorted[mid - 1] + comps_sorted[mid]) / 2
    )

    extra = {
        "mode": "sentence_mean",
        "compound_mean": mean_compound,
        "compound_median": median_compound,
        "pos_mean": mean_pos,
        "neg_mean": mean_neg,
        "neu_mean": mean_neu,
        "pct_negative": pct_negative,
        "n_sentences": len(comps),
    }

    return float(mean_compound), json.dumps(extra)


def textblob_score(text: str) -> tuple[float, str]:
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)         # [-1, 1]
    subjectivity = float(blob.sentiment.subjectivity) # [0, 1]
    extra = {"polarity": polarity, "subjectivity": subjectivity}
    return polarity, json.dumps(extra)


def upsert_result(
    conn: sqlite3.Connection,
    transcript_id: int,
    section: str,
    model_name: str,
    score: float,
    extra_json: str,
) -> None:
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO sentiment_results
            (transcript_id, section, model_name, sentiment_score, extra_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(transcript_id, section, model_name) DO UPDATE SET
            sentiment_score=excluded.sentiment_score,
            extra_json=excluded.extra_json,
            created_at=excluded.created_at;
        """,
        (transcript_id, section, model_name, score, extra_json, now),
    )


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    vader_analyzer = SentimentIntensityAnalyzer()

    with sqlite3.connect(DB_PATH) as conn:
        init_sentiment_table(conn)

        rows = conn.execute(
            """
            SELECT transcript_id, prepared_text, qa_text
            FROM cleaned_transcripts;
            """
        ).fetchall()

        processed = 0

        for transcript_id, prepared_text, qa_text in rows:
            # prepared
            if prepared_text and prepared_text.strip():
                v_score, v_extra = vader_score(prepared_text, vader_analyzer)
                upsert_result(conn, transcript_id, "prepared", "vader", v_score, v_extra)

                t_score, t_extra = textblob_score(prepared_text)
                upsert_result(conn, transcript_id, "prepared", "textblob", t_score, t_extra)

            # qa
            if qa_text and qa_text.strip():
                v_score, v_extra = vader_score(qa_text, vader_analyzer)
                upsert_result(conn, transcript_id, "qa", "vader", v_score, v_extra)

                t_score, t_extra = textblob_score(qa_text)
                upsert_result(conn, transcript_id, "qa", "textblob", t_score, t_extra)

            processed += 1

        conn.commit()

        total = conn.execute("SELECT COUNT(*) FROM sentiment_results;").fetchone()[0]

    print("✅ Sentiment scoring finished")
    print(f"Transcripts processed: {processed}")
    print(f"Total rows in sentiment_results: {total}")
    print("Expected: 5 transcripts × 2 sections × 2 models = 20 rows")


if __name__ == "__main__":
    main()