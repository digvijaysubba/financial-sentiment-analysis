from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "pipeline.db"
RETURNS_CSV = BASE_DIR / "outputs" / "results_with_returns.csv"

# Reuse the exact scoring logic from the pipeline so /analyze matches stored scores
sys.path.insert(0, str(BASE_DIR / "scripts"))
from sentiment_models import vader_score, textblob_score  # noqa: E402
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: E402

app = FastAPI(
    title="Financial Sentiment API",
    description="Serves earnings-call sentiment scores, tone shifts and post-earnings returns.",
    version="1.0.0",
)

_vader = SentimentIntensityAnalyzer()

Model = Literal["vader", "textblob"]
Section = Literal["prepared", "qa"]


# ── helpers ──────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(503, f"Database not found: {DB_PATH.name}. Run the pipeline first.")
    # Read-only: the API never mutates pipeline data
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: tuple = ()) -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def load_results() -> pd.DataFrame:
    if not RETURNS_CSV.exists():
        raise HTTPException(503, f"{RETURNS_CSV.name} not found. Run compute_returns.py first.")
    return pd.read_csv(RETURNS_CSV)


def records(df: pd.DataFrame) -> list[dict]:
    # NaN is not valid JSON; convert to None
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


# ── schemas ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Any passage of transcript text")


class AnalyzeResponse(BaseModel):
    vader: float
    textblob: float
    vader_details: dict
    textblob_details: dict


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "db": DB_PATH.exists(), "results": RETURNS_CSV.exists()}


@app.get("/companies")
def companies():
    rows = query(
        """
        SELECT company, COUNT(*) AS n_transcripts,
               MIN(report_date) AS first_report, MAX(report_date) AS last_report
        FROM raw_transcripts GROUP BY company ORDER BY company
        """
    )
    return rows


@app.get("/transcripts")
def transcripts(company: Optional[str] = None):
    sql = """
        SELECT r.id, r.company, r.report_date, r.source_file,
               c.qa_found,
               LENGTH(c.prepared_text) AS prepared_chars,
               LENGTH(c.qa_text) AS qa_chars
        FROM raw_transcripts r
        LEFT JOIN cleaned_transcripts c ON c.transcript_id = r.id
    """
    params: tuple = ()
    if company:
        sql += " WHERE r.company = ?"
        params = (company.upper(),)
    sql += " ORDER BY r.company, r.report_date"
    return query(sql, params)


@app.get("/transcripts/{transcript_id}")
def transcript(transcript_id: int, include_text: bool = False):
    rows = query(
        """
        SELECT r.id, r.company, r.report_date, r.source_file, c.qa_found,
               c.prepared_text, c.qa_text
        FROM raw_transcripts r
        LEFT JOIN cleaned_transcripts c ON c.transcript_id = r.id
        WHERE r.id = ?
        """,
        (transcript_id,),
    )
    if not rows:
        raise HTTPException(404, f"Transcript {transcript_id} not found")
    row = rows[0]
    row["sentiment"] = query(
        "SELECT section, model_name, sentiment_score FROM sentiment_results "
        "WHERE transcript_id = ? ORDER BY model_name, section",
        (transcript_id,),
    )
    if not include_text:
        row.pop("prepared_text")
        row.pop("qa_text")
    return row


@app.get("/sentiment")
def sentiment(
    company: Optional[str] = None,
    model: Optional[Model] = None,
    section: Optional[Section] = None,
):
    """Raw per-section scores, one row per transcript × section × model."""
    sql = """
        SELECT r.id AS transcript_id, r.company, r.report_date,
               s.section, s.model_name, s.sentiment_score
        FROM sentiment_results s
        JOIN raw_transcripts r ON r.id = s.transcript_id
        WHERE 1 = 1
    """
    params: list = []
    if company:
        sql += " AND r.company = ?"
        params.append(company.upper())
    if model:
        sql += " AND s.model_name = ?"
        params.append(model)
    if section:
        sql += " AND s.section = ?"
        params.append(section)
    sql += " ORDER BY r.company, r.report_date, s.model_name, s.section"
    return query(sql, tuple(params))


@app.get("/results")
def results(company: Optional[str] = None):
    """One row per earnings call: prepared/Q&A scores, tone shift and post-earnings returns."""
    df = load_results()
    if company:
        df = df[df["company"] == company.upper()]
        if df.empty:
            raise HTTPException(404, f"No results for {company}")
    return records(df.sort_values(["company", "report_date"]))


@app.get("/summary")
def summary():
    """Per-company averages of tone shift and returns."""
    df = load_results()
    cols = ["vader_shift", "textblob_shift", "gap_ret", "ret_1d", "ret_3d", "ret_5d"]
    out = df.groupby("company")[cols].mean().reset_index()
    out.insert(1, "n_calls", df.groupby("company").size().values)
    return records(out)


@app.get("/correlations")
def correlations(model: Model = "vader"):
    """Pearson r between tone shift and each return horizon (small sample — illustrative)."""
    df = load_results()
    shift = f"{model}_shift"
    out = {}
    for col in ["gap_ret", "ret_1d", "ret_3d", "ret_5d"]:
        pair = df[[shift, col]].dropna()
        out[col] = {
            "r": None if len(pair) < 3 else round(float(pair[shift].corr(pair[col])), 4),
            "n": int(len(pair)),
        }
    return {"model": model, "shift_column": shift, "correlations": out}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Score arbitrary text with the same VADER and TextBlob logic used by the pipeline."""
    v, v_extra = vader_score(req.text, _vader)
    t, t_extra = textblob_score(req.text)
    return AnalyzeResponse(
        vader=v, textblob=t,
        vader_details=json.loads(v_extra), textblob_details=json.loads(t_extra),
    )
