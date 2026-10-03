import os
import sqlite3
import pandas as pd


DB_PATH = os.path.join("db", "pipeline.db")
OUT_DIR = "outputs"
OUT_CSV = os.path.join(OUT_DIR, "results.csv")


def main():
    # 1) Ensure outputs/ exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # 2) Connect to SQLite
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # 3) Pull all sentiment rows
    # Assumptions about sentiment_results columns:
    # - transcript_id
    # - model_name  (e.g., 'vader', 'textblob')
    # - section     (e.g., 'prepared', 'qa')
    # - score       (numeric)
    query = """
    SELECT
    transcript_id,
    model_name,
    section,
    sentiment_score AS score
    FROM sentiment_results
    """
    df = pd.read_sql_query(query, conn)

    # 4) Also pull transcript metadata (company/date) from raw_transcripts
    # Assumptions about raw_transcripts columns:
    # - id (primary key)
    # - company
    # - report_date
    meta_query = """
    SELECT
        id AS transcript_id,
        company,
        report_date
    FROM raw_transcripts
    """
    meta = pd.read_sql_query(meta_query, conn)

    conn.close()

    # 5) Pivot: one row per transcript_id, columns per (model_name, section)
    # This gives columns like ('vader','prepared'), ('vader','qa'), ...
    pivot = df.pivot_table(
        index="transcript_id",
        columns=["model_name", "section"],
        values="score",
        aggfunc="mean"
    )

    # Flatten multi-index columns into single names like vader_prepared
    pivot.columns = [f"{m}_{s}" for (m, s) in pivot.columns]
    pivot = pivot.reset_index()

    # 6) Merge metadata (company, date)
    out = meta.merge(pivot, on="transcript_id", how="left")

    # 7) Create “shift” columns (qa - prepared) for each model if both exist
    for model in ["vader", "textblob"]:
        prepared_col = f"{model}_prepared"
        qa_col = f"{model}_qa"
        shift_col = f"{model}_shift"
        if prepared_col in out.columns and qa_col in out.columns:
            out[shift_col] = out[qa_col] - out[prepared_col]

    # 8) Optional: nice column order (if present)
    preferred_order = [
        "company", "report_date",
        "vader_prepared", "vader_qa", "vader_shift",
        "textblob_prepared", "textblob_qa", "textblob_shift",
        "transcript_id"
    ]
    cols = [c for c in preferred_order if c in out.columns] + \
           [c for c in out.columns if c not in preferred_order]
    out = out[cols]

    # 9) Save CSV
    out.to_csv(OUT_CSV, index=False)

    print(f"✅ Saved: {OUT_CSV}")
    print(f"Rows: {len(out)}")
    print("Preview:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()