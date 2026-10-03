import os
import pandas as pd

IN_CSV  = os.path.join("outputs", "results_with_returns.csv")
OUT_CSV = os.path.join("outputs", "financial_sentiment_powerbi.csv")

# GICS sector classifications for each ticker
SECTORS = {
    "AAPL":  "Technology",
    "GOOGL": "Communication Services",
    "MSFT":  "Technology",
    "NVDA":  "Technology",
    "TSLA":  "Consumer Discretionary",
}


def quarter_label(date_str):
    """'2026-01-29'  ->  'Q1 2026'"""
    dt = pd.to_datetime(date_str)
    q = (dt.month - 1) // 3 + 1
    return f"Q{q} {dt.year}"


def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Run compute_returns.py first: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # ── derived / renamed columns ─────────────────────────────────────────────

    # date: keep YYYY-MM-DD, just rename
    df = df.rename(columns={"report_date": "date"})

    # quarter
    df["quarter"] = df["date"].apply(quarter_label)

    # sector
    df["sector"] = df["company"].map(SECTORS)

    # sentiment_score: average of VADER and TextBlob shifts, already -1..1 scale
    df["sentiment_score"] = (df["vader_shift"] + df["textblob_shift"]) / 2

    # return_5d: rename from ret_5d (decimal, e.g. 0.068 = 6.8%)
    df = df.rename(columns={
        "ret_1d": "return_1d",
        "ret_3d": "return_3d",
        "ret_5d": "return_5d",
    })

    # ── column order ──────────────────────────────────────────────────────────
    cols = [
        "company", "date", "quarter", "sector",
        "sentiment_score",
        "vader_shift", "textblob_shift",
        "gap_ret", "return_1d", "return_3d", "return_5d",
        "vader_prepared", "vader_qa",
        "textblob_prepared", "textblob_qa",
        "transcript_id",
    ]
    df = df[[c for c in cols if c in df.columns]]

    df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Rows: {len(df)}")
    print()
    print(df[["company", "date", "quarter", "sector",
              "sentiment_score", "return_5d"]].to_string(index=False))


if __name__ == "__main__":
    main()
