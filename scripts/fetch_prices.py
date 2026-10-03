import os
import sqlite3
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

DB_PATH = os.path.join("db", "pipeline.db")
PRICES_DIR = os.path.join("data", "prices")
WINDOW_DAYS = 15  # trading days before/after each report date


def main():
    os.makedirs(PRICES_DIR, exist_ok=True)

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    meta = pd.read_sql_query(
        "SELECT company, report_date FROM raw_transcripts", conn
    )
    conn.close()

    meta["report_date"] = pd.to_datetime(meta["report_date"])

    # One CSV per ticker, wide enough to cover all its earnings dates
    for ticker, group in meta.groupby("company"):
        start = (group["report_date"].min() - BDay(WINDOW_DAYS)).date()
        # +1 extra day because yfinance end is exclusive
        end = (group["report_date"].max() + BDay(WINDOW_DAYS + 1)).date()

        print(f"Downloading {ticker}: {start} -> {end} ...")

        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=str(start), end=str(end), auto_adjust=True)

        if df.empty:
            print(f"  WARNING: no data returned for {ticker}")
            continue

        # Drop timezone from index so CSV is clean
        df.index = df.index.tz_localize(None)
        df.index.name = "Date"

        out_path = os.path.join(PRICES_DIR, f"{ticker}.csv")
        df.to_csv(out_path)
        print(f"  Saved {len(df)} rows -> {out_path}")

    print("\nDone. Files in data/prices/:")
    for f in sorted(os.listdir(PRICES_DIR)):
        size = os.path.getsize(os.path.join(PRICES_DIR, f))
        print(f"  {f}  ({size} bytes)")


if __name__ == "__main__":
    main()
