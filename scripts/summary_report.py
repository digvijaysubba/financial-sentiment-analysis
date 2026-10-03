from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "outputs" / "results.csv"

def pct_negative(series: pd.Series) -> float:
    # % of values below 0 (as a percent, not fraction)
    if len(series) == 0:
        return float("nan")
    return 100.0 * (series < 0).mean()

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}. Run build_results.py first.")

    df = pd.read_csv(CSV_PATH)

    rows = []
    for model in ["vader", "textblob"]:
        prep = f"{model}_prepared"
        qa = f"{model}_qa"
        shift = f"{model}_shift"

        if prep not in df.columns or qa not in df.columns or shift not in df.columns:
            continue

        rows.append({
            "model": model.upper(),
            "mean_prepared": df[prep].mean(),
            "mean_qa": df[qa].mean(),
            "mean_shift": df[shift].mean(),
            "pct_negative_shift": pct_negative(df[shift]),
        })

    summary = pd.DataFrame(rows)

    print("\n=== Summary Report (Across Companies) ===")
    # nicer formatting
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    # -----------------------------
    # Add correlation metrics here
    # -----------------------------
    print("\n=== Model Correlation ===")

    if "vader_prepared" in df.columns and "textblob_prepared" in df.columns:
        print("Prepared correlation:",
          df["vader_prepared"].corr(df["textblob_prepared"]))

    if "vader_qa" in df.columns and "textblob_qa" in df.columns:
        print("Q&A correlation:",
          df["vader_qa"].corr(df["textblob_qa"]))
        
    # Optional: show which tickers had negative shift for each model
    print("\n=== Negative Shift Companies (Q&A < Prepared) ===")
    for model in ["vader", "textblob"]:
        shift = f"{model}_shift"
        if shift in df.columns:
            neg_names = df.loc[df[shift] < 0, "company"].tolist()
            print(f"{model.upper()}: {neg_names}")

if __name__ == "__main__":
    main()