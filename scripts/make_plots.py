from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "outputs" / "results.csv"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"


def _save_current_fig(filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Saved plot: {out_path}")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}. Run build_results.py first.")

    df = pd.read_csv(CSV_PATH)
    # -----------------------------
    # Quick numeric agreement stats
    # -----------------------------
    if "vader_prepared" in df.columns and "textblob_prepared" in df.columns:
        print("\nCorrelation (Prepared):",
            df["vader_prepared"].corr(df["textblob_prepared"]))

    if "vader_qa" in df.columns and "textblob_qa" in df.columns:
        print("Correlation (Q&A):",
            df["vader_qa"].corr(df["textblob_qa"]))
    # -----------------------------
    # Plot 1: Tone shift bar charts
    # -----------------------------
    for model in ["vader", "textblob"]:
        shift_col = f"{model}_shift"
        if shift_col not in df.columns:
            continue

        df_sorted = df.sort_values(shift_col)

        plt.figure()
        plt.bar(df_sorted["company"], df_sorted[shift_col])
        plt.axhline(0)
        plt.title(f"Tone Shift (Q&A - Prepared) — {model.upper()}")
        plt.xlabel("Company")
        plt.ylabel("Shift (positive = Q&A more positive)")
        _save_current_fig(f"{model}_tone_shift.png")

    # --------------------------------------
    # Plot 2: Prepared vs Q&A scatter (each model)
    # --------------------------------------
    for model in ["vader", "textblob"]:
        prep = f"{model}_prepared"
        qa = f"{model}_qa"
        if prep not in df.columns or qa not in df.columns:
            continue

        plt.figure()
        plt.scatter(df[prep], df[qa])
        for _, row in df.iterrows():
            plt.annotate(row["company"], (row[prep], row[qa]))
        plt.axhline(0)
        plt.axvline(0)
        plt.title(f"Prepared vs Q&A Sentiment — {model.upper()}")
        plt.xlabel("Prepared")
        plt.ylabel("Q&A")
        _save_current_fig(f"{model}_prepared_vs_qa.png")

    # ------------------------------------------------
    # Plot 3: VADER vs TextBlob agreement (Prepared + Q&A)
    # ------------------------------------------------
    if "vader_prepared" in df.columns and "textblob_prepared" in df.columns:
        plt.figure()
        plt.scatter(df["vader_prepared"], df["textblob_prepared"])
        for _, row in df.iterrows():
            plt.annotate(row["company"], (row["vader_prepared"], row["textblob_prepared"]))
        plt.axhline(0)
        plt.axvline(0)
        plt.title("Model Agreement — Prepared (VADER vs TextBlob)")
        plt.xlabel("VADER prepared")
        plt.ylabel("TextBlob prepared")
        _save_current_fig("agreement_prepared_vader_vs_textblob.png")

    if "vader_qa" in df.columns and "textblob_qa" in df.columns:
        plt.figure()
        plt.scatter(df["vader_qa"], df["textblob_qa"])
        for _, row in df.iterrows():
            plt.annotate(row["company"], (row["vader_qa"], row["textblob_qa"]))
        plt.axhline(0)
        plt.axvline(0)
        plt.title("Model Agreement — Q&A (VADER vs TextBlob)")
        plt.xlabel("VADER Q&A")
        plt.ylabel("TextBlob Q&A")
        _save_current_fig("agreement_qa_vader_vs_textblob.png")

    print("\n✅ All plots generated.")
    print(f"Open this folder: {PLOTS_DIR}")


if __name__ == "__main__":
    main()