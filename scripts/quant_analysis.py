import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

IN_CSV  = os.path.join("outputs", "results_with_returns.csv")
PLOT_DIR = os.path.join("outputs", "plots")

RETURN_COLS = ["gap_ret", "ret_1d", "ret_3d", "ret_5d"]
RETURN_LABELS = {
    "gap_ret": "Gap Return\n(Open[t+1]/Close[t]-1)",
    "ret_1d":  "1-Day Return\n(Close[t+1]/Close[t]-1)",
    "ret_3d":  "3-Day Return\n(Close[t+3]/Close[t]-1)",
    "ret_5d":  "5-Day Return\n(Close[t+5]/Close[t]-1)",
}
MODELS = {
    "vader":    "VADER Tone Shift",
    "textblob": "TextBlob Tone Shift",
}
CAVEAT = f"n=6  |  p-values are illustrative only — no statistical power at this sample size"


# ── helpers ──────────────────────────────────────────────────────────────────

def corr_row(x, y):
    """Pearson r and p-value, handling NaNs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p


def regression(x, y):
    """OLS via scipy.stats.linregress, returns dict."""
    mask = ~(np.isnan(x) | np.isnan(y))
    res = stats.linregress(x[mask], y[mask])
    return {
        "slope":     res.slope,
        "intercept": res.intercept,
        "r2":        res.rvalue ** 2,
        "p":         res.pvalue,
        "n":         int(mask.sum()),
    }


# ── terminal output ───────────────────────────────────────────────────────────

def print_correlations(df):
    print("=" * 62)
    print("CORRELATION: tone shift vs post-earnings returns")
    print(CAVEAT)
    print("=" * 62)

    header = f"{'':>12}  {'vader_shift':>14}  {'textblob_shift':>16}"
    print(header)
    print("-" * 62)

    for ret in RETURN_COLS:
        y = df[ret].values.astype(float)
        r_v, p_v = corr_row(df["vader_shift"].values.astype(float), y)
        r_t, p_t = corr_row(df["textblob_shift"].values.astype(float), y)

        def fmt(r, p):
            stars = "***" if p < 0.01 else "** " if p < 0.05 else "*  " if p < 0.10 else "   "
            return f"{r:+.3f}{stars} (p={p:.2f})"

        print(f"{ret:>12}  {fmt(r_v, p_v):>17}  {fmt(r_t, p_t):>17}")

    print()


def print_regressions(df):
    print("=" * 62)
    print("REGRESSION: ret_1d ~ tone_shift")
    print(CAVEAT)
    print("=" * 62)

    for model, label in MODELS.items():
        x = df[f"{model}_shift"].values.astype(float)
        y = df["ret_1d"].values.astype(float)
        r = regression(x, y)
        print(f"\n  {label}")
        print(f"    slope     = {r['slope']:+.4f}")
        print(f"    intercept = {r['intercept']:+.4f}")
        print(f"    R²        = {r['r2']:.4f}")
        print(f"    p-value   = {r['p']:.4f}")

    print()


def print_raw_table(df):
    print("=" * 62)
    print("RAW DATA")
    print("=" * 62)
    cols = ["company", "report_date", "vader_shift", "textblob_shift",
            "gap_ret", "ret_1d", "ret_3d", "ret_5d"]
    display = df[cols].copy()
    for c in cols[2:]:
        display[c] = display[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "n/a")
    print(display.to_string(index=False))
    print()


# ── plots ─────────────────────────────────────────────────────────────────────

def make_scatter_plot(df, model, shift_col, label):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        f"Earnings Tone Shift vs Post-Earnings Returns\nModel: {label}",
        fontsize=13, fontweight="bold", y=0.98
    )

    for ax, ret_col in zip(axes.flat, RETURN_COLS):
        x = df[shift_col].values.astype(float)
        y = df[ret_col].values.astype(float)
        companies = df["company"].values

        ax.scatter(x, y, color="steelblue", s=80, zorder=3)

        for xi, yi, name in zip(x, y, companies):
            ax.annotate(name, (xi, yi), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)

        # regression line
        res = stats.linregress(x, y)
        x_line = np.linspace(x.min() - 0.01, x.max() + 0.01, 100)
        ax.plot(x_line, res.slope * x_line + res.intercept,
                color="tomato", linewidth=1.4, linestyle="--", zorder=2)

        r, p = corr_row(x, y)
        ax.set_title(
            f"{RETURN_LABELS[ret_col].split(chr(10))[0]}  "
            f"r={r:+.2f}  p={p:.2f}",
            fontsize=9
        )
        ax.set_xlabel("Tone Shift (Q&A − Prepared)", fontsize=8)
        ax.set_ylabel("Return", fontsize=8)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.grid(True, alpha=0.3)

    fig.text(0.5, 0.01, CAVEAT, ha="center", fontsize=7, color="grey")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(PLOT_DIR, f"shift_vs_return_{model}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Run compute_returns.py first: {IN_CSV}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(IN_CSV)

    print_raw_table(df)
    print_correlations(df)
    print_regressions(df)

    for model, label in MODELS.items():
        make_scatter_plot(df, model, f"{model}_shift", label)

    print("Done.")


if __name__ == "__main__":
    main()
