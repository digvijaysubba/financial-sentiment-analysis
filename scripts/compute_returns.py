import os
import pandas as pd

RESULTS_CSV = os.path.join("outputs", "results.csv")
PRICES_DIR = os.path.join("data", "prices")
OUT_CSV = os.path.join("outputs", "results_with_returns.csv")


def load_prices(ticker):
    path = os.path.join(PRICES_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No price file for {ticker}: {path}")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df[["Open", "Close"]].sort_index()


def compute_returns(ticker, report_date, prices):
    """
    Earnings calls happen after market close, so:
      t   = report_date close  (market has NOT yet heard the call)
      t+1 = first day market can react
    All returns are measured from close[t] as the baseline.
    """
    dates = prices.index

    # Find t: last trading day on or before report_date
    mask = dates <= pd.Timestamp(report_date)
    if not mask.any():
        raise ValueError(f"No trading day on/before {report_date} for {ticker}")
    t = int(mask.sum()) - 1

    close_t = prices.iloc[t]["Close"]

    def close_ret(offset):
        pos = t + offset
        if pos >= len(prices):
            return None
        return (prices.iloc[pos]["Close"] / close_t) - 1

    def gap_ret():
        pos = t + 1
        if pos >= len(prices):
            return None
        return (prices.iloc[pos]["Open"] / close_t) - 1

    return {
        "gap_ret": gap_ret(),
        "ret_1d":  close_ret(1),
        "ret_3d":  close_ret(3),
        "ret_5d":  close_ret(5),
    }


def main():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Run build_results.py first: {RESULTS_CSV}")

    results = pd.read_csv(RESULTS_CSV)
    rows = []

    for _, row in results.iterrows():
        ticker = row["company"]
        report_date = row["report_date"]

        print(f"Computing returns for {ticker} {report_date} ...")
        prices = load_prices(ticker)
        ret = compute_returns(ticker, report_date, prices)

        combined = row.to_dict()
        combined.update(ret)
        rows.append(combined)

        print(
            f"  gap={ret['gap_ret']:+.2%}  "
            f"1d={ret['ret_1d']:+.2%}  "
            f"3d={ret['ret_3d']:+.2%}  "
            f"5d={ret['ret_5d']:+.2%}"
        )

    out = pd.DataFrame(rows)

    preferred = [
        "company", "report_date",
        "vader_shift", "textblob_shift",
        "gap_ret", "ret_1d", "ret_3d", "ret_5d",
        "vader_prepared", "vader_qa",
        "textblob_prepared", "textblob_qa",
        "transcript_id",
    ]
    cols = [c for c in preferred if c in out.columns] + \
           [c for c in out.columns if c not in preferred]
    out = out[cols]

    out.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")
    print("\nFull results:")
    print(
        out[["company", "report_date", "vader_shift", "textblob_shift",
             "gap_ret", "ret_1d", "ret_3d", "ret_5d"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
