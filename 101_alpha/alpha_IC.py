"""
WorldQuant "101 Formulaic Alphas" (Kakushadze, 2015) on HK stocks - single-file version.

Reads HK OHLCV (+ cap, sector/industry) data from a CSV built by
download_hk_data.py, computes all (or a subset of) the 101 alphas, and
backtests each one individually using rank IC (mean, std, t-stat, hit-rate,
ICIR) resampled to a configurable holding period (--holding-period, a pandas
resample offset alias; default below is '1D' = every trading day), plus a
rolling-window IC chart (--window holding periods) for the top alphas. This
script never touches the network - its only input is the CSV; run
download_hk_data.py first (or whenever you want fresh data).

Usage
-----
Edit the CONFIG block below (MARKET, TICKERS, ALPHAS, YEARS, HOLDING_PERIOD,
WINDOW, ...), then just run:
    python hk_alpha101_single.py

Every CONFIG setting can also be overridden on the command line with the
matching --flag, without editing the file, e.g.:
    python download_hk_data.py --years 10                  # build hk_universe_data.csv once
    python hk_alpha101_single.py --out results.csv
    python hk_alpha101_single.py --tickers 0700.HK 9988.HK 0005.HK --alphas alpha001 alpha012 alpha101
    python hk_alpha101_single.py --data custom_data.csv --start 2022-01-01
    python hk_alpha101_single.py --holding-period 1ME --window 12  # monthly IC, 12-month rolling chart
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from alpha101_engine import MarketData, ALL_ALPHAS

# =============================================================================
# CONFIG - edit these, then just run: python hk_alpha101_single.py
# Every setting here can also be overridden on the command line with the
# matching --flag (e.g. --tickers, --years), which takes precedence.
# =============================================================================

# Market being backtested - switch with MARKET/--market {hk,us}, the same way alpha_ML.py
# does. Each market gets its own default DATA/OUT/PLOT path so an HK run and a US run
# never overwrite each other's CSV/results/chart unless you override those yourself.
MARKETS = {
    "hk": dict(data="hk_universe_data.csv", out="hk_full_ic_results.csv", plot="hk_full_ic_plot.png"),
    "us": dict(data="us_universe_data.csv", out="us_full_ic_results.csv", plot="us_full_ic_plot.png"),
}
MARKET = "hk"  # or "us"; override with --market

TICKERS = None  # subset of tickers to test, e.g. ["0700.HK", "9988.HK", "0005.HK"]; None = every ticker in the CSV; override with --tickers
ALPHAS = None   # subset, e.g. ["alpha001", "alpha012", "alpha101"]; None = all 101; override with --alphas

DATA = None  # CSV database built by download_hk_data.py/download_us_data.py; None = MARKET's own default above; override with --data
OUT = None   # output path for the IC summary CSV; None = MARKET's own default above; override with --out
PLOT = None  # output path for the rolling-IC chart; None = MARKET's own default above; override with --plot

START = None  # filter to dates on/after this "YYYY-MM-DD"; overrides YEARS; override with --start
END = None    # filter to dates on/before this "YYYY-MM-DD"; override with --end
YEARS = 2    # use only the most recent N years of the CSV's own date range; 0, negative, or None = everything; override with --years

HOLDING_PERIOD = '1D'  # pandas resample offset alias: the IC holding period ('1D' daily, '1W' weekly, '1ME' month-end, ...); override with --holding-period
WINDOW = 5  # rolling window, in holding periods, for the IC chart; override with --window
MIN_NAMES = 5  # minimum stocks with valid data on a given day for that day's cross-sectional IC to count; override with --min-names
MIN_PERIODS = 100  # an alpha needs at least this many valid IC periods to be eligible for the rolling-IC chart; override with --min-periods
TOP_N = 5  # number of top alphas (by mean IC) to plot; override with --top-n

# =============================================================================
# Data loader - reads the CSV database built by download_hk_data.py
#
# Expects a long-format CSV, one row per (date, ticker): columns
# date, ticker, open, high, low, close, volume, and optionally cap, sector,
# industry (sector/industry are static per ticker; cap varies by date). This
# script never touches the network - build/refresh the CSV separately with
# `python download_hk_data.py`.
#
# vwap is approximated as (H+L+C)/3 (typical price) inside MarketData, since
# daily bars carry no true intraday VWAP.
# =============================================================================

def load_data(csv_path, tickers=None, start=None, end=None, years=None):
    """`years`: use only the most recent `years` of whatever's in the CSV,
    measured from the CSV's own last date (not today's real-world date, since
    the CSV may be stale) - a convenience for "I have 10 years cached, just
    use the last 5" without computing an absolute --start date by hand.
    0 or negative means no filter (everything in the CSV). Ignored if --start
    is given explicitly.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Build it first with: python download_hk_data.py"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    if start is None and years is not None and years > 0 and not df.empty:
        start = (df["date"].max() - pd.Timedelta(days=years * 365.25)).date().isoformat()
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]

    open_df = df.pivot(index="date", columns="ticker", values="open")
    high_df = df.pivot(index="date", columns="ticker", values="high")
    low_df = df.pivot(index="date", columns="ticker", values="low")
    close_df = df.pivot(index="date", columns="ticker", values="close")
    volume_df = df.pivot(index="date", columns="ticker", values="volume")

    cap_df = df.pivot(index="date", columns="ticker", values="cap") if "cap" in df.columns else None
    sector = df.groupby("ticker")["sector"].first() if "sector" in df.columns else None
    industry = df.groupby("ticker")["industry"].first() if "industry" in df.columns else None

    return open_df, high_df, low_df, close_df, volume_df, cap_df, sector, industry


# =============================================================================
# Factor testing - rank-IC backtest over a configurable holding period
#
# IC is computed once per holding period (--holding-period, a pandas resample
# offset alias - default set by `HOLDING_PERIOD` above): cross-sectional
# rank correlation between each alpha's value on the last trading day of
# period t and that stock's realized return from period t's close to period
# t+1's close. Reports the standard IC summary stats per alpha: mean, std,
# t-stat (mean / (std / sqrt(n))), hit-rate (fraction of periods with IC > 0),
# and ICIR (mean/std).
# =============================================================================

def period_label(freq):
    """Human-readable name for a pandas resample offset alias, e.g. '1M' ->
    'month', '1W' -> 'week', '1D' -> 'day'. Falls back to the raw alias for
    frequencies without a friendly name."""
    unit = re.sub(r"^[0-9]+", "", str(freq)).upper()
    labels = {
        "D": "day", "B": "day", "W": "week",
        "M": "month", "ME": "month", "MS": "month", "BM": "month",
        "Q": "quarter", "QE": "quarter", "QS": "quarter", "BQ": "quarter",
        "A": "year", "Y": "year", "YE": "year", "AS": "year", "YS": "year",
    }
    return labels.get(unit, str(freq))


def information_coefficient(factor, fwd_ret, min_names=MIN_NAMES):
    """Cross-sectional rank (Spearman) IC per row, vectorized as a row-wise
    Pearson correlation between cross-sectional ranks of factor and forward
    return (mathematically identical to Spearman's rho). Frequency-agnostic -
    used here at whatever frequency `fwd_ret` is sampled at (see
    `periodic_ic_series`), but works on any date-indexed panel."""
    idx = factor.index.intersection(fwd_ret.index)
    factor = factor.loc[idx]
    fwd_ret = fwd_ret.loc[idx]

    valid = factor.notna() & fwd_ret.notna()
    n = valid.sum(axis=1)

    rf = factor.where(valid).rank(axis=1)
    rr = fwd_ret.where(valid).rank(axis=1)
    rf_c = rf.sub(rf.mean(axis=1), axis=0)
    rr_c = rr.sub(rr.mean(axis=1), axis=0)

    num = (rf_c * rr_c).sum(axis=1)
    den = np.sqrt((rf_c ** 2).sum(axis=1) * (rr_c ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = num / den
    ic = ic.where(n >= min_names)
    return ic.dropna()


def periodic_ic_series(factor, close, min_names=MIN_NAMES, freq=HOLDING_PERIOD):
    """Rank IC per holding period: factor value on the last trading day of
    each `freq`-resampled period vs that stock's return over the following
    period (e.g. freq='1M' gives monthly IC, freq='1W' gives weekly IC).

    `.resample(freq)` bins by calendar time, so at daily/sub-weekly
    frequencies it inserts an empty bin for every non-trading calendar day
    (weekends, holidays) - `.last()` on an empty bin is NaN, not the prior
    trading day's value. Left unhandled, `close_p.shift(-1)` on a Friday (or
    any day right before a gap) would then grab that NaN weekend/holiday
    bin instead of the next real trading day, silently turning that day's
    forward return - and its whole IC observation - into NaN. Dropping
    fully-empty bins before shifting makes shift(-1) land on the next bin
    that actually has data, i.e. the next trading day, as intended.
    """
    factor_p = factor.resample(freq).last().dropna(how="all")
    close_p = close.resample(freq).last().dropna(how="all")
    fwd_ret_p = close_p.shift(-1) / close_p - 1
    return information_coefficient(factor_p, fwd_ret_p, min_names=min_names)


def evaluate_alpha(name, factor, close, min_names=MIN_NAMES, freq=HOLDING_PERIOD):
    ic = periodic_ic_series(factor, close, min_names, freq=freq)
    n = len(ic)
    ic_mean = ic.mean()
    ic_std = ic.std()
    ic_tstat = np.nan
    icir = np.nan
    if n > 1 and pd.notna(ic_std) and ic_std != 0:
        ic_tstat = ic_mean / (ic_std / np.sqrt(n))
        icir = ic_mean / ic_std
    row = {
        "name": name,
        "n_periods": n,
        "IC_mean": ic_mean,
        "IC_std": ic_std,
        "IC_tstat": ic_tstat,
        "IC_hitrate": (ic > 0).mean() if n else np.nan,
        "ICIR": icir,
    }
    return row, ic


def evaluate_all(factors: dict, close, min_names=MIN_NAMES, freq=HOLDING_PERIOD):
    rows = []
    ic_series = {}
    for name, factor in factors.items():
        try:
            row, ic = evaluate_alpha(name, factor, close, min_names, freq=freq)
            rows.append(row)
            ic_series[name] = ic
        except Exception as e:
            rows.append({"name": name, "error": str(e)})
    df = pd.DataFrame(rows).set_index("name")
    if "IC_mean" in df.columns:
        df = df.reindex(df["IC_mean"].sort_values(ascending=False).index)
    return df, ic_series


def plot_rolling_ic(ic_series: dict, top_names, window=WINDOW, out_path="rolling_ic.png",
                     freq=HOLDING_PERIOD):
    """Rolling `window`-period mean IC for each alpha in `top_names`, one line
    each (a "period" is one holding period of length `freq`, e.g. a month)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = period_label(freq).capitalize()
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in top_names:
        ic = ic_series.get(name)
        if ic is None or ic.empty:
            continue
        rolling = ic.rolling(window).mean()
        ax.plot(rolling.index, rolling.values, label=name)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {window}-{label} IC - Top {len(top_names)} Alphas by Mean IC")
    ax.set_xlabel(label)
    ax.set_ylabel(f"Rolling {window}-{label} Mean IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Test WorldQuant 101 formulaic alphas on HK or US stocks (reads a local CSV, no network)")
    p.add_argument("--market", choices=list(MARKETS), default=MARKET,
                    help="which market's default --data/--out/--plot paths to use (default: %(default)s); "
                         "explicitly passing --data/--out/--plot overrides the market default for that flag")
    p.add_argument("--data", default=DATA, help="CSV database built by download_hk_data.py/download_us_data.py (default: market's own CSV)")
    p.add_argument("--tickers", nargs="*", default=TICKERS, help="subset of tickers to test, e.g. 0700.HK 9988.HK")
    p.add_argument("--start", default=START, help="filter to dates on/after this (default: everything in the CSV)")
    p.add_argument("--end", default=END, help="filter to dates on/before this")
    p.add_argument("--years", type=float, default=YEARS,
                    help="use only the most recent N years of the CSV's own date range (e.g. --years 5 out of "
                         "10 years cached) - a convenience alternative to computing --start by hand; ignored if "
                         "--start is given (default: %(default)s, pass 0 or a negative number for everything in the CSV)")
    p.add_argument("--alphas", nargs="*", default=ALPHAS, help="subset, e.g. alpha001 alpha012 alpha101 (default: all 101)")
    p.add_argument("--out", default=OUT, help="output path for the IC summary CSV (default: market's own path)")
    p.add_argument("--plot", default=PLOT, help="output path for the rolling-IC chart (default: market's own path)")
    p.add_argument("--top-n", type=int, default=TOP_N, help="number of top alphas (by mean IC) to plot")
    p.add_argument("--holding-period", default=HOLDING_PERIOD,
                    help="pandas resample offset alias setting the IC holding period, e.g. "
                         "'1D' (daily), '1W' (weekly), '1ME' (monthly) (default: %(default)s)")
    p.add_argument("--window", type=int, default=WINDOW,
                    help="rolling window, in holding periods, for the IC chart (default: %(default)s)")
    p.add_argument("--min-names", type=int, default=MIN_NAMES,
                    help="minimum stocks with valid data on a day for that day's IC to count (default: %(default)s)")
    p.add_argument("--min-periods", type=int, default=MIN_PERIODS,
                    help="minimum valid IC periods an alpha needs to be eligible for the rolling-IC chart (default: %(default)s)")
    args = p.parse_args()
    label = period_label(args.holding_period)
    market_defaults = MARKETS[args.market]
    data_path = args.data or market_defaults["data"]
    out_path = args.out or market_defaults["out"]
    plot_path = args.plot or market_defaults["plot"]

    print(f"Market: {args.market.upper()} - loading {data_path} ...")
    open_df, high_df, low_df, close_df, volume_df, cap_df, sector, industry = load_data(
        data_path, args.tickers, args.start, args.end, years=args.years
    )
    print(f"Loaded {close_df.shape[1]} tickers, {close_df.shape[0]} trading days "
          f"({close_df.index.min().date()} to {close_df.index.max().date()}).")
    if close_df.shape[1] == 0:
        print("No data loaded - check --data path/tickers/date range.")
        return

    md = MarketData(
        open_df, high_df, low_df, close_df, volume_df,
        cap=cap_df, sector=sector, industry=industry,
    )

    alpha_fns = ALL_ALPHAS if not args.alphas else {k: ALL_ALPHAS[k] for k in args.alphas}
    print(f"Computing {len(alpha_fns)} alpha(s)...")

    factors = {}
    for name, fn in alpha_fns.items():
        try:
            factors[name] = fn(md)
        except Exception as e:
            print(f"  {name} FAILED to compute: {e}")

    print(f"Evaluating {len(factors)} alpha(s) with rank IC "
          f"(holding period: {args.holding_period}, i.e. one IC per {label})...")
    summary, ic_series = evaluate_all(factors, close_df, min_names=args.min_names, freq=args.holding_period)
    summary.to_csv(out_path)
    print(f"\nSaved summary to {out_path}\n")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.head(20))

    top_names = summary.loc[summary['n_periods'] >= args.min_periods, "IC_mean"].dropna().sort_values(ascending=False).head(args.top_n).index.tolist()
    print(f"\nPlotting rolling {args.window}-{label} IC for top {len(top_names)} alphas by mean IC: {', '.join(top_names)}")
    plot_rolling_ic(ic_series, top_names, window=args.window, out_path=plot_path, freq=args.holding_period)
    print(f"Saved chart to {plot_path}")


if __name__ == "__main__":
    main()
