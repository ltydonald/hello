"""
Test WorldQuant 101 alpha performance by actually trading a hand-picked list
of HK stocks, rebalanced every HOLDING_DAYS trading days, net of transaction
cost and execution slippage. The evaluation metric is realized portfolio
return (not rank IC, not an ML model) - compared against a benchmark index
(default: ^HSI).

Each alpha is computed across the WHOLE universe in DATA_CSV (not just
TICKERS), so a stock's cross-sectional percentile rank reflects the entire
market on that day - TICKERS only decides which of those universe-wide
percentiles actually get traded/reported. For each ticker in TICKERS,
independently: long if its alpha ranks at/above LONG_PERCENTILE of the whole
universe that day, short if at/below SHORT_PERCENTILE (never, if LONG_ONLY),
flat otherwise. Produces one equity curve per (ticker, alpha) pair, all
plotted together (one chart per alpha).

Unlike hk_alpha101_single.py (rank IC across the whole universe) or
alpha_ML.py (LightGBM-combined alphas across the whole universe), this
answers a narrower, more concrete question: "if I actually traded exactly
these N stocks using this one alpha - ranked against the whole market, not
just against each other - what return would I have made?"

Reads a CSV built by download_hk_data.py (default: hk_universe_data.csv) - no
network access beyond the benchmark fetch via yfinance. Since the alpha is
computed over the whole universe, this is as heavy as hk_alpha101_single.py/
alpha_ML.py (thousands of tickers x 101 alphas), not a quick small-list-only
computation.

Usage
-----
Edit the CONFIG block below (TICKERS, ALPHAS, HOLDING_DAYS, LONG_PERCENTILE/
SHORT_PERCENTILE, TRANSACTION_COST, EXECUTION_LAG_DAYS, ...), then just run:
    python hk_alpha_stock_backtest.py

HK only for now - US support could be added the same way alpha_ML.py did (a
MARKETS dict + a market switch) if/when needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from alpha101_engine import MarketData, ALL_ALPHAS, INDNEUTRALIZE_ALPHAS, rank

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG - edit these
# =============================================================================

# The stocks to actually TRADE - not the universe used to compute the alpha.
# Every ticker in DATA_CSV is loaded and every alpha is computed across all of
# them (see MIN_ADV in download_hk_data.py for what's already excluded at
# download time); TICKERS just selects which of those universe-wide
# percentile ranks actually get traded/reported.
#
# BENCHMARK_TICKER (e.g. "^HSI") is normally excluded from that universe - an
# index isn't a stock, so it wouldn't get its own alpha/percentile the way a
# real ticker does. Add it here explicitly if you want to trade the index
# itself too: it'll be kept in the universe, get its own alpha computed and
# ranked against everything else, and be tradeable like any other name - on
# top of still being read separately as the comparison benchmark below.
TICKERS = [
    "0700.HK", "0005.HK", "0388.HK", '^HSI'
]

# Which alpha(s) to test - each one is backtested independently (its own set
# of per-ticker equity curves, its own performance stats), so you can compare
# them side by side in the output summary/chart (one chart per alpha). Use
# list(ALL_ALPHAS) (imported above) for every alpha, or list a subset, e.g.
# ["alpha001", "alpha012", "alpha101"].
ALPHAS = list(ALL_ALPHAS)

DATA_CSV = "hk_universe_data.csv"  # built by download_hk_data.py - the WHOLE universe, not just TICKERS
YEARS = 10     # most recent N years of the CSV's own date range; 0, negative, or None = everything in the CSV
START = None    # explicit start date "YYYY-MM-DD", overrides YEARS
END = None      # explicit end date "YYYY-MM-DD"

HOLDING_DAYS = 1    # rebalance/holding period, in trading days (also the forward-return horizon)

# For each ticker in TICKERS, independently: long if its alpha value ranks
# at/above this percentile of the WHOLE universe that day (cross-sectional,
# not that ticker's own history), short if at/below SHORT_PERCENTILE (unless
# LONG_ONLY), flat otherwise. E.g. the defaults below mean "long the ticker if
# it's in the top 10% of the whole market that day, short if in the bottom 10%".
LONG_PERCENTILE = 0.90
SHORT_PERCENTILE = 0.10
LONG_ONLY = False   # True: never take the short side, only ever long or flat

# Same slippage convention as alpha_ML.py: trade at EXECUTION_PRICE_FIELD,
# EXECUTION_LAG_DAYS trading days after the signal date (0 = same-day close,
# an unrealistic zero-latency idealization). Default assumes next day's open.
EXECUTION_LAG_DAYS = 1
EXECUTION_PRICE_FIELD = "open"   # "open" or "close"

TRANSACTION_COST = 0.0008   # cost rate per unit of portfolio weight actually traded (turnover); matches alpha_ML.py's default

# BENCHMARK_TICKER must already be saved in DATA_CSV - download_hk_data.py's own
# BENCHMARK_TICKER (default "^HSI") does this automatically, so a normal
# `python download_hk_data.py` run is enough; no live yfinance call happens here.
BENCHMARK_TICKER = "^HSI"

# If fewer than this fraction of the WHOLE universe has sector data,
# indneutralize-based alphas in ALPHAS are skipped (with a warning) rather
# than silently running on an unreliable sector map.
SECTOR_COVERAGE_MIN = 0.5

OUT_CSV = "hk_stock_backtest_results.csv"
OUT_PLOT = "hk_stock_backtest.png"  # one PNG is saved per alpha, named "hk_stock_backtest_<alpha>.png"
OUT_BEST_PLOT = "hk_stock_backtest_best.png"  # whichever (ticker, alpha) pair had the highest annualized mean
                                               # return: strategy vs buy-and-hold for that ticker, with buy/sell
                                               # signal markers



# =============================================================================
# Data loading - reads DATA_CSV. tickers=None loads the WHOLE universe (used
# to compute the alpha); pass an explicit list to restrict it instead.
# =============================================================================

def load_data(csv_path, tickers=None, start=None, end=None, years=None):
    """`years`: use only the most recent `years` of whatever's in the CSV,
    measured from the CSV's own last date (not today's real-world date, since
    the CSV may be stale). 0, negative, or None means no filter (everything
    in the CSV). Ignored if `start` is given explicitly.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Build it first with: python download_hk_data.py"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    if tickers:
        df = df[df["ticker"].isin(tickers)]
        missing = sorted(set(tickers) - set(df["ticker"].unique()))
        if missing:
            print(f"WARNING: {len(missing)} requested ticker(s) not found in {csv_path}: {missing}")

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
# Trading simulation - each ticker in TICKERS is scored against the WHOLE
# universe's alpha distribution that day (cross-sectional, computed once
# across every ticker in DATA_CSV), then backtested independently: long if
# its percentile rank is at/above LONG_PERCENTILE, short if at/below
# SHORT_PERCENTILE, flat otherwise, rebalanced every HOLDING_DAYS trading
# days, net of transaction cost on turnover.
# =============================================================================

def whole_universe_signal(alpha_values, tickers_to_trade, long_pct, short_pct, long_only=False):
    """+1/-1/0 signal for just `tickers_to_trade`, based on each date's
    CROSS-SECTIONAL percentile rank of the alpha value across the WHOLE
    universe (every column in `alpha_values`, not just tickers_to_trade) -
    long if a traded ticker's percentile that day is at/above long_pct, short
    if at/below short_pct (never, if long_only), flat otherwise.

    rank(x) - from alpha101_engine, already used as an alpha building block -
    is exactly "cross-sectional percentile rank in [0, 1]" (x.rank(axis=1,
    pct=True)), computed across every column passed in. Computing it on the
    FULL alpha_values (whole universe) before slicing down to
    tickers_to_trade is what makes "top 90th percentile" mean "top 10% of the
    whole market", not "top 10% of this hand-picked list".
    """
    pct_rank = rank(alpha_values)[tickers_to_trade]
    signal = pd.DataFrame(0.0, index=pct_rank.index, columns=pct_rank.columns)
    signal[pct_rank >= long_pct] = 1.0
    if not long_only:
        signal[pct_rank <= short_pct] = -1.0
    return signal


def per_ticker_backtest(signal, fwd_ret, holding_days, cost_rate):
    """For each ticker independently: return per rebalance date = signal *
    forward return, net of transaction cost on the change in its own
    position from the previous rebalance (a single-name position isn't split
    into separate long/short legs the way a combined portfolio would be, so
    turnover here is simply |signal_t - signal_(t-1)|). Returns a dict
    ticker -> return Series (empty if a ticker never has enough data)."""
    all_dates = list(signal.index)
    rebal_dates = all_dates[::holding_days]  # non-overlapping holding periods
    signal = signal.reindex(rebal_dates)
    fwd_ret = fwd_ret.reindex(rebal_dates)

    results = {}
    for ticker in signal.columns:
        s = signal[ticker]
        r = fwd_ret[ticker]
        valid = s.notna() & r.notna()
        s, r = s[valid], r[valid]
        if s.empty:
            continue
        prev = 0.0
        rets = {}
        for date, sig, ret in zip(s.index, s.values, r.values):
            rets[date] = sig * ret - cost_rate * abs(sig - prev)
            prev = sig
        results[ticker] = pd.Series(rets)
    return results


def calc_max_drawdown(cum_returns):
    cum_max = np.maximum.accumulate(cum_returns)
    return ((cum_returns - cum_max) / cum_max).min()


def cal_annualized_return(s, periods_per_year):
    cumulative_return = (1 + s).prod() - 1
    n_periods = len(s) / periods_per_year
    return (1 + cumulative_return) ** (1 / n_periods) - 1 if n_periods > 0 else np.nan


def perf(s, periods_per_year, name=""):
    mu = s.mean() * periods_per_year
    vol = s.std() * np.sqrt(periods_per_year)
    sr = mu / vol if vol > 0 else 0
    cum = (1 + s).cumprod()
    mdd = calc_max_drawdown(cum.values)
    return {"Strategy": name, "SR": round(sr, 2), "Ann Mean": f"{mu:.1%}",
            "Ann Vol": f"{vol:.1%}", "MDD": f"{mdd:.1%}", "Total": f"{cum.iloc[-1] - 1:.0%}"}


def plot_strats(strat_dict, benchmark, benchmark_name, periods_per_year, title, out_path):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for name, s in strat_dict.items():
        cum = (1 + s).cumprod()
        sr = s.mean() / s.std() * np.sqrt(periods_per_year) if s.std() > 0 else 0
        ax.plot(cum.index, cum.values, label=f"{name} (SR={sr:.2f})", linewidth=1.3)
    if benchmark is not None and len(benchmark) > 0:
        cum_b = (1 + benchmark).cumprod()
        sr_b = benchmark.mean() / benchmark.std() * np.sqrt(periods_per_year) if benchmark.std() > 0 else 0
        ax.plot(cum_b.index, cum_b.values, color="gray", linestyle="--", linewidth=2.2,
                 label=f"{benchmark_name} (SR={sr_b:.2f})", alpha=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Cumulative Wealth ($1)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_alpha_plot_path(base_path, alpha_name):
    """Insert the alpha name before the extension, e.g. "hk_stock_backtest.png"
    -> "hk_stock_backtest_alpha001.png", so each alpha gets its own chart file."""
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}_{alpha_name}{p.suffix}"))


def plot_best_pair(ticker, alpha_name, strat_ret, price, signal, periods_per_year, out_path):
    """Strategy equity curve vs. plain buy-and-hold for `ticker`, with markers
    where the signal actually entered a new position - a Buy (^) the first
    rebalance date a long position starts, a Sell (v) the first rebalance
    date a short position starts (transitions only, not every day held, so
    a multi-period hold doesn't clutter the chart with repeat markers).

    `price` is `ticker`'s own EXECUTION_PRICE_FIELD series (any date range -
    trimmed here to strat_ret's own dates); `signal` is the raw (all-dates,
    not yet reindexed to rebalance dates) +1/-1/0 series for `ticker` under
    `alpha_name`, as returned by whole_universe_signal().
    """
    fig, ax = plt.subplots(figsize=(13, 5.5))

    cum_strat = (1 + strat_ret).cumprod()
    sr = strat_ret.mean() / strat_ret.std() * np.sqrt(periods_per_year) if strat_ret.std() > 0 else 0
    ax.plot(cum_strat.index, cum_strat.values, label=f"Strategy: {ticker} ({alpha_name}) (SR={sr:.2f})",
             linewidth=1.6, color="tab:blue", zorder=3)

    buy_hold = price.loc[cum_strat.index.min():cum_strat.index.max()].dropna()
    if not buy_hold.empty:
        buy_hold_cum = buy_hold / buy_hold.iloc[0]
        ax.plot(buy_hold_cum.index, buy_hold_cum.values, label=f"Buy & hold {ticker}",
                 linewidth=1.3, color="gray", linestyle="--", alpha=0.85, zorder=2)

    # Signal transitions, restricted to the dates that actually made it into the equity
    # curve (rebalance dates with valid signal+return - see per_ticker_backtest) - a date
    # dropped there (e.g. NaN return) can't be a real trade, so it can't start a new leg.
    sig_at_trades = signal.reindex(cum_strat.index)
    prev = 0.0
    buys, sells = [], []
    for d, s in sig_at_trades.items():
        if pd.isna(s):
            continue
        if s == 1.0 and prev != 1.0:
            buys.append(d)
        elif s == -1.0 and prev != -1.0:
            sells.append(d)
        prev = s

    if buys:
        ax.scatter(buys, cum_strat.reindex(buys), marker="^", color="green", s=110,
                   zorder=5, label="Buy (long entry)", edgecolors="black", linewidths=0.5)
    if sells:
        ax.scatter(sells, cum_strat.reindex(sells), marker="v", color="red", s=110,
                   zorder=5, label="Sell (short entry)", edgecolors="black", linewidths=0.5)

    ax.set_title(f"Best (ticker, alpha) pair by annualized mean return: {ticker} / {alpha_name}", fontsize=13)
    ax.set_ylabel("Cumulative Wealth ($1)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def benchmark_returns_from_csv(csv_path, ticker, dates, holding_days,
                                execution_lag_days=EXECUTION_LAG_DAYS, execution_price_field=EXECUTION_PRICE_FIELD):
    """Forward `holding_days`-return of `ticker`, reindexed to `dates`, using
    the same entry/exit timing convention as the strategy (see
    EXECUTION_LAG_DAYS in CONFIG). Reads the benchmark's price straight out of
    DATA_CSV (no network access) - download_hk_data.py's own BENCHMARK_TICKER
    saves it into the same CSV as the tradeable universe, exempt from the
    --min-adv liquidity screen, precisely so this can be a plain CSV read
    instead of a live yfinance call on every backtest run.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[df["ticker"] == ticker]
    if df.empty:
        raise ValueError(
            f"Benchmark ticker {ticker!r} not found in {csv_path}. Re-run download_hk_data.py "
            f"(its BENCHMARK_TICKER config controls which index gets saved into the CSV)."
        )
    price = df.set_index("date")[execution_price_field].sort_index()
    entry_price = price.shift(-execution_lag_days)
    exit_price = price.shift(-(execution_lag_days + holding_days))
    fwd = exit_price / entry_price - 1
    return fwd.reindex(dates)


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Loading the WHOLE universe from {DATA_CSV} (alpha computed across every ticker; "
          f"only {len(TICKERS)} will actually be traded) ...")
    open_df, high_df, low_df, close_df, volume_df, cap_df, sector, industry = load_data(
        DATA_CSV, tickers=None, start=START, end=END, years=YEARS
    )
    # BENCHMARK_TICKER lives in the same CSV as a row. By default it's excluded from the
    # universe used for alpha/percentile computation (it's an index, not a tradeable
    # stock, and benchmark_returns_from_csv() reads it independently below regardless) -
    # but if you deliberately add it to TICKERS, that's a request to trade it too, so it
    # stays in the universe (gets its own alpha computed, ranked against everything else
    # including itself, and traded like any other name) alongside still being read as
    # the comparison benchmark.
    if BENCHMARK_TICKER in close_df.columns and BENCHMARK_TICKER not in TICKERS:
        open_df, high_df, low_df, close_df, volume_df = (
            open_df.drop(columns=[BENCHMARK_TICKER]), high_df.drop(columns=[BENCHMARK_TICKER]),
            low_df.drop(columns=[BENCHMARK_TICKER]), close_df.drop(columns=[BENCHMARK_TICKER]),
            volume_df.drop(columns=[BENCHMARK_TICKER]),
        )
        if cap_df is not None and BENCHMARK_TICKER in cap_df.columns:
            cap_df = cap_df.drop(columns=[BENCHMARK_TICKER])
        if sector is not None and BENCHMARK_TICKER in sector.index:
            sector = sector.drop(BENCHMARK_TICKER)
        if industry is not None and BENCHMARK_TICKER in industry.index:
            industry = industry.drop(BENCHMARK_TICKER)

    print(f"Loaded {close_df.shape[1]} universe ticker(s), {close_df.shape[0]} trading days "
          f"({close_df.index.min().date()} to {close_df.index.max().date()}).")
    if close_df.shape[1] == 0:
        print("No data loaded - check DATA_CSV/date range.")
        return

    missing = sorted(set(TICKERS) - set(close_df.columns))
    if missing:
        print(f"WARNING: {len(missing)} of TICKERS not found in the loaded universe (won't be traded): {missing}")
    tickers_to_trade = [t for t in TICKERS if t in close_df.columns]
    if not tickers_to_trade:
        print("None of TICKERS are present in the loaded universe - nothing to trade.")
        return

    md = MarketData(open_df, high_df, low_df, close_df, volume_df,
                     cap=cap_df, sector=sector, industry=industry)

    sector_coverage = sector.notna().mean() if sector is not None else 0.0
    alphas_to_run = [a for a in ALPHAS if a in ALL_ALPHAS]
    unknown = [a for a in ALPHAS if a not in ALL_ALPHAS]
    if unknown:
        print(f"  unknown alpha name(s), skipping: {unknown}")
    if sector_coverage < SECTOR_COVERAGE_MIN:
        skipped = [a for a in alphas_to_run if a in INDNEUTRALIZE_ALPHAS]
        if skipped:
            print(f"Universe sector coverage {sector_coverage:.0%} is below {SECTOR_COVERAGE_MIN:.0%} - "
                  f"skipping indneutralize-based alpha(s): {skipped}")
            alphas_to_run = [a for a in alphas_to_run if a not in skipped]

    periods_per_year = 252 / HOLDING_DAYS
    exec_price = getattr(md, EXECUTION_PRICE_FIELD)
    entry_price = exec_price.shift(-EXECUTION_LAG_DAYS)
    exit_price = exec_price.shift(-(EXECUTION_LAG_DAYS + HOLDING_DAYS))
    fwd_ret = (exit_price / entry_price - 1)[tickers_to_trade]

    print(f"Long a traded ticker if its alpha ranks at/above the {LONG_PERCENTILE:.0%} percentile of the "
          f"WHOLE universe that day" +
          ("" if LONG_ONLY else f", short if at/below the {SHORT_PERCENTILE:.0%} percentile") +
          f", flat otherwise - rebalanced/held every {HOLDING_DAYS} trading day(s). "
          f"One equity curve per (ticker, alpha).")

    strat_returns = {}
    curves_by_alpha = {}
    signal_by_alpha = {}
    pair_records = []  # (ticker, alpha_name, annualized_mean_return) for every traded pair - used to find the best one below
    summary_rows = []

    for name in alphas_to_run:
        try:
            alpha_values = ALL_ALPHAS[name](md)
        except Exception as e:
            print(f"  {name} FAILED to compute: {e}")
            continue

        signal = whole_universe_signal(alpha_values, tickers_to_trade, LONG_PERCENTILE, SHORT_PERCENTILE, LONG_ONLY)
        signal_by_alpha[name] = signal
        per_ticker = per_ticker_backtest(signal, fwd_ret, HOLDING_DAYS, TRANSACTION_COST)
        for ticker, ret in per_ticker.items():
            if ret.empty:
                continue
            label = f"{ticker} ({name})"
            strat_returns[label] = ret
            curves_by_alpha.setdefault(name, {})[ticker] = ret
            stats = perf(ret, periods_per_year, name=label)
            pair_records.append((ticker, name, ret.mean() * periods_per_year))
            summary_rows.append(stats)
            print(f"  {label}: {stats}")

    if not strat_returns:
        print("No alpha produced a valid backtest - check ALPHAS/TICKERS/date range.")
        return

    all_rebal_dates = sorted(set().union(*[s.index for s in strat_returns.values()]))
    print(f"Reading benchmark ({BENCHMARK_TICKER}) from {DATA_CSV} ...")
    benchmark_ret = benchmark_returns_from_csv(
        DATA_CSV, BENCHMARK_TICKER, all_rebal_dates, HOLDING_DAYS,
        execution_lag_days=EXECUTION_LAG_DAYS, execution_price_field=EXECUTION_PRICE_FIELD,
    ).dropna()
    if not benchmark_ret.empty:
        bench_stats = perf(benchmark_ret, periods_per_year, name=BENCHMARK_TICKER)
        summary_rows.append(bench_stats)
        print(f"  {BENCHMARK_TICKER} (benchmark): {bench_stats}")

    summary = pd.DataFrame(summary_rows).set_index("Strategy")
    summary.to_csv(OUT_CSV)
    print(f"\nSaved summary to {OUT_CSV}\n")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary)

    for name, curves in curves_by_alpha.items():
        plot_path = per_alpha_plot_path(OUT_PLOT, name)
        plot_strats(curves, benchmark_ret, BENCHMARK_TICKER, periods_per_year,
                    title=f"{name} ({len(tickers_to_trade)} tickers, {HOLDING_DAYS}d holding)",
                    out_path=plot_path)
        print(f"Saved chart to {plot_path}")

    best_ticker, best_alpha, best_ann_mean = max(pair_records, key=lambda r: r[2])
    print(f"\nBest (ticker, alpha) pair by annualized mean return: {best_ticker} / {best_alpha} "
          f"(Ann Mean={best_ann_mean:.1%})")
    plot_best_pair(
        best_ticker, best_alpha, strat_returns[f"{best_ticker} ({best_alpha})"],
        exec_price[best_ticker], signal_by_alpha[best_alpha][best_ticker],
        periods_per_year, OUT_BEST_PLOT,
    )
    print(f"Saved chart to {OUT_BEST_PLOT}")


if __name__ == "__main__":
    main()