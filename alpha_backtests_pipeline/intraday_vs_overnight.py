"""
Intraday (open->close) vs. overnight (close->open) return backtest for a
user-specified list of individual stocks.

Two strategies, each a same-day round trip (never held longer than the two
data points involved, so their transaction-cost treatment is identical):
  - Intraday : buy at today's OPEN,  sell at today's CLOSE  (in the market for
               the trading session; flat overnight).
  - Overnight: buy at yesterday's CLOSE, sell at today's OPEN (the "gap"; flat
               during the trading session). This is the well-known "overnight
               drift" effect - much of a stock's long-run return has
               historically come from the close-to-open gap rather than the
               regular session.

The two strategies are mutually exclusive views of the SAME daily bar (there's
no double-counting): open->close and close->open together reconstruct the
whole close-to-close move.

A third, no-cost "Buy & Hold" curve (plain close-to-close, held continuously) is
also included as a benchmark - the two round-trip strategies are implicitly
competing against just holding the stock.

Each ticker in TICKERS is backtested SEPARATELY (its own returns, own metrics,
own chart) rather than combined into one basket - so results for a 5-ticker
list are 5 independent one-by-one backtests, not a single blended portfolio.
For each ticker, both strategies x both with/without transaction cost (4
combinations) plus Buy & Hold (1 more, no-cost only) produce:
  - printed return metrics: total return, annualized mean/vol, Sharpe ratio,
    max drawdown (same SR/Ann Mean/Ann Vol/MDD convention as alpha_ML.py)
  - one chart (per ticker) with all 5 equity curves overlaid.

Also prints, per ticker, a liquidity figure: average daily volume over the
same lookback window divided by the time-weighted-average SHARES
OUTSTANDING over that window (see floating_share_turnover_ratio() below;
uses shares outstanding rather than floating shares, since no historical
float series is freely available - only shares outstanding has real
history, via yfinance's get_shares_full()).

A second mode (RUN_MODE = "quintile_groups") backtests a whole universe
instead of TICKERS: it loads the previously-built us_universe_data.csv (see
download_us_data.py), ranks every ticker ONCE by whole-window average daily
volume / current shares outstanding, and splits the universe into N_GROUPS
equal-count groups by that ratio. For each group it builds three
equal-weight (daily-rebalanced, "buy the whole group") equity curves - Buy &
Hold, Intraday, and Overnight, the same three return types as the
single-stock mode - and saves ONE CHART PER GROUP (N_GROUPS charts total,
each with its own 5 curves: Buy & Hold plus Intraday/Overnight x
no-cost/with-cost), plus a printed metrics table per group.

Also saves one more chart overlaying all N_GROUPS groups' "Overnight (with
cost)" curves against an overall equal-weight "buy & hold all stocks"
market curve, and prints a table of each group's average turnover ratio
(the same volume/shares-outstanding metric used to rank them).

Usage
-----
Edit TICKERS (and the rest of CONFIG) below, then:
    python intraday_overnight_backtest.py
For the universe/quintile mode: set RUN_MODE = "quintile_groups" below.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import data_download_common as common

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# CONFIG - edit these
# =============================================================================
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]  # any yfinance tickers, e.g. HK: "0700.HK"

YEARS = 5             # lookback, used if START is None
START = None          # "YYYY-MM-DD" or None to use YEARS back from today
END = None            # "YYYY-MM-DD" or None for today

TRANSACTION_COST_BPS = 1   # ONE-WAY cost in basis points (1 = 0.01%) per trade;
                           # each strategy does one buy + one sell per day, so
                           # the "with cost" curves deduct 2x this per day.

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0       # annual; subtracted from Ann Mean before the Sharpe ratio

OUT_CHART_TEMPLATE = "intraday_overnight_equity_curves_{ticker}.png"  # {ticker} filled in per ticker, sanitized for filenames

# --- quintile-group (universe) mode ---
RUN_MODE = "quintile_groups"  # "tickers" (single-stock mode above) or "quintile_groups" (universe mode below)

UNIVERSE_CSV = "us_universe_data.csv"  # previously-built universe, see download_us_data.py
N_GROUPS = 5
SHARES_CACHE_FILE = Path(__file__).parent / "us_alpha101_cache" / "shares_outstanding.json"  # same cache file/format as data_download_common.fetch_cap()
SHARES_FETCH_PAUSE = 0.1  # seconds between per-ticker yfinance calls (courtesy rate-limit delay, matches fetch_cap's default)
OUT_GROUP_CHART_TEMPLATE = "us_turnover_quintile_group{group}_equity_curves.png"  # {group} filled in per group (1..N_GROUPS)
OUT_OVERNIGHT_VS_MARKET_CHART = "us_turnover_quintile_overnight_vs_market.png"  # all N_GROUPS "Overnight (with cost)" curves + overall buy&hold market curve


# =============================================================================
# Data
# =============================================================================

def load_open_close(tickers, start, end):
    """(open_df, close_df, volume_df) wide date x ticker frames via the
    project's shared yfinance fetcher (auto_adjust=True - total-return
    prices, consistent with the rest of this project - so an overnight
    return correctly captures dividends paid between yesterday's close and
    today's open). volume is share-count, already split-adjusted by
    yfinance regardless of auto_adjust (only dividend-adjustment is gated by
    that flag), so it's on a consistent per-share basis across any splits in
    the window."""
    open_df, _high, _low, close_df, volume_df = common.fetch_ohlcv(
        tickers, start, end, interval="1d", min_obs=1, auto_adjust=True,
    )
    return open_df, close_df, volume_df


# =============================================================================
# Returns
# =============================================================================

def daily_returns(open_df, close_df):
    """(intraday_df, overnight_df): per-ticker daily returns, same shape as
    open_df/close_df.
        intraday  = close_t / open_t - 1                (today's session)
        overnight = open_t / close_t-1 - 1               (yesterday close -> today open)
    overnight's first row is NaN (no prior close to gap from). The two
    compound EXACTLY back into the plain close-to-close (buy & hold) return:
    (1+intraday)*(1+overnight) - 1 == close_t/close_t-1 - 1 == buy_and_hold_return()."""
    intraday = close_df / open_df - 1
    overnight = open_df / close_df.shift(1) - 1
    return intraday, overnight


def buy_and_hold_return(close_df):
    """Plain close-to-close daily return, i.e. holding continuously through
    both the session and the overnight gap - the benchmark the two "in and out
    every day" strategies are implicitly competing against. Traded once (buy at
    the start, hold, optionally sell at the end), so it's shown cost-free -
    a single round trip's cost is negligible next to a daily-rebalanced one."""
    return close_df / close_df.shift(1) - 1


def apply_cost(returns, one_way_cost_bps):
    """Subtract a same-day round-trip's transaction cost (one buy + one sell,
    i.e. 2x the one-way cost) from a daily return series - every non-NaN day
    gets exactly `2 * one_way` subtracted, so the resulting series is NEVER
    identical to the uncosted one (as long as one_way_cost_bps > 0 and there's
    at least one non-NaN day)."""
    one_way = one_way_cost_bps / 10_000
    return returns - 2 * one_way


# =============================================================================
# Liquidity: avg daily volume / floating shares
# =============================================================================

def time_weighted_avg_shares_outstanding(ticker, start, end):
    """Time-weighted average SHARES OUTSTANDING over the calendar window
    [start, end], from yfinance's get_shares_full() - a real historical
    series sourced from SEC/exchange filings (sparse, irregular filing
    dates), already reflecting any stock splits since it's the same
    filing-based share count used to compute EPS. It's a step function
    between filings, so time-weighting = holding each value constant day by
    day and averaging - equivalent to forward-filling onto a daily calendar
    index and taking the mean (every day gets equal weight).

    Queried from 400 days before `start` so there's a carry-in value even if
    the first filing inside [start, end] isn't right at the window's edge.
    Returns NaN if yfinance has no shares data for this ticker."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    query_start = (start_ts - pd.Timedelta(days=400)).date().isoformat()
    try:
        raw = yf.Ticker(ticker).get_shares_full(start=query_start, end=end_ts.date().isoformat())
    except Exception:
        return np.nan
    if raw is None or len(raw) == 0:
        return np.nan
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    calendar = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D")
    daily = raw.reindex(raw.index.union(calendar)).sort_index().ffill().reindex(calendar)
    return daily.mean() if not daily.isna().all() else np.nan


def floating_share_turnover_ratio(ticker, volume, start, end):
    """dict with the breakdown of avg daily volume / time-weighted-average
    SHARES OUTSTANDING over [start, end]:
      avg_daily_volume            - mean daily share volume in the window
      tw_avg_shares_outstanding   - time_weighted_avg_shares_outstanding()
      turnover_ratio              - avg_daily_volume / tw_avg_shares_outstanding
    Uses shares outstanding directly (not floating shares) - no historical
    float series is freely available anywhere, only shares outstanding has
    real history via get_shares_full(). Any missing piece propagates as NaN
    rather than raising, so one ticker's missing shares data doesn't stop
    the rest of the run."""
    avg_daily_volume = volume.dropna().mean() if volume is not None else np.nan
    tw_avg_shares_out = time_weighted_avg_shares_outstanding(ticker, start, end)
    turnover_ratio = (avg_daily_volume / tw_avg_shares_out
                      if np.isfinite(avg_daily_volume) and np.isfinite(tw_avg_shares_out) and tw_avg_shares_out > 0
                      else np.nan)
    return {
        "avg_daily_volume": avg_daily_volume,
        "tw_avg_shares_outstanding": tw_avg_shares_out,
        "turnover_ratio": turnover_ratio,
    }


# =============================================================================
# Metrics
# =============================================================================

def backtest_metrics(returns, trading_days_per_year, risk_free_rate):
    """SR/Ann Mean/Ann Vol/MDD/Total for one daily-return series (same
    convention as alpha_ML.py: Ann Mean = mean*periods, Ann Vol =
    std*sqrt(periods), SR = (Ann Mean - risk_free) / Ann Vol). Returns a dict of
    the formatted display fields plus "n_days" and the cumulative "equity"
    curve (the latter is what plot_equity_curves actually plots)."""
    r = returns.dropna()
    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min() if len(drawdown) else np.nan

    ann_mean = r.mean() * trading_days_per_year
    ann_vol = r.std() * np.sqrt(trading_days_per_year)
    sharpe = (ann_mean - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    total_return = equity.iloc[-1] - 1 if len(equity) else np.nan

    return {
        "SR": round(sharpe, 2) if np.isfinite(sharpe) else np.nan,
        "Ann Mean": f"{ann_mean:.1%}",
        "Ann Vol": f"{ann_vol:.1%}",
        "MDD": f"{max_dd:.1%}",
        "Total": f"{total_return:.1%}",
        "n_days": len(r),
        "equity": equity,
    }


# =============================================================================
# Plot
# =============================================================================

def plot_equity_curves(curves, ticker, out_path):
    """curves: dict of {label: equity Series (starts at 1.0)}. One chart, all
    4 curves overlaid, y-axis as cumulative return %."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    colors = {"Intraday": "tab:blue", "Overnight": "tab:orange", "Buy & Hold": "tab:green"}
    styles = {"no cost": "-", "with cost": "--"}

    for label, equity in curves.items():
        strategy, cost_label = label.split(" (")
        cost_label = cost_label.rstrip(")")
        ax.plot((equity - 1) * 100, color=colors[strategy], linestyle=styles[cost_label],
                linewidth=1.4, label=label)

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("date")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_title(f"{ticker}: intraday vs. overnight equity curves", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _safe_filename(ticker):
    """Sanitize a ticker symbol (e.g. "0700.HK", "^GSPC") into a filesystem-safe
    filename fragment."""
    return "".join(c if c.isalnum() or c in ".-_" else "_" for c in ticker)


# =============================================================================
# Quintile-group (universe) mode
# =============================================================================

def load_universe(csv_path):
    """(open_df, close_df, volume_df): wide date x ticker frames from a
    long-format universe CSV built by download_us_data.py/download_hk_data.py
    (date, ticker, open, high, low, close, volume[, cap, sector, industry]).
    Benchmark columns (e.g. "^GSPC") are dropped - not a stock, doesn't
    belong in a turnover-ranked universe.

    Non-positive prices are masked to NaN (treated as missing, not a real
    price) - found empirically via a real data glitch: yfinance's cached
    OHLCV for ticker "DEC" holds a literal close of 0.0 for a few days in
    Nov 2023 before jumping back to a real price. pct_change() off a 0.0
    prior close is +inf, and since a group's return is a cross-sectional
    mean, that single inf poisons the WHOLE group's return that day -
    cumprod() then stays inf forever after, wrecking every metric
    (Sharpe/Ann Vol/Total all NaN or inf). Masking the bad price to NaN
    turns both adjacent transitions (into and out of the glitch) into NaN
    instead, which skipna correctly excludes."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    open_df = df.pivot(index="date", columns="ticker", values="open").sort_index()
    close_df = df.pivot(index="date", columns="ticker", values="close").sort_index()
    volume_df = df.pivot(index="date", columns="ticker", values="volume").sort_index()
    open_df = open_df.mask(open_df <= 0)
    close_df = close_df.mask(close_df <= 0)
    universe_cols = [c for c in close_df.columns if not c.startswith("^")]
    return open_df[universe_cols], close_df[universe_cols], volume_df[universe_cols]


def fetch_shares_outstanding(tickers, cache_file, pause=0.1, max_retries=1, retry_backoff=15.0):
    """Current shares-outstanding snapshot per ticker (yfinance fast_info -
    no bulk/multi-ticker form exists, so this is one network call per
    uncached ticker), cached to disk as {ticker: shares}. Uses the SAME
    cache file/format as data_download_common.fetch_cap(), so any ticker
    already priced by a prior download_us_data.py run is free here.

    Deliberately NOT time-weighted like the single-ticker mode's
    time_weighted_avg_shares_outstanding() - fetching a full historical
    shares-outstanding series (get_shares_full) for a ~2,500-ticker universe
    would mean ~2,500 extra sequential API calls; a current snapshot is the
    practical tradeoff for ranking a whole universe. Saves progress to disk
    periodically so a long first-time fetch isn't lost if interrupted."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    shares = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    to_fetch = [t for t in tickers if t not in shares]
    if to_fetch:
        print(f"Fetching shares outstanding for {len(to_fetch)}/{len(tickers)} ticker(s) "
              f"not already cached in {cache_file.name} (~{len(to_fetch) * pause / 60:.1f} min minimum) ...")
    for i, t in enumerate(to_fetch):
        def call(t=t):
            info = yf.Ticker(t).fast_info
            return info.get("shares") if hasattr(info, "get") else getattr(info, "shares", None)
        so = common._fetch_one(t, call, max_retries, retry_backoff)
        shares[t] = float(so) if so else np.nan
        time.sleep(pause)
        if (i + 1) % 200 == 0:
            cache_file.write_text(json.dumps(shares))
            print(f"  ... {i + 1}/{len(to_fetch)}")
    cache_file.write_text(json.dumps(shares))
    return pd.Series(shares).reindex(tickers)


def assign_turnover_groups(volume_df, shares, n_groups):
    """Static quintile assignment: rank tickers ONCE by whole-window average
    daily volume / current shares outstanding (a single number per ticker -
    unlike hk_turnover_concentration.py's daily-reranked concentration
    metric), then bucket into n_groups equal-COUNT groups via pd.qcut
    (group 1 = lowest turnover ratio, group n_groups = highest). Tickers
    with missing/zero volume or shares data are dropped before ranking.
    Returns (groups, ratio): groups is a Series {ticker: group (1..n_groups)},
    int-labeled; ratio is the per-ticker turnover ratio (avg daily volume /
    shares outstanding) used to rank/bucket them, for reporting group
    averages separately from the grouping itself."""
    avg_volume = volume_df.mean()
    ratio = (avg_volume / shares).replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    groups = pd.qcut(ratio, n_groups, labels=range(1, n_groups + 1)).astype(int)
    return groups, ratio


def group_returns_by_strategy(open_df, close_df, groups):
    """{group: {label: Series}}, 5 labels per group - "Buy & Hold (no cost)",
    "Intraday (no cost)"/"(with cost)", "Overnight (no cost)"/"(with cost)" -
    mirroring the single-stock mode's `series` dict exactly. Each group's
    return per strategy is the cross-sectional mean (skipna) of its member
    tickers' per-strategy return (daily_returns()/buy_and_hold_return(), the
    same per-ticker formulas as the single-stock mode above), i.e. an
    equal-weight, daily-rebalanced basket ("buy the whole group at the same
    time"). Cost is applied via apply_cost() AFTER averaging across members -
    equivalent to applying it per-ticker first since apply_cost subtracts a
    constant from every day (mean(r_i - c) == mean(r_i) - c)."""
    intraday_df, overnight_df = daily_returns(open_df, close_df)
    buy_and_hold_df = buy_and_hold_return(close_df)
    result = {}
    for g in sorted(groups.unique()):
        members = [t for t in groups[groups == g].index if t in close_df.columns]
        intraday_g = intraday_df[members].mean(axis=1, skipna=True)
        overnight_g = overnight_df[members].mean(axis=1, skipna=True)
        result[g] = {
            "Buy & Hold (no cost)": buy_and_hold_df[members].mean(axis=1, skipna=True),
            "Intraday (no cost)": intraday_g,
            "Intraday (with cost)": apply_cost(intraday_g, TRANSACTION_COST_BPS),
            "Overnight (no cost)": overnight_g,
            "Overnight (with cost)": apply_cost(overnight_g, TRANSACTION_COST_BPS),
        }
    return result


def plot_group_strategy_curves(curves, group, tag, out_path):
    """curves: {label: equity Series (starts at 1.0)}, 5 labels as produced
    by group_returns_by_strategy(). One chart per group, 5 curves overlaid -
    same colour/style convention as plot_equity_curves() (Intraday blue,
    Overnight orange, Buy & Hold green; solid = no cost, dashed = with cost)."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    colors = {"Intraday": "tab:blue", "Overnight": "tab:orange", "Buy & Hold": "tab:green"}
    styles = {"no cost": "-", "with cost": "--"}
    for label, equity in curves.items():
        strategy, cost_label = label.split(" (")
        cost_label = cost_label.rstrip(")")
        ax.plot((equity - 1) * 100, color=colors[strategy], linestyle=styles[cost_label],
                linewidth=1.4, label=label)

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("date")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_title(f"US universe Group {group} ({tag} turnover): equal-weight equity curves", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def overall_market_return(close_df):
    """Equal-weight, daily-rebalanced Buy & Hold return across EVERY ticker
    in the universe (not restricted to any one group) - "buy and hold all
    the stocks", the market-wide benchmark the 5 quintile portfolios are
    compared against."""
    return buy_and_hold_return(close_df).mean(axis=1, skipna=True)


def plot_overnight_vs_market(group_overnight_curves, market_curve, n_groups, out_path):
    """group_overnight_curves: {group: "Overnight (with cost)" equity Series}
    for all n_groups; market_curve: the overall-market Buy & Hold equity
    Series. One chart: n_groups quintile curves (viridis, low-to-high
    turnover) plus one black dashed "Market (buy & hold all stocks)" line."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    cmap = plt.get_cmap("viridis")
    for g, equity in group_overnight_curves.items():
        tag = "lowest" if g == min(group_overnight_curves) else "highest" if g == max(group_overnight_curves) else "mid"
        ax.plot((equity - 1) * 100, color=cmap((g - 1) / max(n_groups - 1, 1)),
                linewidth=1.5, label=f"Group {g} ({tag} turnover) Overnight, with cost")
    ax.plot((market_curve - 1) * 100, color="black", linestyle="--", linewidth=1.6,
            label="Market (buy & hold all stocks)")

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("date")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_title("US universe: overnight (with cost) equity curves by turnover quintile vs. buy & hold market", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_quintile_group_backtest(start=None, end=None):
    """start/end (as resolved by main() from YEARS/START/END) trim the
    loaded universe CSV to that window - the CSV itself isn't re-downloaded,
    so this can only narrow the window to what's already cached, not extend
    it past the CSV's own date range (re-run download_us_data.py --years N
    for that)."""
    print(f"Loading universe from {UNIVERSE_CSV} ...")
    open_df, close_df, volume_df = load_universe(UNIVERSE_CSV)
    if close_df.empty:
        print(f"{UNIVERSE_CSV} not found or empty - run download_us_data.py first.")
        return
    print(f"  {close_df.shape[1]} tickers, full CSV range {close_df.index.min().date()} to {close_df.index.max().date()}")

    if start or end:
        open_df, close_df, volume_df = open_df.loc[start:end], close_df.loc[start:end], volume_df.loc[start:end]
        if close_df.empty:
            print(f"  Requested window {start} to {end or 'today'} has no overlap with the CSV - nothing to backtest.")
            return
        print(f"  Trimmed to requested window ({start} to {end or 'today'}): "
              f"{close_df.index.min().date()} to {close_df.index.max().date()}")

    shares = fetch_shares_outstanding(list(close_df.columns), SHARES_CACHE_FILE, pause=SHARES_FETCH_PAUSE)
    groups, ratio = assign_turnover_groups(volume_df, shares, N_GROUPS)
    print(f"Ranked {len(groups)}/{close_df.shape[1]} tickers into {N_GROUPS} groups "
          f"(dropped {close_df.shape[1] - len(groups)} with missing/zero volume or shares data).")
    print(groups.groupby(groups).size().rename("n_tickers").to_string())

    print(f"\n=== Average turnover ratio (avg daily volume / shares outstanding) by group ===")
    turnover_table = pd.DataFrame({
        "n_tickers": groups.groupby(groups).size(),
        "avg_turnover_ratio": ratio.groupby(groups).mean(),
    })
    turnover_table["avg_turnover_ratio"] = turnover_table["avg_turnover_ratio"].map(lambda x: f"{x:.3%}")
    print(turnover_table.to_string())

    group_returns = group_returns_by_strategy(open_df, close_df, groups)
    all_rows = []
    overnight_with_cost_curves = {}
    for g, series in group_returns.items():
        tag = "lowest" if g == min(group_returns) else "highest" if g == max(group_returns) else "mid"
        curves, rows = {}, []
        for label, r in series.items():
            m = backtest_metrics(r, TRADING_DAYS_PER_YEAR, RISK_FREE_RATE)
            curves[label] = m.pop("equity")
            rows.append({"Strategy": label, "n_tickers": int((groups == g).sum()), **m})
        all_rows.extend({"Group": g, **row} for row in rows)
        overnight_with_cost_curves[g] = curves["Overnight (with cost)"]

        print(f"\n=== Group {g} ({tag} turnover) ===")
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(pd.DataFrame(rows).to_string(index=False))

        out_path = OUT_GROUP_CHART_TEMPLATE.format(group=g)
        plot_group_strategy_curves(curves, g, tag, out_path)
        print(f"Saved chart to {out_path}")

    print("\n=== All groups ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(pd.DataFrame(all_rows).to_string(index=False))

    market_return = overall_market_return(close_df)
    plot_overnight_vs_market(overnight_with_cost_curves, (1 + market_return.dropna()).cumprod(), N_GROUPS, OUT_OVERNIGHT_VS_MARKET_CHART)
    print(f"Saved chart to {OUT_OVERNIGHT_VS_MARKET_CHART}")


# =============================================================================
def main():
    if START:
        start = START
    else:
        start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=YEARS * 365.25)).date().isoformat()

    if RUN_MODE == "quintile_groups":
        run_quintile_group_backtest(start, END)
        return

    print(f"Fetching {len(TICKERS)} ticker(s): {', '.join(TICKERS)} ...")
    open_df, close_df, volume_df = load_open_close(TICKERS, start, END)
    if close_df.empty:
        print("No data retrieved - check TICKERS/date range/network access.")
        return
    print(f"Got {close_df.shape[1]}/{len(TICKERS)} ticker(s), "
          f"{close_df.index.min().date()} to {close_df.index.max().date()}")
    missing = set(TICKERS) - set(close_df.columns)
    if missing:
        print(f"  (no usable data for: {', '.join(sorted(missing))})")

    intraday_df, overnight_df = daily_returns(open_df, close_df)
    buy_and_hold_df = buy_and_hold_return(close_df)
    liq_start, liq_end = close_df.index.min(), close_df.index.max()

    all_rows, liquidity_rows = [], []
    for ticker in close_df.columns:  # only tickers that actually returned data
        series = {
            "Intraday (no cost)": intraday_df[ticker],
            "Intraday (with cost)": apply_cost(intraday_df[ticker], TRANSACTION_COST_BPS),
            "Overnight (no cost)": overnight_df[ticker],
            "Overnight (with cost)": apply_cost(overnight_df[ticker], TRANSACTION_COST_BPS),
            "Buy & Hold (no cost)": buy_and_hold_df[ticker],
        }

        rows, curves = [], {}
        for label, r in series.items():
            m = backtest_metrics(r, TRADING_DAYS_PER_YEAR, RISK_FREE_RATE)
            curves[label] = m.pop("equity")
            rows.append({"Ticker": ticker, "Strategy": label, **m})
        all_rows.extend(rows)

        # sanity check, printed every run: the with/without-cost equity curves
        # must differ (by construction - see apply_cost) whenever cost > 0 and
        # there's at least one valid day; this is NOT just a visual check.
        if TRANSACTION_COST_BPS > 0:
            for strat in ("Intraday", "Overnight"):
                no_cost_end = curves[f"{strat} (no cost)"].iloc[-1]
                with_cost_end = curves[f"{strat} (with cost)"].iloc[-1]
                gap_pp = (no_cost_end - with_cost_end) * 100
                print(f"  [{ticker}] {strat}: no-cost vs. with-cost final equity differ by {gap_pp:.2f}pp "
                      f"({'OK, distinct' if abs(gap_pp) > 1e-9 else 'BUG: IDENTICAL'})")

        print(f"\n=== {ticker} ===")
        table = pd.DataFrame(rows).drop(columns=["Ticker"])
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(table.to_string(index=False))

        out_path = OUT_CHART_TEMPLATE.format(ticker=_safe_filename(ticker))
        plot_equity_curves(curves, ticker, out_path)
        print(f"Saved chart to {out_path}")

        liq = floating_share_turnover_ratio(ticker, volume_df[ticker], liq_start, liq_end)
        liquidity_rows.append({"Ticker": ticker, **liq})
        print(f"  Avg daily volume: {liq['avg_daily_volume']:,.0f} shares | "
              f"time-wtd avg shares outstanding: {liq['tw_avg_shares_outstanding']:,.0f}"
              if np.isfinite(liq["tw_avg_shares_outstanding"])
              else f"  Avg daily volume: {liq['avg_daily_volume']:,.0f} shares | shares-outstanding data unavailable for {ticker}")
        print(f"  -> {liq_start.date()} to {liq_end.date()} avg daily volume / time-wtd avg shares outstanding: "
              f"{liq['turnover_ratio']:.3%}" if np.isfinite(liq["turnover_ratio"]) else "  -> turnover ratio: n/a")

    if len(all_rows) > 4:  # more than one ticker - also show everything side by side
        print("\n=== All tickers ===")
        combined = pd.DataFrame(all_rows)
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(combined.to_string(index=False))

    print(f"\n=== Liquidity: {liq_start.date()} to {liq_end.date()} avg daily volume / time-weighted avg shares outstanding ===")
    liq_table = pd.DataFrame(liquidity_rows)
    liq_table["turnover_ratio"] = liq_table["turnover_ratio"].map(lambda x: f"{x:.3%}" if np.isfinite(x) else "n/a")
    for col in ("avg_daily_volume", "tw_avg_shares_outstanding"):
        liq_table[col] = liq_table[col].map(lambda x: f"{x:,.0f}" if np.isfinite(x) else "n/a")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(liq_table.to_string(index=False))


if __name__ == "__main__":
    main()