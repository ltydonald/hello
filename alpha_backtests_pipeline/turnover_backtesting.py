"""
HK market turnover concentration vs. the Hang Seng Index.

Plots two lines over time on shared dates:
  - the % of total HK stock-market turnover contributed by that day's TOP_N
    highest-turnover stocks (re-ranked every day - "today's top 5", not a fixed
    watchlist), and
  - the HSI close price (right-hand axis), for visual context on the same chart.

Data source
-----------
HKEX does NOT publish this metric, or the underlying per-stock daily turnover,
as a clean downloadable dataset:
  - the "Securities Statistics Archive" only has AGGREGATE daily market turnover
    (no per-stock breakdown);
  - the "Monthly Bulletin" has a "top 20 by turnover" section, but it's MONTHLY
    and delivered as a PDF document viewer, not a machine-readable file.
So this script computes the metric itself from the daily close+volume this
project already downloads for the whole HK equity/REIT universe (~2,790
tickers) via download_hk_data.py - turnover = close x volume per ticker per
day (the same dollar-turnover proxy already used for this project's ADV
liquidity screen), summed for the TOP_N names and divided by the summed
turnover of the whole universe. HSI's own close price is pulled from the same
CSV (download_hk_data.py saves ^HSI into it as BENCHMARK_TICKER).

This is an approximation of HKEX's own turnover figure (close x volume vs. the
exchange's actual traded value), but is accurate enough for a concentration
RANKING/SHARE metric - which 5 names dominate a given day is a stable, obvious
signal even with a proxy.

Usage
-----
Edit the CONFIG block below, then:
    python hk_turnover_concentration.py
Needs hk_universe_data.csv already built (python download_hk_data.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# CONFIG - edit these
# =============================================================================
# "hk" (HK universe vs. HSI) or "us" (Russell 3000 universe vs. S&P 500) - picks
# the data CSV, benchmark ticker, and every "HK"/"HSI" label below automatically.
MARKET = "us"
_DATA_CSV_BY_MARKET = {"hk": "hk_universe_data.csv", "us": "us_universe_data.csv"}
_BENCHMARK_TICKER_BY_MARKET = {"hk": "^HSI", "us": "^GSPC"}
_BENCHMARK_NAME_BY_MARKET = {"hk": "HSI", "us": "S&P 500"}
_MARKET_LABEL_BY_MARKET = {"hk": "HK", "us": "US"}

DATA_CSV = _DATA_CSV_BY_MARKET[MARKET]              # from download_hk_data.py / download_us_data.py
BENCHMARK_TICKER = _BENCHMARK_TICKER_BY_MARKET[MARKET]  # must match that script's own BENCHMARK_TICKER
BENCHMARK_NAME = _BENCHMARK_NAME_BY_MARKET[MARKET]      # display name used in every chart/print label
MARKET_LABEL = _MARKET_LABEL_BY_MARKET[MARKET]          # e.g. "HK market turnover concentration..."

TOP_N = 20             # how many highest-turnover stocks per day to sum ("top 5")
START = None          # "YYYY-MM-DD" or None for the dataset's own start
END = None            # "YYYY-MM-DD" or None for the dataset's own end

# A day needs at least this many tickers reporting turnover to be included -
# guards against a nonsense ratio on days with very sparse coverage (e.g. right
# at the edge of the downloaded history, before most tickers have data yet).
MIN_TICKERS_PER_DAY = 50

# Three independent window knobs, kept separate on purpose so you can test e.g.
# "does a 5-day concentration reading predict HSI's NEXT 3-day return" instead of
# forcing everything onto one shared window:
#
# CONCENTRATION_DAYS: window (trading days) for the concentration x-axis -
#   REGRESSION_X_MODE == "level"  -> trailing rolling-AVERAGE of the daily share.
#   REGRESSION_X_MODE == "change" -> trailing rolling-CUMULATIVE %-change of it.
CONCENTRATION_DAYS = 20

# HSI_CUM_DAYS: window (trading days) for HSI's own cumulative return, i.e.
# (close_t / close_t-HSI_CUM_DAYS - 1) x 100 - independent of CONCENTRATION_DAYS.
HSI_CUM_DAYS = 30

# SHIFT_DAYS: shifts HSI's return window FORWARD by this many trading days
# relative to the concentration measurement date - 0 (default) = no shift, the
# HSI window ends on the SAME date as the concentration reading (contemporaneous/
# overlapping - tests correlation, not prediction). SHIFT_DAYS > 0 moves that
# window into the future (via a plain .shift(-SHIFT_DAYS)), so day t's y becomes
# the return that will actually happen AFTER day t - this is what actually tests
# whether concentration has PREDICTIVE power over HSI's subsequent move. Set
# SHIFT_DAYS == HSI_CUM_DAYS for a clean non-overlapping forward N-day return
# starting exactly at the concentration reading; any other combination is fine
# too (e.g. a 1-day-ahead 5-day return), it just means the forward window starts
# SHIFT_DAYS after t rather than exactly at t.
SHIFT_DAYS = 30

# Output filenames are per-market (MARKET=="hk" keeps the original names so
# nothing changes for existing HK runs; MARKET=="us" gets its own "us_..." set
# so the two markets' outputs never overwrite each other).
_FILE_PREFIX_BY_MARKET = {"hk": "hk", "us": "us"}
_PREFIX = _FILE_PREFIX_BY_MARKET[MARKET]

OUT_CHART = f"{_PREFIX}_turnover_concentration.png"
OUT_STACKED = f"{_PREFIX}_turnover_concentration_stacked.png"  # benchmark's own price (top panel) + concentration curve (bottom panel), stacked, shared date axis

# Regression x-axis mode:
#   "level"  = the smoothed concentration LEVEL (%) - "how concentrated was
#              trading, on average, over the last CONCENTRATION_DAYS days".
#   "change" = the rolling CUMULATIVE percentage change of the concentration
#              percentage over the same window, i.e. (share_t / share_t-N - 1)
#              x 100 - "how much did that concentration level itself move, in
#              total, over the last CONCENTRATION_DAYS days" (same compounded-
#              not-averaged construction as the benchmark's own cumulative return).
REGRESSION_X_MODE = "level"
OUT_REGRESSION_LEVEL = f"{_PREFIX}_turnover_concentration_regression.png"           # REGRESSION_X_MODE == "level" (unchanged filename for MARKET=="hk")
OUT_REGRESSION_CHANGE = f"{_PREFIX}_turnover_concentration_regression_change.png"   # REGRESSION_X_MODE == "change"

# Degree of the fitted curve: 1 = ordinary linear regression (a straight line,
# the original behaviour); 2 = quadratic; 3 = cubic; etc. Fit via least-squares
# polynomial (numpy.polyfit); R^2 and the p-value (an overall F-test of the fit
# vs. a flat mean) are computed the same way regardless of degree, and match
# scipy.stats.linregress exactly when REGRESSION_DEGREE == 1.
REGRESSION_DEGREE = 1


# =============================================================================
# Data loading
# =============================================================================

def load_data(csv_path, benchmark_ticker):
    """(close_df, volume_df, hsi_close) - close_df/volume_df are wide date x
    ticker frames for the tradeable universe (the benchmark ticker excluded),
    hsi_close is the benchmark's own close price Series. Returns (None, None,
    None) if the CSV doesn't exist."""
    path = Path(csv_path)
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path, parse_dates=["date"])
    close_df = df.pivot(index="date", columns="ticker", values="close").sort_index()
    volume_df = df.pivot(index="date", columns="ticker", values="volume").sort_index()
    hsi_close = close_df[benchmark_ticker] if benchmark_ticker in close_df.columns else None
    universe_cols = [c for c in close_df.columns if not c.startswith("^")]
    return close_df[universe_cols], volume_df[universe_cols], hsi_close


# =============================================================================
# Concentration metric
# =============================================================================

def top_n_turnover_share(close_df, volume_df, top_n, min_tickers):
    """For every date, the % of total turnover (close x volume, summed across
    the universe) contributed by that date's TOP_N highest-turnover tickers.
    Returns (share_pct, n_reporting) as same-indexed Series; share_pct is NaN
    on any date with fewer than min_tickers tickers ACTIVELY TRADED (nonzero
    turnover) - not merely having a recorded price. Many HK GEM micro-caps carry
    a valid close on days they don't trade at all (zero volume); counting those
    as "reporting" let a handful of genuinely active names swallow ~100% of a
    gate-passing day's total, which is a data-coverage artifact, not real
    concentration."""
    turnover = close_df.to_numpy(dtype=float) * volume_df.to_numpy(dtype=float)
    n_reporting = np.sum(turnover > 0, axis=1)  # NaN compares False, so this also excludes missing data

    filled = np.where(np.isnan(turnover), -np.inf, turnover)
    top_vals = np.sort(filled, axis=1)[:, -top_n:]           # top_n largest per row (ascending order)
    top_vals = np.where(np.isneginf(top_vals), 0.0, top_vals)  # pad value when fewer than top_n reported
    top_sum = top_vals.sum(axis=1)
    total = np.nansum(turnover, axis=1)

    share = np.full(len(total), np.nan)
    ok = total > 0
    share[ok] = top_sum[ok] / total[ok] * 100

    share = pd.Series(share, index=close_df.index)
    n_reporting = pd.Series(n_reporting, index=close_df.index)
    share = share.where(n_reporting >= min_tickers)
    return share, n_reporting


def smooth(series, smooth_days):
    """Trailing rolling-average smoothing shared by the concentration line and
    the HSI return line, so both are on the same footing (0/1 = no smoothing)."""
    if smooth_days <= 1:
        return series
    return series.rolling(smooth_days, min_periods=max(1, smooth_days // 2)).mean()


def cum_pct_change(series, window):
    """Rolling window-day CUMULATIVE (compounded, not averaged) percentage
    change of `series`, in %: (x_t / x_t-window - 1) x 100. Used for both HSI's
    price (-> its window-day cumulative return) and, in "change" regression
    mode, the concentration line itself (-> how much concentration itself moved
    over the window, vs. smooth()'s day-to-day average change). The first
    `window` rows are NaN (a genuine window-day change needs `window` prior
    values; there's no meaningful partial-window version of it)."""
    return (series / series.shift(window) - 1) * 100


def hsi_forward_return(hsi_close, cum_days, shift_days):
    """HSI's cum_days-day cumulative return (cum_pct_change), then pulled
    shift_days into the future via .shift(-shift_days) so day t's value is the
    return that actually happens AFTER day t, not the one ending at t. shift_days
    == 0 is a no-op (contemporaneous, current behaviour). The LAST shift_days
    rows become NaN (there's no future data yet to pull in for those days) -
    that's correct, not a bug: you can't test predictive power on days you don't
    have a future outcome for."""
    ret = cum_pct_change(hsi_close, cum_days)
    return ret.shift(-shift_days) if shift_days else ret


def window_label(cum_days, shift_days):
    """Shared 'HSI Nd cumulative return[, shifted +Md forward]' phrasing used by
    both the chart and the regression, so their labels always describe the same
    thing the underlying series actually is."""
    base = f"{cum_days}-day" if cum_days > 1 else "daily"
    return f"{base}{f', shifted +{shift_days}d forward' if shift_days else ''}"


# =============================================================================
# Plot
# =============================================================================

def plot_chart(share_pct, hsi_ret, top_n, concentration_days, x_mode, hsi_cum_days, shift_days,
               benchmark_name, market_label, out_path):
    """The blue concentration line uses the SAME x_mode/CONCENTRATION_DAYS as the
    regression scatter's x-axis (regression_scatter), so the two outputs describe
    the exact same concentration series - just one over time, one against the
    benchmark's return. hsi_ret is already forward-shifted (or not) by the caller."""
    fig, ax1 = plt.subplots(figsize=(13, 6.5))

    if x_mode == "level":
        plot_share = smooth(share_pct, concentration_days)
        share_label = f"Top {top_n} turnover share, {concentration_days}d avg" if concentration_days > 1 \
            else f"Top {top_n} turnover share (daily)"
        share_ylabel = f"Top {top_n} stocks' turnover, % of total market turnover"
        y1_decimals = 0
    elif x_mode == "change":
        plot_share = cum_pct_change(share_pct, concentration_days)
        c_window_note = f"{concentration_days}-day" if concentration_days > 1 else "daily"
        share_label = f"Top {top_n} turnover share, {c_window_note} cumulative % change"
        share_ylabel = f"Top {top_n} turnover share, cumulative % change (over {c_window_note})"
        y1_decimals = 1
    else:
        raise ValueError(f"x_mode must be 'level' or 'change', got {x_mode!r}")

    l1, = ax1.plot(plot_share.index, plot_share.values, color="tab:blue", linewidth=1.2,
                   label=share_label)
    ax1.set_ylabel(share_ylabel, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=y1_decimals))
    if x_mode == "change":
        ax1.axhline(0, color="tab:blue", linewidth=0.6, alpha=0.4)
    ax1.grid(alpha=0.3)

    hsi_note = window_label(hsi_cum_days, shift_days)
    ax2 = ax1.twinx()
    l2, = ax2.plot(hsi_ret.index, hsi_ret.values, color="tab:red", linewidth=1.0, alpha=0.85,
                   label=f"{benchmark_name} {hsi_note} cumulative return")
    ax2.axhline(0, color="tab:red", linewidth=0.6, alpha=0.4)
    ax2.set_ylabel(f"{benchmark_name} {hsi_note} cumulative return (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
    ax1.set_xlabel("date")
    ax1.set_title(f"{market_label} market turnover concentration (top {top_n} stocks, {x_mode}) vs. {benchmark_name} {hsi_note} cumulative return", fontsize=13)
    ax1.legend(handles=[l1, l2], loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stacked(share_pct, hsi_close, top_n, concentration_days, x_mode, benchmark_name, market_label, out_path):
    """Two panels stacked vertically, sharing the date axis, over the whole
    testing period: the benchmark's own closing PRICE on top (for visual context
    - not a return, the actual index level), the concentration curve (same
    x_mode/CONCENTRATION_DAYS view as the other outputs) on the bottom."""
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                            gridspec_kw={"height_ratios": [1, 1]})

    ax_top.plot(hsi_close.index, hsi_close.values, color="tab:red", linewidth=1.0)
    ax_top.set_ylabel(f"{benchmark_name} close price")
    ax_top.set_title(f"{benchmark_name} price vs. {market_label} market turnover concentration (top {top_n} stocks, {x_mode})", fontsize=13)
    ax_top.grid(alpha=0.3)

    if x_mode == "level":
        plot_share = smooth(share_pct, concentration_days)
        share_ylabel = f"Top {top_n} stocks' turnover, % of total market turnover"
        y_decimals = 0
    elif x_mode == "change":
        plot_share = cum_pct_change(share_pct, concentration_days)
        c_window_note = f"{concentration_days}-day" if concentration_days > 1 else "daily"
        share_ylabel = f"Top {top_n} turnover share, cumulative % change (over {c_window_note})"
        y_decimals = 1
    else:
        raise ValueError(f"x_mode must be 'level' or 'change', got {x_mode!r}")

    ax_bottom.plot(plot_share.index, plot_share.values, color="tab:blue", linewidth=1.0)
    ax_bottom.set_ylabel(share_ylabel)
    ax_bottom.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=y_decimals))
    if x_mode == "change":
        ax_bottom.axhline(0, color="tab:blue", linewidth=0.6, alpha=0.4)
    ax_bottom.grid(alpha=0.3)
    ax_bottom.set_xlabel("date")
    ax_bottom.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_bottom.xaxis.get_major_locator()))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_polynomial(x, y, degree):
    """Least-squares fit of y ~ x as an order-`degree` polynomial (1 = a straight
    line, i.e. ordinary linear regression). Returns a dict with:
        coeffs   - polynomial coefficients, HIGHEST power first (np.polyfit order)
        r2       - R^2 of the fit
        p        - p-value of an overall F-test (is the fit better than just
                   predicting the mean of y?) - for degree=1 this is
                   mathematically identical to scipy.stats.linregress's p-value
                   (the F-test on 1 predictor reduces to the t-test on its slope).
        n        - number of points used
    NaN r2/p if there are too few points to support the fit (n <= degree + 1)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if n - degree - 1 > 0 and np.isfinite(r2) and r2 < 1:
        f_stat = (r2 / degree) / ((1 - r2) / (n - degree - 1))
        p = stats.f.sf(f_stat, degree, n - degree - 1)
    else:
        p = np.nan
    return {"coeffs": coeffs, "r2": r2, "p": p, "n": n, "degree": degree}


def poly_equation_str(coeffs):
    """coeffs are HIGHEST power first (np.polyfit order). Builds a signed
    'y = a x^2 + b x + c'-style string, e.g. for [0.01, -0.2, 3.1] ->
    'y = 0.010x^2 - 0.200x + 3.100'."""
    degree = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        power = degree - i
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        if power == 0:
            term = f"{mag:.3f}"
        elif power == 1:
            term = f"{mag:.3f}x"
        else:
            term = f"{mag:.3f}x^{power}"
        parts.append((sign, term))
    eq = parts[0][1] if parts[0][0] == "+" else f"-{parts[0][1]}"
    for sign, term in parts[1:]:
        eq += f" {sign} {term}"
    return "y = " + eq


# =============================================================================
# Regression: HSI cumulative return ~ turnover concentration (both same window)
# =============================================================================

def regression_scatter(share_pct, hsi_ret, concentration_days, x_mode, hsi_cum_days, shift_days, degree,
                       benchmark_name, out_path):
    """Regresses HSI's (possibly forward-shifted) cumulative return on a view of
    the concentration line - x_mode picks which view, each over its own
    CONCENTRATION_DAYS window (independent of HSI_CUM_DAYS/SHIFT_DAYS):
        "level"  - the concentration_days-day rolling-average LEVEL of the
                   turnover share (%) (smooth()). Asks: does a HIGH concentration
                   level coincide with (or, with shift_days > 0, PRECEDE) a big
                   HSI move?
        "change" - the concentration_days-day rolling CUMULATIVE percentage
                   change of that concentration level (cum_pct_change(), the
                   same compounded-not-averaged construction used for HSI's own
                   return). Asks: does concentration itself moving a lot (rather
                   than merely being high) coincide with / precede a big HSI move?
    hsi_ret is expected already computed (via hsi_forward_return) by the caller -
    this function only regresses and plots. degree=1 is ordinary linear
    regression (a straight line); degree>1 fits a polynomial curve instead (see
    fit_polynomial). Returns (fit dict, n) - n is the number of overlapping
    pairs used - or (None, 0) if there aren't enough points for the requested
    degree.

    Caveat: x is built from an overlapping trailing window, so consecutive
    points are serially correlated, which inflates how significant the fit looks
    (p-value/R^2 are optimistic) - treat this as a descriptive scatter/slope,
    not a rigorous hypothesis test. Using shift_days > 0 (non-overlapping x/y
    periods, e.g. shift_days == hsi_cum_days) is the more defensible setup for
    actually claiming predictive power. A higher degree will also always fit the
    SAMPLE at least as well as a lower one even with no real relationship
    (more free parameters) - watch the p-value, not just a rising R^2, before
    reading anything into a curved fit.
    """
    if x_mode == "level":
        x = smooth(share_pct, concentration_days)
        x_desc = "top-N turnover share, % of total market turnover"
    elif x_mode == "change":
        x = cum_pct_change(share_pct, concentration_days)
        x_desc = "cumulative % change in top-N turnover share"
    else:
        raise ValueError(f"x_mode must be 'level' or 'change', got {x_mode!r}")

    y = hsi_ret
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < degree + 2:
        return None, 0

    fit = fit_polynomial(df["x"], df["y"], degree)
    hsi_note = window_label(hsi_cum_days, shift_days)
    c_window_note = f"{concentration_days}-day" if concentration_days > 1 else "daily"
    # level is a rolling AVERAGE ("Nd avg"); change is a rolling CUMULATIVE move
    # over the window ("over N days") - different operations, different phrasing.
    x_note = f", {concentration_days}d avg" if x_mode == "level" and concentration_days > 1 \
        else f" (over {c_window_note})" if x_mode == "change" else ""

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df["x"], df["y"], s=14, alpha=0.5, color="tab:blue")
    xs = np.linspace(df["x"].min(), df["x"].max(), 200)
    ax.plot(xs, np.polyval(fit["coeffs"], xs), color="tab:red", linewidth=2,
            label=f"{poly_equation_str(fit['coeffs'])}\n"
                  f"R² = {fit['r2']:.3f}, p = {fit['p']:.3g}, n = {fit['n']}")
    ax.axhline(0, color="grey", linewidth=0.6)
    if x_mode == "change":
        ax.axvline(0, color="grey", linewidth=0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0 if x_mode == "level" else 1))
    ax.set_xlabel(f"{x_desc}{x_note}")
    ax.set_ylabel(f"{benchmark_name} {hsi_note} cumulative return (%)")
    ax.set_title(f"{benchmark_name} {hsi_note} cumulative return\nvs. turnover concentration ({x_mode})", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fit, fit["n"]


# =============================================================================
def main():
    close_df, volume_df, hsi_close = load_data(DATA_CSV, BENCHMARK_TICKER)
    if close_df is None:
        print(f"{DATA_CSV} not found - build it with: python download_{MARKET}_data.py")
        return
    if hsi_close is None:
        print(f"{BENCHMARK_TICKER} not found in {DATA_CSV} - re-run download_{MARKET}_data.py "
              f"(it saves the benchmark ticker into the same CSV).")
        return

    print(f"Loaded {close_df.shape[1]} tickers, {close_df.index.min().date()} to {close_df.index.max().date()}")

    share_pct, n_reporting = top_n_turnover_share(close_df, volume_df, TOP_N, MIN_TICKERS_PER_DAY)
    # computed on the full series BEFORE any START/END trim, so the first kept day
    # still has HSI_CUM_DAYS of real prior closes to compute its window return
    # from, and the last kept day still has SHIFT_DAYS of real future closes to
    # shift in (if SHIFT_DAYS > 0).
    hsi_ret = hsi_forward_return(hsi_close, HSI_CUM_DAYS, SHIFT_DAYS)

    if START or END:
        mask = pd.Series(True, index=share_pct.index)
        if START:
            mask &= share_pct.index >= pd.Timestamp(START)
        if END:
            mask &= share_pct.index <= pd.Timestamp(END)
        share_pct, n_reporting = share_pct[mask], n_reporting[mask]
        hsi_ret = hsi_ret[hsi_ret.index.isin(share_pct.index)]

    valid = share_pct.dropna()
    dropped = len(share_pct) - len(valid)
    print(f"{len(valid)} usable date(s)" +
          (f" ({dropped} dropped for < {MIN_TICKERS_PER_DAY} tickers reporting)" if dropped else "") +
          f", {valid.index.min().date()} to {valid.index.max().date()}")
    print(f"Top {TOP_N} turnover share: mean {valid.mean():.1f}%, min {valid.min():.1f}%, "
          f"max {valid.max():.1f}%, latest {valid.iloc[-1]:.1f}% ({valid.index[-1].date()})")

    # align HSI return to the same dates for the chart
    hsi_plot = hsi_ret.reindex(share_pct.index).dropna()

    plot_chart(share_pct, hsi_plot, TOP_N, CONCENTRATION_DAYS, REGRESSION_X_MODE,
               HSI_CUM_DAYS, SHIFT_DAYS, BENCHMARK_NAME, MARKET_LABEL, OUT_CHART)
    print(f"Saved chart to {OUT_CHART}")

    # benchmark's own price, trimmed to the same (START/END-restricted) testing period
    hsi_price_plot = hsi_close.reindex(share_pct.index)
    plot_stacked(share_pct, hsi_price_plot, TOP_N, CONCENTRATION_DAYS, REGRESSION_X_MODE,
                BENCHMARK_NAME, MARKET_LABEL, OUT_STACKED)
    print(f"Saved stacked chart to {OUT_STACKED}")

    out_regression = OUT_REGRESSION_LEVEL if REGRESSION_X_MODE == "level" else OUT_REGRESSION_CHANGE
    reg, n = regression_scatter(share_pct, hsi_ret, CONCENTRATION_DAYS, REGRESSION_X_MODE,
                                HSI_CUM_DAYS, SHIFT_DAYS, REGRESSION_DEGREE, BENCHMARK_NAME, out_regression)
    if reg is None:
        print("Not enough overlapping data points for the regression.")
    else:
        x_desc = f"{CONCENTRATION_DAYS}-day avg concentration" if REGRESSION_X_MODE == "level" \
            else f"{CONCENTRATION_DAYS}-day cumulative %-change in concentration"
        degree_desc = "linear" if REGRESSION_DEGREE == 1 else f"degree-{REGRESSION_DEGREE} polynomial"
        print(f"Regression ({degree_desc}) {BENCHMARK_NAME} {window_label(HSI_CUM_DAYS, SHIFT_DAYS)} cumulative return ~ {x_desc}: "
              f"{poly_equation_str(reg['coeffs'])}, "
              f"R^2={reg['r2']:.4f}, p={reg['p']:.4g}, n={n}")
        print(f"Saved regression scatter to {out_regression}")


if __name__ == "__main__":
    main()