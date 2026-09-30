"""
Earnings-gap study: bucket every earnings event by its reaction-day gap into
BIN_WIDTH-wide bins across [GAP_MIN, GAP_MAX] (default 10% bins from -50% to
+50%), and look at the subsequent RETURN_DAYS-day return in each bin. Outputs:
  - a boxplot: one box per gap bin (x = gap bin, y = forward-return range)
  - a table (CSV + printed): average positive return and average negative
    return per bin.
  - a streak-probability table (printed): the empirical probability that a gap is
    followed by another gap in the same direction, by streak length.

Set CONSECUTIVE_GAPS = N (default 1) to focus the boxplot/table on events that
are the N-th consecutive gap in the same direction for a stock (N gap-ups, or N
gap-downs, on successive earnings); N = 1 is the original single-gap study.

How each event is measured
--------------------------
- Gap = reaction-day open / prior close - 1. The "reaction day" is whichever
  of {the first trading day on/after the earnings date, the one after it} has
  the larger |overnight move|. This robustly captures the earnings jump
  without needing to know whether the release was before-open (BMO, reacts
  same day) or after-close (AMC, reacts next day), and sidesteps yfinance's
  inconsistent earnings-timestamp timezones.
- Forward return = close RETURN_DAYS trading days later / reaction-day close - 1
  (enter at the gap day's close, after the gap is already known, then hold).

Data
----
- Prices come from the universe CSVs built by download_hk_data.py /
  download_us_data.py, so the study's depth is bounded by that CSV's own date
  range - download more history (e.g. `python download_us_data.py --years 10`)
  for a richer study.
- Earnings dates come from yfinance, one .earnings_dates call per ticker,
  cached to JSON so re-runs don't re-fetch. yfinance serves ~5-6 years of
  earnings history per ticker.

Usage
-----
Edit the CONFIG block below (RETURN_DAYS, GAP_MIN/GAP_MAX/BIN_WIDTH, MARKETS,
START/END, ...), then:
    python earnings_gap_backtest.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# =============================================================================
# CONFIG - edit these
# =============================================================================
MARKETS = ["us"]  # which universes to include (their events are pooled into the same bins); add "hk" to also study HK
DATA_CSVS = {"us": "us_universe_data.csv", "hk": "hk_universe_data.csv"}  # price source per market
EARNINGS_CACHE = {"us": "us_earnings_dates.json", "hk": "hk_earnings_dates.json"}  # per-ticker earnings-date cache

RETURN_DAYS = 200    # forward-return horizon, in trading days

# Consecutive-gap study. CONSECUTIVE_GAPS = N restricts the forward-return
# boxplot/table to earnings events that are the N-th (or later) gap in the SAME
# direction in a row for that stock - i.e. N consecutive gap-ups, or N
# consecutive gap-downs, on successive earnings. N = 1 (the default) means every
# event qualifies, i.e. the original single-gap study, unchanged. A separate
# streak-probability report (printed) shows how often gaps string together in the
# same direction - the empirical probability of a 2nd, 3rd, ... consecutive
# same-direction gap.
# CONSEC_GAP_THRESHOLD sets how big a gap must be to count as an up/down for the
# streak logic: |gap| <= threshold is "neutral" and breaks a streak (0.0 = use
# the raw sign of the gap).
CONSECUTIVE_GAPS = 1
CONSEC_GAP_THRESHOLD = 0.0

# Volume filter on the gaps. Keep only gaps whose reaction-day volume is at least
# RVOL_MIN times its own trailing RVOL_WINDOW-day average volume (relative volume,
# "RVOL") - a genuine earnings gap normally trades on a big volume spike, so this
# screens out low-conviction gaps. Default RVOL >= 3 over a 50-day baseline. Set
# RVOL_MIN = 0 to disable.
RVOL_MIN = 0
RVOL_WINDOW = 50

# Market-cap group filter. None = all stocks (no filter). Otherwise pick one of
# "small" / "mid" (alias "middle") / "large", split by the two thresholds below
# on each event's reaction-day market cap (close x shares). NOTE: caps are in the
# market's local currency (US = USD, HK = HKD), so if you pool both markets these
# absolute thresholds mix currencies - run a single market, or adjust the
# thresholds, for a clean cap split.
MARKET_CAP_GROUP = 'large'
CAP_SMALL_MAX = 2_000_000_000     # cap < this               -> "small"
CAP_MID_MAX = 10_000_000_000      # CAP_SMALL_MAX <= cap < this -> "mid"; cap >= this -> "large"

# Stop loss on the trade. Options:
#   None       = no stop loss (hold to the RETURN_DAYS exit - the original behaviour).
#   "day_low"  = stop at the ENTRY day's low.
#   "min"      = minimum-LOSS stop: cap the loss at STOP_LOSS_PCT. Stop at the TIGHTER
#                (higher price) of the entry-day low and entry x (1 - STOP_LOSS_PCT) - so
#                if the entry-day low is more than STOP_LOSS_PCT below entry the stop is
#                pulled up to STOP_LOSS_PCT (loss capped); a shallower day low is used
#                as-is. (A day low equal to entry, i.e. close = low, still doesn't arm.)
#   a float f  = fixed stop f below entry, e.g. 0.08 = 8% below entry.
# All percentage stops are measured against the ENTRY price (not a trailing high). The
# stop fires the day AFTER entry if the low breaches the level, and only arms when the
# level sits strictly below entry (a day-low equal to entry is skipped). Fills are
# assumed exactly at the stop level (ignores gap-through slippage).
STOP_LOSS = 0.05
STOP_LOSS_PCT = 0.05   # fraction below ENTRY used by the "min" mode (and standalone as a float above)

# Gap bins for the boxplot/table: one box per BIN_WIDTH-wide gap bucket across [GAP_MIN, GAP_MAX].
# Defaults = 10% bins from -50% to +50% (10 boxes). Events with a gap outside this range are dropped.
GAP_MIN = -0.50
GAP_MAX = 0.50
BIN_WIDTH = 0.10

# Restrict the study to earnings events whose gap (reaction) day falls in [START, END].
# "YYYY-MM-DD" strings, or None for the dataset's own edge. Must be within the CSV's date
# range; the forward return still uses the actual prices after the event (which may run
# slightly past END, as long as they're in the dataset).
START = None
END = None
MAX_TICKERS = None  # cap tickers processed per market (None = all in the CSV); handy for a quick test
INFO_PAUSE = 0.1    # seconds between per-ticker earnings-date API calls (rate-limit courtesy)

OUT_BOXPLOT = "earnings_gap_boxplot.png"      # boxplot: forward-return range per gap bin
OUT_TABLE = "earnings_gap_return_table.csv"   # per-bin avg positive / avg negative forward return

# Optionally dump the individual earnings events that end up counted in the
# boxplot/table (i.e. after ALL filters - period, market cap, RVOL, consecutive -
# and only those whose gap lands in a [GAP_MIN, GAP_MAX] bin) to a CSV, one row
# per event with ticker/date/gap/rvol/cap/streak/forward return. Off by default.
OUTPUT_EVENTS = True
OUT_EVENTS = "earnings_gap_events.csv"

# Boxplot y-axis uses a log scale so the crowded near-zero returns spread out and
# large outliers stop dominating. Forward returns can be negative, so it's a *symlog*
# scale (log away from zero, linear within +-Y_LINTHRESH_PCT % around zero) - the axis
# still labels the real return values, not their logarithms.
Y_LINTHRESH_PCT = 5.0


# =============================================================================
# Data loading
# =============================================================================

def load_prices(csv_path):
    """(open_df, close_df, low_df, volume_df, cap_df) wide date x ticker frames
    from a universe CSV, or 5x None if the file doesn't exist. cap_df is None if
    the CSV has no 'cap' column (e.g. downloaded with --no-cap). Excludes index/
    benchmark tickers (those starting with '^') - indices don't report earnings.
    volume_df feeds the relative-volume (RVOL) gap filter; low_df feeds the
    stop-loss check."""
    path = Path(csv_path)
    if not path.exists():
        return None, None, None, None, None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[~df["ticker"].astype(str).str.startswith("^")]
    open_df = df.pivot(index="date", columns="ticker", values="open").sort_index()
    close_df = df.pivot(index="date", columns="ticker", values="close").sort_index()
    low_df = df.pivot(index="date", columns="ticker", values="low").sort_index()
    volume_df = df.pivot(index="date", columns="ticker", values="volume").sort_index()
    cap_df = df.pivot(index="date", columns="ticker", values="cap").sort_index() if "cap" in df.columns else None
    return open_df, close_df, low_df, volume_df, cap_df


def _fetch_earnings_one(ticker, max_retries, retry_backoff):
    """One ticker's earnings dates as a sorted ISO-date list (empty = the
    ticker genuinely has no earnings coverage on yfinance), or None if it
    kept hitting the rate limit. Distinguishing None (transient) from []
    (genuinely none) matters: the caller caches [] but leaves None uncached
    so a bulk run that got rate-limited retries those tickers next time,
    instead of silently freezing thousands of them as empty."""
    for attempt in range(max_retries):
        try:
            ed = yf.Ticker(ticker).earnings_dates
            return sorted({d.date().isoformat() for d in ed.index}) if ed is not None and not ed.empty else []
        except YFRateLimitError:
            if attempt == max_retries - 1:
                return None
            time.sleep(retry_backoff * (attempt + 1))
        except Exception:
            return []  # bad/delisted ticker or no earnings section - genuinely nothing, cache empty
    return None


def fetch_earnings_dates(tickers, cache_file, pause=INFO_PAUSE, max_retries=4, retry_backoff=15.0):
    """ticker -> sorted list of ISO earnings dates, via yfinance
    .earnings_dates (one call per uncached ticker), cached to JSON. Only the
    DATE is kept - yfinance's earnings-timestamp time/timezone is unreliable
    across markets, and the reaction-day logic doesn't need it. Tickers that
    keep hitting the rate limit are left UNCACHED (re-run to retry them),
    rather than being frozen as empty."""
    cache_path = Path(cache_file)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    to_fetch = [t for t in tickers if t not in cache]
    n_failed = 0
    for i, t in enumerate(to_fetch, 1):
        result = _fetch_earnings_one(t, max_retries, retry_backoff)
        if result is None:
            n_failed += 1  # rate-limited - leave uncached so a re-run retries it
        else:
            cache[t] = result
        time.sleep(pause)
        if i % 100 == 0:
            print(f"    fetched earnings for {i}/{len(to_fetch)} new tickers "
                  f"({n_failed} rate-limited, left to retry) ...")
            cache_path.write_text(json.dumps(cache))  # checkpoint periodically
    cache_path.write_text(json.dumps(cache))
    if n_failed:
        print(f"  NOTE: {n_failed} ticker(s) kept hitting the rate limit and were left uncached - "
              f"re-run to retry just those (already-fetched tickers are skipped).")
    return cache


# =============================================================================
# Event extraction
# =============================================================================

def gap_events(open_df, close_df, low_df, volume_df, cap_df, earnings, return_days, rvol_window=50,
               entry_next_open=False, stop_loss=None, stop_loss_pct=0.10):
    """Return a list of per-event dicts - one for EVERY earnings event (no gap /
    volume / cap filter here; callers apply those where they want them) - each with:
        ticker        : the stock
        earnings_date : the reported earnings date (from yfinance)
        date          : the reaction (gap/entry) trading day
        gap           : signed reaction-day gap (fraction; +0.15 = gapped up 15%, can be ~0)
        rvol          : relative volume = reaction-day volume / its own trailing
                        rvol_window-day average volume (NaN if not computable)
        market_cap    : reaction-day market cap (close x shares), or NaN if no cap data
        fwd_ret       : return_days-day forward return. Entry depends on entry_next_open:
                        False -> enter at the reaction-day close (c[r]);
                        True  -> enter at the NEXT day's open (o[r+1]), because RVOL
                        can only be computed after the gap day closes, so entering
                        at the gap-day close would be look-ahead. Exit is the close
                        return_days trading days after the entry day, UNLESS stop_loss
                        triggers first (see below).
        stopped       : True if stop_loss fired before the timed exit.
    stop_loss: None (hold to the timed exit), "day_low" (stop at the entry day's
    low), "min" (minimum-loss: the tighter of the entry-day low and stop_loss_pct
    below entry, capping the loss at stop_loss_pct), or a float (fixed fractional
    stop below entry, e.g. 0.08 = 8%). All
    percentage levels are relative to the entry price. When the holding window's low
    breaches the stop level, the trade exits at the stop.
    """
    events = []
    dates = close_df.index
    n = len(dates)
    for ticker, ed_list in earnings.items():
        if ticker not in close_df.columns:
            continue
        o = open_df[ticker].to_numpy()
        c = close_df[ticker].to_numpy()
        low = low_df[ticker].to_numpy() if low_df is not None and ticker in low_df.columns else None
        v = volume_df[ticker].to_numpy() if volume_df is not None and ticker in volume_df.columns else None
        cap = cap_df[ticker].to_numpy() if cap_df is not None and ticker in cap_df.columns else None
        for ed_str in ed_list:
            ed = pd.Timestamp(ed_str)
            pos = int(dates.searchsorted(ed))  # first trading day on/after the earnings date
            # Skip earnings that fall outside the price data's range. If ed is before the
            # first cached date, searchsorted returns 0 (no prior close), and without this
            # guard the code would fall back to r=1 for EVERY such out-of-range earnings -
            # giving them all the identical (dates[1] open / dates[0] close) gap and return.
            # ed after the last date is handled by the r+return_days>=n check below.
            if pos < 1 or ed < dates[0] or ed > dates[-1]:
                continue
            # the earnings jump reacts either that session (BMO) or the next (AMC) -
            # take whichever candidate day has the bigger |overnight gap|.
            # next-open entry needs one extra day of forward data (entry is r+1),
            # so the forward window must reach r+1+return_days.
            horizon = return_days + (1 if entry_next_open else 0)
            best = None
            for r in (pos, pos + 1):
                if r < 1 or r + horizon >= n:
                    continue
                prev_close, open_r = c[r - 1], o[r]
                if not np.isfinite(prev_close) or not np.isfinite(open_r) or prev_close <= 0:
                    continue
                gap = open_r / prev_close - 1
                if best is None or abs(gap) > abs(best[1]):
                    best = (r, gap)
            if best is None:
                continue
            r, gap = best
            # entry day e: the gap day for close entry, or the next day for next-open
            # entry. exit day x: return_days trading days after the entry day.
            e = r + 1 if entry_next_open else r
            x = e + return_days
            entry_px = o[e] if entry_next_open else o[e]
            timed_exit = c[x]
            if not np.isfinite(entry_px) or not np.isfinite(timed_exit) or entry_px <= 0:
                continue
            # stop loss: takes effect only the day AFTER entry - if the low breaches
            # the stop level on any holding day from e+1 up to and incl. the exit day,
            # exit at the stop. The stop must sit strictly BELOW the entry price to
            # arm: a stop at/above entry is degenerate (e.g. the entry-day close equals
            # that day's low, so the day-low stop == entry) and is skipped, not fired.
            exit_px, stopped = timed_exit, False
            if stop_loss is not None:
                day_low = low[e] if low is not None and np.isfinite(low[e]) else np.nan
                pct_level = entry_px * (1 - float(stop_loss_pct))
                if stop_loss == "day_low":
                    stop_level = day_low
                elif stop_loss == "min":
                    # minimum-LOSS stop: the TIGHTER (higher price) of the entry-day low
                    # and STOP_LOSS_PCT below entry, so a day low deeper than that % is
                    # capped at the % (loss limited); a shallower day low is used as-is.
                    # nanmax falls back to the % level if the day low is missing.
                    stop_level = np.nanmax([day_low, pct_level])
                else:  # numeric fraction below entry
                    stop_level = entry_px * (1 - float(stop_loss))
                if np.isfinite(stop_level) and stop_level < entry_px and low is not None:
                    window_low = low[e + 1:x + 1]
                    if np.isfinite(window_low).any() and np.nanmin(window_low) <= stop_level:
                        exit_px, stopped = stop_level, True
            market_cap = cap[r] if cap is not None and np.isfinite(cap[r]) else np.nan
            # relative volume: reaction-day volume vs its own trailing average (the
            # rvol_window days BEFORE the gap, so the spike day itself isn't in its
            # own baseline). NaN when there's no usable prior-volume history.
            rvol = np.nan
            if v is not None:
                base = v[max(0, r - rvol_window):r]
                base = base[np.isfinite(base)]
                if base.size and np.isfinite(v[r]) and base.mean() > 0:
                    rvol = v[r] / base.mean()
            events.append({
                "ticker": ticker,
                "earnings_date": ed_str,
                "date": dates[r].date().isoformat(),
                "gap": gap,
                "rvol": rvol,
                "market_cap": market_cap,
                "fwd_ret": exit_px / entry_px - 1,
                "stopped": stopped,
            })
    return events


# =============================================================================
# Binning + boxplot
# =============================================================================

def gap_bin_edges_labels(gap_min, gap_max, bin_width):
    """Bin edges (fractions) and matching labels like '-50..-40%'."""
    edges = np.round(np.arange(gap_min, gap_max + bin_width / 2, bin_width), 10)
    # int(round(...)) avoids "-0" from floating-point negative zero near the boundary
    labels = [f"{int(round(edges[i] * 100))}..{int(round(edges[i + 1] * 100))}%" for i in range(len(edges) - 1)]
    return edges, labels


def plot_boxplot(returns_by_bin, labels, out_path):
    """One box per gap bin, aligned horizontally: x = gap bin, y = the spread
    of RETURN_DAYS-day forward returns (%) for events in that bin. Empty bins
    just leave a gap in the row."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    data, positions = [], []
    for i, vals in enumerate(returns_by_bin):
        vals = np.asarray(vals, dtype=float)
        if len(vals):
            data.append(vals * 100)
            positions.append(i + 1)
    if data:
        ax.boxplot(data, positions=positions, widths=0.6, showfliers=True,
                   medianprops=dict(color="red", linewidth=1.5))
    ax.axhline(0, color="black", linewidth=1)

    # Log (symlog) y-axis: linear within +-Y_LINTHRESH_PCT of zero, logarithmic beyond,
    # so it works with negative returns. Ticks are placed at real return values and
    # labelled as plain numbers (e.g. -50, -20, 0, 20, 50) - not their logarithms.
    ax.set_yscale("symlog", linthresh=Y_LINTHRESH_PCT)
    all_vals = np.concatenate(data) if data else np.array([0.0])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    candidate_ticks = np.array([-500, -200, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 200, 500], dtype=float)
    ticks = candidate_ticks[(candidate_ticks >= lo - Y_LINTHRESH_PCT) & (candidate_ticks <= hi + Y_LINTHRESH_PCT)]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{v:g}"))

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlim(0.5, len(labels) + 0.5)
    entry_lbl = "next-open entry" if RVOL_MIN else "close entry"
    ax.set_xlabel("earnings gap bin")
    ax.set_ylabel(f"{RETURN_DAYS}-day forward return (%, log scale, {entry_lbl})")
    consec_note = "" if CONSECUTIVE_GAPS <= 1 else f" (after {CONSECUTIVE_GAPS} consecutive same-direction gaps)"
    ax.set_title(f"{RETURN_DAYS}-day forward return by earnings-gap bin{consec_note}", fontsize=13)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Consecutive-gap streaks
# =============================================================================

def add_consecutive_runs(df, threshold):
    """Tag each event with the same-direction streak it belongs to, per ticker in
    date order. Adds two columns:
        run_dir : +1 if the gap is an up (> +threshold), -1 if a down
                  (< -threshold), 0 if neutral (|gap| <= threshold)
        run_len : length of the same-direction streak ending at this event
                  (1 for the first gap of a run, 2 for the next same-direction
                  gap, ...); 0 for a neutral gap. A neutral gap breaks a streak.
    So an event with run_dir=+1, run_len=3 is the 3rd consecutive gap-up in a row.
    """
    df = df.sort_values(["market", "ticker", "date"]).reset_index(drop=True)
    mkt = df["market"].to_numpy()
    tkr = df["ticker"].to_numpy()
    gap = df["gap"].to_numpy()
    dirs = np.where(gap > threshold, 1, np.where(gap < -threshold, -1, 0))
    run_len = np.zeros(len(df), dtype=int)
    prev_key, prev_dir, cur = None, 0, 0
    for i in range(len(df)):
        key = (mkt[i], tkr[i])
        d = int(dirs[i])
        if key != prev_key:            # new ticker - reset the streak
            prev_dir, cur = 0, 0
        if d == 0:                     # neutral gap breaks any streak
            cur = 0
        elif d == prev_dir:            # same direction as the previous gap - extend
            cur += 1
        else:                          # opposite direction (or first gap) - fresh streak
            cur = 1
        run_len[i] = cur
        prev_dir, prev_key = d, key
    df["run_dir"] = dirs
    df["run_len"] = run_len
    return df


def streak_probability_table(df):
    """From the run_dir/run_len tags, build the empirical probability that a gap
    continues in the same direction. For each direction and streak length k:
        n_reaching_k   : how many streaks reached at least length k
        p_continue     : P(reach k+1 | reached k) = n_reaching_(k+1) / n_reaching_k
    A run of maximal length L contributes exactly one event at each run_len in
    1..L, so the count of events with run_len == k equals the number of streaks
    that reached length k - which is all this needs."""
    rows = []
    for d, name in [(1, "up"), (-1, "down")]:
        lens = df.loc[df["run_dir"] == d, "run_len"]
        if lens.empty:
            continue
        max_k = int(lens.max())
        n_reaching = {k: int((lens == k).sum()) for k in range(1, max_k + 1)}
        for k in range(1, max_k + 1):
            nk, nk1 = n_reaching[k], n_reaching.get(k + 1, 0)
            rows.append({
                "direction": name,
                "streak_len": k,
                "n_reaching": nk,
                "p_continue_to_next": round(nk1 / nk, 4) if nk else np.nan,
            })
    return pd.DataFrame(rows, columns=["direction", "streak_len", "n_reaching", "p_continue_to_next"])


def cap_group_mask(cap, group, small_max, mid_max):
    """Boolean mask selecting events in the requested market-cap group. Events
    with a NaN cap are always excluded (can't be classified). `group` is
    "small" / "mid" (or "middle") / "large"; anything else raises."""
    g = group.lower()
    if g == "small":
        return cap < small_max
    if g in ("mid", "middle"):
        return (cap >= small_max) & (cap < mid_max)
    if g == "large":
        return cap >= mid_max
    raise ValueError(f"MARKET_CAP_GROUP must be None/'small'/'mid'/'large', got {group!r}")


def main():
    events = []
    for market in MARKETS:
        csv = DATA_CSVS.get(market)
        open_df, close_df, low_df, volume_df, cap_df = load_prices(csv)
        if close_df is None:
            print(f"[{market}] {csv} not found - skipping (build it with download_{market}_data.py)")
            continue
        tickers = list(close_df.columns)
        if MAX_TICKERS:
            tickers = tickers[:MAX_TICKERS]
        print(f"[{market}] {len(tickers)} tickers, prices "
              f"{close_df.index.min().date()} to {close_df.index.max().date()}")
        print(f"[{market}] fetching earnings dates (cached to {EARNINGS_CACHE[market]}) ...")
        earnings = fetch_earnings_dates(tickers, EARNINGS_CACHE[market])
        earnings = {t: earnings.get(t, []) for t in tickers}
        market_events = gap_events(open_df, close_df, low_df, volume_df, cap_df, earnings,
                                   RETURN_DAYS, rvol_window=RVOL_WINDOW,
                                   entry_next_open=bool(RVOL_MIN), stop_loss=STOP_LOSS,
                                   stop_loss_pct=STOP_LOSS_PCT)
        for ev in market_events:
            ev["market"] = market
        print(f"[{market}] {len(market_events)} earnings events with usable price data")
        events += market_events

    df = pd.DataFrame(events, columns=["market", "ticker", "earnings_date", "date",
                                       "gap", "rvol", "market_cap", "fwd_ret", "stopped"])

    # restrict to events whose gap (reaction) day falls in [START, END]
    if START or END:
        d = pd.to_datetime(df["date"])
        mask = pd.Series(True, index=df.index)
        if START:
            mask &= d >= pd.Timestamp(START)
        if END:
            mask &= d <= pd.Timestamp(END)
        before = len(df)
        df = df[mask]
        print(f"Period filter [{START or 'start'} .. {END or 'end'}]: kept {len(df)}/{before} events")

    # market-cap group filter: scopes the whole study (streaks + boxplot) to
    # small / mid / large names. None = all stocks. Applied before the streak
    # tagging so "consecutive gaps" is measured within the chosen cap universe.
    if MARKET_CAP_GROUP:
        before = len(df)
        df = df[cap_group_mask(df["market_cap"], MARKET_CAP_GROUP, CAP_SMALL_MAX, CAP_MID_MAX)]
        print(f"Market-cap filter [{MARKET_CAP_GROUP}] "
              f"(small<{CAP_SMALL_MAX:,.0f}<=mid<{CAP_MID_MAX:,.0f}<=large): kept {len(df)}/{before} events")

    if df.empty:
        print("No earnings events found - check DATA_CSVS/date range/earnings cache "
              "(or a MARKET_CAP_GROUP that matched nothing).")
        return

    # Tag each event with its same-direction streak (per ticker, in date order),
    # then report how often gaps continue in the same direction and, when
    # CONSECUTIVE_GAPS > 1, keep only events that are the N-th consecutive gap.
    df = add_consecutive_runs(df, CONSEC_GAP_THRESHOLD)
    streak = streak_probability_table(df)

    n_up = int((df["run_dir"] == 1).sum())
    n_dn = int((df["run_dir"] == -1).sum())
    n_dir = n_up + n_dn
    thr_note = "" if CONSEC_GAP_THRESHOLD == 0 else f" (|gap| > {CONSEC_GAP_THRESHOLD:.0%} counts as a gap)"
    print(f"\nConsecutive-gap streaks{thr_note}: {n_up} up-gaps, {n_dn} down-gaps"
          + (f"; base rate P(up) = {n_up / n_dir:.1%}, P(down) = {n_dn / n_dir:.1%}" if n_dir else ""))
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(streak.to_string(index=False))
    # focus line for the chosen N: probability of an N-in-a-row streak from a
    # single gap (trivially 100% for N=1, so only worth showing for N>=2).
    if CONSECUTIVE_GAPS >= 2:
        for d, name in [(1, "up"), (-1, "down")]:
            sd = streak[streak["direction"] == name].set_index("streak_len")["n_reaching"]
            if len(sd) and 1 in sd.index and sd.loc[1] > 0:
                reach_n = int(sd.loc[CONSECUTIVE_GAPS]) if CONSECUTIVE_GAPS in sd.index else 0
                print(f"  P({CONSECUTIVE_GAPS} consecutive {name}-gaps | a {name}-gap) = "
                      f"{reach_n}/{int(sd.loc[1])} = {reach_n / sd.loc[1]:.1%}")

    # relative-volume filter on the studied gaps (applied after the streak stats,
    # so it screens which gaps enter the forward-return boxplot without redefining
    # what counts as a "consecutive" earnings gap). NaN-RVOL events are dropped.
    if RVOL_MIN:
        before = len(df)
        df = df[df["rvol"] >= RVOL_MIN]
        print(f"\nRVOL filter: keeping {len(df)}/{before} gaps with reaction-day volume "
              f">= {RVOL_MIN:g}x their trailing {RVOL_WINDOW}-day average.")
        print("  (RVOL is only known after the gap-day close, so entry = NEXT day's open, "
              "not the gap-day close.)")
        if df.empty:
            print("No gaps pass the RVOL filter - lower RVOL_MIN/RVOL_WINDOW.")
            return

    if CONSECUTIVE_GAPS > 1:
        before = len(df)
        df = df[df["run_len"] >= CONSECUTIVE_GAPS]
        print(f"\nCONSECUTIVE_GAPS = {CONSECUTIVE_GAPS}: keeping {len(df)}/{before} events that are the "
              f"{CONSECUTIVE_GAPS}-th+ consecutive same-direction gap (up-gaps land in the positive gap "
              f"bins, down-gaps in the negative bins).")
        if df.empty:
            print("No events with a streak that long - lower CONSECUTIVE_GAPS.")
            return

    # bin every event by its reaction-day gap into [GAP_MIN, GAP_MAX] buckets.
    # pd.cut default is (lo, hi], with include_lowest folding the bottom edge in; gaps
    # outside the range become NaN and are dropped.
    edges, labels = gap_bin_edges_labels(GAP_MIN, GAP_MAX, BIN_WIDTH)
    df["gap_bin"] = pd.cut(df["gap"], bins=edges, labels=labels, include_lowest=True)
    n_binned = int(df["gap_bin"].notna().sum())
    print(f"\nTOTAL: {len(df)} earnings events; {n_binned} with a gap in "
          f"[{GAP_MIN:.0%}, {GAP_MAX:.0%}] binned into {len(labels)} boxes ({RETURN_DAYS}-day forward return)")
    if STOP_LOSS is not None:
        binned = df[df["gap_bin"].notna()]
        n_stopped = int(binned["stopped"].sum())
        if STOP_LOSS == "day_low":
            stop_desc = "entry-day low"
        elif STOP_LOSS == "min":
            stop_desc = f"entry-day low, loss capped at {STOP_LOSS_PCT:.0%}"
        else:
            stop_desc = f"{float(STOP_LOSS):.0%} below entry"
        pct = f" ({n_stopped / len(binned):.1%})" if len(binned) else ""
        print(f"Stop loss ({stop_desc}): {n_stopped}/{len(binned)} binned trades{pct} exited at the stop.")

    # optionally dump the exact events counted in the boxplot/table (post-filter,
    # in-range gap only) so they can be inspected ticker-by-ticker.
    if OUTPUT_EVENTS:
        cols = ["market", "ticker", "earnings_date", "date", "gap", "rvol",
                "market_cap", "run_dir", "run_len", "gap_bin", "fwd_ret", "stopped"]
        events_out = df.loc[df["gap_bin"].notna(), [c for c in cols if c in df.columns]]
        events_out = events_out.sort_values(["date", "ticker"])
        events_out.to_csv(OUT_EVENTS, index=False)
        print(f"Saved {len(events_out)} filtered event(s) to {OUT_EVENTS}")

    returns_by_bin, rows = [], []
    for label in labels:
        r = df.loc[df["gap_bin"] == label, "fwd_ret"].to_numpy()
        returns_by_bin.append(r)
        pos, neg = r[r > 0], r[r < 0]
        rows.append({
            "gap_bin": label,
            "n_events": len(r),
            "n_positive": len(pos),
            "avg_positive_return_%": round(pos.mean() * 100, 2) if len(pos) else np.nan,
            "n_negative": len(neg),
            "avg_negative_return_%": round(neg.mean() * 100, 2) if len(neg) else np.nan,
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT_TABLE, index=False)
    print()
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table.to_string(index=False))

    plot_boxplot(returns_by_bin, labels, OUT_BOXPLOT)
    print(f"\nSaved boxplot to {OUT_BOXPLOT}")
    print(f"Saved table to {OUT_TABLE}")


if __name__ == "__main__":
    main()