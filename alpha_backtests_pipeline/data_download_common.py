"""
Shared download/reshape logic for download_hk_data.py and download_us_data.py.

Both scripts fetch OHLCV (+ approximate market cap, sector/industry) via
yfinance for a fixed ticker list and reshape it into one long-format CSV
"database" (date, ticker, open, high, low, close, volume, cap, sector,
industry). This module holds the market-agnostic pieces; each script only
supplies its own ticker list and market-specific CONFIG (interval, cache
directory, output filename, lookback default).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

HK_UNIVERSE_FILE = Path(__file__).parent / "hk_equity_universe.csv"
US_UNIVERSE_FILE = Path(__file__).parent / "us_equity_universe.csv"


def load_hk_universe(path=HK_UNIVERSE_FILE):
    """All currently-listed HKD-denominated ordinary equities + REITs on HKEX
    Main Board/GEM (2,790 tickers as of the snapshot below) - the single
    source of truth for the HK ticker universe, shared by download_hk_data.py
    and alpha_ML.py so the two can't silently drift apart the way they used to.

    Provenance: HKEX's official "List of Securities" (updated as at
    2026-08-03), downloaded from
    https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx,
    filtered to Category in {Equity, Real Estate Investment Trusts},
    Sub-Category in {Equity Securities (Main Board), Equity Securities (GEM),
    Investment Companies} (REITs have no Sub-Category), Trading Currency ==
    HKD. Excludes Derivative Warrants, CBBCs, Debt Securities, Exchange
    Traded Products, Equity Warrants, and Depositary Receipts (DRs report the
    underlying foreign parent's market cap, not a genuine HK-listed float -
    the same reasoning the prior hand-curated list used to drop the
    Cisco/Intel/Applied Materials DR pilot codes, just applied consistently
    to all DRs instead of only those three).

    This is a much broader, less performance-conditioned universe than the
    old "top ~500 by today's market cap" list: it isn't restricted to
    survivors that grew large enough to still rank today, which is exactly
    the "must be a current winner" bias that selection criterion added on
    top of survivorship. It does NOT fix true survivorship bias, though -
    delisted/renamed names that dropped off HKEX before this snapshot are
    still absent, since this is still today's listed universe, not a
    point-in-time reconstruction. Refresh periodically by re-running the
    same fetch+filter against a fresh copy of the HKEX file.
    """
    return pd.read_csv(path)["ticker"].tolist()


def load_us_universe(path=US_UNIVERSE_FILE):
    """Russell 3000 constituents (2,580 tickers as of the snapshot below) -
    the single source of truth for the US ticker universe, mirroring
    load_hk_universe() below.

    Provenance: iShares' Russell 3000 ETF (IWV) full holdings file -
    https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/latest-holdings.csv
    (as at 2026-07-31) - filtered to Asset Class == Equity and a standard
    ticker pattern (drops ~5 non-tradable private-placement/escrow/CVR lines
    that don't carry a real ticker). IWV's holdings are a live ETF proxy for
    the index, not the index's own official constituent file, but Russell
    itself doesn't publish that for free - this is the standard practical
    substitute.

    An earlier version of this universe was "every currently-listed US
    common stock" (~5,626 tickers, all of Nasdaq/NYSE/etc.) - broader and
    less performance-conditioned, but 2-3x slower to fetch OHLCV/cap/sector
    for. Russell 3000 was chosen as a faster, still-broad (covers ~98% of
    US investable market cap) middle ground - it's an index-membership cutoff,
    so it reintroduces a mild version of the same "must be large/liquid
    enough to qualify" conditioning the broader list was built to avoid, just
    a much less restrictive one than a S&P-500-style top-few-hundred list.

    Not a point-in-time reconstruction either way - delisted/renamed names
    from before this snapshot are absent, and Russell 3000 membership itself
    drifts with each reconstitution. Refresh periodically by re-downloading
    the IWV holdings file and re-running the same filter.
    """
    return pd.read_csv(path)["ticker"].tolist()


def fetch_ohlcv(tickers, start, end=None, interval="1d", min_obs=100, auto_adjust=True,
                 batch_size=150, batch_pause=1.0, max_retries=1, retry_backoff=20.0):
    """Download OHLCV and reshape into 5 date x ticker DataFrames.

    Drops any ticker with fewer than `min_obs` non-NaN closes (delisted /
    barely-traded / bad ticker), so downstream alphas don't choke on all-NaN
    columns.

    `auto_adjust` defaults to True: split-adjustment always applies in
    yfinance regardless of this flag, but dividend adjustment doesn't - with
    auto_adjust=False, a stock's ex-dividend price drop shows up as a pure
    (and spurious) negative return, since the dividend cash flow that
    actually offsets it is never added back. Defaulting to True keeps this
    consistent with alpha_ML.py's PRICE_ADJUST.

    Fetched in batches of `batch_size` tickers (one yf.download call per
    batch) rather than all at once: yfinance/Yahoo rate-limits large
    multi-ticker requests, and a single request for thousands of tickers at
    once was observed to silently fail ~50% of them. Tickers with no usable
    data after a batch are retried up to `max_retries` times with a growing
    backoff before being given up on as genuinely unavailable.
    """
    opens, highs, lows, closes, volumes = {}, {}, {}, {}, {}

    def _extract(raw, batch):
        ok, missing = [], []
        for t in batch:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                missing.append(t)
                continue
            if sub is None or sub.empty or sub["Close"].dropna().empty:
                missing.append(t)
                continue
            opens[t] = sub["Open"]
            highs[t] = sub["High"]
            lows[t] = sub["Low"]
            closes[t] = sub["Close"]
            volumes[t] = sub["Volume"]
            ok.append(t)
        return missing

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        pending = batch
        for attempt in range(max_retries):
            raw = yf.download(
                pending, start=start, end=end, interval=interval,
                auto_adjust=auto_adjust, group_by="ticker", threads=True, progress=False,
            )
            pending = _extract(raw, pending)
            if not pending:
                break
            time.sleep(retry_backoff * (attempt + 1))
        if i + batch_size < len(tickers):
            time.sleep(batch_pause)

    min_obs_ok = {t for t, s in closes.items() if s.dropna().shape[0] >= min_obs}
    opens = {t: v for t, v in opens.items() if t in min_obs_ok}
    highs = {t: v for t, v in highs.items() if t in min_obs_ok}
    lows = {t: v for t, v in lows.items() if t in min_obs_ok}
    closes = {t: v for t, v in closes.items() if t in min_obs_ok}
    volumes = {t: v for t, v in volumes.items() if t in min_obs_ok}

    close_df = pd.DataFrame(closes)
    cols = close_df.columns
    return (pd.DataFrame(opens)[cols], pd.DataFrame(highs)[cols], pd.DataFrame(lows)[cols],
            close_df, pd.DataFrame(volumes)[cols])


def _pivot_wide(long_df, tickers):
    """Pivot a subset of a long-format (date, ticker, open, high, low, close,
    volume, ...) CSV into the same 5-tuple of wide date x ticker DataFrames
    fetch_ohlcv() returns."""
    df = long_df[long_df["ticker"].isin(tickers)]
    if df.empty:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        return empty, empty.copy(), empty.copy(), empty.copy(), empty.copy()
    return (
        df.pivot(index="date", columns="ticker", values="open"),
        df.pivot(index="date", columns="ticker", values="high"),
        df.pivot(index="date", columns="ticker", values="low"),
        df.pivot(index="date", columns="ticker", values="close"),
        df.pivot(index="date", columns="ticker", values="volume"),
    )


def _concat_wide(wide_a, wide_b):
    """Row-concat two same-shaped 5-tuples of wide frames (e.g. old cached
    dates + newly-fetched dates for the same tickers), keeping the most
    recently fetched value on any date the two happen to share."""
    out = []
    for a, b in zip(wide_a, wide_b):
        combined = pd.concat([a, b])
        combined = combined[~combined.index.duplicated(keep="last")]
        out.append(combined.sort_index())
    return tuple(out)


def _combine_first_wide(wide_a, wide_b):
    """Cell-wise merge of two same-shaped 5-tuples of wide frames: keep wide_a's
    value wherever it's non-NaN, and fill the gaps (plus any dates/tickers only
    wide_b has) from wide_b. Unlike _concat_wide's whole-row 'keep last', this
    fills individual NaN cells - needed when tickers have DIFFERENT cached date
    ranges, so a short-history ticker's freshly-backfilled older cells aren't
    clobbered by the NaNs the wide pivot padded it with (a row-wise keep-last
    would take the padded-NaN row from the cached frame and wipe the backfill)."""
    return tuple(a.combine_first(b) for a, b in zip(wide_a, wide_b))


def _union_wide(wide_frames):
    """Column-union same-shaped 5-tuples of wide frames that cover disjoint
    ticker sets (e.g. brand-new tickers vs. previously-cached ones), aligning
    on the combined date index."""
    wide_frames = [w for w in wide_frames if not w[3].empty]
    if not wide_frames:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        return empty, empty.copy(), empty.copy(), empty.copy(), empty.copy()
    return tuple(pd.concat([w[i] for w in wide_frames], axis=1) for i in range(5))


def fetch_ohlcv_incremental(tickers, existing_csv, start, end=None, min_obs=1, **fetch_kwargs):
    """Like fetch_ohlcv(), but reuses whatever's already cached in
    `existing_csv` (a long-format CSV a prior run of this same script wrote)
    instead of re-downloading it from scratch:

    - Tickers already in the CSV fetch only the days OUTSIDE their cached
      range: the days since their last cached date (forward, to `end`), AND -
      when the requested window reaches further back than the cache - the
      older slice from `start` up to the first cached date (backfill). So
      widening --years (e.g. 3 -> 5) now fills in the extra older history
      instead of leaving the ticker stuck at its cached span. A ticker whose
      cache already covers the whole [start, end] window isn't re-fetched.
    - Tickers not yet in the CSV fetch their full requested window.
    - Tickers in the CSV but not in the current `tickers` request are
      dropped from the result entirely - the "delete what I don't need"
      half of an incremental update.
    - The combined result is then trimmed to [start, end], dropping any
      cached history outside the currently-requested window, and min_obs is
      re-applied to the final merged-and-trimmed series (a ticker survives
      only if its *total* history meets the bar, not just the newly-fetched
      slice).

    Caveat: with auto_adjust=True, dividend/split adjustment is anchored to the
    end of each yfinance request, so a backfilled (or forward-extended) slice is
    on a slightly different adjustment basis than the originally-cached slice if
    a corporate action falls between them - a minor discontinuity at the join.
    Use --full-refresh when you need every ticker on one consistent basis.

    Returns the same 5-tuple of wide DataFrames as fetch_ohlcv().
    """
    existing_long = None
    if existing_csv is not None and Path(existing_csv).exists():
        existing_long = pd.read_csv(existing_csv, parse_dates=["date"])
        if existing_long.empty:
            existing_long = None

    if existing_long is None:
        return fetch_ohlcv(tickers, start, end, min_obs=min_obs, **fetch_kwargs)

    cached_tickers = set(existing_long["ticker"].unique())
    keep_existing = [t for t in tickers if t in cached_tickers]
    new_tickers = [t for t in tickers if t not in cached_tickers]

    frame_sets = []

    if new_tickers:
        print(f"  {len(new_tickers)} ticker(s) not yet cached - fetching full history for those ...")
        frame_sets.append(fetch_ohlcv(new_tickers, start, end, min_obs=0, **fetch_kwargs))

    if keep_existing:
        old_wide = _pivot_wide(existing_long, keep_existing)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

        # Use each ticker's OWN cached span, not the global one: tickers can have
        # different first/last cached dates (a recent IPO, or a ticker whose history
        # was backfilled further in an earlier run than its peers). Backfill must
        # reach the ticker that starts LATEST (max first date) and forward-extend from
        # the ticker that ends EARLIEST (min last date), so a mix of e.g. 3yr and 5yr
        # tickers still fills the 3yr ones instead of skipping backfill because the
        # global earliest date already reaches `start`. Merging is cell-wise
        # (_combine_first_wide): cached real values win, only NaN gaps get filled.
        spans = existing_long[existing_long["ticker"].isin(keep_existing)].groupby("ticker")["date"]
        max_first = spans.min().max()   # latest per-ticker first cached date
        min_last = spans.max().min()    # earliest per-ticker last cached date

        merged = old_wide
        did_fetch = False

        if start_ts < max_first:
            print(f"  backfilling older history {start_ts.date()} .. {max_first.date()} "
                  f"for {len(keep_existing)} cached ticker(s) (per-ticker, up to whichever starts latest) ...")
            back_wide = fetch_ohlcv(keep_existing, start, max_first.date().isoformat(),
                                    min_obs=0, **fetch_kwargs)
            merged = _combine_first_wide(merged, back_wide)
            did_fetch = True

        incr_start = min_last + pd.Timedelta(days=1)
        if incr_start <= end_ts:
            print(f"  extending forward {incr_start.date()} .. {end_ts.date()} "
                  f"for {len(keep_existing)} cached ticker(s) ...")
            new_wide = fetch_ohlcv(keep_existing, incr_start.date().isoformat(), end, min_obs=0, **fetch_kwargs)
            merged = _combine_first_wide(merged, new_wide)
            did_fetch = True

        if not did_fetch:
            print(f"  {len(keep_existing)} ticker(s) already cover the requested window - not re-fetched.")
        frame_sets.append(merged)

    open_df, high_df, low_df, close_df, volume_df = _union_wide(frame_sets)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else None
    mask = (close_df.index >= start_ts) & (close_df.index <= end_ts if end_ts is not None else True)
    open_df, high_df, low_df, close_df, volume_df = (
        open_df.loc[mask], high_df.loc[mask], low_df.loc[mask], close_df.loc[mask], volume_df.loc[mask]
    )

    keep_cols = close_df.columns[close_df.notna().sum(axis=0) >= min_obs]
    return (open_df[keep_cols], high_df[keep_cols], low_df[keep_cols],
            close_df[keep_cols], volume_df[keep_cols])


def _fetch_one(t, call, max_retries=1, retry_backoff=15.0):
    """Run `call()` for a single ticker, retrying only on YFRateLimitError
    (with growing backoff) - any other exception means the ticker itself is
    the problem (delisted, bad symbol, etc.), not the rate limit, so retrying
    it would just waste time. Returns None if every retry is exhausted."""
    for attempt in range(max_retries):
        try:
            return call()
        except YFRateLimitError:
            if attempt == max_retries - 1:
                return None
            time.sleep(retry_backoff * (attempt + 1))
        except Exception:
            return None
    return None


def fetch_cap(tickers, close_df, cache_file, pause=0.1, max_retries=1, retry_backoff=15.0):
    """Approximate market cap = latest shares outstanding x historical close.

    One yfinance call per ticker (fast_info has no bulk/multi-ticker form
    the way OHLCV does), so this is the most rate-limit-exposed step in the
    whole pipeline if left unpaced - `pause` is a mandatory courtesy delay
    between every single call, not just an option.
    """
    shares = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    for t in tickers:
        if t in shares:
            continue

        def call(t=t):
            info = yf.Ticker(t).fast_info
            # yfinance's FastInfo key is "shares" (not "shares_outstanding")
            return info.get("shares") if hasattr(info, "get") else getattr(info, "shares", None)

        so = _fetch_one(t, call, max_retries, retry_backoff)
        shares[t] = float(so) if so else np.nan
        time.sleep(pause)

    cache_file.write_text(json.dumps(shares))
    shares_s = pd.Series(shares).reindex(close_df.columns)
    return close_df.mul(shares_s, axis=1)


def fetch_industry(tickers, cache_file, pause=0.1, max_retries=1, retry_backoff=15.0):
    """Best-effort sector/industry classification for indneutralize()."""
    data = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    for t in tickers:
        if t in data:
            continue

        def call(t=t):
            info = yf.Ticker(t).info
            return {"sector": info.get("sector"), "industry": info.get("industry")}

        result = _fetch_one(t, call, max_retries, retry_backoff)
        data[t] = result if result is not None else {"sector": None, "industry": None}
        time.sleep(pause)

    cache_file.write_text(json.dumps(data))
    sector = pd.Series({t: v.get("sector") for t, v in data.items()})
    industry = pd.Series({t: v.get("industry") for t, v in data.items()})
    return sector, industry


def to_long_format(open_df, high_df, low_df, close_df, volume_df, cap_df=None, sector=None, industry=None):
    """Reshape the 5 (+cap) wide date x ticker DataFrames into one long CSV-ready frame."""
    frames = []
    for t in close_df.columns:
        frame = pd.DataFrame({
            "date": close_df.index,
            "ticker": t,
            "open": open_df[t].values,
            "high": high_df[t].values,
            "low": low_df[t].values,
            "close": close_df[t].values,
            "volume": volume_df[t].values,
        })
        if cap_df is not None and t in cap_df.columns:
            frame["cap"] = cap_df[t].values
        if sector is not None:
            frame["sector"] = sector.get(t)
        if industry is not None:
            frame["industry"] = industry.get(t)
        frames.append(frame)

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["close"])
    return long_df.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_arg_parser(*, market_label, default_tickers_help, default_out, default_years, default_days, default_min_adv=0):
    p = argparse.ArgumentParser(description=f"Download {market_label} stock data into a long-format CSV database (schema-compatible with hk_alpha101_single.py)")
    p.add_argument("--tickers", nargs="*", default=None, help=default_tickers_help)
    p.add_argument("--years", type=float, default=None, help=f"years of history to download (default: {default_years})")
    p.add_argument("--days", type=int, default=None, help=f"days of history to download, overrides --years (default: {default_days})")
    p.add_argument("--start", default=None, help="explicit start date (YYYY-MM-DD), overrides --years/--days")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default=None, help="yfinance bar interval, e.g. '1d', '1h' (default: market-specific)")
    p.add_argument("--min-obs-frac", type=float, default=0.5, help="drop a ticker with fewer than this fraction x expected bars of history")
    p.add_argument("--min-adv", type=float, default=default_min_adv,
                    help="drop a ticker whose average daily dollar volume (close * volume, meaned over the "
                         "downloaded date range) is below this, in the market's local currency - screens out "
                         "illiquid penny/microcap names before they're ever saved to the CSV (default: "
                         f"{default_min_adv:,.0f}, use 0 to disable)")
    p.add_argument("--info-pause", type=float, default=0.1, help="seconds to sleep between yfinance per-ticker .info/fast_info calls (cap + sector/industry fetch) - the most rate-limit-exposed step, since those have no bulk/multi-ticker form")
    p.add_argument("--no-adjust", dest="adjust", action="store_false",
                    help="fetch raw (ex-dividend) close instead of the default dividend/split-adjusted total-return price. "
                         "Splits are always adjusted by yfinance regardless of this flag - only dividend adjustment toggles.")
    p.set_defaults(adjust=True)
    p.add_argument("--no-cap", action="store_true", help="skip shares-outstanding fetch (used by alpha056)")
    p.add_argument("--no-industry", action="store_true", help="skip sector/industry fetch (used by indneutralize alphas)")
    p.add_argument("--full-refresh", action="store_true",
                    help="ignore any existing --out CSV and redownload everything from scratch. Default is "
                         "incremental: tickers already in the CSV only fetch days since their last cached date "
                         "(or aren't re-fetched at all if already current), new tickers fetch their full window, "
                         "and the result is trimmed to the currently-requested --years/--days/--start window and "
                         "ticker list either way - so a narrower rerun both adds what's missing and drops what's "
                         "no longer wanted.")
    p.add_argument("--out", default=default_out)
    return p


def _bars_per_year(interval):
    """Approximate trading bars per year for a yfinance --interval string,
    used only to size the min-history cutoff (min_obs) below - doesn't need
    to be exact, just the right order of magnitude. Previously this assumed
    any non-"1d" interval was intraday (~6.5 trading hrs/day), which silently
    set an impossibly high min_obs for --interval 1wk/1mo/3mo (e.g. weekly
    data has ~52 bars/year, not ~1,764) and dropped every single ticker."""
    return {
        "1d": 252, "5d": 252 / 5, "1wk": 52, "1mo": 12, "3mo": 4,
    }.get(interval, 252 * 7)  # anything else (1m/5m/15m/30m/60m/1h/90m/...) is intraday


def _sniff_cached_bar_gap(existing_csv):
    """Median day-gap between consecutive cached bars for one ticker in
    `existing_csv`, or None if there's no usable existing file. Used to catch
    an --interval switch on an existing --out file before incremental mode
    would silently mix granularities (see the ValueError in run())."""
    if existing_csv is None or not Path(existing_csv).exists():
        return None
    df = pd.read_csv(existing_csv, usecols=["date", "ticker"], parse_dates=["date"])
    if df.empty:
        return None
    first_ticker = df["ticker"].iloc[0]
    dates = df.loc[df["ticker"] == first_ticker, "date"].sort_values()
    if len(dates) < 2:
        return None
    gap = dates.diff().dt.days.median()
    return gap if gap and gap > 0 else None


def run(*, tickers, args, interval_default, cache_dir, market_name, benchmark_ticker=None):
    """`benchmark_ticker`: an index ticker (e.g. "^HSI") to fetch and save into
    the same CSV alongside `tickers`, so a backtest script can read the
    benchmark's price series straight out of the CSV instead of making its own
    live yfinance call every run. Exempted from the --min-adv liquidity screen
    below (an index reports 0 volume in yfinance, so it would otherwise always
    fail any positive dollar-volume floor) and from the ticker-count logging,
    since it isn't part of the tradeable universe.
    """
    cache_dir.mkdir(exist_ok=True)
    interval = args.interval or interval_default
    fetch_tickers = list(tickers)
    if benchmark_ticker and benchmark_ticker not in fetch_tickers:
        fetch_tickers.append(benchmark_ticker)

    if args.start:
        start = args.start
    elif args.days is not None:
        start = (pd.Timestamp.today().normalize() - pd.DateOffset(days=args.days)).date().isoformat()
    elif args.years is not None:
        start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=args.years * 365.25)).date().isoformat()
    else:
        raise ValueError("caller must resolve a default --years/--days before calling run()")

    bars_per_year = _bars_per_year(interval)  # only used for the min-history cutoff (min_obs) below
    period_years = (args.days / 365.25) if args.days is not None else (args.years if args.years is not None else 1)
    min_obs = max(1, int(period_years * bars_per_year * args.min_obs_frac))

    if not args.full_refresh:
        cached_gap = _sniff_cached_bar_gap(args.out)
        if cached_gap is not None:
            expected_gap = 365.25 / bars_per_year
            if cached_gap / expected_gap > 2 or expected_gap / cached_gap > 2:
                raise ValueError(
                    f"{args.out} looks like it was built with a different --interval than {interval!r}: "
                    f"its cached bars are spaced ~{cached_gap:.1f} days apart, but --interval {interval!r} "
                    f"expects ~{expected_gap:.1f} days apart. Incremental mode has no way to tell the two "
                    f"apart - it would just extend the existing (wrong-granularity) history forward instead "
                    f"of re-fetching at the new interval, silently mixing bar spacings. Re-run with "
                    f"--full-refresh to rebuild {args.out} at {interval!r}, or point --out at a different file."
                )

    fetch_settings = dict(interval=interval, auto_adjust=args.adjust)
    bench_note = f" + benchmark ({benchmark_ticker})" if benchmark_ticker else ""
    if args.full_refresh:
        print(f"Downloading {interval} OHLCV for {len(tickers)} {market_name} tickers{bench_note}, {start} to {args.end or 'today'} (--full-refresh) ...")
        open_df, high_df, low_df, close_df, volume_df = fetch_ohlcv(
            fetch_tickers, start, args.end, min_obs=min_obs, **fetch_settings,
        )
    else:
        print(f"Updating {interval} OHLCV for {len(tickers)} {market_name} tickers{bench_note}, "
              f"final window {start} to {args.end or 'today'} (incremental - reusing {args.out} where possible) ...")
        open_df, high_df, low_df, close_df, volume_df = fetch_ohlcv_incremental(
            fetch_tickers, args.out, start, args.end, min_obs=min_obs, **fetch_settings,
        )
    # close_df.shape[0] is the UNION of every distinct date across every ticker, not any one
    # ticker's own bar count - for --interval 1d that's the same thing (every ticker shares the
    # same trading calendar), but for 1wk/1mo a stray ticker or two anchored to a different
    # weekday/month-day inflates it well past what any individual ticker actually has (e.g. 2
    # tickers out of 2307 anchored to Wednesday instead of Monday turned "522 bars/ticker" into
    # a misleading "867 bars" here). Report the per-ticker median instead - what you'd actually
    # expect a normal ticker to have - and keep the union count too, just clearly labeled.
    per_ticker_bars = close_df.notna().sum(axis=0)
    median_bars = int(per_ticker_bars.median()) if len(per_ticker_bars) else 0
    print(f"Got {close_df.shape[1]}/{len(fetch_tickers)} ticker(s), median {median_bars} bars/ticker "
          f"({close_df.shape[0]} distinct dates across all tickers combined).")
    if close_df.shape[1] == 0:
        print("No data retrieved - check tickers/date range/network access.")
        return

    if args.min_adv:
        avg_dv = (close_df * volume_df).mean(axis=0, skipna=True)
        keep = avg_dv[avg_dv >= args.min_adv].index
        if benchmark_ticker and benchmark_ticker in avg_dv.index and benchmark_ticker not in keep:
            keep = keep.union([benchmark_ticker])  # never screen out the benchmark on liquidity
        dropped = len(avg_dv) - len(keep)
        if dropped:
            print(f"ADV screen: dropping {dropped}/{len(avg_dv)} tickers below "
                  f"{args.min_adv:,.0f} average daily dollar volume (before the cap/sector .info fetch, "
                  f"so screened-out tickers never cost a per-ticker API call) ...")
            open_df, high_df, low_df, close_df, volume_df = (
                open_df[keep], high_df[keep], low_df[keep], close_df[keep], volume_df[keep]
            )

    cap_df = None
    if not args.no_cap:
        print("Fetching shares outstanding (for cap-based alphas, e.g. alpha056)...")
        cap_df = fetch_cap(list(close_df.columns), close_df, cache_dir / "shares_outstanding.json", pause=args.info_pause)

    sector = industry = None
    if not args.no_industry:
        print("Fetching sector/industry classification (for indneutralize-based alphas)...")
        sector, industry = fetch_industry(list(close_df.columns), cache_dir / "industry.json", pause=args.info_pause)

    print("Reshaping into a long-format CSV ...")
    long_df = to_long_format(open_df, high_df, low_df, close_df, volume_df, cap_df, sector, industry)
    long_df.to_csv(args.out, index=False)
    note = ("" if market_name == "HK" else
            " (hk_alpha101_single.py only reads the CSV schema - it works fine on non-HK tickers despite the name)")
    print(
        f"\nSaved {len(long_df):,} rows ({long_df['ticker'].nunique()} tickers, "
        f"{long_df['date'].min().date()} to {long_df['date'].max().date()}) to {args.out}\n"
        f"Run backtests against it with: python hk_alpha101_single.py --data {args.out}{note}"
    )