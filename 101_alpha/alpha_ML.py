"""
WorldQuant "101 Formulaic Alphas" (Kakushadze, 2015) as LightGBM features for a
long/short portfolio, switchable between an HK and a US stock universe.

End-to-end, single-file pipeline:
  1. Universe: --market hk (~2,790 currently-listed HK equities/REITs) or
     --market us (~2,580 Russell 3000 constituents), shared with
     download_hk_data.py/download_us_data.py (see MARKETS/resolve_market()
     below, and load_hk_universe()/load_us_universe() in
     data_download_common.py).
  2. Download daily OHLCV (+ approximate market cap, sector/industry) via
     yfinance, caching it to the same hk_universe_data.csv/us_universe_data.csv
     that download_hk_data.py/download_us_data.py use (see resolve_market()) -
     whichever script builds it first, the other reuses it instead of
     re-downloading. Everything else ML-specific (predictions, chart, tuning
     log) still lives under alpha_ml_cache/<market>/.
  3. Compute all 101 formulaic alphas from the paper (operators in Appendix
     A.2, formulas in Appendix A.1), cross-sectionally standardize each one
     ("_xs" features).
  4. Build a forward-return target over a configurable holding period
     (HOLDING_DAYS trading days).
  5. Walk-forward train a LightGBM regressor on the alphas and backtest a
     top-K / bottom-K long/short portfolio, rebalanced every HOLDING_DAYS
     trading days, against the selected market's benchmark (^HSI for hk,
     ^GSPC for us).

Everything you're likely to want to tweak is in the CONFIG block below.

Usage
-----
    python alpha_ML.py                    # --market hk (default)
    python alpha_ML.py --market us
    python alpha_ML.py --market us --stage test

First run downloads and caches data to hk_universe_data.csv/us_universe_data.csv
(skipped entirely if download_hk_data.py/download_us_data.py already built it -
can take a *long* while otherwise at these universe sizes: OHLCV for thousands
of tickers x YEARS_HISTORY years, plus two yfinance ``.info``-style calls per
ticker for cap and sector/industry - expect this to run for hours, not
minutes, and to be far more likely to hit yfinance rate limits than a
hand-picked few-hundred-ticker list was. Subsequent runs reuse the CSV unless
you delete it or flip USE_PRICE_CACHE off.

Caveats
-------
- vwap: yfinance daily bars carry no intraday VWAP, so it is approximated as
  the typical price (high + low + close) / 3.
- cap: no historical market-cap series is free; it is approximated as
  (today's shares outstanding) x (historical close), i.e. share-count drift
  from buybacks/issuance is ignored. This is also a look-ahead bias, not just
  an approximation: today's share count is applied to every past date.
- sector/industry: current yfinance classification is used for all history
  (no point-in-time GICS reclassification, i.e. another look-ahead bias),
  and yfinance has no "subindustry" level, so subindustry falls back to
  industry.
- universe survivorship: the ticker list is everything *currently* listed as
  of the load_hk_universe()/load_us_universe() snapshot, so names that
  delisted/renamed before that snapshot are still missing (survivorship
  bias). Broader than a hand-picked "top names by today's market cap/index
  membership" list - which stacks an extra "must still be a big winner
  today" selection on top of survivorship - but still not a true
  point-in-time universe. Fine for a features/ML demo; not fine for a
  publication-grade backtest.
- transaction costs: charged on turnover at TRANSACTION_COST (0.08%) per unit of
  weight traded (see walk_forward_ls) - this is a flat estimate, not a real cost
  model, so it excludes market impact, bid-ask spread beyond the flat rate, and
  short-borrow financing cost for the short leg.
- dividend adjustment: stock OHLCV and the benchmark are fetched with
  different `auto_adjust` settings by default (see PRICE_ADJUST in CONFIG),
  which biases the strategy-vs-benchmark comparison unless aligned.
"""

import argparse
import hashlib
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
import data_download_common as common
from alpha101_engine import (
    MarketData, ALL_ALPHAS, INDNEUTRALIZE_ALPHAS,
    rank, scale, indneutralize, delay, delta, correlation, covariance, sum_, product_,
    stddev, ts_min, ts_max, ts_argmax, ts_argmin, ts_rank, decay_linear,
    abs_, log_, sign_, signedpower, iif, emax, emin,
)

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (13, 5.5), 'font.size': 11})

# =============================================================================
# CONFIG — edit these
# =============================================================================

# Market being backtested - switch with --market {hk,us}, the same way hk_alpha101_single.py
# switches universes via --data. Everything market-specific (ticker universe, benchmark,
# cache location) lives here; every CONFIG value below this dict is a strategy
# hyperparameter and applies to either market equally.
# shared_csv matches download_hk_data.py's/download_us_data.py's own --out default,
# so both scripts read and write the exact same file - run either one first (or
# both, at different times) and the other reuses whatever's already there instead
# of redownloading it. See resolve_market()'s docstring for what that implies.
MARKETS = {
    "hk": dict(universe_loader=common.load_hk_universe, benchmark_ticker="^HSI", shared_csv="hk_universe_data.csv"),
    "us": dict(universe_loader=common.load_us_universe, benchmark_ticker="^GSPC", shared_csv="us_universe_data.csv"),
}
MARKET_DEFAULT = "hk"

HOLDING_DAYS       = 1          # duration: forward-return horizon & rebalance spacing (trading days)

# Execution timing/slippage: alpha(t) only uses data through close(t), so the earliest you
# could realistically act on it is the next trading session - not literally at close(t)
# itself, which would mean zero-latency execution on the same closing print your signal was
# computed from. EXECUTION_LAG_DAYS is how many trading days after the signal date you
# actually trade; EXECUTION_PRICE_FIELD is which price you trade at on that day ("open" or
# "close"). Both entry and exit use this same field/lag convention, HOLDING_DAYS apart, so a
# full round trip is e.g. open-to-open rather than mixing an open entry with a close exit.
# Applied consistently to the ML label/backtest AND the benchmark fetch, for the same reason
# PRICE_ADJUST is applied to both - a mismatched convention on either side biases the
# strategy-vs-benchmark comparison. Set EXECUTION_LAG_DAYS=0, EXECUTION_PRICE_FIELD="close"
# to restore the old (unrealistic, zero-latency) same-day-close assumption.
EXECUTION_LAG_DAYS = 1
EXECUTION_PRICE_FIELD = "open"

K                  = 10         # number of stocks in each leg (top-K long / bottom-K short)
LONG_ONLY          = False      # True: equal-weight only the top-K names each rebalance (no shorting, no
                                 # leverage - gross exposure 1.0, vs 2.0 for long+short = 100% long + 100% short)
RETRAIN_EVERY      = 126        # retrain LightGBM every N trading days (~1y); use a big number to train once
YEARS_HISTORY      = 10         # years of daily OHLCV history to download
YEARS_DEFAULT      = None         # default for --years, which trims an already-cached CSV's own date range
                                 # (independent of YEARS_HISTORY - see --years help below); None or <=0 for everything cached
OOS_START          = None       # "YYYY-MM-DD", or None -> use OOS_FRACTION of the downloaded history instead
OOS_FRACTION       = 0.2        # only used when OOS_START is None: last this-fraction of dates becomes the OOS test window

# --stage tune: hold out and never even load the OOS window (rows with date >= oos_start
# are dropped right after the split, not just "not looked at"); of what's left, the last
# VALIDATION_FRACTION of dates becomes a validation window used to compare hyperparameter
# configs (edit HOLDING_DAYS/K/RETRAIN_EVERY/LGB_PARAMS/TRANSACTION_COST above, rerun
# `--stage tune`, compare). Each run's config + validation result is appended to that
# market's tuning_log_file (see resolve_market()). --stage test: fit on all pre-OOS
# data (train+validation) and evaluate once, for real, on the true OOS window - run
# this after tuning is done, and only once.
STAGE              = "tune"     # "tune" or "test"; override with --stage
VALIDATION_FRACTION = 0.2       # fraction of the pre-OOS dates held out as the validation window, in --stage tune
MIN_OBS_FRAC       = 0.5        # drop a ticker if it has fewer than this fraction x expected trading days
SECTOR_COVERAGE_MIN = 0.5       # if fewer than this fraction of tickers have a sector label, skip indneutralize alphas
TRADING_DAYS_PER_YEAR = 252     # used to convert YEARS_HISTORY -> min observations and to annualize performance stats

TRAIN_TARGET       = "y_xs"     # label the model is trained on (cross-sectionally standardized forward return)
EVAL_TARGET        = "y_raw"    # label used for portfolio P&L / IC evaluation (un-standardized forward return)

# Cost per unit of portfolio weight actually traded (0.08% = 8bps), charged only on
# turnover between rebalances - a position held unchanged since the prior rebalance
# costs nothing, even though naively it looks like 1/K of the book "traded" again.
# See walk_forward_ls() for the turnover accounting. Excludes market impact/slippage
# and short-borrow cost - see the docstring caveats for other unmodeled costs.
TRANSACTION_COST   = 0.0008

# Cache the computed alpha panel (all 101 alphas + target) on disk, keyed on the underlying
# price data + whatever actually affects the panel's contents (see build_feature_panel_cached()
# below for exactly what's in the key). Hyperparameters that only affect what happens
# *downstream* of the panel - K, RETRAIN_EVERY, LGB_PARAMS, TRANSACTION_COST, OOS_FRACTION,
# VALIDATION_FRACTION, STAGE - are deliberately NOT in the key, so tuning those reuses the
# cached panel instead of recomputing all 101 alphas every run. Editing an alpha formula or
# operator IS automatically detected (via a source-code hash) and busts the cache; deleting
# alpha_ml_cache/<market>/panel_cache_*.pkl also forces a recompute.
PANEL_CACHE = True

# yfinance auto_adjust: True = dividend+split-adjusted "total return" prices, False = raw close.
# Applied consistently to BOTH the stock universe and the benchmark so the L/S-vs-benchmark
# comparison isn't biased by one side including dividends and the other not. Changing this
# invalidates the existing price cache (delete the market's cache dir, or flip USE_PRICE_CACHE
# off, to force a re-download).
PRICE_ADJUST       = True

INFO_FETCH_PAUSE   = 0.1        # seconds to sleep between yfinance per-ticker .info/fast_info calls (cap + sector/industry
                                 # fetch) - the most rate-limit-exposed step, since those have no bulk/multi-ticker form

USE_PRICE_CACHE    = True       # reuse cached OHLCV/cap/industry (delete the cache dir to force a refresh)

# Each market gets its own cache subdirectory (alpha_ml_cache/hk/, alpha_ml_cache/us/) so
# switching --market can't mix up or overwrite the other market's cached data, predictions,
# tuning log, or chart. See resolve_market() below for how these paths get built per run.
CACHE_ROOT = Path(__file__).parent / "alpha_ml_cache"


def resolve_market(market):
    """Build every market-specific path/setting for this run: ticker universe,
    benchmark, and a dedicated cache subdirectory (so --market hk and --market us
    never read or write each other's cached data).

    price_cache_file points at the shared root-level CSV (hk_universe_data.csv /
    us_universe_data.csv) rather than a private alpha_ml_cache copy, so this
    reuses whatever download_hk_data.py/download_us_data.py already downloaded
    instead of independently re-fetching the same OHLCV/cap/sector data a second
    time. download_and_cache_data() only falls back to its own fresh download
    (into that same shared path) if the file doesn't exist yet - one script can
    build it, the other reuses it, either order.

    Caveat: once that shared CSV exists, its own history window (whatever
    --years/--days the download script was last run with) is what you get,
    regardless of this file's YEARS_HISTORY - the cache-hit path doesn't
    check or care about YEARS_HISTORY at all. Delete the CSV (or point
    price_cache_file elsewhere) if you need alpha_ML.py to pull a different
    window than the download scripts last built.
    """
    cfg = MARKETS[market]
    cache_dir = CACHE_ROOT / market
    cache_dir.mkdir(parents=True, exist_ok=True)
    return dict(
        tickers=cfg["universe_loader"](),
        benchmark_ticker=cfg["benchmark_ticker"],
        cache_dir=cache_dir,
        price_cache_file=Path(__file__).parent / cfg["shared_csv"],
        shares_cache_file=cache_dir / "shares_outstanding.json",
        industry_cache_file=cache_dir / "industry.json",
        tuning_log_file=cache_dir / "tuning_log.csv",
    )

# Any lightgbm.LGBMRegressor keyword can go here - commonly tuned ones beyond the defaults
# below: num_leaves, subsample (+ subsample_freq), colsample_bytree, min_child_samples.
LGB_PARAMS = dict(
    max_depth=5, num_leaves=16, learning_rate=0.01, n_estimators=1000,
    min_samples_leaf=200, reg_lambda=3.0, random_state=42, verbose=-1
)

# GPU training - off by default so this exact file behaves identically on a plain-CPU
# machine (e.g. this Windows box) with zero setup. Only flip USE_GPU to True in an
# environment with a CUDA-enabled LightGBM build (e.g. Google Colab with a T4 GPU
# attached, Runtime > Change runtime type > T4 GPU). The stock PyPI `lightgbm` wheel is
# CPU-only; you need to install a CUDA build first, e.g. in a Colab cell:
#   !pip uninstall -y lightgbm
#   !pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON
# (this compiles from source against Colab's preinstalled CUDA toolkit, several minutes).
# Setting USE_GPU=True without a CUDA-enabled build will error out immediately when
# LightGBM tries to initialize the GPU device - if that happens, either the install
# above didn't take, or this Colab instance doesn't actually have a GPU runtime attached.
USE_GPU = False
GPU_LGB_PARAMS = dict(device_type="cuda", gpu_use_dp=False)  # merged into LGB_PARAMS only if USE_GPU


def _lgb_params():
    return {**LGB_PARAMS, **GPU_LGB_PARAMS} if USE_GPU else dict(LGB_PARAMS)



# =============================================================================
# 1. Universe — resolved per --market at runtime, see resolve_market() above
#
# HK: ~2,790 currently-listed HKD-denominated equities + REITs on HKEX Main
# Board/GEM (load_hk_universe() in data_download_common.py). US: ~2,580
# Russell 3000 constituents (load_us_universe()). Both replaced a hand-curated
# "top names by today's market cap/index membership" list that used to be
# hardcoded here - besides
# being a narrower, more performance-conditioned universe (only stocks that
# both survived and grew large/prominent enough to still qualify today),
# keeping independent copies of "the ticker list" in different files was a
# maintenance hazard in its own right (they had already silently drifted
# apart before this change - see download_hk_data.py/download_us_data.py,
# which now load from the same two functions). Neither list fixes true
# survivorship bias - see the two loaders' docstrings for what would.
# =============================================================================

# =============================================================================
# 2. Data download — OHLCV, approximate cap, sector/industry (all via yfinance)
# =============================================================================

# fetch_ohlcv/fetch_cap/fetch_industry/to_long_format used to be duplicated here with
# their own (once-buggy) copies - now delegated to data_download_common.py, the same
# functions download_hk_data.py/download_us_data.py use, so a fix in one place can't
# silently miss the other the way the fetch_cap "shares_outstanding" vs "shares" key
# bug did.

def download_and_cache_data(tickers, price_cache_file, shares_cache_file, industry_cache_file,
                             years=YEARS_HISTORY, use_cache=USE_PRICE_CACHE):
    if use_cache and price_cache_file.exists():
        print(f"Loading cached price data from {price_cache_file} ...")
        return pd.read_csv(price_cache_file, parse_dates=["date"])

    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=years)).date().isoformat()
    min_obs = int(years * TRADING_DAYS_PER_YEAR * MIN_OBS_FRAC)
    print(f"Downloading OHLCV for {len(tickers)} tickers, {start} to today ...")
    open_df, high_df, low_df, close_df, volume_df = common.fetch_ohlcv(
        tickers, start, min_obs=min_obs, auto_adjust=PRICE_ADJUST)
    print(f"  got {close_df.shape[1]}/{len(tickers)} tickers, {close_df.shape[0]} trading days.")
    if close_df.shape[1] == 0:
        raise RuntimeError("No data retrieved - check tickers/date range/network access.")

    print("Fetching shares outstanding (approximate cap, used by alpha056) ...")
    cap_df = common.fetch_cap(list(close_df.columns), close_df, shares_cache_file, pause=INFO_FETCH_PAUSE)

    print("Fetching sector/industry classification (used by indneutralize alphas) ...")
    sector, industry = common.fetch_industry(list(close_df.columns), industry_cache_file, pause=INFO_FETCH_PAUSE)

    long_df = common.to_long_format(open_df, high_df, low_df, close_df, volume_df, cap_df, sector, industry)
    long_df.to_csv(price_cache_file, index=False)
    print(f"  cached {len(long_df):,} rows to {price_cache_file}")
    return long_df


# =============================================================================
# 6. Data loader — reshapes the cached long CSV/parquet into wide panels
# =============================================================================

def load_data(long_df, tickers=None, start=None, end=None):
    df = long_df
    if tickers:
        df = df[df["ticker"].isin(tickers)]
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
# 7. Feature panel — compute alphas, standardize cross-sectionally, add target
# =============================================================================

def standardize_xs(x):
    """Cross-sectional z-score per day."""
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


def _stack_all(x):
    """DataFrame.stack() that keeps every (date, ticker) cell, including NaNs,
    so every alpha lines up on the same index. pandas < 2.1 needs
    dropna=False for that; pandas >= 3.0 removed the kwarg because its new
    stack() already keeps NaNs by default."""
    try:
        s = x.stack(dropna=False)
    except (TypeError, ValueError):
        s = x.stack()
    s.index = s.index.set_names(["date", "ticker"])
    return s


def build_feature_panel(md, holding_days=HOLDING_DAYS, skip_indneutralize=False,
                         execution_lag_days=EXECUTION_LAG_DAYS, execution_price_field=EXECUTION_PRICE_FIELD):
    """Compute all 101 alphas, cross-sectionally standardize each ("_xs"), add
    a forward-return target over `holding_days` trading days, and return one
    long DataFrame: date, ticker, y_raw, y_xs, me, sector, alpha001_xs, ...

    The target is entry at `execution_price_field`, `execution_lag_days`
    trading days after the signal date, held for `holding_days` more trading
    days and exited at the same field (e.g. open-to-open, `execution_lag_days`
    days later than a naive same-day-close assumption) - see EXECUTION_LAG_DAYS
    in CONFIG for why.
    """
    alpha_fns = ALL_ALPHAS
    if skip_indneutralize:
        alpha_fns = {k: v for k, v in ALL_ALPHAS.items() if k not in INDNEUTRALIZE_ALPHAS}
        print(f"  sector/industry coverage too low - skipping {len(INDNEUTRALIZE_ALPHAS)} "
              f"indneutralize-based alphas: {sorted(INDNEUTRALIZE_ALPHAS)}")

    print(f"Computing {len(alpha_fns)} alphas ...")
    raw_results = {}
    for name, fn in alpha_fns.items():
        try:
            raw_results[name] = fn(md)
        except Exception as e:
            print(f"  {name} FAILED to compute: {e}")

    series_list = []
    for name in alpha_fns:
        if name in raw_results:
            xs = standardize_xs(raw_results[name])
            series_list.append(_stack_all(xs).rename(f"{name}_xs"))

    panel = pd.concat(series_list, axis=1)
    panel.index.set_names(["date", "ticker"], inplace=True)

    exec_price = getattr(md, execution_price_field)
    entry_price = exec_price.shift(-execution_lag_days)
    exit_price = exec_price.shift(-(execution_lag_days + holding_days))
    fwd_ret = exit_price / entry_price - 1
    y_raw = _stack_all(fwd_ret).rename("y_raw")
    y_xs = _stack_all(standardize_xs(fwd_ret)).rename("y_xs")
    panel = panel.join(y_raw).join(y_xs)

    if md.cap is not None:
        panel = panel.join(_stack_all(md.cap).rename("me"))
    else:
        panel["me"] = np.nan

    panel = panel.reset_index()
    if md.sector is not None:
        panel["sector"] = panel["ticker"].map(md.sector)
    else:
        panel["sector"] = None

    return panel


def _alpha_code_fingerprint():
    """Hash of the source code that actually determines what build_feature_panel()
    produces - the operators, all 101 alpha formulas, MarketData, standardize_xs, and
    build_feature_panel itself - so editing an alpha's logic (or an operator, or how
    the target is built) invalidates the panel cache automatically. Deliberately does
    NOT include the rest of the file (K, RETRAIN_EVERY, LGB_PARAMS, TRANSACTION_COST,
    etc. all live in the same file but don't affect this hash), since those are exactly
    the hyperparameters this cache exists to let you tune without a recompute.
    """
    funcs = list(ALL_ALPHAS.values()) + [
        MarketData, standardize_xs, build_feature_panel,
        rank, scale, indneutralize, delay, delta, correlation, covariance, sum_, product_,
        stddev, ts_min, ts_max, ts_argmax, ts_argmin, ts_rank, decay_linear,
        abs_, log_, sign_, signedpower, iif, emax, emin,
    ]
    source = "".join(inspect.getsource(f) for f in funcs)
    return hashlib.md5(source.encode()).hexdigest()[:16]


def _file_content_hash(path):
    """MD5 of a file's actual bytes (not its path or mtime), so the cache is portable
    across machines/environments - e.g. uploading the price CSV to Google Drive resets
    its mtime and changes its absolute path, which would otherwise bust the cache even
    though the content (and therefore the correct panel) is identical. Costs well under
    a second even on an ~85MB price CSV - negligible next to a multi-minute recompute.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_feature_panel_cached(md, cache_dir, price_cache_file, years, holding_days=HOLDING_DAYS,
                                skip_indneutralize=False, execution_lag_days=EXECUTION_LAG_DAYS,
                                execution_price_field=EXECUTION_PRICE_FIELD, use_cache=PANEL_CACHE):
    """Same result as build_feature_panel(), but cached on disk under cache_dir, keyed on
    the underlying price CSV's actual content (a content hash, not path/mtime - see
    _file_content_hash - so the cache still hits after copying the CSV and/or the cache
    dir to a different machine or environment, e.g. Colab), `years` (--years trims the
    data before the panel is built), holding_days/skip_indneutralize/execution_lag_days/
    execution_price_field (all directly change the panel's contents), and a source-code
    fingerprint of the alpha formulas (see _alpha_code_fingerprint()).

    K, RETRAIN_EVERY, LGB_PARAMS, TRANSACTION_COST, OOS_FRACTION, VALIDATION_FRACTION, and
    STAGE are NOT part of the cache key - they only affect what happens downstream of the
    panel (model training, portfolio construction, evaluation window), so tuning those
    hits the cache instead of recomputing all 101 alphas.
    """
    if not use_cache:
        return build_feature_panel(md, holding_days, skip_indneutralize, execution_lag_days, execution_price_field)

    content_hash = _file_content_hash(price_cache_file) if price_cache_file.exists() else None
    key_parts = (
        content_hash,
        years, holding_days, skip_indneutralize, execution_lag_days, execution_price_field,
        _alpha_code_fingerprint(),
    )
    digest = hashlib.md5(repr(key_parts).encode()).hexdigest()[:16]
    cache_file = cache_dir / f"panel_cache_{digest}.pkl"

    if cache_file.exists():
        print(f"Loading cached alpha panel from {cache_file} ...")
        return pd.read_pickle(cache_file)

    panel = build_feature_panel(md, holding_days, skip_indneutralize, execution_lag_days, execution_price_field)
    panel.to_pickle(cache_file)
    print(f"Cached alpha panel to {cache_file}")
    return panel


# =============================================================================
# 8. Backtest — walk-forward LightGBM top-K/bottom-K long/short
#    (rebalanced every HOLDING_DAYS trading days)
# =============================================================================

def topk_ls(sub, pred_col, eval_col, k, long_only=LONG_ONLY):
    """EW top-K long minus bottom-K short (or top-K long only, if long_only).
    Returns (gross_return, weights), where weights maps ticker -> signed
    portfolio weight (+1/k long, -1/k short) so the caller can measure
    turnover against the prior rebalance."""
    sub = sub.dropna(subset=[pred_col, eval_col])
    needed = k if long_only else 2 * k
    if len(sub) < needed:
        return np.nan, pd.Series(dtype=float)
    top = sub.nlargest(k, pred_col)
    weights = pd.Series(1.0 / k, index=top["ticker"].values)
    if long_only:
        return top[eval_col].mean(), weights
    bot = sub.nsmallest(k, pred_col)
    ret = top[eval_col].mean() - bot[eval_col].mean()
    weights = pd.concat([weights, pd.Series(-1.0 / k, index=bot["ticker"].values)])
    return ret, weights


def walk_forward_ls(preds_df, pred_col, eval_col, k, cost_rate=TRANSACTION_COST, long_only=LONG_ONLY):
    """Equal-weighted top-K/bottom-K long-short return per rebalance date
    (or top-K long-only, if long_only), net of transaction costs.

    Cost is charged on turnover, not on the whole book every period: a
    position held unchanged from the prior rebalance (same ticker, same
    leg, same weight) costs nothing, since its trade value is 0 even though
    its portfolio weight is nonzero. Only entries, exits, and long<->short
    flips incur `cost_rate` per unit of weight actually traded (a flip
    trades 2/k - the full round trip - since the old leg must be closed and
    the new leg opened).
    """
    rets = {}
    prev_weights = pd.Series(dtype=float)
    for m, grp in preds_df.groupby("date"):
        r, weights = topk_ls(grp, pred_col, eval_col, k, long_only)
        if np.isnan(r):
            continue
        idx = weights.index.union(prev_weights.index)
        turnover = (weights.reindex(idx, fill_value=0.0)
                    - prev_weights.reindex(idx, fill_value=0.0)).abs().sum()
        rets[m] = r - cost_rate * turnover
        prev_weights = weights
    return pd.Series(rets).sort_index()


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


def fetch_benchmark_returns(ticker, dates, holding_days, auto_adjust=PRICE_ADJUST,
                             execution_lag_days=EXECUTION_LAG_DAYS, execution_price_field=EXECUTION_PRICE_FIELD):
    """Forward `holding_days`-return of `ticker`, reindexed to `dates`, using
    the same entry/exit timing convention as the strategy (see
    EXECUTION_LAG_DAYS in CONFIG) - a benchmark computed same-day-close while
    the strategy trades next-day-open would be as biased a comparison as the
    dividend-adjustment mismatch PRICE_ADJUST already guards against.

    `auto_adjust` defaults to the same PRICE_ADJUST used for the stock
    universe (see CONFIG) so the benchmark comparison isn't biased by one
    side including dividends/splits and the other not.
    """
    start = (min(dates) - pd.DateOffset(days=5)).date().isoformat()
    end = (max(dates) + pd.DateOffset(days=5)).date().isoformat()
    raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=auto_adjust, progress=False)
    field = execution_price_field.capitalize()  # yfinance columns are "Open"/"Close"
    price = raw[field].iloc[:, 0] if isinstance(raw.columns, pd.MultiIndex) else raw[field]
    entry_price = price.shift(-execution_lag_days)
    exit_price = price.shift(-(execution_lag_days + holding_days))
    fwd = exit_price / entry_price - 1
    return fwd.reindex(dates)


def run_walk_forward(panel, x_cols, train_target, eval_target, oos_start,
                      holding_days=HOLDING_DAYS, k=K, retrain_every=RETRAIN_EVERY,
                      execution_lag_days=EXECUTION_LAG_DAYS):
    all_dates = sorted(panel["date"].unique())
    rebal_dates = all_dates[::holding_days]
    oos_dates = [dt for dt in rebal_dates if dt >= pd.Timestamp(oos_start)]

    model = None
    last_train_idx = -10 ** 9
    date_to_idx = {dt: i for i, dt in enumerate(all_dates)}
    preds = []
    purge_window = execution_lag_days + holding_days

    for j, m in enumerate(oos_dates):
        idx = date_to_idx[m]
        if model is None or (idx - last_train_idx) >= retrain_every:
            # Purge the `purge_window` dates immediately before m: a training row dated d
            # has a label = forward return from d+execution_lag_days to
            # d+execution_lag_days+holding_days, so any d within purge_window of m has a
            # label reaching into [m, m+holding_days) - the same window the model is about
            # to be evaluated on. Purging stops that leak, and (as a side effect) enforces
            # the gap at every train/validation/test boundary too, since the first m in
            # each stage gets purged the same way.
            purge_cutoff = all_dates[max(0, idx - purge_window)]
            train = panel[panel["date"] < purge_cutoff].dropna(subset=[train_target])
            model = lgb.LGBMRegressor(**_lgb_params())
            model.fit(train[x_cols].astype(np.float64).values,
                      train[train_target].astype(np.float64).values)
            last_train_idx = idx
            print(f"  retrained {m.date()}, train={len(train):,}")

        test = panel[panel["date"] == m].dropna(subset=[eval_target])
        if len(test) < 2 * k:
            continue
        test = test.copy()
        test["pred"] = model.predict(test[x_cols].astype(np.float64).values)
        preds.append(test[["ticker", "date", eval_target, "me", "sector", "pred"]])

    return pd.concat(preds, ignore_index=True)


# =============================================================================
# 9. Main
# =============================================================================

def log_tuning_run(stats, ls_ret, market, tuning_log_file):
    """Append this --stage tune run's config + validation result to tuning_log_file,
    so hyperparameter search leaves an honest trail of every config tried (needed if
    you ever want to apply a multiple-testing correction to the final test result)."""
    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "market": market,
        "holding_days": HOLDING_DAYS, "k": K, "retrain_every": RETRAIN_EVERY,
        "transaction_cost": TRANSACTION_COST, "validation_fraction": VALIDATION_FRACTION,
        "lgb_params": json.dumps(_lgb_params()),
        "n_periods": len(ls_ret),
        "sharpe": stats["SR"], "ann_mean": stats["Ann Mean"], "ann_vol": stats["Ann Vol"],
        "mdd": stats["MDD"], "total_return": stats["Total"],
    }])
    row.to_csv(tuning_log_file, mode="a", header=not tuning_log_file.exists(), index=False)


def main():
    parser = argparse.ArgumentParser(description="Run the 101-alpha LightGBM long/short backtest")
    parser.add_argument("--market", choices=list(MARKETS), default=MARKET_DEFAULT,
                         help="which ticker universe + benchmark to use (default: %(default)s); "
                              "each market gets its own cache subdirectory under alpha_ml_cache/")
    parser.add_argument("--stage", choices=["tune", "test"], default=STAGE,
                         help="'tune': drop the OOS window entirely and evaluate on a held-out validation "
                              "slice of what's left, for hyperparameter search (edit HOLDING_DAYS/K/"
                              "RETRAIN_EVERY/LGB_PARAMS/TRANSACTION_COST above and rerun to compare); "
                              "'test': fit on all pre-OOS data (train+validation) and evaluate once on "
                              "the true OOS window - run this only after tuning is done (default: %(default)s)")
    parser.add_argument("--years", type=float, default=YEARS_DEFAULT,
                         help="use only the most recent N years of the cached data's own date range (e.g. "
                              "--years 5 out of 10 years cached) - independent of YEARS_HISTORY, which only "
                              "controls how much gets downloaded in the first place, not how much of an "
                              "already-cached CSV gets used for this run (default: %(default)s, pass 0 or a "
                              "negative number for everything cached)")
    parser.add_argument("--no-panel-cache", dest="panel_cache", action="store_false",
                         help="force recomputing all 101 alphas instead of reusing a cached panel from a "
                              "prior run with the same data/holding period/execution settings (default: use "
                              "the cache - see PANEL_CACHE in CONFIG)")
    parser.set_defaults(panel_cache=PANEL_CACHE)
    args = parser.parse_args()
    stage = args.stage
    market = resolve_market(args.market)

    print(f"Market: {args.market.upper()} - universe: {len(market['tickers'])} tickers, "
          f"benchmark: {market['benchmark_ticker']}, cache: {market['cache_dir']}")

    long_df = download_and_cache_data(
        market["tickers"], market["price_cache_file"], market["shares_cache_file"], market["industry_cache_file"])
    if args.years is not None and args.years > 0 and not long_df.empty:
        cutoff = long_df["date"].max() - pd.Timedelta(days=args.years * 365.25)
        long_df = long_df[long_df["date"] >= cutoff]
    print(f"Loaded {long_df['ticker'].nunique()} tickers, "
          f"{long_df['date'].nunique()} trading days "
          f"({long_df['date'].min().date()} to {long_df['date'].max().date()}).")

    open_df, high_df, low_df, close_df, volume_df, cap_df, sector, industry = load_data(long_df)
    md = MarketData(open_df, high_df, low_df, close_df, volume_df,
                     cap=cap_df, sector=sector, industry=industry)

    sector_coverage = sector.notna().mean() if sector is not None else 0.0
    skip_indneutralize = sector_coverage < SECTOR_COVERAGE_MIN
    print(f"Sector coverage: {sector_coverage:.0%} of tickers "
          f"({'OK' if not skip_indneutralize else 'below threshold'})")

    panel = build_feature_panel_cached(md, market["cache_dir"], market["price_cache_file"], args.years,
                                        holding_days=HOLDING_DAYS, skip_indneutralize=skip_indneutralize,
                                        use_cache=args.panel_cache)
    x_cols = [c for c in panel.columns if c.endswith("_xs") and c != TRAIN_TARGET]
    print(f"Feature panel: {len(panel):,} rows, {len(x_cols)} alpha features.")

    all_dates = sorted(panel["date"].unique())
    oos_start = OOS_START or all_dates[int(len(all_dates) * (1 - OOS_FRACTION))].date().isoformat()
    print(f"OOS start: {oos_start}")

    if stage == "tune":
        # The OOS window is dropped outright here, not just "not looked at" - it never
        # reaches run_walk_forward, so it's structurally impossible for this run to leak
        # into or be influenced by the test window.
        panel = panel[panel["date"] < pd.Timestamp(oos_start)]
        pretest_dates = sorted(panel["date"].unique())
        eval_start = pretest_dates[int(len(pretest_dates) * (1 - VALIDATION_FRACTION))].date().isoformat()
        print(f"[tune] Validation start: {eval_start} (OOS window at {oos_start} onward was dropped, not loaded)")
    else:
        eval_start = oos_start
        print("[test] Training on all pre-OOS data (train+validation); evaluating once on the true OOS window")

    print(f"Running walk-forward LightGBM backtest (holding period: {HOLDING_DAYS} trading day"
          f"{'s' if HOLDING_DAYS != 1 else ''}) ...")
    preds_df = run_walk_forward(panel, x_cols, TRAIN_TARGET, EVAL_TARGET, eval_start,
                                 holding_days=HOLDING_DAYS, k=K, retrain_every=RETRAIN_EVERY)
    ls_ret = walk_forward_ls(preds_df, "pred", EVAL_TARGET, K, cost_rate=TRANSACTION_COST, long_only=LONG_ONLY)
    print(f"Transaction cost: {TRANSACTION_COST:.2%} per unit of weight traded "
          f"(charged on turnover only - see walk_forward_ls)" +
          (" | LONG_ONLY: top-K long, no shorting" if LONG_ONLY else ""))

    periods_per_year = TRADING_DAYS_PER_YEAR / HOLDING_DAYS
    eval_label = "validation" if stage == "tune" else "OOS/test"
    strategy_label = "Long-only" if LONG_ONLY else "L/S"
    stats = perf(ls_ret, periods_per_year, name=f"LightGBM 101-Alpha {strategy_label} ({eval_label}, net of cost)")
    print(f"\nLightGBM {strategy_label} ({eval_label}): {stats}")
    print(f"Annualized return: {cal_annualized_return(ls_ret, periods_per_year):.2%}")

    if stage == "tune":
        log_tuning_run(stats, ls_ret, args.market, market["tuning_log_file"])
        print(f"\nLogged this configuration + validation result to {market['tuning_log_file']}")
        return

    benchmark_ticker = market["benchmark_ticker"]
    print(f"Fetching {benchmark_ticker} benchmark ...")
    bench_ret = fetch_benchmark_returns(benchmark_ticker, ls_ret.index, HOLDING_DAYS).dropna()
    bench_stats = perf(bench_ret, periods_per_year, name=benchmark_ticker)
    print(f"{benchmark_ticker} benchmark: {bench_stats}")

    out_path = market["cache_dir"] / "lightgbm_ls_vs_benchmark.png"
    plot_strats({f"LightGBM 101-Alpha {strategy_label}": ls_ret}, bench_ret, benchmark_ticker,
                periods_per_year, f"LightGBM {strategy_label} (K={K}, hold={HOLDING_DAYS}d) vs {benchmark_ticker}", out_path)
    print(f"Saved chart to {out_path}")

    preds_path = market["cache_dir"] / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    main()