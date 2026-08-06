"""
Shared WorldQuant "101 Formulaic Alphas" (Kakushadze, 2015) engine: the
operators (Appendix A.2), the MarketData panel container, and all 101 alpha
formulas (Appendix A.1) - transcribed as literally as possible from the
paper. Used by hk_alpha101_single.py (rank-IC backtest), alpha_ML.py
(LightGBM-combined backtest), and hk_alpha_stock_backtest.py (hand-picked
stock-list backtest), so a formula/operator fix or edit only needs to happen
in one place instead of three.

This module has no CLI and no __main__ - it's imported, not run directly.
"""

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Several formulas legitimately hit NaN/inf edge cases (log of a negative
# spread, 0/0 when high == low, etc.) - expected, not worth the noise/cost.
# Applies globally to any process that imports this module.
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Operators - Appendix A.2 of the paper
#
# All inputs/outputs are pandas DataFrames indexed by date (rows) x ticker
# (columns), unless noted otherwise. "Cross-sectional" operators act across
# tickers on a given date (across columns, per row); "time-series" operators
# act across dates for a given ticker (down a column).
# =============================================================================

def _w(d):
    """Window length: non-integer d is floored, per the paper's convention."""
    return max(1, int(np.floor(d)))


def _windows(values, w):
    """(n, w) view of trailing w-windows for a single ticker's 1D series,
    NaN-padded so row i ends at values[i]. A zero-copy view via sliding_window_view."""
    padded = np.concatenate([np.full(w - 1, np.nan), values.astype(float)])
    return sliding_window_view(padded, w)


def _rolling_np(x, w, kernel):
    """Apply a (n, w) -> (n,) kernel one ticker at a time (a Python loop over columns)."""
    out = {}
    for col in x.columns:
        out[col] = kernel(_windows(x[col].to_numpy(), w))
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _as_frame(x, like):
    """Broadcast a scalar to a DataFrame shaped like `like`; pass DataFrames through."""
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x, index=like.index, columns=like.columns)


# ---- elementwise ----------------------------------------------------------

def abs_(x):
    return x.abs()


def log_(x):
    return np.log(x)


def sign_(x):
    return np.sign(x)


def signedpower(x, a):
    return np.sign(x) * np.abs(x) ** a


def iif(cond, a, b):
    """cond ? a : b"""
    a = _as_frame(a, cond)
    b = _as_frame(b, cond)
    return a.where(cond.fillna(False), b)


def emax(a, b):
    """Elementwise max of two same-shaped frames (paper's max(x, y) form)."""
    return a.where((a >= b) | b.isna(), b)


def emin(a, b):
    """Elementwise min of two same-shaped frames (paper's min(x, y) form)."""
    return a.where((a <= b) | b.isna(), b)


# ---- cross-sectional --------------------------------------------------------

def rank(x):
    """Cross-sectional percentile rank in [0, 1]."""
    return x.rank(axis=1, pct=True)


def scale(x, a=1.0):
    """Rescale each row so sum(abs(x)) == a.

    A single ticker with +-inf (e.g. a rolling correlation blowing up because
    its price hasn't moved for the whole window - a real, expected edge case
    for thinly-traded penny stocks) would otherwise poison every other
    ticker's value that day: summing abs(x) across the row makes the
    denominator inf, so every finite ticker divides to 0 and the offending
    ticker itself becomes inf/inf = NaN. Treating inf as NaN up front keeps
    the damage isolated to the one ticker, matching how every other NaN/inf
    edge case in these formulas is handled (see the module docstring).
    """
    x = x.replace([np.inf, -np.inf], np.nan)
    denom = x.abs().sum(axis=1).replace(0, np.nan)
    return x.div(denom, axis=0) * a


def indneutralize(x, group):
    """Cross-sectionally demean x within each group (sector/industry/subindustry).

    group: pd.Series mapping ticker -> group label, or None (falls back to
    demeaning across the whole universe).

    Same inf-isolation reasoning as scale(): a group-wide mean() would
    otherwise let one ticker's +-inf contaminate every other ticker in its
    group.
    """
    x = x.replace([np.inf, -np.inf], np.nan)
    if group is None:
        return x.sub(x.mean(axis=1), axis=0)
    out = x.copy() * np.nan
    g = group.reindex(x.columns)
    for label, cols in g.groupby(g).groups.items():
        cols = [c for c in cols if c in x.columns]
        if not cols:
            continue
        sub = x[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out


# ---- time-series ------------------------------------------------------------

def delay(x, d):
    return x.shift(_w(d))


def delta(x, d):
    return x - x.shift(_w(d))


def correlation(x, y, d):
    return x.rolling(_w(d)).corr(y)


def covariance(x, y, d):
    return x.rolling(_w(d)).cov(y)


def sum_(x, d):
    return x.rolling(_w(d)).sum()


def product_(x, d):
    w = _w(d)

    def kernel(windows):  # windows: (n, w)
        valid = ~np.isnan(windows)
        with np.errstate(invalid="ignore"):
            prod = np.nanprod(windows, axis=1)
        return np.where(valid.any(axis=1), prod, np.nan)

    return _rolling_np(x, w, kernel)


def stddev(x, d):
    return x.rolling(_w(d)).std()


def ts_min(x, d):
    return x.rolling(_w(d)).min()


def ts_max(x, d):
    return x.rolling(_w(d)).max()


min_ = ts_min  # paper: min(x, d) = ts_min(x, d)
max_ = ts_max  # paper: max(x, d) = ts_max(x, d)


def ts_argmax(x, d):
    """1-based position of the max within the window, oldest day = 1."""
    w = _w(d)

    def kernel(windows):  # windows: (n, w)
        valid = ~np.isnan(windows)
        filled = np.where(valid, windows, -np.inf)
        pos = np.argmax(filled, axis=1) + 1
        return np.where(valid.any(axis=1), pos, np.nan)

    return _rolling_np(x, w, kernel)


def ts_argmin(x, d):
    """1-based position of the min within the window, oldest day = 1."""
    w = _w(d)

    def kernel(windows):  # windows: (n, w)
        valid = ~np.isnan(windows)
        filled = np.where(valid, windows, np.inf)
        pos = np.argmin(filled, axis=1) + 1
        return np.where(valid.any(axis=1), pos, np.nan)

    return _rolling_np(x, w, kernel)


def ts_rank(x, d):
    """Percentile rank (0, 1] of the most recent (last) value within the trailing window."""
    w = _w(d)

    def kernel(windows):  # windows: (n, w)
        valid = ~np.isnan(windows)
        today = windows[:, -1]
        le_today = np.where(valid, windows <= today[:, None], False)
        counts = valid.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ranks = le_today.sum(axis=1) / counts
        return np.where((counts > 0) & ~np.isnan(today), ranks, np.nan)

    return _rolling_np(x, w, kernel)


def decay_linear(x, d):
    """Weighted moving average; today (the last day in the window) gets weight d,
    decaying linearly to 1 for the oldest day (rescaled to sum to 1)."""
    w = _w(d)
    weights = np.arange(1, w + 1, dtype=float)  # weights[-1] (today) == w

    def kernel(windows):  # windows: (n, w); weights broadcasts over the last axis
        valid = ~np.isnan(windows)
        wsum = (valid * weights).sum(axis=1)
        contrib = np.where(valid, windows * weights, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(wsum > 0, contrib / wsum, np.nan)

    return _rolling_np(x, w, kernel)


# =============================================================================
# MarketData - panel container feeding the alpha formulas
# =============================================================================

class MarketData:
    """Panel container (date x ticker DataFrames) feeding the alpha formulas.

    vwap defaults to the typical price (H+L+C)/3 when not supplied, since daily
    bars don't carry a true intraday VWAP.
    """

    def __init__(self, open_, high, low, close, volume, vwap=None, cap=None,
                 sector=None, industry=None, subindustry=None):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap = vwap if vwap is not None else (high + low + close) / 3.0
        self.returns = close.pct_change(fill_method=None)
        self.cap = cap
        self.sector = sector
        self.industry = industry
        self.subindustry = subindustry if subindustry is not None else industry
        self._adv_cache = {}

    def adv(self, d):
        """Average daily dollar volume over the past d days."""
        w = _w(d)
        if w not in self._adv_cache:
            dollar_vol = self.close * self.volume
            self._adv_cache[w] = dollar_vol.rolling(w).mean()
        return self._adv_cache[w]


# =============================================================================
# The 101 formulaic alphas - Appendix A.1 of the paper
#
# Each alphaNNN(d) takes a MarketData instance and returns a date x ticker
# DataFrame. Formulas are transcribed as literally as possible from the paper
# (including redundant sub-expressions like `x*w + x*(1-w)`, which are
# mathematically equal to `x` but kept for fidelity to the published formula
# text).
#
# Alphas using indneutralize(x, IndClass.*) need a sector/industry/subindustry
# mapping on `d`; if none was supplied, indneutralize() falls back to
# demeaning across the whole universe.
# =============================================================================

def alpha001(d):
    x = iif(d.returns < 0, stddev(d.returns, 20), d.close)
    inner = signedpower(x, 2.0)
    return rank(ts_argmax(inner, 5)) - 0.5


def alpha002(d):
    x = rank(delta(log_(d.volume), 2))
    y = rank((d.close - d.open) / d.open)
    return -1 * correlation(x, y, 6)


def alpha003(d):
    return -1 * correlation(rank(d.open), rank(d.volume), 10)


def alpha004(d):
    return -1 * ts_rank(rank(d.low), 9)


def alpha005(d):
    return rank(d.open - (sum_(d.vwap, 10) / 10)) * (-1 * abs_(rank(d.close - d.vwap)))


def alpha006(d):
    return -1 * correlation(d.open, d.volume, 10)


def alpha007(d):
    adv20 = d.adv(20)
    cond = adv20 < d.volume
    true_val = (-1 * ts_rank(abs_(delta(d.close, 7)), 60)) * sign_(delta(d.close, 7))
    return iif(cond, true_val, -1)


def alpha008(d):
    x = sum_(d.open, 5) * sum_(d.returns, 5)
    return -1 * rank(x - delay(x, 10))


def _alpha009_10_core(d, window):
    dclose1 = delta(d.close, 1)
    cond_a = 0 < ts_min(dclose1, window)
    cond_b = ts_max(dclose1, window) < 0
    return iif(cond_a, dclose1, iif(cond_b, dclose1, -1 * dclose1))


def alpha009(d):
    return _alpha009_10_core(d, 5)


def alpha010(d):
    return rank(_alpha009_10_core(d, 4))


def alpha011(d):
    vc = d.vwap - d.close
    return (rank(ts_max(vc, 3)) + rank(ts_min(vc, 3))) * rank(delta(d.volume, 3))


def alpha012(d):
    return sign_(delta(d.volume, 1)) * (-1 * delta(d.close, 1))


def alpha013(d):
    return -1 * rank(covariance(rank(d.close), rank(d.volume), 5))


def alpha014(d):
    return (-1 * rank(delta(d.returns, 3))) * correlation(d.open, d.volume, 10)


def alpha015(d):
    return -1 * sum_(rank(correlation(rank(d.high), rank(d.volume), 3)), 3)


def alpha016(d):
    return -1 * rank(covariance(rank(d.high), rank(d.volume), 5))


def alpha017(d):
    a = -1 * rank(ts_rank(d.close, 10))
    b = rank(delta(delta(d.close, 1), 1))
    c = rank(ts_rank(d.volume / d.adv(20), 5))
    return a * b * c


def alpha018(d):
    x = stddev(abs_(d.close - d.open), 5) + (d.close - d.open) + correlation(d.close, d.open, 10)
    return -1 * rank(x)


def alpha019(d):
    a = -1 * sign_((d.close - delay(d.close, 7)) + delta(d.close, 7))
    b = 1 + rank(1 + sum_(d.returns, 250))
    return a * b


def alpha020(d):
    a = -1 * rank(d.open - delay(d.high, 1))
    b = rank(d.open - delay(d.close, 1))
    c = rank(d.open - delay(d.low, 1))
    return a * b * c


def alpha021(d):
    adv20 = d.adv(20)
    cond1 = (sum_(d.close, 8) / 8 + stddev(d.close, 8)) < (sum_(d.close, 2) / 2)
    cond2 = (sum_(d.close, 2) / 2) < (sum_(d.close, 8) / 8 - stddev(d.close, 8))
    ratio = d.volume / adv20
    cond3 = (ratio > 1) | (ratio == 1)
    branch3 = iif(cond3, 1, -1)
    branch2 = iif(cond2, 1, branch3)
    return iif(cond1, -1, branch2)


def alpha022(d):
    x = delta(correlation(d.high, d.volume, 5), 5)
    return -1 * (x * rank(stddev(d.close, 20)))


def alpha023(d):
    cond = (sum_(d.high, 20) / 20) < d.high
    return iif(cond, -1 * delta(d.high, 2), 0)


def alpha024(d):
    x = delta(sum_(d.close, 100) / 100, 100) / delay(d.close, 100)
    cond = (x < 0.05) | (x == 0.05)
    return iif(cond, -1 * (d.close - ts_min(d.close, 100)), -1 * delta(d.close, 3))


def alpha025(d):
    return rank(((-1 * d.returns) * d.adv(20)) * d.vwap * (d.high - d.close))


def alpha026(d):
    x = correlation(ts_rank(d.volume, 5), ts_rank(d.high, 5), 5)
    return -1 * ts_max(x, 3)


def alpha027(d):
    x = rank(sum_(correlation(rank(d.volume), rank(d.vwap), 6), 2) / 2.0)
    return iif(0.5 < x, -1, 1)


def alpha028(d):
    adv20 = d.adv(20)
    return scale(correlation(adv20, d.low, 5) + ((d.high + d.low) / 2) - d.close)


def alpha029(d):
    step1 = rank(delta(d.close - 1, 5))
    step3 = rank(rank(-1 * step1))
    step4 = ts_min(step3, 2)
    step5 = sum_(step4, 1)
    step6 = log_(step5)
    step7 = scale(step6)
    step8 = rank(rank(step7))
    step9 = product_(step8, 1)
    part1 = ts_min(step9, 5)
    part2 = ts_rank(delay(-1 * d.returns, 6), 5)
    return part1 + part2


def alpha030(d):
    s1 = sign_(d.close - delay(d.close, 1))
    s2 = sign_(delay(d.close, 1) - delay(d.close, 2))
    s3 = sign_(delay(d.close, 2) - delay(d.close, 3))
    x = (1.0 - rank(s1 + s2 + s3)) * sum_(d.volume, 5)
    return x / sum_(d.volume, 20)


def alpha031(d):
    adv20 = d.adv(20)
    a = rank(rank(rank(decay_linear(-1 * rank(rank(delta(d.close, 10))), 10))))
    b = rank(-1 * delta(d.close, 3))
    c = sign_(scale(correlation(adv20, d.low, 12)))
    return a + b + c


def alpha032(d):
    a = scale(sum_(d.close, 7) / 7 - d.close)
    b = 20 * scale(correlation(d.vwap, delay(d.close, 5), 230))
    return a + b


def alpha033(d):
    return rank(-1 * (1 - (d.open / d.close)))


def alpha034(d):
    a = 1 - rank(stddev(d.returns, 2) / stddev(d.returns, 5))
    b = 1 - rank(delta(d.close, 1))
    return rank(a + b)


def alpha035(d):
    a = ts_rank(d.volume, 32)
    b = 1 - ts_rank((d.close + d.high) - d.low, 16)
    c = 1 - ts_rank(d.returns, 32)
    return a * b * c


def alpha036(d):
    adv20 = d.adv(20)
    t1 = 2.21 * rank(correlation(d.close - d.open, delay(d.volume, 1), 15))
    t2 = 0.7 * rank(d.open - d.close)
    t3 = 0.73 * rank(ts_rank(delay(-1 * d.returns, 6), 5))
    t4 = rank(abs_(correlation(d.vwap, adv20, 6)))
    t5 = 0.6 * rank(((sum_(d.close, 200) / 200) - d.open) * (d.close - d.open))
    return t1 + t2 + t3 + t4 + t5


def alpha037(d):
    a = rank(correlation(delay(d.open - d.close, 1), d.close, 200))
    b = rank(d.open - d.close)
    return a + b


def alpha038(d):
    return (-1 * rank(ts_rank(d.close, 10))) * rank(d.close / d.open)


def alpha039(d):
    adv20 = d.adv(20)
    a = -1 * rank(delta(d.close, 7) * (1 - rank(decay_linear(d.volume / adv20, 9))))
    b = 1 + rank(sum_(d.returns, 250))
    return a * b


def alpha040(d):
    return (-1 * rank(stddev(d.high, 10))) * correlation(d.high, d.volume, 10)


def alpha041(d):
    return (d.high * d.low) ** 0.5 - d.vwap


def alpha042(d):
    return rank(d.vwap - d.close) / rank(d.vwap + d.close)


def alpha043(d):
    a = ts_rank(d.volume / d.adv(20), 20)
    b = ts_rank(-1 * delta(d.close, 7), 8)
    return a * b


def alpha044(d):
    return -1 * correlation(d.high, rank(d.volume), 5)


def alpha045(d):
    a = rank(sum_(delay(d.close, 5), 20) / 20)
    b = correlation(d.close, d.volume, 2)
    c = rank(correlation(sum_(d.close, 5), sum_(d.close, 20), 2))
    return -1 * (a * b * c)


def alpha046(d):
    x = (delay(d.close, 20) - delay(d.close, 10)) / 10 - (delay(d.close, 10) - d.close) / 10
    cond1 = 0.25 < x
    cond2 = x < 0
    return iif(cond1, -1, iif(cond2, 1, -1 * (d.close - delay(d.close, 1))))


def alpha047(d):
    adv20 = d.adv(20)
    a = (rank(1 / d.close) * d.volume) / adv20
    b = (d.high * rank(d.high - d.close)) / (sum_(d.high, 5) / 5)
    c = rank(d.vwap - delay(d.vwap, 5))
    return a * b - c


def alpha048(d):
    num = indneutralize(
        (correlation(delta(d.close, 1), delta(delay(d.close, 1), 1), 250) * delta(d.close, 1)) / d.close,
        d.subindustry,
    )
    den = sum_((delta(d.close, 1) / delay(d.close, 1)) ** 2, 250)
    return num / den


def alpha049(d):
    x = (delay(d.close, 20) - delay(d.close, 10)) / 10 - (delay(d.close, 10) - d.close) / 10
    cond = x < (-1 * 0.1)
    return iif(cond, 1, -1 * (d.close - delay(d.close, 1)))


def alpha050(d):
    x = rank(correlation(rank(d.volume), rank(d.vwap), 5))
    return -1 * ts_max(x, 5)


def alpha051(d):
    x = (delay(d.close, 20) - delay(d.close, 10)) / 10 - (delay(d.close, 10) - d.close) / 10
    cond = x < (-1 * 0.05)
    return iif(cond, 1, -1 * (d.close - delay(d.close, 1)))


def alpha052(d):
    a = (-1 * ts_min(d.low, 5)) + delay(ts_min(d.low, 5), 5)
    b = rank((sum_(d.returns, 240) - sum_(d.returns, 20)) / 220)
    c = ts_rank(d.volume, 5)
    return a * b * c


def alpha053(d):
    x = ((d.close - d.low) - (d.high - d.close)) / (d.close - d.low)
    return -1 * delta(x, 9)


def alpha054(d):
    num = -1 * ((d.low - d.close) * (d.open ** 5))
    den = (d.low - d.high) * (d.close ** 5)
    return num / den


def alpha055(d):
    x = (d.close - ts_min(d.low, 12)) / (ts_max(d.high, 12) - ts_min(d.low, 12))
    return -1 * correlation(rank(x), rank(d.volume), 6)


def alpha056(d):
    if d.cap is None:
        return pd.DataFrame(np.nan, index=d.close.index, columns=d.close.columns)
    a = rank(sum_(d.returns, 10) / sum_(sum_(d.returns, 2), 3))
    b = rank(d.returns * d.cap)
    return 0 - (1 * (a * b))


def alpha057(d):
    x = (d.close - d.vwap) / decay_linear(rank(ts_argmax(d.close, 30)), 2)
    return 0 - (1 * x)


def alpha058(d):
    x = correlation(indneutralize(d.vwap, d.sector), d.volume, 3.92795)
    return -1 * ts_rank(decay_linear(x, 7.89291), 5.50322)


def alpha059(d):
    vwap_mix = (d.vwap * 0.728317) + (d.vwap * (1 - 0.728317))
    x = correlation(indneutralize(vwap_mix, d.industry), d.volume, 4.25197)
    return -1 * ts_rank(decay_linear(x, 16.2289), 8.19648)


def alpha060(d):
    x = rank((((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)) * d.volume)
    a = 2 * scale(x)
    b = scale(rank(ts_argmax(d.close, 10)))
    return 0 - (1 * (a - b))


def alpha061(d):
    a = rank(d.vwap - ts_min(d.vwap, 16.1219))
    b = rank(correlation(d.vwap, d.adv(180), 17.9282))
    return (a < b).astype(float)


def alpha062(d):
    adv20 = d.adv(20)
    a = rank(correlation(d.vwap, sum_(adv20, 22.4101), 9.91009))
    inner_cond = (rank(d.open) + rank(d.open)) < (rank((d.high + d.low) / 2) + rank(d.high))
    b = rank(inner_cond.astype(float))
    return (a < b).astype(float) * -1


def alpha063(d):
    a = rank(decay_linear(delta(indneutralize(d.close, d.industry), 2.25164), 8.22237))
    open_mix = (d.vwap * 0.318108) + (d.open * (1 - 0.318108))
    b = rank(decay_linear(correlation(open_mix, sum_(d.adv(180), 37.2467), 13.557), 12.2883))
    return (a - b) * -1


def alpha064(d):
    ol_mix = (d.open * 0.178404) + (d.low * (1 - 0.178404))
    a = rank(correlation(sum_(ol_mix, 12.7054), sum_(d.adv(120), 12.7054), 16.6208))
    hlv_mix = (((d.high + d.low) / 2) * 0.178404) + (d.vwap * (1 - 0.178404))
    b = rank(delta(hlv_mix, 3.69741))
    return (a < b).astype(float) * -1


def alpha065(d):
    ov_mix = (d.open * 0.00817205) + (d.vwap * (1 - 0.00817205))
    a = rank(correlation(ov_mix, sum_(d.adv(60), 8.6911), 6.40374))
    b = rank(d.open - ts_min(d.open, 13.635))
    return (a < b).astype(float) * -1


def alpha066(d):
    a = rank(decay_linear(delta(d.vwap, 3.51013), 7.23052))
    low_mix = (d.low * 0.96633) + (d.low * (1 - 0.96633))
    x = (low_mix - d.vwap) / (d.open - ((d.high + d.low) / 2))
    b = ts_rank(decay_linear(x, 11.4157), 6.72611)
    return (a + b) * -1


def alpha067(d):
    a = rank(d.high - ts_min(d.high, 2.14593))
    b = rank(correlation(indneutralize(d.vwap, d.sector), indneutralize(d.adv(20), d.subindustry), 6.02936))
    return (a ** b) * -1


def alpha068(d):
    a = ts_rank(correlation(rank(d.high), rank(d.adv(15)), 8.91644), 13.9333)
    cl_mix = (d.close * 0.518371) + (d.low * (1 - 0.518371))
    b = rank(delta(cl_mix, 1.06157))
    return (a < b).astype(float) * -1


def alpha069(d):
    a = rank(ts_max(delta(indneutralize(d.vwap, d.industry), 2.72412), 4.79344))
    cv_mix = (d.close * 0.490655) + (d.vwap * (1 - 0.490655))
    b = ts_rank(correlation(cv_mix, d.adv(20), 4.92416), 9.0615)
    return (a ** b) * -1


def alpha070(d):
    a = rank(delta(d.vwap, 1.29456))
    b = ts_rank(correlation(indneutralize(d.close, d.industry), d.adv(50), 17.8256), 17.9171)
    return (a ** b) * -1


def alpha071(d):
    a = ts_rank(decay_linear(correlation(ts_rank(d.close, 3.43976), ts_rank(d.adv(180), 12.0647), 18.0175), 4.20501), 15.6948)
    x = rank((d.low + d.open) - (d.vwap + d.vwap)) ** 2
    b = ts_rank(decay_linear(x, 16.4662), 4.4388)
    return emax(a, b)


def alpha072(d):
    adv40 = d.adv(40)
    a = rank(decay_linear(correlation((d.high + d.low) / 2, adv40, 8.93345), 10.1519))
    b = rank(decay_linear(correlation(ts_rank(d.vwap, 3.72469), ts_rank(d.volume, 18.5188), 6.86671), 2.95011))
    return a / b


def alpha073(d):
    a = rank(decay_linear(delta(d.vwap, 4.72775), 2.91864))
    ol_mix = (d.open * 0.147155) + (d.low * (1 - 0.147155))
    x = (delta(ol_mix, 2.03608) / ol_mix) * -1
    b = ts_rank(decay_linear(x, 3.33829), 16.7411)
    return emax(a, b) * -1


def alpha074(d):
    a = rank(correlation(d.close, sum_(d.adv(30), 37.4843), 15.1365))
    hv_mix = (d.high * 0.0261661) + (d.vwap * (1 - 0.0261661))
    b = rank(correlation(rank(hv_mix), rank(d.volume), 11.4791))
    return (a < b).astype(float) * -1


def alpha075(d):
    a = rank(correlation(d.vwap, d.volume, 4.24304))
    b = rank(correlation(rank(d.low), rank(d.adv(50)), 12.4413))
    return (a < b).astype(float)


def alpha076(d):
    a = rank(decay_linear(delta(d.vwap, 1.24383), 11.8259))
    x = ts_rank(correlation(indneutralize(d.low, d.sector), d.adv(81), 8.14941), 19.569)
    b = ts_rank(decay_linear(x, 17.1543), 19.383)
    return emax(a, b) * -1


def alpha077(d):
    adv40 = d.adv(40)
    a = rank(decay_linear((((d.high + d.low) / 2) + d.high) - (d.vwap + d.high), 20.0451))
    b = rank(decay_linear(correlation((d.high + d.low) / 2, adv40, 3.1614), 5.64125))
    return emin(a, b)


def alpha078(d):
    lv_mix = (d.low * 0.352233) + (d.vwap * (1 - 0.352233))
    a = rank(correlation(sum_(lv_mix, 19.7428), sum_(d.adv(40), 19.7428), 6.83313))
    b = rank(correlation(rank(d.vwap), rank(d.volume), 5.77492))
    return a ** b


def alpha079(d):
    co_mix = (d.close * 0.60733) + (d.open * (1 - 0.60733))
    a = rank(delta(indneutralize(co_mix, d.sector), 1.23438))
    b = rank(correlation(ts_rank(d.vwap, 3.60973), ts_rank(d.adv(150), 9.18637), 14.6644))
    return (a < b).astype(float)


def alpha080(d):
    oh_mix = (d.open * 0.868128) + (d.high * (1 - 0.868128))
    a = rank(sign_(delta(indneutralize(oh_mix, d.industry), 4.04545)))
    b = ts_rank(correlation(d.high, d.adv(10), 5.11456), 5.53756)
    return (a ** b) * -1


def alpha081(d):
    x = rank(correlation(d.vwap, sum_(d.adv(10), 49.6054), 8.47743)) ** 4
    a = rank(log_(product_(rank(x), 14.9655)))
    b = rank(correlation(rank(d.vwap), rank(d.volume), 5.07914))
    return (a < b).astype(float) * -1


def alpha082(d):
    a = rank(decay_linear(delta(d.open, 1.46063), 14.8717))
    open_mix = (d.open * 0.634196) + (d.open * (1 - 0.634196))
    x = correlation(indneutralize(d.volume, d.sector), open_mix, 17.4842)
    b = ts_rank(decay_linear(x, 6.92131), 13.4283)
    return emin(a, b) * -1


def alpha083(d):
    x = (d.high - d.low) / (sum_(d.close, 5) / 5)
    a = rank(delay(x, 2)) * rank(rank(d.volume))
    b = x / (d.vwap - d.close)
    return a / b


def alpha084(d):
    return signedpower(ts_rank(d.vwap - ts_max(d.vwap, 15.3217), 20.7127), delta(d.close, 4.96796))


def alpha085(d):
    hc_mix = (d.high * 0.876703) + (d.close * (1 - 0.876703))
    a = rank(correlation(hc_mix, d.adv(30), 9.61331))
    b = rank(correlation(ts_rank((d.high + d.low) / 2, 3.70596), ts_rank(d.volume, 10.1595), 7.11408))
    return a ** b


def alpha086(d):
    a = ts_rank(correlation(d.close, sum_(d.adv(20), 14.7444), 6.00049), 20.4195)
    b = rank((d.open + d.close) - (d.vwap + d.open))
    return (a < b).astype(float) * -1


def alpha087(d):
    cv_mix = (d.close * 0.369701) + (d.vwap * (1 - 0.369701))
    a = rank(decay_linear(delta(cv_mix, 1.91233), 2.65461))
    x = abs_(correlation(indneutralize(d.adv(81), d.industry), d.close, 13.4132))
    b = ts_rank(decay_linear(x, 4.89768), 14.4535)
    return emax(a, b) * -1


def alpha088(d):
    a = rank(decay_linear((rank(d.open) + rank(d.low)) - (rank(d.high) + rank(d.close)), 8.06882))
    x = correlation(ts_rank(d.close, 8.44728), ts_rank(d.adv(60), 20.6966), 8.01266)
    b = ts_rank(decay_linear(x, 6.65053), 2.61957)
    return emin(a, b)


def alpha089(d):
    low_mix = (d.low * 0.967285) + (d.low * (1 - 0.967285))
    a = ts_rank(decay_linear(correlation(low_mix, d.adv(10), 6.94279), 5.51607), 3.79744)
    b = ts_rank(decay_linear(delta(indneutralize(d.vwap, d.industry), 3.48158), 10.1466), 15.3012)
    return a - b


def alpha090(d):
    a = rank(d.close - ts_max(d.close, 4.66719))
    b = ts_rank(correlation(indneutralize(d.adv(40), d.subindustry), d.low, 5.38375), 3.21856)
    return (a ** b) * -1


def alpha091(d):
    x = decay_linear(decay_linear(correlation(indneutralize(d.close, d.industry), d.volume, 9.74928), 16.398), 3.83219)
    a = ts_rank(x, 4.8667)
    b = rank(decay_linear(correlation(d.vwap, d.adv(30), 4.01303), 2.6809))
    return (a - b) * -1


def alpha092(d):
    x = (((d.high + d.low) / 2) + d.close) < (d.low + d.open)
    a = ts_rank(decay_linear(x.astype(float), 14.7221), 18.8683)
    b = ts_rank(decay_linear(correlation(rank(d.low), rank(d.adv(30)), 7.58555), 6.94024), 6.80584)
    return emin(a, b)


def alpha093(d):
    a = ts_rank(decay_linear(correlation(indneutralize(d.vwap, d.industry), d.adv(81), 17.4193), 19.848), 7.54455)
    cv_mix = (d.close * 0.524434) + (d.vwap * (1 - 0.524434))
    b = rank(decay_linear(delta(cv_mix, 2.77377), 16.2664))
    return a / b


def alpha094(d):
    a = rank(d.vwap - ts_min(d.vwap, 11.5783))
    b = ts_rank(correlation(ts_rank(d.vwap, 19.6462), ts_rank(d.adv(60), 4.02992), 18.0926), 2.70756)
    return (a ** b) * -1


def alpha095(d):
    a = rank(d.open - ts_min(d.open, 12.4105))
    x = rank(correlation(sum_((d.high + d.low) / 2, 19.1351), sum_(d.adv(40), 19.1351), 12.8742)) ** 5
    b = ts_rank(x, 11.7584)
    return (a < b).astype(float)


def alpha096(d):
    a = ts_rank(decay_linear(correlation(rank(d.vwap), rank(d.volume), 3.83878), 4.16783), 8.38151)
    x = ts_argmax(correlation(ts_rank(d.close, 7.45404), ts_rank(d.adv(60), 4.13242), 3.65459), 12.6556)
    b = ts_rank(decay_linear(x, 14.0365), 13.4143)
    return emax(a, b) * -1


def alpha097(d):
    lv_mix = (d.low * 0.721001) + (d.vwap * (1 - 0.721001))
    a = rank(decay_linear(delta(indneutralize(lv_mix, d.industry), 3.3705), 20.4523))
    x = ts_rank(correlation(ts_rank(d.low, 7.87871), ts_rank(d.adv(60), 17.255), 4.97547), 18.5925)
    b = ts_rank(decay_linear(x, 15.7152), 6.71659)
    return (a - b) * -1


def alpha098(d):
    a = rank(decay_linear(correlation(d.vwap, sum_(d.adv(5), 26.4719), 4.58418), 7.18088))
    x = ts_rank(ts_argmin(correlation(rank(d.open), rank(d.adv(15)), 20.8187), 8.62571), 6.95668)
    b = rank(decay_linear(x, 8.07206))
    return a - b


def alpha099(d):
    a = rank(correlation(sum_((d.high + d.low) / 2, 19.8975), sum_(d.adv(60), 19.8975), 8.8136))
    b = rank(correlation(d.low, d.volume, 6.28259))
    return (a < b).astype(float) * -1


def alpha100(d):
    x = rank((((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)) * d.volume)
    a = 1.5 * scale(indneutralize(indneutralize(x, d.subindustry), d.subindustry))
    y = correlation(d.close, rank(d.adv(20)), 5) - rank(ts_argmin(d.close, 30))
    b = scale(indneutralize(y, d.subindustry))
    return 0 - (1 * ((a - b) * (d.volume / d.adv(20))))


def alpha101(d):
    return (d.close - d.open) / ((d.high - d.low) + 0.001)


ALL_ALPHAS = {
    f"alpha{i:03d}": globals()[f"alpha{i:03d}"]
    for i in range(1, 102)
}


# Alphas that use indneutralize(x, IndClass.*) - need a sector/industry
# mapping on MarketData to be meaningful (falls back to whole-universe/
# whole-subset demeaning otherwise, which can degenerate badly for a small
# ticker list - see SECTOR_COVERAGE_MIN in the scripts that use this).
INDNEUTRALIZE_ALPHAS = {
    "alpha048", "alpha058", "alpha059", "alpha063", "alpha067", "alpha069",
    "alpha070", "alpha076", "alpha079", "alpha080", "alpha082", "alpha087",
    "alpha089", "alpha090", "alpha091", "alpha093", "alpha097", "alpha100",
}