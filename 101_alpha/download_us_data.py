"""
Download US stock data (OHLCV + market cap + sector/industry) for a fixed
US-ticker universe and save it as one long-format CSV "database", so
downstream scripts (e.g. hk_alpha101_single.py, which despite its name only
cares about the CSV schema) never need to touch the network - they only
ever read this file.

Output schema (one row per date x ticker):
    date, ticker, open, high, low, close, volume, cap, sector, industry
(cap/sector/industry columns are omitted if fetched with --no-cap/--no-industry)

Usage
-----
python download_us_data.py                                 # 90d (see DAYS_DEFAULT below), ~2,580-ticker default universe -> us_universe_data.csv
python download_us_data.py --days 365 --out data1y.csv
python download_us_data.py --tickers AAPL MSFT --years 3
python download_us_data.py --no-cap --no-industry           # OHLCV only, much faster (skips per-ticker .info calls)

Shared download/reshape logic (fetch_ohlcv, fetch_cap, fetch_industry,
to_long_format) lives in data_download_common.py - download_hk_data.py uses
the same functions with a different ticker list and CONFIG below.

The default universe is ~2,580 Russell 3000 tickers (see load_us_universe()
in data_download_common.py for exact filter criteria and provenance) - still
enough tickers that a full run takes a while and uses a lot of yfinance
requests. Use --tickers to test on a small subset first.
"""

from pathlib import Path

import data_download_common as common

# =============================================================================
# CONFIG - edit these
# =============================================================================
INTERVAL_DEFAULT = "1d"  # daily bars by default (unlike download_hk_data.py's hourly default) - override with --interval
YEARS_DEFAULT = 10         # yfinance only serves ~2 years of "1h" bars; switch to --interval 1d for a multi-year pull
OUT_DEFAULT = "us_universe_data.csv"
CACHE_DIR = Path(__file__).parent / "us_alpha101_cache"
MIN_ADV_DEFAULT = 1_000_000  # USD average daily dollar volume floor - screens out illiquid microcap names;
                              # see the GEM-microcap liquidity exploitation finding from ML backtesting

# Russell 3000 constituents (~2,580 tickers, via iShares' IWV ETF holdings) -
# see load_us_universe() in data_download_common.py for the exact filter and
# provenance. This replaced a hand-curated ~600-ticker Dow/Nasdaq-100/S&P 500
# sample, which added a "must be a large, currently-index-eligible winner"
# bias on top of survivorship. Russell 3000 is a broader index cutoff (covers
# ~98% of US investable market cap) chosen as a faster-to-fetch middle ground
# than an even broader "every currently-listed US common stock" universe
# would be - it still doesn't fix true survivorship bias (delisted names
# before this snapshot are still absent); see load_us_universe()'s docstring
# for what would. Pass --tickers to override with your own list, or refresh
# via a fresh IWV holdings file periodically.
tickers = common.load_us_universe()


def main():
    p = common.build_arg_parser(
        market_label="US",
        default_tickers_help="US tickers to download (default: ~2,580 common stocks on Russell 3000)",
        default_out=OUT_DEFAULT,
        default_years=YEARS_DEFAULT,
        default_days=None,
        default_min_adv=MIN_ADV_DEFAULT,
    )
    args = p.parse_args()
    if args.start is None and args.days is None and args.years is None:
        args.years = YEARS_DEFAULT
    ticker_list = args.tickers or tickers
    common.run(tickers=ticker_list, args=args, interval_default=INTERVAL_DEFAULT,
               cache_dir=CACHE_DIR, market_name="US")


if __name__ == "__main__":
    main()
