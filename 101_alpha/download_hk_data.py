"""
Download HK stock data (OHLCV + market cap + sector/industry) for the 101-alphas
universe and save it as one long-format CSV "database", so hk_alpha101_single.py
never needs to touch the network - it only ever reads this file.

Output schema (one row per date x ticker):
    date, ticker, open, high, low, close, volume, cap, sector, industry
(cap/sector/industry columns are omitted if fetched with --no-cap/--no-industry)

Usage
-----
python download_hk_data.py                                # 1d bars, 10y (see YEARS_DEFAULT below), ~2,790-ticker default universe -> hk_universe_data.csv
python download_hk_data.py --years 20 --out data20y.csv
python download_hk_data.py --tickers 0700.HK 9988.HK --years 3
python download_hk_data.py --no-cap --no-industry          # OHLCV only, much faster (skips per-ticker .info calls)

Shared download/reshape logic (fetch_ohlcv, fetch_cap, fetch_industry,
to_long_format) lives in data_download_common.py - download_us_data.py uses
the same functions with a different ticker list and CONFIG below.

The default universe is ~2,790 tickers (see load_hk_universe() in
data_download_common.py for exact filter criteria and provenance) - a full
run at this size takes much longer and uses far more yfinance requests than
the old ~500-ticker list did. Use --tickers to test on a small subset first.
"""

from pathlib import Path

import data_download_common as common

# =============================================================================
# CONFIG - edit these
# =============================================================================
INTERVAL_DEFAULT = "1d"
YEARS_DEFAULT = 10       # matches alpha_ML.py's YEARS_HISTORY
OUT_DEFAULT = "hk_universe_data.csv"
CACHE_DIR = Path(__file__).parent / "hk_alpha101_cache"
MIN_ADV_DEFAULT = 5_000_000  # HKD average daily dollar volume floor - screens out illiquid GEM/penny names;
                              # see the GEM-microcap liquidity exploitation finding from ML backtesting
BENCHMARK_TICKER = "^HSI"    # saved into the same CSV (exempt from the ADV screen) so backtest scripts
                              # (e.g. hk_alpha_stock_backtest.py) can read its price series straight out of
                              # the CSV instead of making their own live yfinance call every run

# All currently-listed HKD-denominated ordinary equities + REITs on HKEX Main
# Board/GEM (~2,790 tickers) - see load_hk_universe() in data_download_common.py
# for the exact filter and provenance. This replaced a hand-curated "top ~500
# by today's market cap" list: that selection criterion added a "must still be
# a big winner today" bias on top of survivorship, since only stocks that both
# survived AND grew large enough to rank in the top ~500 were included. This
# broader list isn't conditioned on size or performance - it's just everything
# currently tradable in the relevant categories. It does NOT fix true
# survivorship bias (delisted names before this snapshot are still absent);
# see load_hk_universe()'s docstring for what would. Pass --tickers to
# override with your own list, or refresh via a fresh HKEX file periodically.
tickers = common.load_hk_universe()


def main():
    p = common.build_arg_parser(
        market_label="HK",
        default_tickers_help="HK tickers to download (default: ~2,790 HKD-denominated equities/REITs on HKEX Main Board/GEM)",
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
               cache_dir=CACHE_DIR, market_name="HK", benchmark_ticker=BENCHMARK_TICKER)


if __name__ == "__main__":
    main()