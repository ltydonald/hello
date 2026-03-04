import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import yfinance as yf
tickers = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", 
    "WMT", "ASML", "MU", "COST", "NFLX", "AMD", "CSCO", "LRCX", 
    "AMAT", "TMUS", "LIN", "PEP", "INTC", "AMGN", "KLAC", "TXN", "ISRG", 
    "GILD", "ADI", "SHOP", "QCOM", "HON", "BKNG", 
    "VRTX", "PANW", "SBUX", "CMCSA", "INTU", "ADBE","WDC", 
    "MAR", "STX", "MELI", "ADP", "MNST", "SNPS", "REGN", "CDNS", 
    "CSX", "CTAS", "ORLY", "MDLZ", "AEP", "WBD", "MRVL", "ROST", 
    "PCAR", "FTNT", "NXPI", "MPWR", "IDXX", "FAST", "EA", 
    "EXC", "ADSK", "XEL", "CCEP", "FANG", "MSTR", "TRI", "ALNY", "AXON", 
    "ODFL", "MCHP", "TTWO", "ROP", "WDAY", 
    "CPRT", "PAYX", "INSM", "CTSH", "KHC", "CHTR", "DXCM", "VRSK", "CSGP"
]
capital = 1000000
portfolio_capital = None
# Download historical data for the NASDAQ-100 stocks
data = yf.download(tickers, start="2015-01-01", end="2025-12-31")['Close'].pct_change(fill_method=None).dropna()
# Calculate the correlation matrix for each 6-month period
corr1 = data.groupby(pd.Grouper(freq='182D', origin='start', label='right')).corr()
corr2 = data.groupby(pd.Grouper(freq='182D', origin='start', offset='91D', label='right')).corr()
corr = pd.concat([corr1, corr2]).sort_index()
# Shift the index of the correlation DataFrame to prevent look-ahead bias
corr.index = corr.index.set_levels(corr.index.levels[0] + pd.DateOffset(days=1), level=0)
# For each period, find the stock with the lowest average absolute correlation to all other stocks
def create_portfolio(group):
    portfolio = []
    stock1 = group.abs().mean().idxmin()
    stock2 = group[stock1].drop(group[group.index.get_level_values(1) == stock1].index).abs().idxmin(axis=0)[-1]
    portfolio.append(stock1)
    portfolio.append(stock2)
    for i in range(3, 11):
        stock = group[portfolio].drop(group[group.index.get_level_values(1).isin(portfolio)].index).abs().mean(axis=1).idxmin()[-1]
        portfolio.append(stock)
    return portfolio
portfolio = corr.groupby(level=0).apply(create_portfolio)
# Allocate capital equally among the 10 stocks in the portfolio and track the capital over time
returns = data[portfolio.iloc[0]].loc[portfolio.index[0]:portfolio.index[1]-pd.Timedelta(days=1)]
portfolio_capital = capital/10 * (1 + returns).cumprod()
portfolio_capital.columns = range(10)
# Initialize the portfolio tracker with the first period's capital
portfolio_tracker = portfolio_capital
capital = portfolio_capital.iloc[-1].sum()
# For each subsequent period, calculate the returns for the new portfolio and update the capital accordingly
for i in range(1, len(portfolio)-1):
    if portfolio.index[i] > data.index[-1]:
        break
    returns = data[portfolio.iloc[i]].loc[portfolio.index[i]:portfolio.index[i+1]-pd.Timedelta(days=1)]
    portfolio_capital = capital/10 * (1 + returns).cumprod()
    portfolio_capital.columns = range(10)
    portfolio_tracker = pd.concat([portfolio_tracker, portfolio_capital])
    capital = portfolio_capital.iloc[-1].sum()
# Calculate performance metrics
overall_portfolio_track = portfolio_tracker.sum(axis=1)
# Calculate drawdown
max_drawdown=((overall_portfolio_track.cummax()-overall_portfolio_track)/overall_portfolio_track.cummax()).max()
print(f'Max Drawdown: {max_drawdown:.2%}')
# Calculate Sharpe Ratio
sharpe_ratio = (overall_portfolio_track.pct_change().dropna().mean() / overall_portfolio_track.pct_change().dropna().std()) * (252**0.5)
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
# # Plot the portfolio capital over time and compare it to buy and hold QQQ
nasdaq = yf.download('QQQ', start= '2015-01-01', end = '2025-12-31')['Close'].pct_change().dropna().loc[overall_portfolio_track.index[0]:overall_portfolio_track.index[-1]].apply(lambda x: (1+x)).cumprod().mul(1000000)
max_nasdaq_drawdown=(((nasdaq.cummax()-nasdaq)/nasdaq.cummax()).max()).iloc[0]
print(f'NASDAQ-100 Max Drawdown: {max_nasdaq_drawdown:.2%}')
nasdaq_sharpe_ratio = ((nasdaq.pct_change().dropna().mean() / nasdaq.pct_change().dropna().std()) * (252**0.5)).iloc[0]
print(f'NASDAQ-100 Sharpe Ratio: {nasdaq_sharpe_ratio:.2f}')
print(portfolio)
plt.figure(figsize=(20,5))
plt.plot(overall_portfolio_track.index, overall_portfolio_track.values, label='Portfolio')
plt.plot(nasdaq.index, nasdaq.values, label='NASDAQ-100')
plt.title('Portfolio Capital Over Time')
plt.xlabel('Date')
plt.ylabel('Capital')
plt.legend()
plt.show()