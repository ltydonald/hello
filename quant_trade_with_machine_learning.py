import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.inspection import permutation_importance
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Parameters ----
tickers = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'AVGO', 'TSLA', 'AMD',
    'ADBE', 'CRM', 'CSCO', 'NFLX', 'INTC', 'ORCL', 'QCOM', 'ACN', 'IBM', 'TXN',
    'AMAT', 'INTU', 'NOW', 'MU', 'LRCX', 'UBER', 'PYPL', 'ADI', 'SHOP', 'PANW',
    'SNPS', 'BKNG', 'CDNS', 'ABNB', 'MSCI', 'ADP', 'VRTX', 'MCHP', 'REGN', 'FTNT',
    'KLAC', 'ROP', 'FISV', 'WDAY', 'ZS', 'TEAM', 'MAR', 'DDOG', 'ANET', 'CRWD'
]
period = '150d'
interval = '1d'
short_term_MA_window = 5
long_term_MA_window = 20
rsi_window = 5
future_window = 1  # Number of days to predict into the future
n_lags = 8

# === Set your target stock here ===
TARGET_STOCK = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'AVGO', 'TSLA', 'AMD']

def add_lagged_features(data, feature_names, n_lags):
    for feature in feature_names:
        for lag in range(1, n_lags+1):
            data[f'{feature}_lag{lag}'] = data[feature].shift(lag)
    return data

def calculate_MA(data):
    data['MA20'] = data['Close'].rolling(window=short_term_MA_window).mean()
    data['MA60'] = data['Close'].rolling(window=long_term_MA_window).mean()
    data['MA20_MA60_diff'] = data['MA20'] - data['MA60']
    return data

def calculate_rsi(data):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_MACD(data):
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'] - data['Signal_Line']
    return data

def calculate_bolinger_bands(data, window=20, num_std=2):
    data['BB_Middle'] = data['Close'].rolling(window=window).mean()
    data['BB_Upper'] =  data['Close'].rolling(window=window).mean() + (data['Close'].rolling(window=window).std() * num_std)
    data['BB_Lower'] = data['Close'].rolling(window=window).mean() - (data['Close'].rolling(window=window).std() * num_std)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower'])/ data['BB_Middle']
    return data

def calculate_stochastic_oscillator(data, k_window=14, d_window=3):
    data['Lowest_Low'] = data['Low'].rolling(window=k_window).min()
    data['Highest_High'] = data['High'].rolling(window=k_window).max()
    data['%K'] = 100 * (data['Close'] - data['Low'].rolling(window=k_window).min()) / (data['High'].rolling(window=k_window).max() - data['Low'].rolling(window=k_window).min())
    data['%D'] = data['%K'].rolling(window=d_window).mean()
    data['%K_%D_diff'] = data['%K'] - data['%D']
    return data

def add_ml_features(data):
    data['Return'] = data['Close'].pct_change()
    data['Future_Cum_Return'] = data['Close'].shift(-future_window) / data['Close'] - 1
    return data

def prepare_ml_data(data, n_lags):
    lag_features = [
        'RSI', 'MACD_Signal', 'Return', 'MA20_MA60_diff', '%K_%D_diff', 'Volume'
    ]
    data = add_lagged_features(data, lag_features, n_lags)
    feature_columns = []
    for f in lag_features:
        feature_columns += [f"{f}_lag{lag}" for lag in range(1, n_lags+1)]
    data = data.dropna()
    X = data[feature_columns]
    y = data['Future_Cum_Return']
    return X, y, data, feature_columns

def predict_future_change_for_stock(
    tickers,
    svr_model,
    svc_model,
    scaler_svr,
    scaler_svc,
    n_lags=3,
    period='150d',
    interval='1d'
):
    for ticker in tickers:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty or len(data) < 80:
            print(f"Not enough data for {ticker}. Skipping.")
            continue

        data = calculate_MA(data)
        data = calculate_rsi(data)
        data = calculate_MACD(data)
        data = calculate_bolinger_bands(data)
        data = calculate_stochastic_oscillator(data)
        data = add_ml_features(data)
        data['Future_Cum_Return'] = data['Future_Cum_Return'].fillna(0)  # Fill NaN values with 0
        lag_features = [
            'RSI', 'MACD_Signal', 'Return', 'MA20_MA60_diff', '%K_%D_diff', 'Volume'
        ]
        data = add_lagged_features(data, lag_features, n_lags)
        feature_columns = []
        for f in lag_features:
            feature_columns += [f"{f}_lag{lag}" for lag in range(1, n_lags+1)]
        
        data = data.dropna()
        if data.empty:
            print("No valid rows after feature/lags for prediction.")
            continue

        # Get the last date and prediction
        last_row = data.iloc[[-1]]
        last_date = last_row.index[-1].strftime('%Y-%m-%d')
        X_pred = last_row[feature_columns].astype(np.float32)

        # SVR prediction
        X_pred_scaled_svr = scaler_svr.transform(X_pred)
        pred_return_svr = svr_model.predict(X_pred_scaled_svr)[0]

        # SVC prediction
        X_pred_scaled_svc = scaler_svc.transform(X_pred)
        pred_class_svc = svc_model.predict(X_pred_scaled_svc)[0]
        pred_prob_svc = svc_model.predict_proba(X_pred_scaled_svc)[0][1]

        # Example signal logic
        if pred_class_svc == 1:
            signal = "Buy"
        else:
            signal = "Sell"

        print(f"\nPrediction for {ticker} on {last_date}:")
        print(f"  SVR predicted future cumulative return (next {future_window} days): {pred_return_svr:.4f}")
        print(f"  SVC predicted direction (1=up, 0=down): {pred_class_svc} (prob up: {pred_prob_svc:.2f})")
        print(f"  Recommended Action on {last_date}: {signal}")

    # Optionally, return or store signals for further use

def backtest_signals_for_stock(
    ticker,
    svc_model,
    scaler_svc,
    n_lags=3,
    period='150d',
    interval='1d'
):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty or len(data) < 80:
        print(f"Not enough data for {ticker}. Skipping.")
        return None

    data = calculate_MA(data)
    data = calculate_rsi(data)
    data = calculate_MACD(data)
    data = calculate_bolinger_bands(data)
    data = calculate_stochastic_oscillator(data)
    data = add_ml_features(data)

    lag_features = [
        'RSI', 'MACD_Signal', 'Return', 'MA20_MA60_diff', '%K_%D_diff', 'Volume'
    ]
    data = add_lagged_features(data, lag_features, n_lags)
    feature_columns = []
    for f in lag_features:
        feature_columns += [f"{f}_lag{lag}" for lag in range(1, n_lags+1)]
    data = data.dropna()
    if data.empty:
        print("No valid rows after feature/lags for prediction.")
        return None

    X = data[feature_columns].astype(np.float32)
    X_scaled = scaler_svc.transform(X)
    pred_class_svc = svc_model.predict(X_scaled)

    signals = pd.DataFrame({
        'date': data.index.strftime('%Y-%m-%d'),
        'signal': pred_class_svc
    })
    signals.set_index('date', inplace=True)

    # Add signals to data for cumulative return calculation
    data['signal'] = pred_class_svc
    data['date'] = data.index.strftime('%Y-%m-%d')

    # Print buy/sell signals for reference
    print(f"\nBacktested signals for {ticker}:")
    print(signals[signals['signal'] == 1].rename(columns={'signal': 'Buy Signal'}))
    print(signals[signals['signal'] == 0].rename(columns={'signal': 'Sell Signal'}))

    # Return the data with signals for cumulative return calculation
    return data

def plot_cumulative_return(data, ticker):
    """
    Plot cumulative return of ML strategy vs buy & hold using signal column and Close price.
    """
    df = data.copy()
    df['return'] = df['Close'].pct_change().fillna(0)
    df['strategy_return'] = df['return'] * df['signal'].shift(1, fill_value=0)  # lag to avoid lookahead bias
    df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod() - 1
    df['cumulative_buy_hold'] = (1 + df['return']).cumprod() - 1

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_strategy'], label='ML Strategy')
    plt.plot(df.index, df['cumulative_buy_hold'], label='Buy & Hold', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title(f'Cumulative Return: {ticker}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print(f"Strategy Cumulative Return: {df['cumulative_strategy'].iloc[-1]:.2%}")
    print(f"Buy & Hold Cumulative Return: {df['cumulative_buy_hold'].iloc[-1]:.2%}")

def plot_feature_importance(importances, feature_names, model_name, n_top=10):
    """Plot feature importance returned from permutation_importance."""
    importances = np.array(importances)
    indices = np.argsort(importances)[::-1][:n_top]
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_top), importances[indices], align='center')
    plt.xticks(range(n_top), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title(f"Top {n_top} Feature Importances ({model_name})")
    plt.tight_layout()
    plt.show()

def plot_feature_correlation(X, feature_columns):
    corr = X[feature_columns].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def main():
    all_X, all_y = [], []
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty or len(data) < 150:
            print(f"Not enough data for {ticker}. Skipping.")
            continue
        data = calculate_MA(data)
        data = calculate_rsi(data)
        data = calculate_MACD(data)
        data = calculate_bolinger_bands(data)
        data = calculate_stochastic_oscillator(data)
        data = add_ml_features(data)
        X, y, _, feature_columns = prepare_ml_data(data, n_lags)
        all_X.append(X)
        all_y.append(y)
    X_all = pd.concat(all_X, axis=0).astype(np.float32)
    y_all = pd.concat(all_y, axis=0).astype(np.float32)

    # ========== SVM Regression ==========
    print("\n===== Support Vector Regression (SVR) Baseline =====")
    scaler_svr = StandardScaler()
    svr = SVR(kernel='rbf')
    tscv = TimeSeriesSplit(n_splits=5)
    fold = 0
    for train_index, test_index in tscv.split(X_all):
        fold += 1
        X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
        y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index]
        y_train_bin = (y_train > 0).astype(int)
        y_test_bin = (y_test > 0).astype(int)

        X_train_scaled_svr = scaler_svr.fit_transform(X_train)
        X_test_scaled_svr = scaler_svr.transform(X_test)

        svr.fit(X_train_scaled_svr, y_train)
        y_pred_svr = svr.predict(X_test_scaled_svr)

    print("SVR Results:")
    print(f"  MSE:  {mean_squared_error(y_test, y_pred_svr):.6f}")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred_svr):.6f}")
    print(f"  R^2:  {r2_score(y_test, y_pred_svr):.4f}")
    print("  Sample predictions:")
    print(pd.DataFrame({'Pred': y_pred_svr[:5], 'Actual': y_test.values[:5]}))

    # ========== Feature Importance for SVR ==========
    print("\n===== Permutation Feature Importance (SVR) =====")
    svr_perm = permutation_importance(svr, X_test_scaled_svr, y_test, n_repeats=10, random_state=42)
    for f, imp in zip(feature_columns, svr_perm.importances_mean):
        print(f"{f}: {imp:.6f}")
    """
    plot_feature_importance(svr_perm.importances_mean, feature_columns, "SVR")
    """
    # ========== SVM Classification ==========
    print("\n===== Support Vector Classification (SVC) Baseline =====")
    y_train_bin = (y_train > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)
    scaler_svc = scaler_svr  # Use separate scaler if needed
    svc = SVC(kernel='rbf', probability=True)
    svc.fit(X_train_scaled_svr, y_train_bin)
    y_pred_svc = svc.predict(X_test_scaled_svr)
    y_prob_svc = svc.predict_proba(X_test_scaled_svr)[:, 1]

    print("SVC Classification Results:")
    print(f"  Accuracy: {accuracy_score(y_test_bin, y_pred_svc):.4f}")
    print(f"  F1 Score: {f1_score(y_test_bin, y_pred_svc):.4f}")
    print(f"  ROC AUC:  {roc_auc_score(y_test_bin, y_prob_svc):.4f}")
    print("  Classification Report:\n", classification_report(y_test_bin, y_pred_svc, digits=4))
    print("  Sample predictions:")
    print(pd.DataFrame({'Pred': y_pred_svc[:5], 'Prob': y_prob_svc[:5], 'Actual': y_test_bin.values[:5]}))

    # ========== SVM Classification with TimeSeriesSplit ==========
    print("\n===== SVC Classification with TimeSeriesSplit =====")
    scaler = StandardScaler()
    tscv = TimeSeriesSplit(n_splits=5)
    fold = 0
    for train_index, test_index in tscv.split(X_all):
        fold += 1
        X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
        y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index]
        y_train_bin = (y_train > 0).astype(int)
        y_test_bin = (y_test > 0).astype(int)

        # Feature scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svc.fit(X_train_scaled, y_train_bin)
        y_pred = svc.predict(X_test_scaled)
        y_prob = svc.predict_proba(X_test_scaled)[:, 1]

        print(f"\nFold {fold}:")
        print(f"  Accuracy: {accuracy_score(y_test_bin, y_pred):.4f}")
        print(f"  F1 Score: {f1_score(y_test_bin, y_pred):.4f}")
        print(f"  ROC AUC: {roc_auc_score(y_test_bin, y_prob):.4f}")
        print("  Classification Report:\n", classification_report(y_test_bin, y_pred, digits=4))
        print("  Sample predictions:")
        print(pd.DataFrame({'Pred': y_pred[:5], 'Prob': y_prob[:5], 'Actual': y_test_bin.values[:5]}))

    # ========== Prediction for Target Stock ==========
    print(f"\n===== Prediction for target stock: {TARGET_STOCK} =====")
    predict_future_change_for_stock(
        TARGET_STOCK,
        svr_model=svr,
        svc_model=svc,
        scaler_svr=scaler_svr,
        scaler_svc=scaler_svc,
        n_lags=n_lags,
        period=period,
        interval=interval
    )

    #========== Backtest for Target Stock: Buy/Sell Dates and Cumulative Return ==========
    for ticker in TARGET_STOCK:
        print(f"\n===== Backtesting for {ticker} =====")
        backtest_data = backtest_signals_for_stock(
            ticker,
            svc_model=svc,
            scaler_svc=scaler_svc,
            n_lags=n_lags,
            period=period,
            interval=interval
        )
        if backtest_data is not None:
            plot_cumulative_return(backtest_data, ticker)


if __name__ == "__main__":
    main()