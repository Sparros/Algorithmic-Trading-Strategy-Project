import numpy as np
import pandas as pd

from pykalman import KalmanFilter

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.stock_data import fetch_multiple_stock_data

EPS = 1e-9

def _safe_div(a, b):
    """Elementwise safe divide, returns NaN where denom == 0."""
    return a / b.replace(0, np.nan)

def _has_cols(df, cols):
    return all(c in df.columns for c in cols)

def _has_volume(df, prefix):
    return f"Volume_{prefix}" in df.columns

def shift_to_t_minus_1(df: pd.DataFrame, cols, lag: int = 1) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].shift(lag)
    return out

def calculate_daily_returns(df_input, cols_to_process=['Close']):
    """
    Calculate daily returns for specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data.
    cols_to_process (list): List of column names to calculate daily returns for.

    Returns:
    pd.DataFrame: DataFrame with daily returns added as new columns.
    """
    df = df_input.copy()
    for col in cols_to_process:
        if col in df.columns:
            df[f'{col}_daily_return'] = df[col].pct_change()
        else:
            print(f"Warning: Column '{col}' not found for daily return calculation.")
    return df

def create_next_day_targets(df_input, cols_to_process):
    """
    Creates next-day percentage return targets for specified columns.
    Assumes that the input DataFrame already contains daily return columns
    named '{original_col_name}_daily_return'.

    Parameters:
    df_input (pd.DataFrame): DataFrame containing daily return columns.
    cols_to_process (list): List of original column names (e.g., 'LLOY.L', 'BARC.L')
                            for which you want to create next-day targets
                            from their daily return columns.

    Returns:
    pd.DataFrame: DataFrame with additional '{original_col_name}_Next_Day_Return' columns.
    """
    df = df_input.copy()
    print("  - Creating next-day return targets...")

    for col in cols_to_process:
        daily_return_col_name = f'{col}_daily_return'
        next_day_target_col_name = f'{col}_Next_Day_Return'

        if daily_return_col_name in df.columns:
            # Shift the daily return column up by 1.
            # This makes the value at row 'X' the daily return for day 'X+1'.
            df[next_day_target_col_name] = df[daily_return_col_name].shift(-1)
        else:
            print(f"    Warning: Daily return column '{daily_return_col_name}' not found for creating '{next_day_target_col_name}'.")
    return df

def add_lagged_features(df_input, cols_to_lag, lag_periods):
    """
    Adds lagged features for specified columns and lag periods.

    Parameters:
    df_input (pd.DataFrame): DataFrame containing the original features.
    cols_to_lag (list): List of column names (strings) for which lagged features should be created.
                        These columns should already exist in df_input.
                        Examples: ['Close_AAPL', 'Close_AAPL_daily_return', '^FTSE_SMA20']
    lag_periods (list): List of integers representing the number of periods (days) to lag.
                        E.g., [1, 5, 10] would create lags for 1, 5, and 10 periods ago.

    Returns:
    pd.DataFrame: DataFrame with additional lagged feature columns.
    """
    df = df_input.copy()
    print("  - Adding lagged features...")

    # Iterate through each column for which we want to create lags
    for col in cols_to_lag:
        if col in df.columns:
            # Iterate through each specified lag period
            for lag in lag_periods:
                # Construct the new column name (e.g., 'Close_AAPL_lag1', 'Close_AAPL_daily_return_lag5')
                lagged_col_name = f'{col}_lag{lag}'
                
                # Use the .shift() method to create the lagged feature
                # .shift(N) moves data N rows *down*. So, df.loc[i, lagged_col] = df.loc[i-N, original_col]
                # This means at the current row, you get the value from N periods ago.
                df[lagged_col_name] = df[col].shift(lag)
        else:
            print(f"    Warning: Column '{col}' not found in DataFrame for lagging. Skipping.")
    return df

def add_price_range_features(df_input, ticker_prefix):
    """
    Adds price range and strength features (High-Low, Open-Close, Close-to-Range Ratio).

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'High', 'Low', 'Open', 'Close' columns
                             (e.g., High_AAPL, Low_AAPL, etc.).
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').

    Returns:
    pd.DataFrame: DataFrame with additional range features.
    """
    df = df_input.copy()
    H, L, O, C = (f'High_{ticker_prefix}', f'Low_{ticker_prefix}',
                  f'Open_{ticker_prefix}', f'Close_{ticker_prefix}')
    if _has_cols(df, [H,L,O,C]):
        rng = df[H] - df[L]
        df[f'{ticker_prefix}_HighLow_Range'] = rng
        df[f'{ticker_prefix}_OpenClose_Range'] = df[C] - df[O]
        df[f'{ticker_prefix}_Close_to_Range_Ratio'] = _safe_div(df[C] - df[L], rng).fillna(0.5)
    else:
        print(f"Warning: Missing HLOC for {ticker_prefix} → range features skipped.")
    return df

def calculate_true_range(df_input, ticker_prefix):
    """
    Calculates the True Range (TR) for a given ticker and adds it as a new column.

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns for the ticker.
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').

    Returns:
    pd.DataFrame: DataFrame with an additional True Range column.
    """
    df = df_input.copy()
    H, L, C = (f'High_{ticker_prefix}', f'Low_{ticker_prefix}', f'Close_{ticker_prefix}')
    if _has_cols(df, [H,L,C]):
        prev_close = df[C].shift(1)
        tr1 = df[H] - df[L]
        tr2 = (df[H] - prev_close).abs()
        tr3 = (df[L] - prev_close).abs()
        df[f'{ticker_prefix}_True_Range'] = np.maximum.reduce([tr1, tr2, tr3])
    else:
        print(f"Warning: Missing OHLC for {ticker_prefix} → True Range skipped.")
    return df

def calculate_atr(df_input, ticker_prefix, window=14):
    """
    Calculates Average True Range (ATR) for a specific ticker.
    This function assumes that the '{ticker_prefix}_True_Range' column
    has already been calculated and is present in the df_input DataFrame.

    Parameters:
    df_input (pd.DataFrame): DataFrame containing the '{ticker_prefix}_True_Range' column.
                             Must have a DatetimeIndex.
    ticker_prefix (str): The prefix for the ticker (e.g., 'LLOY.L', 'BARC.L').
    window (int): The window size (number of periods) for the ATR calculation.
                  Commonly 14 periods.

    Returns:
    pd.DataFrame: A new DataFrame with an additional '{ticker_prefix}_ATR{window}' column.
    """
    df = df_input.copy()
    tr = f'{ticker_prefix}_True_Range'
    col = f'{ticker_prefix}_ATR{window}'
    if tr in df.columns:
        df[col] = df[tr].ewm(span=window, adjust=False, min_periods=window).mean()
    else:
        print(f"Warning: {tr} not found for {ticker_prefix} → ATR skipped.")
        df[col] = np.nan
    return df

def add_volume_features(df_input, ticker_prefix, window=20):
    """
    Adds volume change and volume-to-moving-average ratio features for a specific ticker.

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'Volume' column (e.g., Volume_AAPL).
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').
    window (int): Window size for volume moving average.

    Returns:
    pd.DataFrame: DataFrame with additional volume features.
    """
    df = df_input.copy()
    V = f'Volume_{ticker_prefix}'
    if V in df.columns:
        df[f'{ticker_prefix}_Volume_Daily_Change'] = df[V].pct_change() * 100
        vma = df[V].rolling(window=window, min_periods=1).mean()
        df[f'{ticker_prefix}_Volume_MA_{window}D'] = vma
        df[f'{ticker_prefix}_Volume_MA_Ratio'] = _safe_div(df[V], vma)
    else:
        print(f"Info: No {V} for {ticker_prefix} → volume features skipped.")
    return df

def calculate_obv(df_input, ticker_prefix):
    """
    Calculates On-Balance Volume (OBV) for a specific ticker.

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'Close' and 'Volume' columns for the ticker.
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').

    Returns:
    pd.DataFrame: DataFrame with an additional '{ticker_prefix}_OBV' column.
    """
    df = df_input.copy()
    C, V = f'Close_{ticker_prefix}', f'Volume_{ticker_prefix}'
    if _has_cols(df, [C, V]):
        price_direction = np.sign(df[C].diff().fillna(0))
        obv = (price_direction * df[V].fillna(0)).cumsum()
        # start at 0 deterministically
        if len(obv):
            obv.iloc[0] = 0
        df[f'{ticker_prefix}_OBV'] = obv
    else:
        print(f"Info: Missing Close/Volume for {ticker_prefix} → OBV skipped.")
    return df

def calculate_rsi(df_input, ticker_prefix, window=14):
    """
    Calculates the Relative Strength Index (RSI) for a specific ticker.

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'Close' column for the ticker.
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').
    window (int): The window size for RSI calculation.

    Returns:
    pd.DataFrame: DataFrame with an additional '{ticker_prefix}_RSI{window}' column.
    """
    df = df_input.copy()
    close_col = f'Close_{ticker_prefix}'

    #print(f"  - Calculating RSI for {ticker_prefix} (window={window})...")

    if close_col in df.columns:
        delta = df[close_col].diff()

        # Calculate gains (positive changes) and losses (absolute negative changes)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate Exponential Moving Average (EMA) of gains and losses
        # Using adjust=False for EMA to match traditional RSI calculation
        avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()

        # Calculate Relative Strength (RS)
        # Handle division by zero for RS where avg_loss is 0
        rs = avg_gain / avg_loss
        rs.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace inf with NaN

        # Calculate RSI
        # If avg_loss is 0 (no losses in the window), RS is inf, RSI becomes 100.
        # If avg_gain is 0 (no gains in the window), RS is 0, RSI becomes 0.
        rsi_val = 100 - (100 / (1 + rs))
        df[f'{ticker_prefix}_RSI{window}'] = rsi_val
        
        # Fill first window_size-1 NaNs (from rolling/ewm) with actual values if possible, otherwise keep NaN
        # The min_periods=window in ewm already handles this, but a final fillna for edge cases can be useful
        # For example, if min_periods wasn't used, you might fill with 50 (neutral) for initial NaNs.
        # With min_periods=window, NaNs will only appear for the first `window-1` entries.
    else:
        print(f"    Warning: Close column for {ticker_prefix} not found for RSI calculation. Skipping RSI.")
    return df

def add_moving_averages(df_input, ticker_prefix, window_sizes=[10, 20, 50, 200], ma_type='SMA'):
    """
    Adds Simple Moving Averages (SMA) or Exponential Moving Averages (EMA) to the DataFrame.

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'Close' column for the ticker.
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').
    window_sizes (list): List of window sizes for MA calculation.
    ma_type (str): 'SMA' for Simple Moving Average, 'EMA' for Exponential Moving Average.

    Returns:
    pd.DataFrame: DataFrame with additional MA columns.
    """
    df = df_input.copy()
    close_col = f'Close_{ticker_prefix}'

    if close_col in df.columns:
        for window in window_sizes:
            if ma_type == 'SMA':
                df[f'{ticker_prefix}_SMA_{window}'] = df[close_col].rolling(window=window).mean()
            elif ma_type == 'EMA':
                df[f'{ticker_prefix}_EMA_{window}'] = df[close_col].ewm(span=window, adjust=False).mean()
            else:
                print(f"Warning: Unsupported MA type '{ma_type}'. Use 'SMA' or 'EMA'.")
    else:
        print(f"Warning: Column '{close_col}' not found for moving average calculation.")
    
    return df

def add_bollinger_bands(df_input, ticker_prefix, window=20, num_std=2):
    """
    Adds Bollinger Bands to the DataFrame.

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'Close' column for the ticker.
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').
    window (int): Window size for the moving average.
    num_std (int): Number of standard deviations for the bands.

    Returns:
    pd.DataFrame: DataFrame with Bollinger Bands added.
    """
    df = df_input.copy()
    C = f'Close_{ticker_prefix}'
    if C in df.columns:
        mid = df[C].rolling(window=window, min_periods=window).mean()
        std = df[C].rolling(window=window, min_periods=window).std()
        up  = mid + num_std*std
        lo  = mid - num_std*std
        df[f'{ticker_prefix}_BB_Middle{window}'] = mid
        df[f'{ticker_prefix}_BB_Upper{window}']  = up
        df[f'{ticker_prefix}_BB_Lower{window}']  = lo
        df[f'{ticker_prefix}_BB_Bandwidth{window}'] = _safe_div(up - lo, mid + EPS)
        df[f'{ticker_prefix}_BB_PctB{window}']     = _safe_div(df[C] - lo, (up - lo))
    else:
        print(f"Warning: {C} missing → Bollinger skipped for {ticker_prefix}.")
    return df

def calculate_stochastic_oscillator(df_input, ticker_prefix, k_period=14, d_period=3):
    """
    Calculates the Stochastic Oscillator (%K and %D).

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns for the ticker.
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').
    k_period (int): Lookback period for %K.
    d_period (int): Smoothing period for %D (SMA of %K).

    Returns:
    pd.DataFrame: DataFrame with additional Stochastic Oscillator features.
    """
    df = df_input.copy()
    H, L, C = f'High_{ticker_prefix}', f'Low_{ticker_prefix}', f'Close_{ticker_prefix}'
    if _has_cols(df, [H,L,C]):
        low_k  = df[L].rolling(k_period, min_periods=k_period).min()
        high_k = df[H].rolling(k_period, min_periods=k_period).max()
        denom  = (high_k - low_k).replace(0, np.nan)
        k = _safe_div(df[C] - low_k, denom) * 100
        df[f'{ticker_prefix}_Stoch_K_{k_period}'] = k
        df[f'{ticker_prefix}_Stoch_D_{k_period}_{d_period}'] = k.rolling(d_period, min_periods=d_period).mean()
    else:
        print(f"Warning: H/L/C missing → Stoch skipped for {ticker_prefix}.")
    return df


def calculate_adx(df_input, ticker_prefix, window=14):
    """
    Calculates the Average Directional Index (ADX), +DI, and -DI.
    Requires True Range to be calculated first for DM.

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' for the ticker.
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').
    window (int): The window size for calculation.

    Returns:
    pd.DataFrame: DataFrame with additional ADX features.
    """
    df = df_input.copy()
    H, L, C = f'High_{ticker_prefix}', f'Low_{ticker_prefix}', f'Close_{ticker_prefix}'
    if _has_cols(df, [H,L,C]):
        # ensure TR exists
        df = calculate_true_range(df, ticker_prefix)
        tr_ewm = df[f'{ticker_prefix}_True_Range'].ewm(span=window, adjust=False, min_periods=window).mean()

        up_move   = df[H].diff()
        down_move = -df[L].diff()
        plusDM  = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minusDM = down_move.where((down_move > up_move) & (down_move > 0), 0)

        plusDI  = _safe_div(plusDM.ewm(span=window, adjust=False).mean(), tr_ewm) * 100
        minusDI = _safe_div(minusDM.ewm(span=window, adjust=False).mean(), tr_ewm) * 100
        df[f'{ticker_prefix}_PlusDI_{window}']  = plusDI
        df[f'{ticker_prefix}_MinusDI_{window}'] = minusDI

        di_sum  = (plusDI + minusDI).replace(0, np.nan)
        dx      = _safe_div((plusDI - minusDI).abs(), di_sum) * 100
        df[f'{ticker_prefix}_DX_{window}']  = dx.fillna(0)
        df[f'{ticker_prefix}_ADX_{window}'] = df[f'{ticker_prefix}_DX_{window}'].ewm(span=window, adjust=False).mean()
    else:
        print(f"Warning: H/L/C missing → ADX skipped for {ticker_prefix}.")
    return df

def calculate_macd(df_input, ticker_prefix, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates the MACD line, Signal line, and Histogram for a specific ticker.

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'Close' column for the ticker.
    ticker_prefix (str): The prefix for the ticker (e.g., 'AAPL').
    fast_period (int): Period for the fast EMA.
    slow_period (int): Period for the slow EMA.
    signal_period (int): Period for the signal line EMA.

    Returns:
    pd.DataFrame: DataFrame with additional MACD features.
    """
    df = df_input.copy()
    close_col = f'Close_{ticker_prefix}'

    #print(f"  - Calculating MACD for {ticker_prefix} (fast={fast_period}, slow={slow_period}, signal={signal_period})...")

    if close_col in df.columns:
        # Calculate Fast and Slow EMAs
        fast_ema = df[close_col].ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
        slow_ema = df[close_col].ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()

        # MACD Line
        df[f'{ticker_prefix}_MACD_Line'] = fast_ema - slow_ema

        # Signal Line (EMA of MACD Line)
        df[f'{ticker_prefix}_MACD_Signal'] = df[f'{ticker_prefix}_MACD_Line'].ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()

        # MACD Histogram
        df[f'{ticker_prefix}_MACD_Hist'] = df[f'{ticker_prefix}_MACD_Line'] - df[f'{ticker_prefix}_MACD_Signal']
    else:
        print(f"    Warning: Close column for {ticker_prefix} not found for MACD calculation. Skipping MACD.")
    return df

def add_time_based_features(df_input):
    """
    Adds day of week and month of year features from the DataFrame index.

    Parameters:
    df_input (pd.DataFrame): DataFrame with a DatetimeIndex.

    Returns:
    pd.DataFrame: DataFrame with additional 'day_of_week' and 'month_of_year' columns.
    """
    df = df_input.copy()
    print("  - Adding time-based features...")

    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
        df['month_of_year'] = df.index.month
    else:
        print("    Warning: DataFrame index is not a DatetimeIndex. Skipping time-based features.")
    return df

def add_relative_strength(df, stock_ticker, benchmark_ticker='^GSPC'):
    """
    Calculates the relative strength of a stock against a benchmark index.
    A higher value indicates the stock is outperforming the benchmark.
    """
    s = df.get(f'Close_{stock_ticker}')
    b = df.get(f'Close_{benchmark_ticker}')
    if s is None or b is None:
        return df
    sr = s.pct_change()
    br = b.pct_change()
    df[f'{stock_ticker}_vs_{benchmark_ticker}_RelStrength'] = _safe_div(sr, (br + EPS))
    return df

def add_interstock_ratios(df, target_ticker, supplier_tickers):
    """
    Calculates the price ratio of a target stock to its suppliers.
    This can indicate potential supply chain health or sentiment shifts.
    """
    for sup in supplier_tickers:
        a = df.get(f'Close_{target_ticker}')
        b = df.get(f'Close_{sup}')
        if a is None or b is None: 
            continue
        df[f'{target_ticker}_vs_{sup}_CloseRatio'] = _safe_div(a, b + EPS)
    return df

def add_volatility_ratios(df, stock_ticker, benchmark_ticker='^GSPC'):
    """
    Calculates the ratio of a stock's volatility (ATR) to the benchmark's volatility.
    A high ratio indicates the stock is more volatile than the market.
    """
    s = df.get(f'{stock_ticker}_ATR14')
    b = df.get(f'{benchmark_ticker}_ATR14')
    if s is None or b is None:
        return df
    df[f'{stock_ticker}_vs_{benchmark_ticker}_ATR_Ratio'] = _safe_div(s, b + EPS)
    return df

def add_volume_ratios(df, stock_ticker, benchmark_ticker='^GSPC'):
    v_s = df.get(f'Volume_{stock_ticker}')
    v_b = df.get(f'Volume_{benchmark_ticker}')
    if v_s is None or v_b is None:
        return df
    df[f'{stock_ticker}_vs_{benchmark_ticker}_VolumeRatio'] = _safe_div(v_s, v_b + EPS)
    return df

def add_volume_volatility_interaction(df, stock_ticker):
    """
    Creates a feature that combines volume and volatility.
    High values can signal high-conviction moves (large volume on volatile days).
    We use the ATR as a measure of volatility.
    """
    v = df.get(f'Volume_{stock_ticker}')
    a = df.get(f'{stock_ticker}_ATR14')
    if v is None or a is None:
        return df
    df[f'{stock_ticker}_Volume_x_ATR'] = v * a
    return df

def add_cross_stock_lagged_correlations(df, target_ticker, source_ticker, window=5):
    """
    Calculates the rolling correlation between the daily returns of two stocks.
    This directly tests the hypothesis that a supplier's movement might predict a retailer's movement.
    """
    tr = df.get(f'Close_{target_ticker}_daily_return')
    sr = df.get(f'Close_{source_ticker}_daily_return')
    if tr is None or sr is None:
        return df
    df[f'{target_ticker}_vs_{source_ticker}_RollingCorr_{window}D'] = (
        tr.rolling(window=window, min_periods=window).corr(sr)
    )
    return df

def add_rolling_mean_convergence(df, tickers, window=20):
    """
    Calculates the Rolling Mean Convergence/Divergence for each ticker.
    This is the ratio of the Close price to its rolling mean.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        tickers (list): A list of ticker symbols (e.g., ['WMT', 'KO']).
        window (int): The window size for the rolling mean.
        
    Returns:
        pd.DataFrame: The DataFrame with new features added.
    """
    for ticker in tickers:
        close_col = f'Close_{ticker}'
        rolling_mean_col = f'{ticker}_RollingMean_{window}'
        convergence_col = f'{ticker}_RollingMean_Convergence_{window}'
        
        # Calculate the rolling mean
        df[rolling_mean_col] = df[close_col].rolling(window=window).mean()
        
        # Calculate the convergence ratio. Add a small epsilon to avoid division by zero.
        df[convergence_col] = df[close_col] / (df[rolling_mean_col] + 1e-6)
        
        # Drop the intermediate rolling mean column
        df.drop(columns=[rolling_mean_col], inplace=True)
        
    return df

def add_intermarket_spread(df, ticker1, ticker2):
    """
    Calculates the spread between two tickers based on their Close prices.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        ticker1 (str): The first ticker symbol.
        ticker2 (str): The second ticker symbol.
        
    Returns:
        pd.DataFrame: The DataFrame with the new spread feature.
    """
    close_col1 = f'Close_{ticker1}'
    close_col2 = f'Close_{ticker2}'
    spread_col = f'{ticker1}_vs_{ticker2}_Spread'
    
    # Calculate the simple spread
    df[spread_col] = df[close_col1] - df[close_col2]
    
    return df

def apply_kalman_filter_with_lag(data_df, target_tickers, lags):
    """
    Forward-only Kalman filter (no smoother), then create lagged features.
    """
    df_copy = data_df.copy()

    for ticker in target_tickers:
        close_col = f'Close_{ticker}'
        if close_col not in df_copy.columns:
            continue

        s = df_copy[close_col].astype(float)
        first_valid = s.first_valid_index()
        if first_valid is None:
            continue

        start_pos = s.index.get_loc(first_valid)
        meas_valid = s.iloc[start_pos:].values.reshape(-1, 1)

        # filter only on valid window
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=float(meas_valid[0, 0]),
            initial_state_covariance=1.0,
            observation_covariance=1.0,
            transition_covariance=0.01
        )
        filtered_valid, _ = kf.filter(meas_valid)
        filtered_series = pd.Series(index=s.index, dtype=float)
        filtered_series.iloc[start_pos:] = filtered_valid.flatten()

        for lag in lags:
            df_copy[f'Kalman_Filtered_Close_{ticker}_lag_{lag}'] = filtered_series.shift(lag)

    return df_copy

def calculate_roc(df, ticker, window=14):
    """
    Calculates the Rate of Change (ROC) for a given ticker.
    Formula: ROC = [(Current Close - Close n periods ago) / (Close n periods ago)] * 100
    """
    close_col = f'Close_{ticker}'
    roc_col = f'{ticker}_ROC_{window}'
    
    # Calculate the percentage change from n periods ago
    df[roc_col] = df[close_col].pct_change(periods=window) * 100
    
    return df

def calculate_mfi(df, ticker, window=14):
    """
    Calculates the Money Flow Index (MFI) for a given ticker.
    MFI combines price and volume to measure buying and selling pressure.
    """
    H,L,C,V = f'High_{ticker}', f'Low_{ticker}', f'Close_{ticker}', f'Volume_{ticker}'
    col = f'{ticker}_MFI_{window}'
    if not _has_cols(df, [H,L,C]) or not _has_volume(df, ticker):
        return df
    tp  = (df[H] + df[L] + df[C]) / 3.0
    mf  = tp * df[V]
    pos = mf.where(tp > tp.shift(1), 0)
    neg = mf.where(tp < tp.shift(1), 0)
    pos_sum = pos.rolling(window, min_periods=window).sum()
    neg_sum = neg.rolling(window, min_periods=window).sum().replace(0, np.nan)
    ratio = _safe_div(pos_sum, neg_sum)
    df[col] = 100 - (100 / (1 + ratio))
    return df

def calculate_cmf(df, ticker, window=21):
    """
    Calculates the Chaikin Money Flow (CMF) for a given ticker.
    CMF measures the amount of money flow over a period.
    """
    H,L,C,V = f'High_{ticker}', f'Low_{ticker}', f'Close_{ticker}', f'Volume_{ticker}'
    col = f'{ticker}_CMF_{window}'
    if not _has_cols(df, [H,L,C]) or not _has_volume(df, ticker):
        return df
    range_ = (df[H] - df[L]).replace(0, np.nan)
    mfm = _safe_div((df[C] - df[L]) - (df[H] - df[C]), range_)
    mfv = mfm * df[V]
    num = mfv.rolling(window, min_periods=window).sum()
    den = df[V].rolling(window, min_periods=window).sum().replace(0, np.nan)
    df[col] = _safe_div(num, den)
    return df

def _first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def add_feature_interactions(df, prefix):
    rsi   = _first_existing(df, [f'{prefix}_RSI14', f'{prefix}_rsi_14'])
    macdl = _first_existing(df, [f'{prefix}_MACD_Line', f'{prefix}_MACD_line'])
    volr  = _first_existing(df, [f'{prefix}_Volume_MA_Ratio'])
    upper = _first_existing(df, [f'{prefix}_BB_Upper20', f'{prefix}_Bollinger_Upper_20'])
    lower = _first_existing(df, [f'{prefix}_BB_Lower20', f'{prefix}_Bollinger_Lower_20'])
    close = f'Close_{prefix}'

    if rsi and volr:
        df[f'{prefix}_RSI_Vol_Interaction'] = df[rsi] * df[volr]
    if macdl and close in df.columns:
        df[f'{prefix}_MACD_Close_Ratio'] = _safe_div(df[macdl], df[close] + EPS)
    if upper and lower and rsi and close in df.columns:
        bb_ratio = _safe_div(df[close] - df[lower], (df[upper] - df[lower] + EPS))
        df[f'{prefix}_BB_RSI_Interaction'] = bb_ratio * df[rsi]
    if macdl and volr:
        df[f'{prefix}_MACD_Vol_Interaction'] = df[macdl] * df[volr]

    tr, vol = f'{prefix}_True_Range', f'Volume_{prefix}'
    if tr in df.columns and vol in df.columns:
        df[f'{prefix}_TrueRange_Vol_Interaction'] = df[tr] * df[vol]
    return df

def add_event_time_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Feature set using ONLY that ticker's own columns (no leakage).
    Expects Close_<prefix> (and optionally Open_<prefix>).
    """
    close = df[f"Close_{prefix}"].astype(float)
    feat = pd.DataFrame(index=df.index)

    ret1 = close.pct_change()
    feat[f"{prefix}_ret_1"]   = ret1
    feat[f"{prefix}_ret_5"]   = close.pct_change(5)
    feat[f"{prefix}_ret_20"]  = close.pct_change(20)

    vol20 = ret1.rolling(20).std()
    vol60 = ret1.rolling(60).std()
    feat[f"{prefix}_vol_20"]    = vol20
    feat[f"{prefix}_vol_60"]    = vol60
    feat[f"{prefix}_vol_ratio"] = vol20 / (vol60 + 1e-9)

    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    std20 = ret1.rolling(20).std()
    feat[f"{prefix}_dist_ma20"] = (close / ma20 - 1)
    feat[f"{prefix}_dist_ma50"] = (close / ma50 - 1)
    feat[f"{prefix}_bb_pos_20"] = (close - ma20) / (2*std20*close.shift(1) + 1e-9)

    # Oscillators / momentum
    roll_min = close.rolling(14).min()
    roll_max = close.rolling(14).max()
    feat[f"{prefix}_stoch_k14"] = (close - roll_min) / (roll_max - roll_min + 1e-9)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal= macd.ewm(span=9, adjust=False).mean()
    feat[f"{prefix}_macd_hist"] = macd - signal

    feat[f"{prefix}_rsi_14"] = (
        ret1.clip(lower=0).rolling(14).mean() /
        (ret1.abs().rolling(14).mean() + 1e-9)
    )

    # Shape of recent returns
    r20 = ret1.rolling(20)
    feat[f"{prefix}_skew_20"] = r20.apply(lambda x: pd.Series(x).skew(), raw=False)
    feat[f"{prefix}_kurt_20"] = r20.apply(lambda x: pd.Series(x).kurt(), raw=False)

    # Overnight gap (if Open available)
    open_col = f"Open_{prefix}"
    if open_col in df.columns:
        o = df[open_col].astype(float)
        feat[f"{prefix}_overnight_ret"] = o.pct_change() - ret1

    # Merge back
    return df.join(feat)

def add_rolling_beta(df: pd.DataFrame, prefix: str, bench_prefix: str, window=60):
    """
    Rolling OLS beta of ticker returns vs benchmark returns.
    Uses Close_<prefix> and Close_<bench_prefix>.
    """
    p = df[f"Close_{prefix}"].astype(float).pct_change()
    b = df[f"Close_{bench_prefix}"].astype(float).pct_change()
    cov = p.rolling(window).cov(b)
    var = b.rolling(window).var()
    beta = cov / (var + 1e-12)
    df[f"{prefix}_beta_{bench_prefix}_{window}"] = beta
    return df

def sector_and_market_relatives(df, target_ticker, sector_etf=None, market_bench="^GSPC", beta_window=60):
    """
    Add relative-strength & rolling beta vs the sector ETF (if provided/available)
    and the market benchmark (default ^GSPC). Falls back to XLP if no sector_etf given
    but XLP is present.
    """
    tgt_close = f'Close_{target_ticker}'
    if tgt_close not in df.columns:
        return df  # nothing to do

    # Sector leg
    if sector_etf and f'Close_{sector_etf}' in df.columns:
        df = add_relative_strength(df, target_ticker, sector_etf)
        df = add_rolling_beta(df, target_ticker, sector_etf, window=beta_window)
    elif 'Close_XLP' in df.columns:  # sensible fallback
        df = add_relative_strength(df, target_ticker, 'XLP')
        df = add_rolling_beta(df, target_ticker, 'XLP', window=beta_window)

    # Market leg
    if market_bench and f'Close_{market_bench}' in df.columns:
        df = add_relative_strength(df, target_ticker, market_bench)
        df = add_rolling_beta(df, target_ticker, market_bench, window=beta_window)
    elif 'Close_^GSPC' in df.columns:  # fallback to S&P if string differs
        df = add_relative_strength(df, target_ticker, '^GSPC')
        df = add_rolling_beta(df, target_ticker, '^GSPC', window=beta_window)

    return df


def add_volume_shock(df, prefix, long_win=60, short_win=5):
    v = df.get(f'Volume_{prefix}')
    if v is None: 
        return df
    mu = v.rolling(long_win, min_periods=long_win).mean()
    sd = v.rolling(long_win, min_periods=long_win).std()
    df[f'{prefix}_vol_z_{long_win}'] = _safe_div(v - mu, sd + EPS)
    df[f'{prefix}_vol_ma_ratio_{short_win}_{long_win}'] = _safe_div(v.rolling(short_win).mean(), mu + EPS)
    return df

def add_gap_features(df, prefix):
    o, c = df.get(f'Open_{prefix}'), df.get(f'Close_{prefix}')
    if o is None or c is None: 
        return df
    prev_c = c.shift(1)
    gap = _safe_div(o - prev_c, prev_c)
    df[f'{prefix}_gap_overnight'] = gap
    atr = df.get(f'{prefix}_ATR14')
    if atr is not None:
        df[f'{prefix}_gap_vs_ATR'] = gap.abs() / _safe_div(atr, prev_c)
    return df

def add_drawdown_features(df, prefix, window=252):
    c = df[f'Close_{prefix}'].astype(float)
    roll_max = c.rolling(window).max()
    df[f'{prefix}_drawdown_{window}'] = c/(roll_max + 1e-9) - 1.0
    return df

def add_streak_features(df, prefix):
    r = df[f'Close_{prefix}'].pct_change()
    s = np.sign(r).fillna(0)

    def _streak(x):
        # length of current consecutive 1’s (or -1’s) ending at t
        grp = (x != x.shift()).cumsum()
        return x.groupby(grp).cumcount() + 1

    up = (s > 0).astype(int)
    down = (s < 0).astype(int)
    df[f'{prefix}_up_streak'] = _streak(up) * up
    df[f'{prefix}_down_streak'] = _streak(down) * down
    return df

def add_range_volatility(df, prefix, window=20):
    h, l = df.get(f'High_{prefix}'), df.get(f'Low_{prefix}')
    if h is None or l is None: return df
    parkinson_var = (1.0/(4*np.log(2))) * (np.log((h / l).clip(lower=1e-9)) ** 2)
    df[f'{prefix}_parkinson_{window}'] = parkinson_var.rolling(window).mean()
    return df

def add_rolling_autocorr(df, prefix, window=20, lag=1):
    r = df[f'Close_{prefix}'].pct_change()
    df[f'{prefix}_autocorr_l{lag}_w{window}'] = r.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag), raw=False
    )
    return df

def add_excess_return(df, stock, bench='^GSPC'):
    s = df[f'Close_{stock}'].pct_change()
    b = df[f'Close_{bench}'].pct_change()
    df[f'{stock}_excess_ret_{bench}'] = s - b
    return df

def add_rolling_corr(df, stock, peer, window=60):
    s = df[f'Close_{stock}'].pct_change()
    p = df[f'Close_{peer}'].pct_change()
    df[f'{stock}_corr_{peer}_{window}'] = s.rolling(window).corr(p)
    return df

def add_equity_liquidity_proxies(df: pd.DataFrame, prefix: str, window: int = 20) -> pd.DataFrame:
    """
    Simple equity liquidity proxies:
    - Amihud-like illiquidity: |ret|/volume (needs volume)
    - Turnover: volume / rolling sum(volume)
    """
    out = df.copy()
    c = f'Close_{prefix}'
    v = f'Volume_{prefix}'
    if c in out.columns and v in out.columns:
        ret = out[c].pct_change().abs()
        vol = out[v].replace(0, np.nan)
        out[f'{prefix}_amihud_{window}'] = (ret / vol).rolling(window, min_periods=window).mean()
        out[f'{prefix}_turnover_{window}'] = vol / vol.rolling(window, min_periods=window).sum()
        # shift to t-1
        out[f'{prefix}_amihud_{window}'] = out[f'{prefix}_amihud_{window}'].shift(1)
        out[f'{prefix}_turnover_{window}'] = out[f'{prefix}_turnover_{window}'].shift(1)
    return out

def build_stock_features_orchestrator(
    tickers, 
    target_ticker, 
    supplier_tickers, 
    benchmark_ticker, 
    start_date=None, 
    end_date=None, 
    output_raw_csv=None, 
    output_engineered_csv=None,
    kalman_lags=None,                 # e.g., [1,5,10] or None
    kalman_targets="all",             # "all" or list of tickers
    sector_etf=None,                  # pass "XLP"/"XLV"/... when you have it
    dropna_frac=0.90                  # % of non-NaN required per row
):
    print("\n--- Starting Stock Feature Pipeline ---")

    df = fetch_multiple_stock_data(
        tickers, start_date=start_date, end_date=end_date, output_filename=output_raw_csv
    )
    if df.empty:
        print("Pipeline stopped: No data fetched or data is empty after initial processing.")
        return pd.DataFrame()

    stock_prefixes = sorted({col.split('_')[1] for col in df.columns if '_' in col})
    print(f"\nDiscovered stock prefixes: {stock_prefixes}")

    # 2) Per-ticker features
    for prefix in stock_prefixes:
        print(f"\nProcessing features for stock prefix: {prefix}")
        df = add_event_time_features(df, prefix)
        df = add_price_range_features(df, prefix)
        df = calculate_true_range(df, prefix)
        df = calculate_atr(df, prefix, window=14)
        df = add_volume_features(df, prefix, window=20)
        df = calculate_obv(df, prefix)
        df = calculate_rsi(df, prefix, window=14)
        df = calculate_macd(df, prefix, fast_period=12, slow_period=26, signal_period=9)
        df = add_moving_averages(df, prefix, window_sizes=[10, 20, 50], ma_type='SMA')
        df = add_moving_averages(df, prefix, window_sizes=[12, 26], ma_type='EMA')
        df = add_bollinger_bands(df, prefix, window=20, num_std=2)
        df = calculate_stochastic_oscillator(df, prefix, k_period=14, d_period=3)
        df = calculate_adx(df, prefix, window=14)
        df = add_rolling_mean_convergence(df, [prefix], window=50)
        df = calculate_roc(df, prefix, window=12)
        df = calculate_mfi(df, prefix, window=14)
        df = calculate_cmf(df, prefix, window=21)
        df = add_feature_interactions(df, prefix)
        df = add_volume_shock(df, prefix)
        df = add_gap_features(df, prefix)
        df = add_drawdown_features(df, prefix, window=252)
        df = add_streak_features(df, prefix)
        df = add_range_volatility(df, prefix, window=20)
        df = add_rolling_autocorr(df, prefix, window=20, lag=1)

    # 3) Cross-ticker features
    print("\nApplying general features...")
    close_cols_for_returns = [c for c in df.columns if c.startswith('Close_')]
    df = calculate_daily_returns(df, close_cols_for_returns)

    if benchmark_ticker in tickers:
        df = add_relative_strength(df, target_ticker, benchmark_ticker)
        df = add_volatility_ratios(df, target_ticker, benchmark_ticker)
        df = add_volume_ratios(df, target_ticker, benchmark_ticker)

    df = add_interstock_ratios(df, target_ticker, supplier_tickers)
    df = add_volume_volatility_interaction(df, target_ticker)
    for supplier in supplier_tickers:
        df = add_cross_stock_lagged_correlations(df, target_ticker, supplier)

    # use sector ETF when provided (so this works for non-staples, too)
    df = sector_and_market_relatives(df, target_ticker, sector_etf=sector_etf, market_bench="^GSPC")

    if f'Close_{target_ticker}' in df.columns and 'Close_KO' in df.columns:
        df = add_intermarket_spread(df, target_ticker, 'KO')

    # 4) OPTIONAL Kalman (do not smooth; we only filter)
    if kalman_lags:
        if kalman_targets == "all":
            kt = stock_prefixes
        elif isinstance(kalman_targets, (list, tuple, set)):
            kt = list(kalman_targets)
        else:
            kt = [target_ticker]
        df = apply_kalman_filter_with_lag(df, kt, kalman_lags)

    # 5) Row pruning
    initial_rows = len(df)
    min_non_na = int(dropna_frac * df.shape[1])
    df = df.dropna(thresh=min_non_na)
    print(f"\nDropped {initial_rows - len(df)} rows due to NaN values after feature engineering.")

    if output_engineered_csv:
        df.to_csv(output_engineered_csv)
        print(f"Final engineered data saved to {output_engineered_csv}")

    return df


def base_sector_and_market_relatives(df, target_ticker, sector_etf=None, market_bench="^GSPC"):
    """Rel-strength & rolling beta vs sector ETF and market (if present)."""
    if sector_etf and f'Close_{sector_etf}' in df.columns and f'Close_{target_ticker}' in df.columns:
        df = add_relative_strength(df, target_ticker, sector_etf)
        df = add_rolling_beta(df, target_ticker, sector_etf, window=60)
    if market_bench and f'Close_{market_bench}' in df.columns and f'Close_{target_ticker}' in df.columns:
        df = add_relative_strength(df, target_ticker, market_bench)
        df = add_rolling_beta(df, target_ticker, market_bench, window=60)
    return df


def build_sector_base_features(
    tickers,
    start_date=None,
    end_date=None,
    kalman_lags=None,
    dropna_frac=0.90,
    output_path=None
):
    """Fetch + compute all per-ticker features ONCE for a sector (no target logic)."""
    print("\n--- Building sector BASE (target-agnostic) ---")
    df = fetch_multiple_stock_data(tickers, start_date=start_date, end_date=end_date)
    if df.empty:
        print("No data fetched."); return pd.DataFrame()

    prefixes = sorted({col.split('_')[1] for col in df.columns if '_' in col})
    print(f"Discovered prefixes: {prefixes}")

    # per-ticker features (heavy)
    for p in prefixes:
        df = add_event_time_features(df, p)
        df = add_price_range_features(df, p)
        df = calculate_true_range(df, p)
        df = calculate_atr(df, p, window=14)
        df = add_volume_features(df, p, window=20)
        df = calculate_obv(df, p)
        df = calculate_rsi(df, p, window=14)
        df = calculate_macd(df, p, fast_period=12, slow_period=26, signal_period=9)
        df = add_moving_averages(df, p, window_sizes=[10,20,50], ma_type='SMA')
        df = add_moving_averages(df, p, window_sizes=[12,26], ma_type='EMA')
        df = add_bollinger_bands(df, p, window=20, num_std=2)
        df = calculate_stochastic_oscillator(df, p, k_period=14, d_period=3)
        df = calculate_adx(df, p, window=14)
        df = add_rolling_mean_convergence(df, [p], window=50)
        df = calculate_roc(df, p, window=12)
        df = calculate_mfi(df, p, window=14)
        df = calculate_cmf(df, p, window=21)
        df = add_feature_interactions(df, p)
        df = add_volume_shock(df, p)
        df = add_gap_features(df, p)
        df = add_drawdown_features(df, p, window=252)
        df = add_streak_features(df, p)
        df = add_range_volatility(df, p, window=20)
        df = add_rolling_autocorr(df, p, window=20, lag=1)
        df = add_equity_liquidity_proxies(df, p, window=20)

        leak_sensitive = [f'{p}_MACD_Line', f'{p}_MACD_Signal', f'{p}_MACD_Hist',
                  f'{p}_RSI14', f'{p}_parkinson_20',
                  f'{p}_BB_Middle20', f'{p}_BB_Upper20', f'{p}_BB_Lower20']
        df = shift_to_t_minus_1(df, [c for c in leak_sensitive if c in df.columns])

    # daily returns for all Close_*
    close_cols = [c for c in df.columns if c.startswith("Close_")]
    df = calculate_daily_returns(df, close_cols)

    # optional Kalman for all tickers (cheap once)
    if kalman_lags:
        df = apply_kalman_filter_with_lag(df, target_tickers=prefixes, lags=kalman_lags)

    # prune rows
    min_non_na = int(dropna_frac * df.shape[1])
    before = len(df)
    df = df.dropna(thresh=min_non_na)
    print(f"Dropped {before - len(df)} rows (base pruning @ {dropna_frac:.0%}).")

    if output_path:
        # parquet recommended for speed + schema
        if str(output_path).lower().endswith(".parquet"):
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=True)
        print(f"Saved BASE → {output_path}")

    return df


def make_target_view(
    base_df, target_ticker, supplier_tickers, benchmark_ticker="^GSPC", sector_etf=None
):
    """Add only the target-specific, cross-ticker features on top of a sector BASE."""
    df = base_df.copy()

    # target vs market + sector
    df = add_relative_strength(df, target_ticker, benchmark_ticker)
    df = add_volatility_ratios(df, target_ticker, benchmark_ticker)
    df = add_volume_ratios(df, target_ticker, benchmark_ticker)
    df = add_volume_volatility_interaction(df, target_ticker)
    df = base_sector_and_market_relatives(df, target_ticker, sector_etf=sector_etf, market_bench=benchmark_ticker)

    # target vs suppliers (light)
    df = add_interstock_ratios(df, target_ticker, supplier_tickers)
    for sup in supplier_tickers:
        df = add_cross_stock_lagged_correlations(df, target_ticker, sup)

    # optional fixed intermarket spread you had
    if f'Close_{target_ticker}' in df.columns and 'Close_KO' in df.columns:
        df = add_intermarket_spread(df, target_ticker, 'KO')

    return df

