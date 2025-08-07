import numpy as np
import pandas as pd
from src.stock_data import fetch_multiple_stock_data

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
    high_col = f'High_{ticker_prefix}'
    low_col = f'Low_{ticker_prefix}'
    open_col = f'Open_{ticker_prefix}'
    close_col = f'Close_{ticker_prefix}'

    if all(col in df.columns for col in [high_col, low_col, open_col, close_col]):
        df[f'{ticker_prefix}_HighLow_Range'] = df[high_col] - df[low_col]
        df[f'{ticker_prefix}_OpenClose_Range'] = df[close_col] - df[open_col]
        
        range_denom = df[high_col] - df[low_col]
        # Use a temporary series, then fillna, then assign
        close_to_range_ratio = (df[close_col] - df[low_col]) / range_denom
        df[f'{ticker_prefix}_Close_to_Range_Ratio'] = close_to_range_ratio.fillna(0.5)
    else:
        print(f"Warning: Missing HLOC columns for {ticker_prefix} to calculate range features.")
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
    high_col = f'High_{ticker_prefix}'
    low_col = f'Low_{ticker_prefix}'
    close_col = f'Close_{ticker_prefix}'

    if all(col in df.columns for col in [high_col, low_col, close_col]):
        prev_close = df[close_col].shift(1)
        tr1 = df[high_col] - df[low_col]
        tr2 = (df[high_col] - prev_close).abs()
        tr3 = (df[low_col] - prev_close).abs()
        df[f'{ticker_prefix}_True_Range'] = np.maximum.reduce([tr1, tr2, tr3])
    else:
        print(f"Warning: Missing OHLC columns for {ticker_prefix} to calculate True Range.")
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
    print(f"  - Calculating ATR for {ticker_prefix} (window={window})...")

    true_range_col = f'{ticker_prefix}_True_Range'
    atr_col_name = f'{ticker_prefix}_ATR{window}'

    if true_range_col in df.columns:
        # ATR is typically an Exponential Moving Average (EMA) of the True Range.
        # The .ewm() method in pandas is used for Exponential Weighted Moving Average.
        # span parameter is equivalent to the window for EMA.
        df[atr_col_name] = df[true_range_col].ewm(span=window, adjust=False, min_periods=window).mean()
        # min_periods=window ensures that ATR is only calculated once enough data points are available,
        # otherwise, the initial values would be based on fewer data points.
    else:
        print(f"    Warning: '{true_range_col}' column not found in DataFrame for {ticker_prefix}.")
        print(f"    Please ensure 'calculate_true_range' is run BEFORE 'calculate_atr'. Skipping ATR calculation.")
        # Fill with NaN if True Range is missing, to maintain column presence
        df[atr_col_name] = np.nan
    
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
    volume_col = f'Volume_{ticker_prefix}'

    if volume_col in df.columns:
        df[f'{ticker_prefix}_Volume_Daily_Change'] = df[volume_col].pct_change() * 100
        df[f'{ticker_prefix}_Volume_MA_{window}D'] = df[volume_col].rolling(window=window, min_periods=1).mean()
        
        # Calculate ratio first, then replace and assign in one step
        volume_ma_ratio = df[volume_col] / df[f'{ticker_prefix}_Volume_MA_{window}D']
        df[f'{ticker_prefix}_Volume_MA_Ratio'] = volume_ma_ratio.replace([np.inf, -np.inf], np.nan)
    else:
        print(f"Warning: Volume column '{volume_col}' not found for {ticker_prefix}. Skipping volume features.")
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
    close_col = f'Close_{ticker_prefix}'
    volume_col = f'Volume_{ticker_prefix}'

    if all(col in df.columns for col in [close_col, volume_col]):
        price_direction = np.sign(df[close_col].diff())
        obv = (price_direction * df[volume_col]).cumsum()
        
        # Correctly set the first value using .loc
        # This avoids the ChainedAssignmentError and SettingWithCopyWarning
        obv.loc[df.index[0]] = 0 
        
        df[f'{ticker_prefix}_OBV'] = obv
    else:
        print(f"Warning: Missing Close or Volume column for {ticker_prefix}. Skipping OBV calculation.")
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

    print(f"  - Calculating RSI for {ticker_prefix} (window={window})...")

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
    close_col = f'Close_{ticker_prefix}'

    if close_col in df.columns:
        middle_band = df[close_col].rolling(window=window).mean()
        std_dev = df[close_col].rolling(window=window).std()

        df[f'{ticker_prefix}_BB_Middle{window}'] = middle_band
        df[f'{ticker_prefix}_BB_Upper{window}'] = middle_band + (std_dev * num_std)
        df[f'{ticker_prefix}_BB_Lower{window}'] = middle_band - (std_dev * num_std)

        # Bandwidth: Indicates volatility
        df[f'{ticker_prefix}_BB_Bandwidth{window}'] = (df[f'{ticker_prefix}_BB_Upper{window}'] - df[f'{ticker_prefix}_BB_Lower{window}']) / middle_band

        # %B: Indicates where the price is relative to the bands
        # Avoid division by zero
        denom = (df[f'{ticker_prefix}_BB_Upper{window}'] - df[f'{ticker_prefix}_BB_Lower{window}'])
        
        # CORRECTED: Calculate the series and then replace inf values before assigning
        pct_b_series = (df[close_col] - df[f'{ticker_prefix}_BB_Lower{window}']) / denom
        df[f'{ticker_prefix}_BB_PctB{window}'] = pct_b_series.replace([np.inf, -np.inf], np.nan)
        
    else:
        print(f"Warning: Close column for {ticker_prefix} not found for Bollinger Bands calculation.")
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
    high_col = f'High_{ticker_prefix}'
    low_col = f'Low_{ticker_prefix}'
    close_col = f'Close_{ticker_prefix}'

    if all(col in df.columns for col in [high_col, low_col, close_col]):
        # Calculate %K
        lowest_low = df[low_col].rolling(window=k_period).min()
        highest_high = df[high_col].rolling(window=k_period).max()
        # Avoid division by zero
        denom = (highest_high - lowest_low)
        df[f'{ticker_prefix}_Stoch_K_{k_period}'] = ((df[close_col] - lowest_low) / denom) * 100
        df[f'{ticker_prefix}_Stoch_K_{k_period}'].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero

        # Calculate %D (SMA of %K)
        df[f'{ticker_prefix}_Stoch_D_{k_period}_{d_period}'] = df[f'{ticker_prefix}_Stoch_K_{k_period}'].rolling(window=d_period).mean()
    else:
        print(f"Warning: Missing HLC columns for {ticker_prefix} to calculate Stochastic Oscillator.")
    
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
    high_col = f'High_{ticker_prefix}'
    low_col = f'Low_{ticker_prefix}'
    close_col = f'Close_{ticker_prefix}'

    if all(col in df.columns for col in [high_col, low_col, close_col]):
        # Calculate Directional Movement (DM)
        df[f'{ticker_prefix}_PlusDM'] = df[high_col].diff().where(
            (df[high_col].diff() > df[low_col].diff().abs()) & (df[high_col].diff() > 0), 0
        )
        df[f'{ticker_prefix}_MinusDM'] = df[low_col].diff().abs().where(
            (df[low_col].diff().abs() > df[high_col].diff()) & (df[low_col].diff() < 0), 0
        )
        # Handle cases where current high is lower than prev high and current low is higher than prev low
        # We need to ensure that the +DM is 0 if it's not an "up" move, and -DM is 0 if it's not a "down" move.
        # This is more complex than simple diffs. A common way is to compare current high/low to prev high/low.
        # This is a simplified version. A more robust ADX calculation often involves EMA smoothing of DM.

        # Calculate True Range (TR) - assumes calculate_true_range was run before this
        df = calculate_true_range(df, ticker_prefix) # Ensure TR exists

        # Smooth DM and TR
        # Using 14-period Wilder's smoothing (equivalent to EMA with adjust=False for (window*2)-1)
        df[f'{ticker_prefix}_PlusDI_{window}'] = (
            df[f'{ticker_prefix}_PlusDM'].ewm(span=window, adjust=False).mean() /
            df[f'{ticker_prefix}_True_Range'].ewm(span=window, adjust=False).mean()
        ) * 100

        df[f'{ticker_prefix}_MinusDI_{window}'] = (
            df[f'{ticker_prefix}_MinusDM'].ewm(span=window, adjust=False).mean() /
            df[f'{ticker_prefix}_True_Range'].ewm(span=window, adjust=False).mean()
        ) * 100

        # Calculate DX
        df[f'{ticker_prefix}_DI_Diff'] = abs(df[f'{ticker_prefix}_PlusDI_{window}'] - df[f'{ticker_prefix}_MinusDI_{window}'])
        df[f'{ticker_prefix}_DI_Sum'] = df[f'{ticker_prefix}_PlusDI_{window}'] + df[f'{ticker_prefix}_MinusDI_{window}']
        # Avoid division by zero
        denom_dx = df[f'{ticker_prefix}_DI_Sum']
        df[f'{ticker_prefix}_DX_{window}'] = (df[f'{ticker_prefix}_DI_Diff'] / denom_dx) * 100
        df[f'{ticker_prefix}_DX_{window}'].replace([np.inf, -np.inf], np.nan, inplace=True) # Replace inf with NaN
        df[f'{ticker_prefix}_DX_{window}'].fillna(0, inplace=True) # If sum is 0, DX is 0

        # Calculate ADX (smoothed DX)
        df[f'{ticker_prefix}_ADX_{window}'] = df[f'{ticker_prefix}_DX_{window}'].ewm(span=window, adjust=False).mean()

        # Drop intermediate columns if desired (e.g., PlusDM, MinusDM, DI_Diff, DI_Sum)
        df.drop(columns=[f'{ticker_prefix}_PlusDM', f'{ticker_prefix}_MinusDM',
                         f'{ticker_prefix}_DI_Diff', f'{ticker_prefix}_DI_Sum'],
                errors='ignore', inplace=True)
    else:
        print(f"Warning: Missing OHLC/Close column for {ticker_prefix} to calculate ADX.")
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

    print(f"  - Calculating MACD for {ticker_prefix} (fast={fast_period}, slow={slow_period}, signal={signal_period})...")

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
    # Calculate daily returns for both the stock and the benchmark
    stock_returns = df[f'Close_{stock_ticker}'].pct_change()
    benchmark_returns = df[f'Close_{benchmark_ticker}'].pct_change()
    
    # Calculate relative strength as the ratio of their returns
    # Add a small value to the denominator to avoid division by zero
    df[f'{stock_ticker}_vs_{benchmark_ticker}_RelStrength'] = stock_returns / (benchmark_returns + 1e-9)
    return df

def add_interstock_ratios(df, target_ticker, supplier_tickers):
    """
    Calculates the price ratio of a target stock to its suppliers.
    This can indicate potential supply chain health or sentiment shifts.
    """
    for supplier_ticker in supplier_tickers:
        df[f'{target_ticker}_vs_{supplier_ticker}_CloseRatio'] = df[f'Close_{target_ticker}'] / df[f'Close_{supplier_ticker}']
    return df

def add_volatility_ratios(df, stock_ticker, benchmark_ticker='^GSPC'):
    """
    Calculates the ratio of a stock's volatility (ATR) to the benchmark's volatility.
    A high ratio indicates the stock is more volatile than the market.
    """
    # Calculate volatility ratio as the ratio of their ATRs
    df[f'{stock_ticker}_vs_{benchmark_ticker}_ATR_Ratio'] = df[f'{stock_ticker}_ATR14'] / df[f'{benchmark_ticker}_ATR14']
    return df

def add_volume_ratios(df, stock_ticker, benchmark_ticker='^GSPC'):
    """
    Calculates the ratio of a stock's volume to a benchmark's volume.
    High values can indicate unusual trading interest in a specific stock.
    """
    # Assuming the volume column names are like 'Volume_WMT' and 'Volume_^GSPC'
    df[f'{stock_ticker}_vs_{benchmark_ticker}_VolumeRatio'] = df[f'Volume_{stock_ticker}'] / df[f'Volume_{benchmark_ticker}']
    return df

def add_volume_volatility_interaction(df, stock_ticker):
    """
    Creates a feature that combines volume and volatility.
    High values can signal high-conviction moves (large volume on volatile days).
    We use the ATR as a measure of volatility.
    """
    df[f'{stock_ticker}_Volume_x_ATR'] = df[f'Volume_{stock_ticker}'] * df[f'{stock_ticker}_ATR14']
    return df

def add_cross_stock_lagged_correlations(df, target_ticker, source_ticker, window=5):
    """
    Calculates the rolling correlation between the daily returns of two stocks.
    This directly tests the hypothesis that a supplier's movement might predict a retailer's movement.
    """
    # Calculate daily returns for both stocks
    target_returns = df[f'Close_{target_ticker}_daily_return']
    source_returns = df[f'Close_{source_ticker}_daily_return']

    # Calculate a rolling correlation between the two returns
    df[f'{target_ticker}_vs_{source_ticker}_RollingCorr_{window}D'] = target_returns.rolling(window=window).corr(source_returns)

    return df

def prepare_data_for_ml(tickers, start_date, end_date, output_raw_csv=None, output_engineered_csv="engineered_stock_data.csv"):
    """
    Orchestrates the data fetching and feature engineering pipeline.

    Parameters:
    tickers (list): List of stock ticker symbols.
    start_date (str): Start date for data fetching.
    end_date (str): End date for data fetching.
    output_raw_csv (str, optional): Filename to save the raw combined data.
    output_engineered_csv (str, optional): Filename to save the final engineered data.

    Returns:
    pd.DataFrame: A DataFrame with all engineered features and target variables.
    """
    print("\n--- Starting Data Preparation Pipeline ---")

    # 1. Fetch raw historical data for multiple stocks
    df = fetch_multiple_stock_data(tickers, start_date, end_date, output_filename=output_raw_csv)
    if df.empty:
        print("Pipeline stopped: No data fetched or data is empty after initial processing.")
        return pd.DataFrame()

    # Get a list of the ticker prefixes for iteration (e.g., 'AAPL', 'MSFT')
    stock_prefixes = sorted(list(set([col.split('_')[1] for col in df.columns if '_' in col])))
    print(f"\nDiscovered stock prefixes: {stock_prefixes}")

    # 2. Apply stock-specific features inside the loop
    for prefix in stock_prefixes:
        print(f"\nProcessing features for stock prefix: {prefix}")
        
        # Price Range Features (needs High, Low, Open, Close)
        df = add_price_range_features(df, prefix)
        # True Range (needs High, Low, Close) - Required for ATR and ADX
        df = calculate_true_range(df, prefix)
        # ATR (needs True Range)
        df = calculate_atr(df, prefix, window=14)
        # Volume Features (needs Volume)
        df = add_volume_features(df, prefix, window=20)
        # OBV (needs Close, Volume)
        df = calculate_obv(df, prefix)
        # RSI (needs Close)
        df = calculate_rsi(df, prefix, window=14)
        # MACD (needs Close)
        df = calculate_macd(df, prefix, fast_period=12, slow_period=26, signal_period=9)
        # Moving Averages (needs Close)
        df = add_moving_averages(df, prefix, window_sizes=[10, 20, 50], ma_type='SMA')
        df = add_moving_averages(df, prefix, window_sizes=[12, 26], ma_type='EMA')
        # Bollinger Bands (needs Close)
        df = add_bollinger_bands(df, prefix, window=20, num_std=2)
        # Stochastic Oscillator (needs High, Low, Close)
        df = calculate_stochastic_oscillator(df, prefix, k_period=14, d_period=3)
        # ADX (needs High, Low, Close, and True Range)
        df = calculate_adx(df, prefix, window=14)
    
    # 3. Apply features that depend on all stocks or can be applied once
    print("\nApplying general features and targets...")
    
    # Get a list of all 'Close' columns for daily return calculations
    close_cols_for_returns = [col for col in df.columns if col.startswith('Close_')]
    df = calculate_daily_returns(df, close_cols_for_returns)

    # --- ADDING CUSTOM FEATURES HERE ---
    target_ticker = 'WMT'
    supplier_tickers = ['KO', 'PEP']
    benchmark_ticker = '^GSPC'
    
    if benchmark_ticker in tickers: # Check if benchmark data was fetched
        df = add_relative_strength(df, target_ticker, benchmark_ticker)
        df = add_volatility_ratios(df, target_ticker, benchmark_ticker)
        df = add_volume_ratios(df, target_ticker, benchmark_ticker)
        
    df = add_interstock_ratios(df, target_ticker, supplier_tickers)
    df = add_volume_volatility_interaction(df, target_ticker)
    for supplier in supplier_tickers:
        df = add_cross_stock_lagged_correlations(df, target_ticker, supplier)
    # --- END OF CUSTOM FEATURES ---
    
    # Lagged Features (needs features that are already created)
    features_to_lag = []
    for prefix in stock_prefixes:
        features_to_lag.extend([
            f'Close_{prefix}',
            f'Close_{prefix}_daily_return',
            f'{prefix}_RSI14',
            f'{prefix}_Volume_MA_Ratio'
        ])
    lag_periods = [1, 3, 5]
    df = add_lagged_features(df, features_to_lag, lag_periods)

    # Time-based Features (general)
    #df = add_time_based_features(df)

    # 4. Final Data Cleanup
    initial_rows = len(df)
    df.dropna(inplace=True)
    rows_dropped = initial_rows - len(df)
    print(f"\nDropped {rows_dropped} rows due to NaN values after feature engineering.")

    print("\n--- Data Preparation Complete ---")
    
    # 5. Save the final DataFrame
    df.to_csv(output_engineered_csv)
    print(f"Final engineered data saved to {output_engineered_csv}")
    
    return df

def create_target_variable(df, ticker, window=1, threshold=0):
    """
    Creates a target variable based on the cumulative return over a specified window.
    This function is designed to be called outside the main pipeline for tuning.

    Parameters:
    df (pd.DataFrame): The DataFrame with Close prices.
    ticker (str): The stock ticker for the target.
    window (int): The number of days for the return period (e.g., 1 for next day, 5 for a week).
    threshold (float): The minimum return to be considered "up".

    Returns:
    pd.DataFrame: A DataFrame with the new target and target return columns.
    """
    close_col = f'Close_{ticker}'
    target_return_col = f'{ticker}_target_return_{window}D_{threshold}'
    target_col = f'{ticker}_Target'

    # Calculate cumulative return over the window
    df[target_return_col] = df[close_col].pct_change(periods=window).shift(-window)

    # Create the binary target
    df[target_col] = (df[target_return_col] > threshold).astype(int)

    return df