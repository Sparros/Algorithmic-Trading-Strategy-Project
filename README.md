Feature Functions:

1. calculate_daily_returns(df_input, cols_to_process)
What it does: Computes the day-over-day percentage change in price for specified columns (typically Close prices).

Why it's useful: Returns are a fundamental input for ML models, representing actual price movements and profit/loss. They form the basis for many volatility calculations and are frequently used as the model's target variable.

2. create_next_day_targets(df_input, cols_to_process)
What it does: Shifts the calculated daily returns forward by one day, creating a future return value (e.g., LLOY.L_Next_Day_Return).

Why it's useful: This is the crucial y variable your supervised machine learning model aims to predict. For a trading bot, predicting the next day's movement is the primary objective.

3. add_lagged_features(df_input, cols_to_lag, lag_periods)
What it does: Generates new columns containing the values of existing features (e.g., prices, daily returns, indicators) from n days in the past.

Why it's useful: Provides historical context to the model, enabling it to learn patterns and relationships based on recent past performance, which is vital for time-series predictions.

4. add_price_range_features(df_input, ticker_prefix)
What it does: Calculates features derived from the daily Open, High, Low, and Close prices for a specific stock:

HighLow_Range: The total price movement during the trading day.

OpenClose_Range: The net change from the opening to the closing price.

Close_to_Range_Ratio: Indicates where the closing price falls within the day's high-low range (0 = low, 1 = high).

Why it's useful: Captures intraday volatility and the strength/weakness of buying/selling pressure into the close, offering more granular insight than simple daily returns.

5. calculate_true_range(df_input, ticker_prefix)
What it does: Determines the maximum of three values: (Current High - Current Low), absolute (Current High - Previous Close), and absolute (Current Low - Previous Close).

Why it's useful: This is the core component for measuring actual price movement, including overnight gaps, and serves as the foundation for the Average True Range (ATR) indicator.

6. calculate_atr(df_input, ticker_prefix, window)
What it does: Computes the Average True Range (ATR), which is a smoothed average of the True Range over a specified period.

Why it's useful: Provides a consistent measure of a stock's historical volatility (how much it typically moves). Useful for risk management, setting stop-losses, and identifying periods of high or low price fluctuation.

7. add_volume_features(df_input, ticker_prefix, window)
What it does: Generates features related to trading volume:

Volume_Daily_Change: The day-over-day percentage change in trading volume.

Volume_MA_Ratio: Current volume relative to its rolling average volume.

Why it's useful: Volume often confirms price trends; high volume validates strong price moves. These features assess trading activity and liquidity.

8. calculate_obv(df_input, ticker_prefix)
What it does: Calculates On-Balance Volume (OBV), a cumulative indicator that adds volume on "up" days and subtracts it on "down" days.

Why it's useful: Helps confirm price trends. Divergences between price and OBV (e.g., price rising but OBV falling) can signal a weakening trend or potential reversal.

9. calculate_rsi(df_input, ticker_prefix, window)
What it does: Computes the Relative Strength Index (RSI), a momentum oscillator that measures the speed and change of price movements.

Why it's useful: Identifies potential overbought (typically >70) or oversold (typically <30) conditions, suggesting potential price reversals or pullbacks. Also indicates momentum strength.

10. calculate_macd(df_input, ticker_prefix, fast_period, slow_period, signal_period)
What it does: Calculates the Moving Average Convergence Divergence (MACD) indicator, composed of the MACD Line, Signal Line, and Histogram.

Why it's useful: A versatile trend-following momentum indicator that helps identify changes in a trend's direction, strength, momentum, and duration. Crossovers between the MACD and Signal lines are common trading signals.

11. add_moving_averages(df_input, ticker_prefix, window_sizes, ma_type)
What it does: Adds Simple Moving Averages (SMA) or Exponential Moving Averages (EMA) for various lookback periods.

Why it's useful: MAs smooth price data to reveal trends. Different periods help identify short-, medium-, and long-term trends. Their values and crossovers are widely used for trend identification and dynamic support/resistance levels.

12. add_bollinger_bands(df_input, ticker_prefix, window, num_std_dev)
What it does: Creates upper, middle, and lower bands around a stock's price, based on a moving average and standard deviation. Also calculates Bandwidth and %B (price's position within bands).

Why it's useful: Measures volatility by showing how far prices deviate from their average. Provides signals for potential overbought/oversold conditions and helps identify volatility expansion/contraction.

13. calculate_stochastic_oscillator(df_input, ticker_prefix, k_period, d_period)
What it does: Compares a stock's closing price to its price range over a given period, generating %K (fast) and %D (slow) lines.

Why it's useful: Similar to RSI, it's an oscillator used to identify overbought/oversold levels and potential price reversals. Divergences between price and the oscillator can signal trend weakness.

14. calculate_adx(df_input, ticker_prefix, window)
What it does: Calculates the Average Directional Index (ADX) along with Positive Directional Indicator (+DI) and Negative Directional Indicator (-DI).

Why it's useful: ADX quantifies the strength of a trend, not its direction. A rising ADX indicates a strong trend (up or down), while a falling ADX suggests consolidation. +DI and -DI indicate the trend's direction.


16. add_time_based_features(df_input)
What it does: Extracts calendar-based features like the day_of_week and month_of_year from the date index.

Why it's useful: Enables the ML model to capture and exploit known market "calendar effects" or seasonal biases (e.g., specific days or months tending to have higher or lower returns).