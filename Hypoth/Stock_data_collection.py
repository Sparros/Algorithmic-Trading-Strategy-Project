import pandas as pd
import yfinance as yf

# Example UK Tickers
uk_tickers = ['^FTSE', 'LLOY.L', 'BARC.L', 'AV.L', 'PSN.L', 'LAND.L']
start_date = "2000-01-01" # Extend as far back as reliable data exists
end_date = "2024-12-31" # Up to current or desired end

# A list to store the valid Series we download
all_stock_series = []

for ticker in uk_tickers:
    print(f"Attempting to download {ticker}...")
    try:
        # auto_adjust=True is now the default, so 'Adj Close' doesn't exist.
        # Use data['Close'] directly as it will already be adjusted.
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

        if data.empty:
            print(f"Warning: No data found for {ticker}. Skipping.")
            continue # Skip to next ticker if DataFrame is empty

        if 'Close' not in data.columns:
            print(f"Error: 'Close' column not found for {ticker} after download. Skipping.")
            continue # Skip if 'Close' column is missing

        # Extract the 'Close' Series
        ticker_series = data['Close']

        if ticker_series.empty:
            print(f"Warning: 'Close' Series for {ticker} is empty. Skipping.")
            continue # Skip if the Series itself is empty

        # Ensure the index is a DatetimeIndex for proper merging later
        ticker_series.index = pd.to_datetime(ticker_series.index)
        # Give the Series a name, which will become the column name in the final DataFrame
        ticker_series.name = ticker

        all_stock_series.append(ticker_series)
        print(f"Successfully downloaded {ticker} with {len(ticker_series)} data points.")

    except Exception as e:
        print(f"Error downloading or processing {ticker}: {e}")

# Combine all successfully downloaded Series into a single DataFrame
if all_stock_series:
    # pd.concat is more robust than pd.DataFrame(dict) for combining Series
    # with potentially non-identical (but overlapping) date ranges.
    # It will align by index (date) and fill missing values with NaN where data doesn't exist for a given date.
    df_uk_stocks = pd.concat(all_stock_series, axis=1)
    print("\nUK Stock Data Sample (Combined):")
    print(df_uk_stocks.head())
    print(df_uk_stocks.info())
    # Save the DataFrame to a CSV file
    df_uk_stocks.to_csv('uk_stock_data.csv')
else:
    print("\nNo stock data successfully downloaded for any ticker. df_uk_stocks is empty.")
    df_uk_stocks = pd.DataFrame() # Initialize an empty DataFrame if no data was collected