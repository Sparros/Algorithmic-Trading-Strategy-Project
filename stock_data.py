import yfinance as yf
import pandas as pd
import numpy as np # For potential NaN handling if any data is missing

def _format_yfinance_columns(df, ticker_suffix=None):
    """
    Helper function to flatten and format yfinance columns.
    Handles both single-ticker (flat columns) and multi-ticker (MultiIndex) outputs.
    Drops 'Adj Close' columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Case: Multi-ticker download -> MultiIndex columns (e.g., ('Close', 'AAPL'))
        # Flatten to 'Category_Ticker' (e.g., 'Close_AAPL')
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    elif ticker_suffix:
        # Case: Single-ticker download -> Flat columns (e.g., 'Close')
        # Prepend ticker suffix to each column name
        df.columns = [f"{col}_{ticker_suffix}" for col in df.columns]
    
    # Drop any 'Adj Close' columns, regardless of how they were formatted
    # This uses a list comprehension to build a new list of columns to keep
    cols_to_keep = [col for col in df.columns if not col.startswith('Adj Close_')]
    df = df[cols_to_keep]

    return df

def fetch_single_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a single stock using yfinance
    and format its columns as 'Category_Ticker'.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing historical stock data with formatted columns.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            print(f"No data found for {ticker} in the specified date range.")
            return pd.DataFrame() # Return an empty DataFrame
        
        # Format columns: Open -> Open_TICKER, High -> High_TICKER etc.
        stock_data = _format_yfinance_columns(stock_data, ticker_suffix=ticker)
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def fetch_multiple_stock_data(tickers, start_date, end_date, output_filename=None):
    """
    Fetch historical stock data for multiple stocks, merge them into a single
    DataFrame, format columns, and optionally save to a single CSV.

    Parameters:
    tickers (list): List of stock ticker symbols.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    output_filename (str, optional): Name of the CSV file to save the data.
                                     If None, data is not saved.

    Returns:
    pd.DataFrame: A single DataFrame containing historical data for all tickers,
                  with columns formatted as 'Category_Ticker'.
    """
    if not isinstance(tickers, list):
        raise TypeError("Tickers must be a list of strings.")
    if not tickers:
        print("No tickers provided. Returning empty DataFrame.")
        return pd.DataFrame()

    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    try:
        # yfinance.download for multiple tickers returns a MultiIndex DataFrame
        all_stocks_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if all_stocks_data.empty:
            print(f"No data found for any of the tickers in the specified date range.")
            return pd.DataFrame()

        # Format columns: ('Close', 'AAPL') -> 'Close_AAPL' and drop 'Adj Close'
        all_stocks_data = _format_yfinance_columns(all_stocks_data)

        # Ensure the index is named 'Date' for consistency if it's not already
        if all_stocks_data.index.name != 'Date':
             all_stocks_data.index.name = 'Date'

        # Remove rows with any NaN values that might arise from missing data for certain tickers
        # This aligns the data so all tickers have data for the same dates
        all_stocks_data.dropna(inplace=True)
        
        if output_filename:
            all_stocks_data.to_csv(output_filename)
            print(f"Combined data for {len(tickers)} tickers saved to {output_filename}")
        
        return all_stocks_data
    except Exception as e:
        print(f"Error fetching data for multiple tickers: {e}")
        return pd.DataFrame()

# --- Example Usage ---
if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = "2010-12-31" 

    # # --- Test fetch_single_stock_data ---
    # print("--- Testing Single Stock Fetch ---")
    # ticker_single = "AAPL"
    # df_single = fetch_single_stock_data(ticker_single, start_date, end_date)
    # if not df_single.empty:
    #     print(f"\nHead of data for {ticker_single}:")
    #     print(df_single.head())
    #     print(f"\nColumns for {ticker_single}: {df_single.columns.tolist()}")
    #     print(f"Index name: {df_single.index.name}")
    #     # Save to CSV (optional, can be done in main script)
    #     # df_single.to_csv(f"{ticker_single}_historical_data.csv")
    #     # print(f"Data for {ticker_single} saved to {ticker_single}_historical_data.csv")
    # else:
    #     print(f"No data fetched for {ticker_single}.")


    # # --- Test fetch_multiple_stocks_data ---
    # print("\n" + "="*80 + "\n--- Testing Multiple Stocks Fetch ---")
    # tickers_multi = ["AAPL", "MSFT", "GOOG"]
    # output_csv_name = "all_stocks_combined_data.csv"

    # df_multi = fetch_multiple_stocks_data(tickers_multi, start_date, end_date, output_filename=output_csv_name)

    # if not df_multi.empty:
    #     print(f"\nHead of combined data for {tickers_multi}:")
    #     print(df_multi.head())
    #     print(f"\nColumns of combined data: {df_multi.columns.tolist()}")
    #     print(f"Index name: {df_multi.index.name}")
    #     print(f"\nShape of combined data: {df_multi.shape}")
    # else:
    #     print(f"No combined data fetched for {tickers_multi}.")

    # # --- Test with UK tickers (from your previous context) ---
    # print("\n" + "="*80 + "\n--- Testing UK Stocks Fetch ---")
    # uk_tickers = ['LLOY.L', 'BARC.L', 'AV.L', 'PSN.L', 'LAND.L']
    # uk_output_csv_name = "uk_stocks_combined_data.csv"

    # df_uk = fetch_multiple_stocks_data(uk_tickers, start_date, end_date, output_filename=uk_output_csv_name)

    # if not df_uk.empty:
    #     print(f"\nHead of combined data for UK stocks:")
    #     print(df_uk.head())
    #     print(f"\nColumns of combined UK data: {df_uk.columns.tolist()[:10]}...") # Print first 10 for brevity
    #     print(f"\nShape of combined UK data: {df_uk.shape}")
    # else:
    #     print(f"No combined data fetched for UK stocks.")