import yfinance as yf
import pandas as pd

def fetch_single_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a single stock using yfinance.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing historical stock data with flattened columns.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # --- Column Joining Method to Flatten MultiIndex Columns ---
    if isinstance(stock_data.columns, pd.MultiIndex):
        # This part iterates through each column tuple (e.g., ('Close', 'AAPL'))
        # and joins the elements with an underscore, creating a single string.
        # So, ('Close', 'AAPL') becomes 'Close_AAPL'.
        stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
    # --- End of Column Joining Method ---    
    return stock_data

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"  # Apple Inc.
    start_date = "2020-01-01"
    end_date = "2024-12-31" # Up to end of last year for backtesting
    
    df = fetch_single_stock_data(ticker, start_date, end_date)
    print(df.head())

    print(df.columns)
    print(df.index.name)
    print(df.head())
    # Save to CSV
    df.to_csv(f"{ticker}_historical_data.csv")
    print(f"Data for {ticker} saved to {ticker}_historical_data.csv")

    