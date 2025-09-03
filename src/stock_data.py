import yfinance as yf
import pandas as pd

def _format_ticker_data(df, ticker):
    """
    Helper to format columns and remove 'Adj Close' for a single ticker DataFrame.
    """
    df.columns = [f"{col}_{ticker}" for col in df.columns]
    adj_close_col = f'Adj Close_{ticker}'
    if adj_close_col in df.columns:
        df.drop(columns=[adj_close_col], inplace=True)
    return df

def fetch_multiple_stock_data(tickers, start_date=None, end_date=None, output_filename=None):
    """
    Fetch historical stock data for multiple tickers, with optional start and end dates.
    If no dates are provided, it fetches the largest possible date range.
    """
    if not isinstance(tickers, list) or not tickers:
        print("Invalid tickers list. Returning empty DataFrame.")
        return pd.DataFrame()

    print(f"Fetching data for {len(tickers)} tickers...")
    
    merged_df = None
    try:
        # If dates are provided, use the efficient yf.download() method
        if start_date and end_date:
            print(f"Date range: {start_date} to {end_date}")
            all_stocks_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

            # Handle the case where only one ticker is fetched
            if len(tickers) == 1:
                all_stocks_data.columns = [f"{col}_{tickers[0]}" for col in all_stocks_data.columns]
            else:
                all_stocks_data = all_stocks_data.stack(level=1).reset_index(level=1).rename(columns={'level_1': 'Ticker'})
                all_stocks_data.columns = [f"{col}_{all_stocks_data.pop('Ticker')}" if col != 'Ticker' else col for col in all_stocks_data.columns]
                # Fix column names to be consistent with the multi-ticker output
                all_stocks_data.columns = [f"{col.split('_')[0]}_{col.split('_')[1]}" for col in all_stocks_data.columns]

            merged_df = all_stocks_data.drop(columns=[c for c in all_stocks_data.columns if c.startswith('Adj Close_')])

        # If no dates are provided, fetch all available data and find common dates
        else:
            print("Date range: All available history to Current date")
            for ticker in tickers:
                print(f"Fetching full history for {ticker}...")
                ticker_data = yf.Ticker(ticker).history(period="max")
                
                if ticker_data.empty:
                    print(f"No data found for {ticker}. Skipping.")
                    continue
                
                formatted_data = _format_ticker_data(ticker_data, ticker)
                
                if merged_df is None:
                    merged_df = formatted_data
                else:
                    merged_df = pd.merge(merged_df, formatted_data, left_index=True, right_index=True, how='outer')

        if merged_df is None or merged_df.empty:
            print("No data could be fetched.")
            return pd.DataFrame()

        merged_df.index.name = 'Date'
        merged_df.dropna(inplace=True)
        print(f"\nFinal merged DataFrame has {len(merged_df)} common entries.")

        if output_filename:
            merged_df.to_csv(output_filename)
            print(f"Combined data saved to {output_filename}")
        
        return merged_df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()