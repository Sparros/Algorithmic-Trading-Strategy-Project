import requests
import pandas as pd
import time
import os
import pytrends
from typing import List, Optional

# Your Alpha Vantage API key
from config import Alpha_Vantage_Access_Key as ALPHA_API_KEY

def fetch_general_macro_data(function_name: str, interval: Optional[str] = 'monthly') -> pd.DataFrame:
    """
    Fetches macroeconomic data (e.g., CPI, Interest Rate) from Alpha Vantage.

    Args:
        function_name (str): The Alpha Vantage function to call (e.g., 'CPI').
        interval (Optional[str]): The data frequency ('monthly', 'quarterly', etc.).

    Returns:
        pd.DataFrame: A DataFrame with the date as index and the value as a column.
    """
    if ALPHA_API_KEY == "YOUR_ALPHA_API_KEY_HERE":
        raise ValueError("Please replace 'YOUR_ALPHA_API_KEY_HERE' with your actual Alpha Vantage API key.")
        
    url = f'https://www.alphavantage.co/query?function={function_name}&interval={interval}&apikey={ALPHA_API_KEY}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage for {function_name}: {e}")
        return pd.DataFrame()
    
    if not isinstance(data, dict) or 'data' not in data:
        print(f"Unexpected API response for function {function_name}: {data}")
        return pd.DataFrame()

    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.rename(columns={'value': function_name.lower()}, inplace=True)
    df[function_name.lower()] = pd.to_numeric(df[function_name.lower()], errors='coerce')
    
    return df

def fetch_fundamental_data(function_name: str, symbol: str) -> pd.DataFrame:
    """
    Fetches fundamental data (e.g., Income Statement) for a specific ticker.

    Args:
        function_name (str): The Alpha Vantage function to call (e.g., 'INCOME_STATEMENT').
        symbol (str): The stock ticker (e.g., 'WMT').

    Returns:
        pd.DataFrame: A DataFrame containing the fundamental data.
    """    
    url = f'https://www.alphavantage.co/query?function={function_name}&symbol={symbol}&apikey={ALPHA_API_KEY}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage for {function_name} ({symbol}): {e}")
        return pd.DataFrame()
        
    if not data or 'Error Message' in data:
        print(f"Unexpected API response for function {function_name}: {data}")
        return pd.DataFrame()

    # Process the data based on the API function type
    if function_name in ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW', 'EARNINGS_CALENDAR']:
        report_key = 'quarterlyReports' if 'quarterlyReports' in data else 'annualReports'
        if report_key in data:
            df = pd.DataFrame(data[report_key])
            df.set_index('fiscalDateEnding', inplace=True)
            return df
        
    elif function_name == 'OVERVIEW':
        # The overview function returns a single JSON object, not a list
        return pd.DataFrame([data])
        
    return pd.DataFrame()


# compute derived fundamental metrics (e.g., EPS, P/E ratio, debt-to-equity) from the raw data fetched by fetch_fundamental_data
def fundamental_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived fundamental metrics from the raw data.

    Args:
        df (pd.DataFrame): The DataFrame containing fundamental data.

    Returns:
        pd.DataFrame: A DataFrame with derived metrics.
    """
    if df.empty:
        return pd.DataFrame()

    # Example calculations
    df['eps'] = df['netIncome'] / df['weightedAverageSharesOutstanding']
    df['pe_ratio'] = df['totalRevenue'] / df['eps']
    df['debt_to_equity'] = df['totalDebt'] / df['totalEquity']

    return df[['eps', 'pe_ratio', 'debt_to_equity']]

def fetch_news_sentiment(
    symbol: Optional[str] = None, 
    topics: Optional[List[str]] = None,
    sort_by: str = 'RELEVANCE',
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches news sentiment data from Alpha Vantage for a specific symbol or topic.

    Args:
        symbol (str, optional): The stock ticker (e.g., 'WMT'). If None, fetches general market news.
        topics (List[str], optional): A list of topics to filter by (e.g., ['technology', 'finance']).
        sort_by (str): How to sort the results ('RELEVANCE' or 'LATEST').
        time_from (str, optional): The start date/time for the query (YYYYMMDDTHHMM).
        time_to (str, optional): The end date/time for the query (YYYYMMDDTHHMM).

    Returns:
        pd.DataFrame: A DataFrame with daily aggregated sentiment scores.
    """    
    if ALPHA_API_KEY == "YOUR_ALPHA_API_KEY_HERE":
        raise ValueError("Please replace 'YOUR_ALPHA_API_KEY_HERE' with your actual Alpha Vantage API key.")
    
    url = "https://www.alphavantage.co/query?"
    params = {
        'function': 'NEWS_SENTIMENT',
        'apikey': ALPHA_API_KEY,
        'sort': sort_by
    }
    if symbol:
        params['tickers'] = symbol
    if topics:
        params['topics'] = ','.join(topics)
    if time_from:
        params['time_from'] = time_from
    if time_to:
        params['time_to'] = time_to
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news sentiment data: {e}")
        return pd.DataFrame()
    
    if 'feed' not in data or not data['feed']:
        print("No news feed data found.")
        return pd.DataFrame()

    news_df = pd.DataFrame(data['feed'])
    news_df['time_published'] = pd.to_datetime(news_df['time_published'])
    news_df.set_index('time_published', inplace=True)
    
    # Extract overall sentiment score and label
    news_df['overall_sentiment_score'] = news_df['overall_sentiment_score'].astype(float)
    news_df['overall_sentiment_label'] = news_df['overall_sentiment_label'].astype('category')
    
    # Resample to daily and aggregate scores
    daily_sentiment = news_df.resample('D').agg({
        'overall_sentiment_score': 'mean',
        'overall_sentiment_label': lambda x: x.mode()[0] if not x.mode().empty else None,
        'ticker_sentiment_score_WMT': 'mean' # Example for a specific ticker
    })
    
    # Rename columns to be specific and avoid conflicts
    daily_sentiment = daily_sentiment.rename(columns=lambda col: f'news_{col}')
    
    return daily_sentiment

def fetch_fred_data(series_id: str, start_date: str = '1980-01-01') -> pd.DataFrame:
    """
    Fetches macroeconomic data from the FRED API.

    Args:
        series_id (str): The FRED series ID for the data (e.g., 'PAYEMS' for Non-farm Payrolls).
        start_date (str): The start date for the data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame with the date as the index and the value as a column.
    """
  
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from FRED for {series_id}: {e}")
        return pd.DataFrame()
        
    if 'observations' not in data:
        print(f"No observations found for FRED series ID: {series_id}")
        return pd.DataFrame()

    df = pd.DataFrame(data['observations'])
    df = df[['date', 'value']]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.rename(columns={'value': series_id}, inplace=True)
    df[series_id] = pd.to_numeric(df[series_id], errors='coerce')

def fetch_google_trends(
    keywords: List[str], 
    timeframe: str = 'today 5-y', 
    geo: str = 'US'
) -> pd.DataFrame:
    """
    Fetches search interest data from Google Trends for a list of keywords.

    Args:
        keywords (List[str]): A list of keywords to search for.
        timeframe (str): The time period for the search (e.g., 'today 5-y', '2010-01-01 2023-01-01').
        geo (str): The geographic region (e.g., 'US', 'GB', 'worldwide').

    Returns:
        pd.DataFrame: A DataFrame with the search interest data for each keyword.
    """
    try:
        # Create a PyTrends object
        pytrend = TrendReq(hl='en-US', tz=360)
        
        # Build the payload
        pytrend.build_payload(kw_list=keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
        
        # Get interest over time and clean the DataFrame
        df = pytrend.interest_over_time()
        
        # The 'isPartial' column is not needed for analysis
        if 'isPartial' in df.columns:
            df = df.drop('isPartial', axis=1)
            
        # Convert index to a proper datetime
        df.index = pd.to_datetime(df.index)
        
        # Rename columns to be more descriptive
        df.columns = [f'trend_{kw.replace(" ", "_")}' for kw in df.columns]
        
        return df

    except Exception as e:
        print(f"Error fetching Google Trends data: {e}")
        return pd.DataFrame()

def fetch_macro_data_orchestrator(
    general_macro_funcs: List[str],
    fundamental_funcs: List[str],
    fred_series_ids: List[str],
    target_ticker: str,
    monthly_interval: str,
    google_trends_keywords: Optional[List[str]] = None,
    output_filename: Optional[str] = None,
    output_directory: str = "data/processed"
) -> pd.DataFrame:
    """
    Orchestrates fetching of external macro and fundamental data,
    and returns a single merged DataFrame.

    Args:
        general_macro_funcs (List[str]): List of general macro function names.
        fundamental_funcs (List[str]): List of fundamental function names for the target ticker.
        target_ticker (str): The stock ticker to fetch fundamental data for.
        monthly_interval (str): The interval for monthly macro data (e.g., 'monthly', 'quarterly').
        output_filename (Optional[str]): The filename to save the final merged CSV.
                                        If None, the file is not saved.
        output_directory (str): The directory to save the output file.

    Returns:
        pd.DataFrame: A single DataFrame with all external data merged, with a daily frequency.
    """
    print("\n--- Starting External Data Orchestration ---")
    
    all_external_data = pd.DataFrame()

    # 1. Fetch and process general macroeconomic data
    for func_name in general_macro_funcs:
        df = fetch_general_macro_data(func_name, interval=monthly_interval)
        if not df.empty:
            df = df.asfreq('D').ffill()
            if all_external_data.empty:
                all_external_data = df
            else:
                all_external_data = all_external_data.merge(df, left_index=True, right_index=True, how='outer')
        time.sleep(15)

    # 2. Fetch and process fundamental data for the target ticker
    for func_name in fundamental_funcs:
        raw_fundamental_df  = fetch_fundamental_data(func_name, symbol=target_ticker)
        
        # Check if the fetched data is for Income Statement or Balance Sheet
        # and then compute the derived metrics
        if func_name in ['INCOME_STATEMENT', 'BALANCE_SHEET']:
            processed_fundamental_df = fundamental_metrics(raw_fundamental_df)
        else:
            processed_fundamental_df = raw_fundamental_df
        
        if not processed_fundamental_df.empty:
            # Resample to daily and forward-fill to align with stock data
            processed_fundamental_df.index = pd.to_datetime(processed_fundamental_df.index)
            processed_fundamental_df = processed_fundamental_df.asfreq('D').ffill()
            
            if all_external_data.empty:
                all_external_data = processed_fundamental_df
            else:
                all_external_data = all_external_data.merge(processed_fundamental_df, left_index=True, right_index=True, how='outer')
        time.sleep(15)

    # 3. Fetch and process data from FRED
    for series_id in fred_series_ids:
        print(f"Fetching FRED data for: {series_id}")
        fred_df = fetch_fred_data(series_id=series_id, start_date='2000-01-01')
        if not fred_df.empty:
            fred_df = fred_df.asfreq('D').ffill()
            if all_external_data.empty:
                all_external_data = fred_df
            else:
                all_external_data = all_external_data.merge(fred_df, left_index=True, right_index=True, how='outer')
        time.sleep(5) # Be mindful of API rate limits
    
    # 4. Fetch and process Google Trends data
    if google_trends_keywords:
        print("\nFetching Google Trends data...")
        trends_df = fetch_google_trends(keywords=google_trends_keywords)
        
        if not trends_df.empty:
            # Resample to daily and forward-fill to align with other data
            trends_df = trends_df.resample('D').ffill()
            
            if all_external_data.empty:
                all_external_data = trends_df
            else:
                all_external_data = all_external_data.merge(trends_df, left_index=True, right_index=True, how='outer')

    
    # Final cleanup
    if not all_external_data.empty:
        all_external_data.ffill(inplace=True)
        # Drop rows with any remaining NaNs
        all_external_data.dropna(inplace=True)

    print("--- External Data Orchestration Complete ---")

    # Save the DataFrame to a CSV if a filename is provided
    if output_filename:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        full_path = os.path.join(output_directory, output_filename)
        all_external_data.to_csv(full_path)
        print(f"External data saved to {full_path}")
        
    return all_external_data
    return df