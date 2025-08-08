import requests
import pandas as pd
import time
import os
from pytrends.request import TrendReq
from typing import List, Optional

# Import API keys
from src.config import ALPHA_ACCESS_KEY as ALPHA_API_KEY
from src.config import FRED_API_KEY


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
def fundamental_metrics(df) -> pd.DataFrame:
    """
    Computes derived fundamental metrics from raw financial statement data.

    Args:
        df (pd.DataFrame): A DataFrame containing raw financial statement data.

    Returns:
        pd.DataFrame: A DataFrame with the raw data and new calculated metrics.
    """
    if df.empty:
        return pd.DataFrame()

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Check for required columns and compute metrics only if they exist
    if 'netIncome' in df.columns and 'weightedAverageSharesOutstanding' in df.columns:
        df['eps'] = df['netIncome'] / df['weightedAverageSharesOutstanding']
    else:
        # If columns are missing, fill the new column with NaN
        df['eps'] = float('nan')

    if 'totalRevenue' in df.columns and 'eps' in df.columns and 'eps' in df.columns and not df['eps'].isnull().all():
        df['pe_ratio'] = df['totalRevenue'] / df['eps']
    else:
        df['pe_ratio'] = float('nan')

    if 'totalDebt' in df.columns and 'totalEquity' in df.columns:
        df['debt_to_equity'] = df['totalDebt'] / df['totalEquity']
    else:
        df['debt_to_equity'] = float('nan')

    return df

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
    
    # DYNAMIC LOGIC FOR TICKER-SPECIFIC SENTIMENT
    if symbol:
        # Create a dynamic column name based on the input symbol
        dynamic_score_col = f'{symbol}_ticker_sentiment_score'
        
        # Use a list comprehension to extract the specific ticker's sentiment score
        # This will return a list of scores or NaN if the ticker is not found
        def get_ticker_score(sentiments):
            for item in sentiments:
                if item.get('ticker') == symbol:
                    return float(item.get('ticker_sentiment_score', float('nan')))
            return float('nan')
        
        news_df[dynamic_score_col] = news_df['ticker_sentiment'].apply(get_ticker_score)
        
        # Aggregate daily scores using the new dynamic column name
        daily_sentiment = news_df.resample('D').agg({
            'overall_sentiment_score': 'mean',
            'overall_sentiment_label': lambda x: x.mode()[0] if not x.mode().empty else None,
            dynamic_score_col: 'mean'
        })
    else:
        # If no symbol is provided, just aggregate the overall sentiment
        daily_sentiment = news_df.resample('D').agg({
            'overall_sentiment_score': 'mean',
            'overall_sentiment_label': lambda x: x.mode()[0] if not x.mode().empty else None,
        })
    
    # Rename columns to be specific and avoid conflicts
    daily_sentiment = daily_sentiment.rename(columns=lambda col: f'news_{col}')
    
    return daily_sentiment

def fetch_fred_data(series_id: str, start_date: str = None) -> pd.DataFrame:
    """
    Fetches macroeconomic data from the FRED API for a single series ID.
    
    Args:
        series_id (str): The FRED series ID for the data (e.g., 'PAYEMS').
        start_date (str, optional): The start date for the data (YYYY-MM-DD). 
                                    Defaults to None, which fetches all available data.
    
    Returns:
        pd.DataFrame: A DataFrame with the date as the index and the value as a column,
                      or an empty DataFrame if no data is found.
    """
    url = f"https://api.stlouisfed.org/fred/series/observations" 
    
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json'
    }

    # Conditionally add the observation_start parameter if a start_date is provided
    if start_date:
        params['observation_start'] = start_date
    
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
    
    return df

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
    google_trends_keywords: Optional[List[str]] = None,
    monthly_interval: str = 'monthly',
    output_filename: Optional[str] = None,
    output_directory: str = "data/processed"
) -> pd.DataFrame:
    """
    Orchestrates the fetching and merging of diverse external data sources.

    This function collects macroeconomic data from Alpha Vantage and FRED, company fundamental
    data from Alpha Vantage, and web search trends from Google Trends. It merges all
    these data points into a single, comprehensive DataFrame, with all time series
    resampled to a daily frequency for consistency.

    Args:
        general_macro_funcs (List[str]): A list of macroeconomic function names to fetch from Alpha Vantage
            (e.g., ['CPI', 'FEDERAL_FUNDS_RATE']).
        fundamental_funcs (List[str]): A list of financial statement function names to fetch for the
            target ticker from Alpha Vantage (e.g., ['INCOME_STATEMENT', 'BALANCE_SHEET']).
        fred_series_ids (List[str]): A list of FRED series IDs to fetch macroeconomic data from the
            St. Louis Fed (e.g., ['PAYEMS', 'UMCSENT']).
        target_ticker (str): The stock ticker symbol for which to fetch fundamental data.
        google_trends_keywords (Optional[List[str]]): A list of keywords to fetch search interest data
            for from Google Trends. Set to `None` to skip this step.
        monthly_interval (str): The time interval for Alpha Vantage macroeconomic data
            (e.g., 'monthly', 'quarterly').
        output_filename (Optional[str]): The filename to save the final merged CSV. If `None`,
            the DataFrame will not be saved.
        output_directory (str): The directory where the output file will be saved.

    Returns:
        pd.DataFrame: A single DataFrame containing all the merged external data,
        resampled to a daily frequency and ready for modeling.
    """
    print("\n--- Starting External Data Orchestration ---")
    
    all_external_data = pd.DataFrame()

    # 1. Fetch and process general macroeconomic data
    for func_name in general_macro_funcs:
        print(f"Fetching general macroeconomic data for: {func_name}")
        df = fetch_general_macro_data(func_name, interval=monthly_interval)
        if not df.empty:
            df = df.asfreq('D').ffill()
            if all_external_data.empty:
                all_external_data = df
            else:
                all_external_data = all_external_data.merge(df, left_index=True, right_index=True, how='outer')

    # 2. Fetch and process fundamental data for the target ticker
    for func_name in fundamental_funcs:
        print(f"Fetching fundamental data for: {func_name} ({target_ticker})")
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

    print("--- External Data Orchestration Complete ---")

    # Save the DataFrame to a CSV if a filename is provided
    if output_filename:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        full_path = os.path.join(output_directory, output_filename)
        all_external_data.to_csv(full_path)
        print(f"External data saved to {full_path}")
        
    return all_external_data
