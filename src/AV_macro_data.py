import requests
import pandas as pd

from typing import List, Optional

# Import API keys
from src.config import ALPHA_ACCESS_KEY as ALPHA_API_KEY

def AV_fetch_general_macro_data(function_name: str, interval: Optional[str] = 'monthly') -> pd.DataFrame:
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

def AV_fetch_fundamental_data(function_name: str, symbol: str) -> pd.DataFrame:
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
