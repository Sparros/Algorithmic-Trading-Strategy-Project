import requests
import pandas as pd
import time
import os
from pytrends.request import TrendReq
from typing import List, Optional

# Import API keys
from src.config import FRED_API_KEY

def FRED_fetch_macro_data(series_id: str, start_date: str = None) -> pd.DataFrame:
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
    url = "https://api.stlouisfed.org/fred/series/observations" 
    
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json'
    }

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

def macro_data_orchestrator(macro_funcs_to_fetch: list, fred_series_ids_dict: dict, start_date: str = None) -> pd.DataFrame:
    """
    Orchestrates the fetching, cleaning, and merging of all macroeconomic 
    data into a single, time-series-ready DataFrame using the FRED API.

    Args:
        macro_funcs_to_fetch (list): List of macroeconomic indicators to fetch.
        fred_series_ids_dict (dict): A dictionary mapping function names to FRED series IDs.
        start_date (str, optional): The start date for the data (YYYY-MM-DD). 
                                     Defaults to None, which fetches all available data.

    Returns:
        pd.DataFrame: A single, comprehensive DataFrame with all data.
    """
    print("Starting FRED data orchestration pipeline...")
    final_df = pd.DataFrame()

    for func_name in macro_funcs_to_fetch:
        series_id = fred_series_ids_dict.get(func_name)
        if not series_id:
            print(f"Warning: No FRED series ID found for function '{func_name}'. Skipping.")
            continue

        print(f"Fetching and processing data for: {func_name} ({series_id})")
        
        # Pass the start_date to the fetching function
        macro_df = FRED_fetch_macro_data(series_id, start_date=start_date)

        if not macro_df.empty:
            # Resample to daily frequency and forward-fill missing values
            # This is a critical step for data alignment
            macro_df = macro_df.asfreq('D').ffill()

            # Merge with the final_df
            if final_df.empty:
                final_df = macro_df
            else:
                # Use an outer join to ensure all dates are kept
                final_df = final_df.merge(macro_df, left_index=True, right_index=True, how='outer')
    
    # After all data is merged, drop rows with all NaN values to clean up the timeline
    if not final_df.empty:
        final_df.dropna(how='any', inplace=True)

    print("Data orchestration complete.")
    return final_df.sort_index()

# def fetch_macro_data_orchestrator(
#     target_stock: str,
#     fundamental_funcs_to_fetch: list,
#     macro_funcs_to_fetch: list
# ) -> pd.DataFrame:
#     """
#     Orchestrates the fetching, cleaning, and merging of all macroeconomic and
#     fundamental data into a single, time-series-ready DataFrame.
#     """
#     print("Starting data orchestration pipeline...")
#     final_df = pd.DataFrame()

#     # --- Step 1: Fetch and Merge Fundamental Data ---
#     print("Fetching and processing fundamental data...")
#     fundamental_dfs = []
#     for func_name in fundamental_funcs_to_fetch:
#         raw_df = fetch_fundamental_data(func_name, symbol=target_stock)
#         if not raw_df.empty:
#             # Use .copy() to avoid SettingWithCopyWarning
#             df = raw_df.copy()
#             # Convert non-currency columns to numeric types
#             cols_to_convert = [col for col in df.columns if col != 'reportedCurrency']
#             for col in cols_to_convert:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             fundamental_dfs.append(df)

#     if fundamental_dfs:
#         merged_fundamental_df = pd.concat(fundamental_dfs, axis=1, join='outer')
#         processed_fundamental_df = fundamental_metrics(merged_fundamental_df)
#         if 'reportedCurrency' in processed_fundamental_df.columns:
#             processed_fundamental_df = processed_fundamental_df.drop(columns=['reportedCurrency'])
#         processed_fundamental_df = processed_fundamental_df.asfreq('D').ffill()
#         final_df = processed_fundamental_df


#     # --- Step 2: Fetch and Merge General Macro Data ---
#     print("Fetching and processing general macroeconomic data...")
#     for func_name in macro_funcs_to_fetch:
#         macro_df = fetch_general_macro_data(func_name)
#         if not macro_df.empty:
#             macro_df = macro_df.asfreq('D').ffill()
#             if final_df.empty:
#                 final_df = macro_df
#             else:
#                 # Use outer join to keep all dates from both dataframes
#                 final_df = final_df.merge(macro_df, left_index=True, right_index=True, how='outer')

#     # --- Step 3: Remove rows that are all NaN ---
#     # This is the key fix to clean up the DataFrame.
#     if not final_df.empty:
#         final_df.dropna(how='all', inplace=True)

#     print("Data orchestration complete.")
#     return final_df.sort_index()

