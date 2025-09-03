import requests
import pandas as pd
import time
import os
from pytrends.request import TrendReq
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Optional, Sequence, Dict

EPS = 1e-12

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

def normalize_date_col(df, col="Date"):
    # unify column name and type (naive, normalized midnight)
    if col not in df.columns:
        # common alternates
        for c in ["date", "DATE", "Date"]:
            if c in df.columns: 
                df = df.rename(columns={c: "Date"})
                break
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # drop tz if present and normalize to midnight
    if pd.api.types.is_datetime64tz_dtype(df["Date"]):
        df["Date"] = df["Date"].dt.tz_convert(None)
    df["Date"] = df["Date"].dt.normalize()
    return df

def prepare_macro_for_daily_merge(macro_df):
    macro_df = normalize_date_col(macro_df, "Date")
    macro_df = macro_df.sort_values("Date").drop_duplicates("Date")

    # If macro isn’t already daily, resample to business days with forward fill
    # (detect by median spacing > 2 days)
    if macro_df["Date"].diff().median() > pd.Timedelta(days=2):
        macro_df = (macro_df
                    .set_index("Date")
                    .resample("B")   # business days
                    .ffill()
                    .reset_index())
    return macro_df

def merge_stocks_and_macros(stock_df, macro_df, tolerance_days=31):
    stock_df = normalize_date_col(stock_df, "Date").sort_values("Date")
    macro_df = prepare_macro_for_daily_merge(macro_df).sort_values("Date")

    # asof merge: each stock day gets the most recent macro reading at/before it
    merged = pd.merge_asof(
        stock_df, macro_df,
        on="Date",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days)
    )
    return merged

def add_yield_curve_moments(macro_daily: pd.DataFrame,
                            tenors: Dict[str,str] = None) -> pd.DataFrame:
    """
    Compute level/slope/curvature from Treasury yields.
    tenors maps friendly names -> column names in macro_daily.
    Example default expects FRED series columns already present:
        2y: 'DGS2', 10y: 'DGS10', 30y: 'DGS30'
    """
    df = macro_daily.copy()
    if tenors is None:
        tenors = {"2y": "DGS2", "10y": "DGS10", "30y": "DGS30"}
    if not all(col in df.columns for col in tenors.values()):
        # Nothing to do if we don't have yields
        return df

    y2, y10, y30 = df[tenors["2y"]].astype(float), df[tenors["10y"]].astype(float), df[tenors["30y"]].astype(float)
    level = (y2 + y10 + y30) / 3.0
    slope = y10 - y2
    # simple curvature proxy
    curvature = y30 - 2*y10 + y2

    df["YC_Level"] = level
    df["YC_Slope_10y_2y"] = slope
    df["YC_Curvature"] = curvature

    # optional rolling z within time to keep scale stable
    for c in ["YC_Level","YC_Slope_10y_2y","YC_Curvature"]:
        mu = df[c].rolling(252, min_periods=60).mean()
        sd = df[c].rolling(252, min_periods=60).std()
        df[c + "_z"] = (df[c] - mu) / (sd + EPS)

    return df

def add_macro_pca_kmeans_regimes(macro_daily: pd.DataFrame,
                                 cols: Optional[Sequence[str]] = None,
                                 n_components: int = 5,
                                 n_clusters: int = 3,
                                 lookback: int = 252) -> pd.DataFrame:
    """
    Rolling PCA on macro columns, then assign regimes via k-means on the
    current PCA coordinates using only information available up to t-1.
    Outputs:
      - MACRO_PC1..PCk
      - MACRO_Regime (0..k-1)
    """
    df = macro_daily.copy()
    if cols is None:
        # numeric macro cols
        cols = [c for c in df.columns if c != "Date" and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) == 0:
        return df

    pcs = [f"MACRO_PC{i+1}" for i in range(n_components)]
    for p in pcs:
        df[p] = np.nan
    df["MACRO_Regime"] = np.nan

    # rolling expanding window (min 2*lookback to get stable PCs)
    for t in range(len(df)):
        end = t  # up to t-1 for as-of (we will fill PCs at t using data <= t-1)
        if end < 1:
            continue
        start = max(0, end - lookback)
        hist = df.iloc[start:end]  # NOT including row t
        X = hist[cols].astype(float).dropna(how="any")
        if len(X) < max(60, n_components*20):
            continue
        # standardize within history window
        mu = X.mean(axis=0)
        sd = X.std(axis=0).replace(0, np.nan)
        Xz = (X - mu) / sd

        pca = PCA(n_components=n_components, random_state=0)
        Z = pca.fit_transform(Xz)

        # assign clusters on history and then transform current (t) with those params
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        km.fit(Z)

        # now transform the current row (t) using history mu/sd/pca
        x_t = df.iloc[[t]][cols].astype(float)
        if x_t.isna().any(axis=1).iloc[0]:
            continue
        x_tz = (x_t - mu) / sd
        x_tz = x_tz.fillna(0.0)
        z_t = pca.transform(x_tz)
        r_t = km.predict(z_t)[0]

        for i, name in enumerate(pcs):
            df.iat[t, df.columns.get_loc(name)] = float(z_t[0, i])
        df.iat[t, df.columns.get_loc("MACRO_Regime")] = int(r_t)

    return df

def add_risk_appetite_proxies(macro_daily: pd.DataFrame,
                              spread_cols: Optional[Dict[str,str]] = None) -> pd.DataFrame:
    """
    Add risk appetite proxies from credit spreads or VIX if available.
    spread_cols may include:
      - 'HY_OAS' (high yield OAS series)
      - 'IG_OAS' (investment grade OAS)
      - 'VIX'
    Creates levels and 21D z-scores.
    """
    df = macro_daily.copy()
    candidates = spread_cols or {}
    for key, col in candidates.items():
        if col in df.columns:
            mu = df[col].rolling(252, min_periods=60).mean()
            sd = df[col].rolling(252, min_periods=60).std()
            df[f"{key}"] = df[col].astype(float)
            df[f"{key}_z21"] = (df[col] - df[col].rolling(21).mean()) / (df[col].rolling(21).std() + EPS)
            df[f"{key}_z252"] = (df[col] - mu) / (sd + EPS)
    # simple composite if present
    comps = [c for c in ["HY_OAS_z21","VIX_z21"] if c in df.columns]
    if comps:
        df["RiskAppetite_Composite"] = -pd.DataFrame({c: df[c] for c in comps}).mean(axis=1)
    return df

def add_equity_liquidity_proxies(equity_df: pd.DataFrame, ticker: str, window=20) -> pd.DataFrame:
    """
    Add Amihud illiquidity proxy and turnover for a given ticker.
    Amihud = mean_t( |ret| / dollar_vol ), with dollar_vol = price*volume
    """
    df = equity_df.copy()
    c, v = f"Close_{ticker}", f"Volume_{ticker}"
    if c in df.columns and v in df.columns:
        ret = df[c].pct_change().abs()
        dollar_vol = df[c] * df[v]
        illiq = (ret / (dollar_vol + EPS)).rolling(window, min_periods=window).mean()
        df[f"{ticker}_Amihud_{window}"] = illiq
        df[f"{ticker}_Turnover_{window}"] = (df[v] / (df[v].rolling(window).max() + EPS))
    return df

def attach_macro_surprises(macro_daily: pd.DataFrame,
                           surprises_df: pd.DataFrame,
                           on_col: str = "Date",
                           max_back_days: int = 31) -> pd.DataFrame:
    """
    Merge macro 'surprise' columns (actual - expected) to macro_daily via as-of merge.
    surprises_df schema expected:
      - Date (release timestamp normalized to date)
      - one or more surprise columns, e.g., 'CPI_Surprise', 'PAYEMS_Surprise'
    We assume surprises_df is already lagged to availability (i.e., no peeking).
    """
    a = macro_daily.copy()
    b = surprises_df.copy()
    if on_col not in a.columns: a = a.reset_index().rename(columns={a.index.name or "index": on_col})
    if on_col not in b.columns: b = b.reset_index().rename(columns={b.index.name or "index": on_col})
    a[on_col] = pd.to_datetime(a[on_col]).dt.normalize()
    b[on_col] = pd.to_datetime(b[on_col]).dt.normalize()

    merged = pd.merge_asof(
        a.sort_values(on_col),
        b.sort_values(on_col),
        on=on_col, direction="backward",
        tolerance=pd.Timedelta(days=max_back_days)
    )
    return merged

def macro_data_orchestrator(macro_funcs_to_fetch: list, fred_series_ids_dict: dict, start_date: str = None, save_path = None) -> pd.DataFrame:
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
    
    # (1) Yield curve moments if yields available
    final_df = add_yield_curve_moments(final_df)

    # (2) Risk appetite proxies if you have spreads/VIX in your series set
    # map FRED series names to logical keys if present
    spread_map = {}
    for k, cand in [("HY_OAS", "BAMLH0A0HYM2"), ("IG_OAS", "BAMLC0A0CM"), ("VIX", "VIXCLS")]:
        if cand in final_df.columns:
            spread_map[k] = cand
    if spread_map:
        final_df = add_risk_appetite_proxies(final_df, spread_cols=spread_map)

    # (3) Macro PCA + KMeans regimes (uses all numeric macro columns)
    final_df = add_macro_pca_kmeans_regimes(final_df, n_components=5, n_clusters=3, lookback=252)

    # After all data is merged, drop rows with all NaN values to clean up the timeline
    if not final_df.empty:
        final_df.dropna(how='all', inplace=True)

    # Save to CSV if a filename is provided
    if save_path:
        out_path = os.path.join(save_path, f"macros.csv")
        final_df.to_csv(out_path, index=True)

    print("Data orchestration complete.")
    return final_df.sort_index()


def safe_shift(df: pd.DataFrame, cols: Sequence[str], lag: int = 1) -> pd.DataFrame:
    """Shift selected columns by 'lag' to ensure t-1 availability (anti-leakage)."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].shift(lag)
    return df

def compute_yield_curve_features(df: pd.DataFrame,
                                 ten='DGS10', five='DGS5', two='DGS2', three_mo='TB3MS') -> pd.DataFrame:
    """
    Classic level/slope/curvature features. Uses what exists, skips what doesn't.
    All outputs are shifted by 1 day to avoid look-ahead.
    """
    out = df.copy()
    have = set(out.columns)

    # Level (10Y)
    if ten in have:
        out['yc_level_10y'] = out[ten]
    # Slope: 10Y-3M or 10Y-2Y
    if ten in have and three_mo in have:
        out['yc_slope_10y_3m'] = out[ten] - out[three_mo]
    elif ten in have and two in have:
        out['yc_slope_10y_2y'] = out[ten] - out[two]
    # Curvature: 10Y - 2*5Y + 2Y
    if all(c in have for c in [ten, five, two]):
        out['yc_curvature'] = out[ten] - 2.0*out[five] + out[two]

    # Normalize simple diffs/z where appropriate (optional)
    for c in ['yc_level_10y', 'yc_slope_10y_3m', 'yc_slope_10y_2y', 'yc_curvature']:
        if c in out.columns:
            out[c+'_chg_5'] = out[c].diff(5)

    # Shift to t-1 to mimic availability
    shift_cols = [c for c in out.columns if c.startswith('yc_')]
    out = safe_shift(out, shift_cols, lag=1)
    return out

def compute_liquidity_risk_proxies(df: pd.DataFrame,
                                   dff='DFF', three_mo='TB3MS', ten='DGS10') -> pd.DataFrame:
    """
    Simple proxies when 'proper' series (TED, OAS, VIX) are not available:
    - Policy gap: 3M T-bill - FFR (rough funding pressure proxy)
    - Term premium proxy: 10Y - 3M
    - Carry proxy: 10Y - FFR
    All shifted by 1 day.
    """
    out = df.copy()
    if three_mo in out.columns and dff in out.columns:
        out['proxy_ted_like'] = out[three_mo] - out[dff]
    if ten in out.columns and three_mo in out.columns:
        out['term_slope_10y_3m'] = out[ten] - out[three_mo]
    if ten in out.columns and dff in out.columns:
        out['carry_10y_ff'] = out[ten] - out[dff]

    # Changes
    for c in ['proxy_ted_like', 'term_slope_10y_3m', 'carry_10y_ff']:
        if c in out.columns:
            out[c+'_chg_5'] = out[c].diff(5)

    shift_cols = [c for c in out.columns if c.startswith(('proxy_', 'term_', 'carry_'))]
    out = safe_shift(out, shift_cols, lag=1)
    return out

def compute_macro_pca_and_regimes(df: pd.DataFrame,
                                  cols: Optional[Sequence[str]] = None,
                                  n_components: int = 5,
                                  kmeans_k: int = 3,
                                  use_hmm: bool = False,
                                  hmm_states: int = 3,
                                  random_state: int = 42) -> pd.DataFrame:
    """
    PCA over macro columns → rolling standardized PCs → KMeans regimes.
    Optional HMM regimes if hmmlearn available. All regime labels are shifted to t-1.
    """
    out = df.copy()

    # Choose columns: numeric, non-constant
    if cols is None:
        num = out.select_dtypes(include=[np.number])
        cols = [c for c in num.columns if num[c].notna().sum() > 100 and num[c].std(skipna=True) > 0]

    if len(cols) == 0:
        return out

    # Standardize (rolling) to avoid full-sample peek. Use long-ish window.
    Z = pd.DataFrame(index=out.index)
    for c in cols:
        s = out[c].astype(float)
        mu = s.rolling(252, min_periods=60).mean()
        sd = s.rolling(252, min_periods=60).std()
        Z[c] = (s - mu) / (sd + EPS)

    Z = Z.dropna(how='all')
    good = [c for c in cols if c in Z.columns]
    if len(good) == 0:
        return out

    # PCA on available rows only
    Z2 = Z[good].fillna(0.0)
    pca = PCA(n_components=min(n_components, len(good)))
    pcs = pd.DataFrame(pca.fit_transform(Z2.values),
                       index=Z2.index,
                       columns=[f'pc{i+1}' for i in range(pca.n_components_)])
    # Align back
    for c in pcs.columns:
        out[c] = pcs[c]

    # KMeans regimes on PCs
    km = KMeans(n_clusters=kmeans_k, n_init=20, random_state=random_state)
    km_labels = pd.Series(index=pcs.index, data=km.fit_predict(pcs.values), name='regime_km')
    out['regime_km'] = km_labels.reindex(out.index)

    # Optional HMM
    if use_hmm and _HMM_AVAILABLE:
        hmm = GaussianHMM(n_components=hmm_states, covariance_type='full', random_state=random_state, n_iter=200)
        try:
            hmm.fit(pcs.values)
            hmm_labels = hmm.predict(pcs.values)
            out['regime_hmm'] = pd.Series(hmm_labels, index=pcs.index).reindex(out.index)
        except Exception:
            # Fallback if HMM fails to converge
            out['regime_hmm'] = np.nan
    else:
        out['regime_hmm'] = np.nan

    # Shift regime labels to t-1
    out[['regime_km', 'regime_hmm']] = out[['regime_km', 'regime_hmm']].shift(1)

    # Also add regime dummies (one-hot) safely
    if 'regime_km' in out.columns:
        d = pd.get_dummies(out['regime_km'], prefix='regkm', dummy_na=False)
        out = pd.concat([out, d], axis=1)

    return out

def add_macro_surprises(actual_df: pd.DataFrame,
                        expectations_df: Optional[pd.DataFrame] = None,
                        mapping: Optional[Dict[str, str]] = None,
                        publish_lags: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Encode surprises (actual - expected). If expectations are missing, falls back to
    month-over-month z-scores as a proxy.
    - mapping: { 'CPIAUCSL': 'exp_CPIAUCSL', ... } maps actual column → expected column in expectations_df
    - publish_lags: dict of series → days to lag post-release
    Output columns get suffix: *_surprise and *_surprise_z
    """
    out = actual_df.copy()

    if expectations_df is not None and mapping:
        exp = normalize_date_col(expectations_df, col='Date').sort_values('Date').drop_duplicates('Date').set_index('Date')
        for actual_col, exp_col in mapping.items():
            if actual_col in out.columns and exp_col in exp.columns:
                # As-of merge expected values (backward fill)
                tmp = (out[[actual_col]].reset_index()
                       .rename(columns={'index': 'Date'}))
                merged = pd.merge_asof(tmp.sort_values('Date'),
                                       exp[[exp_col]].reset_index().sort_values('Date'),
                                       on='Date', direction='backward')
                surprise = merged[actual_col] - merged[exp_col]
                out[f'{actual_col}_surprise'] = pd.Series(surprise.values, index=out.index)
    # Fallback: MoM z-score proxy when no expectations provided
    # (still useful to mark outsized releases)
    for c in out.select_dtypes(include=[np.number]).columns:
        mom = out[c].pct_change(1)
        mu = mom.rolling(24, min_periods=12).mean()
        sd = mom.rolling(24, min_periods=12).std()
        out[f'{c}_surprise_proxy_z'] = (mom - mu) / (sd + EPS)

    # Apply publish lags to surprise columns
    if publish_lags:
        for c in list(out.columns):
            base = c.replace('_surprise', '').replace('_surprise_proxy_z', '')
            lag = publish_lags.get(base, None)
            if lag is not None and (c.endswith('_surprise') or c.endswith('_surprise_proxy_z')):
                out[c] = out[c].shift(lag)

    # Final day-1 shift to be safe
    s_cols = [c for c in out.columns if c.endswith('_surprise') or c.endswith('_surprise_proxy_z')]
    out = safe_shift(out, s_cols, lag=1)
    return out

def build_macro_state_features(macro_daily: pd.DataFrame,
                               publish_lags: Optional[Dict[str, int]] = None,
                               pca_cols: Optional[Sequence[str]] = None,
                               use_hmm: bool = False) -> pd.DataFrame:
    """
    One-stop enrich: yield-curve features + liquidity/risk proxies + PCA/KMeans regimes + surprises proxy.
    Pass publish_lags to align surprises to post-release availability.
    """
    m = macro_daily.copy()
    m = compute_yield_curve_features(m)
    m = compute_liquidity_risk_proxies(m)
    m = compute_macro_pca_and_regimes(m, cols=pca_cols, use_hmm=use_hmm)

    # Add surprise proxies (if you later provide an expectations_df you can re-run with mapping)
    m = add_macro_surprises(m, expectations_df=None, mapping=None, publish_lags=publish_lags)

    # Final cleanup
    m = m.sort_index()
    return m