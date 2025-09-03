import requests
import pandas as pd

import time
import os
from pytrends.request import TrendReq
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Optional, Sequence, Dict
from pandas.tseries.offsets import BDay

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

def normalize_date_index(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """
    Ensure df has a tz-naive, midnight-normalized DatetimeIndex.
    Accepts a date column (any of ['Date','date','DATE']) or a datetime-like index.
    Returns df with the date set as the index; drops the date column if present.
    """
    out = df.copy()

    # Try to find a date column
    candidates = [col, "date", "Date", "DATE"]
    found = next((c for c in candidates if c in out.columns), None)

    if found is not None:
        # Parse column to datetime
        out[found] = pd.to_datetime(out[found], errors="coerce", utc=False)
        # If tz-aware, strip tz
        if pd.api.types.is_datetime64tz_dtype(out[found]):
            out[found] = out[found].dt.tz_convert(None)
        # Normalize to midnight and set as index
        out[found] = out[found].dt.normalize()
        out = out.set_index(found)
        # Optional: give the index a consistent name
        out.index.name = "date"
        return out

    # No date column—try to use the index
    idx = out.index
    # If it’s not datetime already, try to parse it
    if not pd.api.types.is_datetime64_any_dtype(idx):
        try:
            idx = pd.to_datetime(idx, errors="coerce", utc=False)
        except Exception:
            pass

    if pd.api.types.is_datetime64_any_dtype(idx):
        # Strip tz if needed and normalize
        if pd.api.types.is_datetime64tz_dtype(idx):
            idx = idx.tz_convert(None)
        out.index = idx.normalize()
        out.index.name = "date"
        return out

    # Still no luck—raise a clear error
    raise KeyError(
        "No date column or datetime-like index found. "
        f"Columns present: {list(out.columns)[:10]}{'...' if len(out.columns)>10 else ''}"
    )


def apply_publish_lags(df: pd.DataFrame, lags: dict, date_col: str = "Date") -> pd.DataFrame:
    """
    Shift selected macro columns forward by N business days to mimic post-release availability.
    lags: { column_name: business_days_to_delay }
    Works with either a 'Date' column or a DatetimeIndex. Returns the same layout it received.
    """
    out = df.copy()

    # Standardize to DatetimeIndex for shifting
    had_date_col = date_col in out.columns
    if had_date_col:
        out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
        out = out.set_index(date_col)
    else:
        out.index = pd.to_datetime(out.index).tz_localize(None).normalize()

    for col, bdays in (lags or {}).items():
        if col in out.columns and pd.notna(bdays) and int(bdays) != 0:
            out[col] = out[col].shift(freq=BDay(int(bdays)))   # <-- key fix

    out = out.sort_index()

    # restore original layout
    if had_date_col:
        return out.reset_index().rename(columns={"index": date_col})
    return out

def prepare_macro_for_daily_merge(macro_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure we have a DatetimeIndex (named 'date')
    df = normalize_date_index(macro_df)

    # Drop any stray 'Date' column; index is canonical
    if "Date" in df.columns:
        df = df.drop(columns="Date")

    # De-dup and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # If not already daily-ish, resample to business days and ffill
    median_gap = df.index.to_series().diff().median()
    if pd.isna(median_gap) or median_gap > pd.Timedelta(days=2):
        df = df.resample("B").ffill()

    # merge_asof needs a column, not an index -> return with 'Date' column
    return df.reset_index().rename(columns={"date": "Date"})

def merge_stocks_and_macros(stock_df: pd.DataFrame,
                            macro_df: pd.DataFrame,
                            tolerance_days: int = 31) -> pd.DataFrame:
    """As-of merge on a 'Date' column; macro_df can be index- or column-dated."""
    a = stock_df.copy()
    if "Date" not in a.columns:
        a = a.reset_index().rename(columns={a.index.name or "index": "Date"})
    a["Date"] = pd.to_datetime(a["Date"]).dt.normalize()

    # macro: if it already has 'Date', just normalize; otherwise prep it
    if "Date" in macro_df.columns:
        b = macro_df.copy()
        b["Date"] = pd.to_datetime(b["Date"]).dt.normalize()
    else:
        b = prepare_macro_for_daily_merge(macro_df)  # returns with 'Date' column

    merged = pd.merge_asof(
        a.sort_values("Date"),
        b.sort_values("Date"),
        on="Date",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
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

def add_pca_kmeans_monthly(
    df_monthly: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    n_components: int = 5,
    n_clusters: int = 3,
    train_end: Optional[pd.Timestamp] = None,  # e.g., pd.Timestamp("2015-12-31")
) -> Tuple[pd.DataFrame, PCA, KMeans, pd.Series, pd.Series]:
    """
    Fit PCA and KMeans on a fixed training window (monthly data), then
    transform and classify the entire sample using those fixed parameters.
    Returns (df_with_features, pca, kmeans, mu, sd).
    """
    df = df_monthly.copy()
    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # expanding z-scores computed without leakage
    mu = df[cols].expanding().mean().shift(1)
    sd = df[cols].expanding().std().shift(1).replace(0, np.nan)
    Z = (df[cols] - mu) / sd

    # split train / full matrices
    if train_end is None:
        # use first 70% as train if not provided
        train_end = df.index[int(len(df)*0.7)]
    train_mask = df.index <= train_end
    Z_train = Z.loc[train_mask].dropna(how="any")

    # if not enough data, bail gracefully
    if len(Z_train) < max(60, n_components*20):
        return df, None, None, None, None

    pca = PCA(n_components=n_components, random_state=0).fit(Z_train.values)
    PCs = pca.transform(Z.values)  # will be nan where Z has nans

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=0).fit(PCs[train_mask & Z.notna().all(1)])
    regime = np.full(len(df), np.nan)
    valid_rows = Z.notna().all(1).values
    regime[valid_rows] = km.predict(PCs[valid_rows])

    # attach features
    for i in range(n_components):
        df[f"MACRO_PC{i+1}"] = PCs[:, i]
    df["MACRO_Regime"] = regime.astype("float")  # may include nans

    return df, pca, km, mu.iloc[-1], sd.iloc[-1]

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

def add_simple_regimes(df_m: pd.DataFrame) -> pd.DataFrame:
    """
    Defines a compact, interpretable regime label from macro features.
    Encodes three dimensions: Growth (up/down), Inflation (high/low), Risk (on/off).
    """
    out = df_m.copy()

    growth_up = out.get("PAYEMS_YoY_z")
    infl_hi   = out.get("CPI_YoY_z")
    vix_pct   = out.get("VIX_pctile")
    hy_z      = out.get("HY_OAS_z")
    ig_z      = out.get("IG_OAS_z")

    # Booleans with conservative thresholds
    grow_up = growth_up > 0 if growth_up is not None else pd.Series(False, index=out.index)
    infl_hi = infl_hi   > 0 if infl_hi   is not None else pd.Series(False, index=out.index)

    # Risk-off if VIX in top 20% or credit spreads > +1σ
    risk_off = pd.Series(False, index=out.index)
    if vix_pct is not None:
        risk_off = risk_off | (vix_pct >= 0.80)
    if hy_z is not None:
        risk_off = risk_off | (hy_z >= 1.0)
    if ig_z is not None:
        risk_off = risk_off | (ig_z >= 1.0)

    # Encode regime as 0..5 (3-bit style): (Growth, Inflation, Risk)
    # 0: low growth, low inflation, risk-on … up to 5: high growth, high inflation, risk-off
    out["Regime_GIR"] = (grow_up.astype(int) * 2) + (infl_hi.astype(int)) + (risk_off.astype(int) * 3)

    # Human-readable label (optional)
    labels = {
        0: "Low Growth • Low Inflation • Risk-ON",
        1: "Low Growth • High Inflation • Risk-ON",
        2: "High Growth • Low Inflation • Risk-ON",
        3: "Low Growth • Low Inflation • Risk-OFF",
        4: "High Growth • High Inflation • Risk-ON",
        5: "High Growth • High Inflation • Risk-OFF",
    }
    out["Regime_GIR_Label"] = out["Regime_GIR"].map(labels)

    return out

# ---------- helpers (no leakage) ----------
def expanding_mean(s: pd.Series) -> pd.Series:
    return s.expanding().mean().shift(1)

def expanding_std(s: pd.Series) -> pd.Series:
    return s.expanding().std().shift(1).replace(0, np.nan)

def expanding_z(s: pd.Series) -> pd.Series:
    m, v = expanding_mean(s), expanding_std(s)
    return (s - m) / v

def expanding_percentile(s: pd.Series) -> pd.Series:
    # Percentile of current value within *past* history
    ranks = []
    vals = s.values
    for i in range(len(s)):
        if i == 0 or pd.isna(vals[i]):
            ranks.append(np.nan)
            continue
        hist = pd.Series(vals[:i]).dropna()
        if len(hist) == 0:
            ranks.append(np.nan)
        else:
            ranks.append((hist <= vals[i]).mean())
    return pd.Series(ranks, index=s.index)

def add_simple_macro_features(df_m: pd.DataFrame) -> pd.DataFrame:
    """
    Expects monthly index. Creates common, interpretable features.
    Works even if some inputs are missing.
    """
    out = df_m.copy()

    # Safe getters by FRED id (rename if your columns differ)
    CPI   = out.get("CPIAUCSL")     # CPI level (index)
    PAY   = out.get("PAYEMS")       # Nonfarm payrolls level
    DFF   = out.get("DFF")          # Fed funds (daily, will be resampled)
    DGS10 = out.get("DGS10")        # 10Y Treasury yield (daily, will be resampled)
    VIX   = out.get("VIXCLS")       # VIX (daily, will be resampled)
    HY    = out.get("BAMLH0A0HYM2") # HY OAS (optional)
    IG    = out.get("BAMLC0A0CM")   # IG OAS (optional)

    # YoY rates (use pct_change(12) on monthly)
    if CPI is not None:
        out["CPI_YoY"] = CPI.pct_change(12) * 100
    if PAY is not None:
        out["PAYEMS_YoY"] = PAY.pct_change(12) * 100

    # Yield curve slope (10y - policy/short rate)
    if (DGS10 is not None) and (DFF is not None):
        out["YC_Slope"] = DGS10 - DFF

    # Credit spreads & VIX percentiles (risk appetite)
    if VIX is not None:
        out["VIX_pctile"] = expanding_percentile(VIX)
    if HY is not None:
        out["HY_OAS_z"] = expanding_z(HY)
    if IG is not None:
        out["IG_OAS_z"] = expanding_z(IG)

    # Z-scores for growth/inflation proxies (no leakage)
    if "CPI_YoY" in out:
        out["CPI_YoY_z"] = expanding_z(out["CPI_YoY"])
    if "PAYEMS_YoY" in out:
        out["PAYEMS_YoY_z"] = expanding_z(out["PAYEMS_YoY"])
    if "YC_Slope" in out:
        out["YC_Slope_z"] = expanding_z(out["YC_Slope"])

    return out

def macro_data_orchestrator(
    macro_funcs_to_fetch: list,
    fred_series_ids_dict: dict,
    start_date: str = None,
    save_path: str = None,
    output_freq: str = "M",       # "M" for monthly (recommended) or "D" for daily-ffilled
    add_simple_features: bool = True,
    add_simple_regime: bool = True,
) -> pd.DataFrame:
    """
    Fetches FRED series, merges them into a single DataFrame, builds simple
    macro features/regimes, and returns a modeling-ready time series.

    Pipeline:
      1) Fetch each requested FRED series and concatenate on the date index.
      2) Resample to monthly ('M') using last observation.
      3) Optionally add lightweight features (YoY, yield-curve slope, z-scores,
         VIX/credit percentiles) and an interpretable Growth/Inflation/Risk regime.
      4) Optionally forward-fill to daily at the end if output_freq='D'.
      5) Optionally save the result to CSV.

    Args:
        macro_funcs_to_fetch (list[str]): Logical series names to fetch.
        fred_series_ids_dict (dict[str, str]): Map from logical name to FRED series ID.
        start_date (str | None): Start date ('YYYY-MM-DD'); None fetches all available history.
        save_path (str | None): Directory to write 'macros.csv' (if provided).
        output_freq (str): 'M' (default, recommended) or 'D' (daily via forward-fill).
        add_simple_features (bool): If True, adds Tier-1 macro features. Default True.
        add_simple_regime (bool): If True, adds interpretable GIR regime label. Default True.

    Returns:
        pd.DataFrame: Wide, time-indexed DataFrame at the requested frequency with
        raw series plus any added features/regimes, sorted by date.
    """
    print("Starting FRED data orchestration pipeline...")
    frames = []

    # 1) Fetch (keep native freq; most are monthly; some daily)
    for func_name in macro_funcs_to_fetch:  # make sure this is a LIST, not a set
        series_id = fred_series_ids_dict.get(func_name)
        if not series_id:
            print(f"Warning: No FRED series ID for '{func_name}'. Skipping.")
            continue

        print(f"Fetching and processing data for: {func_name} ({series_id})")
        macro_df = FRED_fetch_macro_data(series_id, start_date=start_date)  # expected to return a datetime index, one column
        if macro_df is None or macro_df.empty:
            continue

        # Standardize column name as its series id if needed
        if macro_df.shape[1] == 1 and macro_df.columns[0] != series_id:
            macro_df = macro_df.rename(columns={macro_df.columns[0]: series_id})

        frames.append(macro_df)

    if not frames:
        print("No data fetched.")
        return pd.DataFrame()

    # 2) Merge
    final_df = pd.concat(frames, axis=1).sort_index()

    # 3) Resample to monthly *once* (fast & appropriate for macro)
    #    - For daily series like DFF/DGS10/VIX, 'last' is fine for macro state
    monthly_df = final_df.resample("M").last()

    # 4) Add interpretable features + regimes
    if add_simple_features:
        monthly_df = add_simple_macro_features(monthly_df)
    # if add_simple_regime:
    #     monthly_df = add_simple_regimes(monthly_df)

    # 5) Optional: compute other light features you already had
    monthly_df = add_yield_curve_moments(monthly_df)  # safe – runs on monthly
    # Risk appetite proxies (if you still want your existing version)
    spread_map = {}
    for k, cand in [("HY_OAS", "BAMLH0A0HYM2"), ("IG_OAS", "BAMLC0A0CM"), ("VIX", "VIXCLS")]:
        if cand in monthly_df.columns:
            spread_map[k] = cand
    if spread_map:
        monthly_df = add_risk_appetite_proxies(monthly_df, spread_cols=spread_map)

    monthly_df.dropna(how="all", inplace=True)

    # 6) Output frequency
    if output_freq == "D":
        # Forward-fill monthly features to daily calendar if you truly need daily rows
        # Use the union of original daily indices to avoid creating massive ranges
        daily_index = final_df.index  # contains all original daily points
        out_df = monthly_df.reindex(daily_index, method="ffill")
    else:
        out_df = monthly_df

    # 7) Save
    if save_path:
        out_path = os.path.join(save_path, "macros.csv")
        out_df.to_csv(out_path, index=True)
        print(f"Saved to {out_path}")

    print("Data orchestration complete.")
    return out_df.sort_index()


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
        exp = normalize_date_index(expectations_df, col='Date').sort_values('Date').drop_duplicates('Date').set_index('Date')
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