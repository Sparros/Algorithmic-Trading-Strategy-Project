from __future__ import annotations
from eli5 import show_weights
from eli5.sklearn import PermutationImportance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from collections import defaultdict
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.base import clone

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


def get_permutation_importance(model, X_test, y_test, scoring='f1_macro'):
    """
    Calculates and returns a DataFrame of permutation importances.
    
    Args:
        model: The trained model (e.g., RandomForestClassifier instance).
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target.
        scoring (str): The scoring metric to use.
        
    Returns:
        pd.DataFrame: A DataFrame with features and their importance scores.
    """
    # Create the PermutationImportance object
    perm_importance = PermutationImportance(model, random_state=42, scoring=scoring).fit(X_test, y_test)
    
    # Get the feature importances and their standard deviations
    feature_importances = perm_importance.feature_importances_
    feature_importances_std = perm_importance.feature_importances_std_
    
    # Create a DataFrame for easy viewing
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': feature_importances,
        'Std Dev': feature_importances_std
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return importance_df

def create_multi_class_target(data: pd.DataFrame, ticker: str, window: int, threshold: float, extreme_threshold: float = None) -> pd.DataFrame:
    """
    Creates a multi-class target variable for a given stock ticker, with optional
    extreme movement classes.

    Args:
        data (pd.DataFrame): The input DataFrame containing stock price data.
        ticker (str): The stock ticker for which to create the target variable.
        window (int): The number of future days to look ahead for the price change.
        threshold (float): The percentage threshold to define 'up' or 'down' movements.
        extreme_threshold (float, optional): A higher percentage threshold for
                                             'extreme' movements. Defaults to None.

    Returns:
        pd.DataFrame: The original DataFrame with an added multi-class target column.
    """
    df = data.copy()
    
    # Calculate the percentage change of the close price over the specified window
    target_return_col_name = f'{ticker}_target_return_{window}D'
    df[target_return_col_name] = df[f'Close_{ticker}'].pct_change(periods=window).shift(-window)

    target_col_name = f'{ticker}_Target_Multi'
    
    # Initialize all values to 'flat'
    df[target_col_name] = -1

    # Label 'up' and 'down' movements
    df.loc[df[target_return_col_name] > threshold, target_col_name] = 1 # up
    df.loc[df[target_return_col_name] < -threshold, target_col_name] = 0 # down

    # If extreme_threshold is provided, create a five-class target
    if extreme_threshold is not None:
        # Re-initialize to handle the new classes
        df[target_col_name] = 2 # up_movement
        df.loc[(df[target_return_col_name] <= threshold) & (df[target_return_col_name] > -threshold), target_col_name] = 1 # flat
        df.loc[df[target_return_col_name] <= -threshold, target_col_name] = 0 # down_movement
        df.loc[df[target_return_col_name] > extreme_threshold, target_col_name] = 3 # extreme_up
        df.loc[df[target_return_col_name] < -extreme_threshold, target_col_name] = 4 # extreme_down


    return df

def create_target_variable(df, ticker, window=1, threshold=0):
    """
    Creates a target variable based on the cumulative return over a specified window.
    This function is designed to be called outside the main pipeline for tuning.

    Parameters:
    df (pd.DataFrame): The DataFrame with Close prices.
    ticker (str): The stock ticker for the target.
    window (int): The number of days for the return period (e.g., 1 for next day, 5 for a week).
    threshold (float): The minimum return to be considered "up".

    Returns:
    pd.DataFrame: A DataFrame with the new target and target return columns.
    """
    target_return_col_name = f'{ticker}_target_return_{window}D_{threshold}'
    df[target_return_col_name] = df[f'Close_{ticker}'].pct_change(periods=window).shift(-window)
    df[f'{ticker}_Target'] = (df[target_return_col_name] > threshold).astype(int)
    
    return df

def calculate_pnl(y_pred, returns):
    """
    Calculates the PnL for a set of predictions.
    
    Args:
        y_pred (np.array): The model's predictions (0 or 1).
        returns (pd.Series): The actual raw returns for the corresponding period.
    
    Returns:
        float: The total PnL.
    """
    # A positive prediction (1) means we "take the trade" and realize the return.
    # A negative prediction (0) means we "do not take the trade" and get 0 return.
    return (y_pred * returns).sum()

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculates the Sharpe Ratio for a series of returns.
    
    Args:
        returns (pd.Series): The daily returns of a strategy.
        risk_free_rate (float): The daily risk-free rate. Default is 0 for simplicity.
    
    Returns:
        float: The Sharpe Ratio. Returns 0 if standard deviation is 0.
    """
    # Annualize the risk-free rate. Assuming 252 trading days.
    annualized_risk_free_rate = (1 + risk_free_rate)**252 - 1
    
    # Calculate daily excess returns
    excess_returns = returns - risk_free_rate
    
    # Calculate the mean and standard deviation of excess returns
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    # Calculate the annualized Sharpe Ratio
    if std_excess_return == 0:
        return 0
    else:
        # Annualizing is often done by multiplying by the square root of the number of periods
        return (mean_excess_return / std_excess_return) * np.sqrt(252)


def calculate_max_drawdown(returns):
    """
    Calculates the maximum drawdown from a series of returns.
    
    Args:
        returns (pd.Series): The daily returns of a strategy.
        
    Returns:
        float: The maximum drawdown.
    """
    # Calculate the cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate the running maximum (peak) of the cumulative returns
    running_max = cumulative_returns.cummax()
    
    # Calculate the drawdown and find the maximum drawdown
    drawdown = (running_max - cumulative_returns) / running_max
    
    return drawdown.max()

def calculate_permutation_importance(model, X, y):
    """
    Calculates the permutation importance for each feature.
    
    Args:
        model: The trained model.
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.
        
    Returns:
        dict: A dictionary of features and their importance scores.
    """
    # Use scikit-learn's permutation_importance for a model-agnostic approach
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Store the results in a dictionary with feature names
    importance_dict = {}
    for i in result.importances_mean.argsort()[::-1]:
        feature_name = X.columns[i]
        importance_dict[feature_name] = {
            'mean_importance': result.importances_mean[i],
            'std_importance': result.importances_std[i]
        }
    return importance_dict

def calculate_sharpe_with_costs(returns, transaction_cost=0.001, random_state=42):
    """Calculate Sharpe ratio with transaction costs"""
    # Simple approximation: subtract cost for each position change
    positions = (returns != 0).astype(int)
    position_changes = positions.diff().fillna(0).abs()
    cost_adjusted_returns = returns - (position_changes * transaction_cost)
    return calculate_sharpe_ratio(cost_adjusted_returns)

def check_feature_stability(X_data, y_data, feature_names, n_periods=4, random_state=42):
    """Check feature importance stability across time periods"""
    print("\n--- Feature Stability Analysis ---")
    
    # Split data into periods
    period_size = len(X_data) // n_periods
    stability_scores = defaultdict(list)
    
    for i in range(n_periods - 1):  # Need overlap for comparison
        start_idx = i * period_size
        end_idx = (i + 2) * period_size
        
        if end_idx > len(X_data):
            break
            
        X_period = X_data.iloc[start_idx:end_idx]
        y_period = y_data.iloc[start_idx:end_idx]
        
        # Quick XGBoost to get feature importance
        temp_model = XGBClassifier(n_estimators=100, random_state=random_state)
        temp_model.fit(X_period, y_period)
        
        importance = temp_model.feature_importances_
        
        # Store top 10 features for this period
        top_features_idx = np.argsort(importance)[-10:]
        for idx in top_features_idx:
            stability_scores[feature_names[idx]].append(importance[idx])
    
    # Calculate stability (coefficient of variation)
    stable_features = []
    for feature, scores in stability_scores.items():
        if len(scores) > 1:
            cv = np.std(scores) / (np.mean(scores) + 1e-8)
            if cv < 0.5:  # Threshold for stability
                stable_features.append((feature, cv))
    
    stable_features.sort(key=lambda x: x[1])
    print(f"Most stable features (CV < 0.5):")
    for feature, cv in stable_features[:10]:
        print(f"  {feature}: CV = {cv:.3f}")
    
    return [f[0] for f in stable_features[:20]]  # Return top 20 stable features

def detect_market_regimes(returns, window=252):
    """Simple regime detection based on volatility"""
    rolling_vol = returns.rolling(window).std()
    vol_median = rolling_vol.median()
    
    regimes = pd.Series(index=returns.index, dtype=int)
    regimes[rolling_vol <= vol_median * 0.8] = 0  # Low volatility
    regimes[(rolling_vol > vol_median * 0.8) & (rolling_vol < vol_median * 1.2)] = 1  # Normal
    regimes[rolling_vol >= vol_median * 1.2] = 2  # High volatility
    
    return regimes.fillna(1)

def regime_aware_validation(X, y, returns, regimes, model_pipeline, transaction_cost=0.001):
    """Test model performance across different market regimes"""
    print("\n--- Regime-Aware Validation ---")
    
    regime_performance = {}
    unique_regimes = regimes.unique()
    
    for regime in unique_regimes:
        regime_mask = regimes == regime
        if regime_mask.sum() < 50:  # Skip if too few samples
            continue
            
        X_regime = X[regime_mask]
        y_regime = y[regime_mask]
        returns_regime = returns[regime_mask]
        
        # Use TimeSeriesSplit for this regime
        tscv = TimeSeriesSplit(n_splits=2)
        sharpe_scores = []
        
        for train_idx, test_idx in tscv.split(X_regime):
            if len(test_idx) < 10:  # Skip if test set too small
                continue
                
            X_train, X_test = X_regime.iloc[train_idx], X_regime.iloc[test_idx]
            y_train, y_test = y_regime.iloc[train_idx], y_regime.iloc[test_idx]
            returns_test = returns_regime.iloc[test_idx]
            
            model_pipeline.fit(X_train, y_train)
            preds = model_pipeline.predict(X_test)
            strategy_returns = preds * returns_test
            sharpe = calculate_sharpe_with_costs(strategy_returns, transaction_cost)
            sharpe_scores.append(sharpe)
        
        if sharpe_scores:
            regime_names = {0: 'Low Vol', 1: 'Normal Vol', 2: 'High Vol'}
            regime_performance[regime] = np.mean(sharpe_scores)
            print(f"  {regime_names.get(regime, f'Regime {regime}')}: Sharpe = {np.mean(sharpe_scores):.3f}")
    
    return regime_performance

"""
Implements (Lopez de Prado style):
- Daily volatility estimator (EWMA)
- Vertical barrier placement (H bars ahead)
- Triple-barrier labeling with optional SIDE (for meta-labeling)
- Helper to build primary (ternary) labels and meta labels
- Sample-weight helper (useful for class imbalance / larger targets)

Notes on leakage safety:
- Labels only use future prices between event time `t` and its vertical barrier `t1`.
- You must DROP the last `H` rows of features before any split or training,
  because those rows don't have a complete label window yet.
- For meta-labeling, the base rule must be computable at time `t` using only
  info available up to the decision time (e.g., close[t]).
"""
# ---------------------------------------------------------------------------
# Volatility & barrier utilities
# ---------------------------------------------------------------------------

def get_daily_vol(close: pd.Series, span: int = 20, min_periods: Optional[int] = None) -> pd.Series:
    """Estimate daily volatility via EWMA of log returns.

    Parameters
    ----------
    close : pd.Series
        Price series indexed by timestamp (or integer index).
    span : int
        EWM span for std of log-returns (default 20).
    min_periods : Optional[int]
        Minimum periods required for a valid value; defaults to `span`.

    Returns
    -------
    pd.Series
        Estimated daily volatility (in return space), aligned with `close`.
    """
    if min_periods is None:
        min_periods = span
    logret = np.log(close).diff()
    vol = logret.ewm(span=span, min_periods=min_periods).std()
    return vol.abs()


def get_vertical_barriers(index: pd.Index, t_events: Iterable[pd.Timestamp], H: int) -> pd.Series:
    """Return the vertical barrier (t1) for each event, H bars into the future.

    If the barrier would exceed the available index, it is set to the last index value.

    Parameters
    ----------
    index : pd.Index
        Full index of the price series (must contain all t_events).
    t_events : Iterable[pd.Timestamp]
        Event times (subset of index) at which to start labeling windows.
    H : int
        Number of bars ahead to place the vertical time barrier.

    Returns
    -------
    pd.Series
        Mapping from event time -> barrier time (t1).
    """
    index = pd.Index(index)
    iloc_map = {ts: i for i, ts in enumerate(index)}
    t1 = {}
    last_i = len(index) - 1
    for t in t_events:
        i0 = iloc_map.get(t)
        if i0 is None:
            continue
        i1 = min(i0 + H, last_i)
        t1[t] = index[i1]
    return pd.Series(t1)


@dataclass
class Events:
    """Container for event specifications used by triple-barrier.

    Attributes
    ----------
    t1 : pd.Series
        Vertical barrier times per event time (index of this Series is event times).
    trgt : pd.Series
        Per-event volatility/target (e.g., daily vol at event time). Must align on index of events.
    side : Optional[pd.Series]
        Direction proposed by a base rule (+1 long, -1 short). If None, labeling is symmetric.
    """
    t1: pd.Series
    trgt: pd.Series
    side: Optional[pd.Series] = None


# ---------------------------------------------------------------------------
# Triple-barrier labeling
# ---------------------------------------------------------------------------

def apply_triple_barrier(
    close: pd.Series,
    events: Events,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
) -> pd.DataFrame:
    """Apply profit-take / stop-loss / vertical barriers and return labels.

    For each event time t, we look forward until t1[t] and inspect the path of
    *side-adjusted* log-returns r(t->u) = side * log(close[u]/close[t]) if side is given,
    else r(t->u) = log(close[u]/close[t]).

    The first time this path crosses +pt (profit) or -sl (stop) determines the label.
    If neither is crossed before t1, outcome is whatever the sign of r(t->t1) is; we also
    record the barrier type ('pt','sl','tl').

    Returns a DataFrame indexed by event times with columns:
        - 't1': barrier time actually used (could be t1 or earlier if hit)
        - 'ret': realized side-adjusted log-return from t to the first barrier (or t1)
        - 'type': which barrier was hit: 'pt', 'sl', or 'tl' (time limit)
        - 'bin': label in {-1, 0, +1} (sign(ret))
    """
    close = close.astype(float)
    idx = close.index
    # Align series
    t1 = events.t1.dropna().copy()
    trgt = events.trgt.reindex(t1.index).astype(float)
    if events.side is not None:
        side = events.side.reindex(t1.index).astype(float)
    else:
        side = pd.Series(1.0, index=t1.index)

    pt_mult, sl_mult = pt_sl

    iloc = {ts: i for i, ts in enumerate(idx)}

    out = []
    for t in t1.index:
        i0 = iloc.get(t)
        i1 = iloc.get(t1.loc[t])
        if i0 is None or i1 is None or i1 <= i0:
            continue
        p0 = close.iloc[i0]
        path = close.iloc[i0 + 1 : i1 + 1]
        # side-adjusted log returns along the path
        r_path = side.loc[t] * np.log(path / p0)
        pt = pt_mult * trgt.loc[t]
        sl = -sl_mult * trgt.loc[t]
        # First touches
        hit_pt = np.where(r_path.values >= pt)[0]
        hit_sl = np.where(r_path.values <= sl)[0]
        hit_type = 'tl'
        j = None
        if hit_pt.size and hit_sl.size:
            j = min(hit_pt[0], hit_sl[0])
            hit_type = 'pt' if hit_pt[0] < hit_sl[0] else 'sl'
        elif hit_pt.size:
            j = hit_pt[0]
            hit_type = 'pt'
        elif hit_sl.size:
            j = hit_sl[0]
            hit_type = 'sl'

        if j is not None:
            ti = path.index[j]
            ret = r_path.iloc[j]
        else:
            ti = t1.loc[t]
            ret = side.loc[t] * np.log(close.loc[ti] / p0)
        out.append((t, ti, float(ret), hit_type))

    res = pd.DataFrame(out, columns=['t', 't1', 'ret', 'type']).set_index('t')
    res['bin'] = np.sign(res['ret']).astype(int)
    # Zero very small returns to avoid noisy labels
    res.loc[res['ret'].abs() < 1e-12, 'bin'] = 0
    return res


# ---------------------------------------------------------------------------
# High-level helpers to build labels (primary + meta)
# ---------------------------------------------------------------------------

def build_primary_labels(
    close: pd.Series,
    span_vol: int = 20,
    H: int = 10,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    min_ret: float = 0.0,
    t_events: Optional[pd.Index] = None,
) -> Tuple[pd.DataFrame, Events]:
    """Convenience wrapper to generate triple-barrier labels (no side).

    Returns
    -------
    bins : pd.DataFrame
        Output of `apply_triple_barrier` with columns ['t1','ret','type','bin'] indexed by event times.
    events : Events
        The Events object used to generate `bins` (t1, trgt, side=None).
    """
    close = close.dropna().astype(float)
    vol = get_daily_vol(close, span=span_vol)
    if t_events is None:
        # By default, create an event at every time where vol is known
        t_events = vol.dropna().index
    t1 = get_vertical_barriers(close.index, t_events, H)
    trgt = vol.reindex(t1.index)
    # filter on minimum return target if desired
    if min_ret > 0:
        keep = trgt[trgt > min_ret].index
        t1 = t1.reindex(keep)
        trgt = trgt.reindex(keep)
    ev = Events(t1=t1, trgt=trgt, side=None)
    bins = apply_triple_barrier(close=close, events=ev, pt_sl=pt_sl)
    return bins, ev


def build_meta_labels(
    close: pd.Series,
    base_side: pd.Series,
    span_vol: int = 20,
    H: int = 10,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    min_ret: float = 0.0,
    t_events: Optional[pd.Index] = None,
) -> Tuple[pd.DataFrame, Events]:
    """Generate meta-labels using a provided base rule `base_side`.

    The base rule should be a Series indexed like `close` with values in {+1, -1, 0} where
    0 means "no trade". We'll create events only where side!=0.

    Returns
    -------
    meta : pd.DataFrame
        Index: event times (subset where base_side != 0 & trgt>min_ret).
        Columns:
            - 't1', 'ret', 'type' (from triple barrier on side-adjusted returns)
            - 'y': binary meta-label: 1 if taking the base trade would be profitable, else 0.
            - 'side': base side (+1/-1) used for each event (for use at inference time)
    events : Events
        The Events object used (includes side).
    """
    close = close.dropna().astype(float)
    base_side = base_side.reindex(close.index).fillna(0.0)
    # consider only times where base rule signals a trade
    t_events_all = base_side[base_side != 0].index
    vol = get_daily_vol(close, span=span_vol)
    if t_events is None:
        t_events = t_events_all.intersection(vol.dropna().index)
    else:
        t_events = pd.Index(t_events).intersection(t_events_all)

    t1 = get_vertical_barriers(close.index, t_events, H)
    trgt = vol.reindex(t1.index)
    # optional min target
    if min_ret > 0:
        keep = trgt[trgt > min_ret].index
        t1 = t1.reindex(keep)
        trgt = trgt.reindex(keep)

    side = base_side.reindex(t1.index).astype(float)
    ev = Events(t1=t1, trgt=trgt, side=side)
    bins = apply_triple_barrier(close=close, events=ev, pt_sl=pt_sl)
    meta = bins.copy()
    # meta-label: was the side-adjusted outcome positive?
    meta['y'] = (meta['ret'] > 0).astype(int)
    meta['side'] = side.reindex(meta.index)
    return meta, ev


# ---------------------------------------------------------------------------
# Sample weights & dataset builders
# ---------------------------------------------------------------------------

def sample_weights_from_trgt(trgt: pd.Series, power: float = 1.0) -> pd.Series:
    """Sample weights proportional to the per-event target magnitude.

    Often useful to give more weight to higher-volatility events.
    """
    w = trgt.abs().pow(power)
    w = w / (w.mean() if w.mean() != 0 else 1.0)
    return w


def build_datasets_for_models(
    X: pd.DataFrame,
    close: pd.Series,
    *,
    span_vol: int = 20,
    H: int = 10,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    min_ret: float = 0.0,
    base_side: Optional[pd.Series] = None,
) -> dict:
    """Produce ready-to-train datasets for primary and (optional) meta models.

    Returns a dict with some of the following keys:
        - 'primary_X', 'primary_y', 'primary_weights', 'primary_t1'
        - 'meta_X', 'meta_y', 'meta_side', 'meta_weights', 'meta_t1'

    The function drops the last H rows of X/close to avoid look-ahead.
    """
    # Drop the last H rows because their labels will use data beyond available features
    if H > 0:
        X = X.iloc[:-H]
        close_trunc = close.iloc[:-H]
    else:
        close_trunc = close

    out = {}

    # Primary labels (symmetric; trade when bin != 0)
    bins, ev = build_primary_labels(
        close=close_trunc, span_vol=span_vol, H=H, pt_sl=pt_sl, min_ret=min_ret
    )
    # Only keep timestamps we have features for
    idx_keep = bins.index.intersection(X.index)
    bins = bins.loc[idx_keep]
    # Train only on decisive events (bin != 0)
    primary_mask = bins['bin'] != 0
    out['primary_X'] = X.loc[idx_keep][primary_mask]
    out['primary_y'] = bins.loc[primary_mask, 'bin']  # -1 / +1
    out['primary_weights'] = sample_weights_from_trgt(ev.trgt.reindex(idx_keep)[primary_mask])
    out['primary_t1'] = bins.loc[primary_mask, 't1']

    # Meta labels (if a base rule is supplied)
    if base_side is not None:
        meta, evm = build_meta_labels(
            close=close_trunc,
            base_side=base_side,
            span_vol=span_vol,
            H=H,
            pt_sl=pt_sl,
            min_ret=min_ret,
        )
        idx_keep_m = meta.index.intersection(X.index)
        meta = meta.loc[idx_keep_m]
        out['meta_X'] = X.loc[idx_keep_m]
        out['meta_y'] = meta['y']  # 1 = take trade, 0 = skip
        out['meta_side'] = meta['side']  # store the direction for use at inference time
        out['meta_weights'] = sample_weights_from_trgt(evm.trgt.reindex(idx_keep_m))
        out['meta_t1'] = meta['t1']

    return out


# ---------------------------------------------------------------------------
# Example base rules (you can replace with your own)
# ---------------------------------------------------------------------------

def base_rule_ma_cross(close: pd.Series, fast: int = 10, slow: int = 20) -> pd.Series:
    """Simple moving-average cross base rule.

    Returns a Series in {+1, -1, 0} with +1 when fast crosses above slow, -1 when fast
    crosses below slow, 0 otherwise. Computed using ONLY close up to time t (no look-ahead).
    """
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    side = pd.Series(0.0, index=close.index)
    # Signals only at cross points (reduce turnover for meta)
    cross_up = (f > s) & (f.shift(1) <= s.shift(1))
    cross_dn = (f < s) & (f.shift(1) >= s.shift(1))
    side[cross_up] = 1.0
    side[cross_dn] = -1.0
    return side


# ---------------------------------------------------------------------------
# Inference-time helper for meta strategy
# ---------------------------------------------------------------------------

def meta_decision(
    proba: np.ndarray,
    side: pd.Series,
    p_star: float = 0.55,
) -> pd.Series:
    """Turn meta model probabilities into trade decisions.

    Parameters
    ----------
    proba : np.ndarray
        Meta model predicted probability of y=1 (trade is profitable) for each event (aligned with `side`).
    side : pd.Series
        Base side per event (+1/-1), index must align with the order of `proba`.
    p_star : float
        Minimum probability to accept the trade.

    Returns
    -------
    pd.Series
        Signal in {-1, 0, +1}: take trade in `side` direction if p>=p_star, else 0.
    """
    s = pd.Series(0, index=side.index)
    s[proba >= p_star] = side[proba >= p_star].astype(int)
    return s

def best_threshold(proba, side, close_series, idx_train):
    grid = np.linspace(0.50, 0.70, 9)
    best_s, best_p = -1e9, 0.55
    daily_ret = close_series.pct_change().shift(-1).reindex(idx_train).fillna(0)
    for p in grid:
        sig = meta_decision(proba, side=side, p_star=p)
        sig_d = sig.reindex(idx_train).fillna(0).astype(float)
        s = calculate_sharpe_with_costs(sig_d * daily_ret, 0.001)
        if s > best_s: best_s, best_p = s, p
    return best_p

def purged_time_series_splits(X_index, t1, n_splits=3, test_size=None, embargo=5):
    """
    Yield (train_idx, test_idx) with:
      - purge: drop train events whose [t, t1] window overlaps the test window
      - embargo: drop train events for `embargo` bars after test window
    X_index : DatetimeIndex (event timestamps of X)
    t1      : pd.Series mapping event time -> vertical barrier time
    """
    N = len(X_index)
    all_idx = np.arange(N)

    # choose a fixed test size if not given (≈ last 20% per fold)
    if test_size is None:
        test_size = max(32, int(0.2 * N))

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    for tr, te in tscv.split(all_idx):
        te_start, te_end = X_index[te[0]], X_index[te[-1]]
        embargo_end_pos = min(te[-1] + embargo, N - 1)
        embargo_end_ts = X_index[embargo_end_pos]

        keep = []
        for i in tr:
            ti = X_index[i]
            ti1 = t1.get(ti, None)
            # purge if event window touches the test window
            overlap = (ti1 is not None) and (ti <= te_end) and (ti1 >= te_start)
            # embargo: drop anything in [te_start, embargo_end_ts]
            in_embargo = (ti >= te_start) and (ti <= embargo_end_ts)
            if not overlap and not in_embargo:
                keep.append(i)
        yield np.array(keep, dtype=int), te

def build_tb_meta_pipeline_xgb(params: dict, selector_threshold: str = "median") -> Pipeline:
    """
    Create a leakage-safe sklearn Pipeline for tb_meta with XGBClassifier.
    - SimpleImputer(mean)
    - SelectFromModel(using a base XGB with same params)
    - XGBClassifier(**params)
    """
    base_est = XGBClassifier(**params)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("selector", SelectFromModel(base_est, threshold=selector_threshold)),
        ("model", XGBClassifier(**params))
    ])
    return pipe


def tb_meta_fold_sharpe(
    pipe: Pipeline,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    side_tr: pd.Series,
    side_va: pd.Series,
    close: pd.Series,
    transaction_cost: float,
    sample_weight_tr: Optional[np.ndarray] = None,
    calibrate: bool = True,
    p_star: Optional[float] = None,
):
    """
    Fit a pipeline on (X_tr, y_tr), calibrate probabilities (optional), choose p_star on train
    if not provided (via best_threshold), then compute daily Sharpe on the validation window.
    """
    fit_kwargs = {}
    if sample_weight_tr is not None:
        fit_kwargs["model__sample_weight"] = sample_weight_tr

    pipe.fit(X_tr, y_tr, **fit_kwargs)
    model = pipe

    if calibrate:
        cal = CalibratedClassifierCV(base_estimator=clone(pipe), method="isotonic", cv=3)
        cal.fit(X_tr, y_tr)  # pass sample_weight here if you need it
        model = cal

    proba_tr = model.predict_proba(X_tr)[:, 1]
    if p_star is None:
        p_star = best_threshold(proba_tr, side_tr, close, X_tr.index)

    proba_va = model.predict_proba(X_va)[:, 1]
    signals = meta_decision(proba_va, side=side_va, p_star=p_star)

    if signals.empty:
        return 0.0

    test_days = close.loc[signals.index.min():signals.index.max()].index
    daily_signal = signals.reindex(test_days).fillna(0).astype(float)
    daily_ret = close.pct_change().shift(-1).reindex(test_days).fillna(0.0)
    pnl = daily_signal * daily_ret

    return float(calculate_sharpe_with_costs(pnl, transaction_cost=transaction_cost))


def make_tb_meta_objective_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    side: pd.Series,
    t1: pd.Series,
    close: pd.Series,
    H: int,
    transaction_cost: float,
    sampler_seed: int = 42,
    use_selector: bool = True,
):
    """
    Build an Optuna objective(Function) for tb_meta + XGBoost.
    Uses purged/embargoed CV splits based on t1 and embargo=H.
    Returns average Sharpe across folds.
    """
    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 20.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "eval_metric": "logloss",
            "random_state": sampler_seed,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        selector_threshold = trial.suggest_categorical(
            "selector_threshold", ["median", "mean", "0.75*mean"]
        ) if use_selector else None

        pipe = build_tb_meta_pipeline_xgb(params, selector_threshold or "median")

        # Purged/embargoed splits
        splits = list(purged_time_series_splits(
            X.index, t1.reindex(X.index), n_splits=3, test_size=None, embargo=H
        ))

        scores = []
        for tr, va in splits:
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            side_tr, side_va = side.iloc[tr], side.iloc[va]

            score = tb_meta_fold_sharpe(
                pipe=pipe,
                X_tr=X_tr, y_tr=y_tr,
                X_va=X_va, y_va=y_va,
                side_tr=side_tr, side_va=side_va,
                close=close,
                transaction_cost=transaction_cost,
                sample_weight_tr=None,  # pass if you have event weights
                calibrate=True,
                p_star=None
            )
            scores.append(score)

        return float(np.nanmean(scores)) if scores else -10.0

    return objective


def fit_tb_meta_final_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    side_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    side_test: pd.Series,
    close: pd.Series,
    transaction_cost: float,
    best_params: dict,
    selector_threshold: str = "median",
    sample_weight_train: Optional[np.ndarray] = None,
    calibrate: bool = True,
) -> tuple:
    """
    Train the final tb_meta pipeline on training data.
    Returns (fitted_model, p_star, test_sharpe, event_signals_test)
    """
    pipe = build_tb_meta_pipeline_xgb(best_params, selector_threshold)

    fit_kwargs = {}
    if sample_weight_train is not None:
        fit_kwargs["model__sample_weight"] = sample_weight_train

    pipe.fit(X_train, y_train, **fit_kwargs)
    model = pipe

    if calibrate:
        cal = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
        cal.fit(X_train, y_train, sample_weight=sample_weight_train)
        model = cal

    proba_train = model.predict_proba(X_train)[:, 1]
    p_star = best_threshold(proba_train, side_train, close, X_train.index)

    proba_test = model.predict_proba(X_test)[:, 1]
    signals = meta_decision(proba_test, side=side_test, p_star=p_star)

    if signals.empty:
        return model, p_star, float("nan"), signals

    test_days = close.loc[signals.index.min():signals.index.max()].index
    daily_signal = signals.reindex(test_days).fillna(0).astype(float)
    daily_ret = close.pct_change().shift(-1).reindex(test_days).fillna(0.0)
    pnl = daily_signal * daily_ret
    sharpe = float(calculate_sharpe_with_costs(pnl, transaction_cost=transaction_cost))

    return model, p_star, sharpe, signals

def pick_split_by_event_count(X_events, train_frac=0.75, val_frac=0.15):
    """
    Choose train_end and val_end by event count.
    E.g., 75% train, 15% val, 10% test.
    Returns strings you can drop into CONFIG or use directly.
    """
    idx = X_events.index.sort_values()
    n = len(idx)
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1

    train_end = idx[int(train_frac * n) - 1]
    val_end   = idx[int((train_frac + val_frac) * n) - 1]
    return str(train_end.date()), str(val_end.date())

def time_blocks_with_purge_embargo(n: int, n_splits: int = 5, purge: int = 5, embargo: int = 5):
    """
    Generator of (train_idx, test_idx) for ordered data [0..n-1] with purge+embargo.
    Use in modelling stage for Lopez de Prado style CV.
    """
    fold = n // n_splits
    for i in range(n_splits):
        test_start = i * fold
        test_end   = (i+1) * fold if i < n_splits - 1 else n
        train_end  = max(0, test_start - purge)
        train_idx  = np.arange(0, train_end)
        test_idx   = np.arange(test_start, test_end)
        # embargo after test
        emb_start  = min(n, test_end + embargo)
        train_idx  = np.concatenate([train_idx, np.arange(emb_start, n)])
        yield train_idx, test_idx