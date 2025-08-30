from eli5 import show_weights
from eli5.sklearn import PermutationImportance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier

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

