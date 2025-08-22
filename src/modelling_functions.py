from eli5 import show_weights
from eli5.sklearn import PermutationImportance
import pandas as pd
import numpy as np

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

def create_multi_class_target(data: pd.DataFrame, ticker: str, window: int, threshold: float) -> pd.DataFrame:
    """
    Creates a multi-class target variable for a given stock ticker.

    The target variable is based on the future percentage change of the stock's closing price.
    It returns a DataFrame with the new target column.

    Args:
        data (pd.DataFrame): The input DataFrame containing stock price data.
        ticker (str): The stock ticker for which to create the target variable.
        window (int): The number of future days to look ahead for the price change.
        threshold (float): The percentage threshold to define 'up' or 'down' movements.
                           (e.g., 0.01 for a 1% movement).

    Returns:
        pd.DataFrame: The original DataFrame with an added multi-class target column.
    """
    df = data.copy()
    
    # Calculate the percentage change of the close price over the specified window
    target_return_col_name = f'{ticker}_target_return_{window}D_{threshold}'
    df[target_return_col_name] = df[f'Close_{ticker}'].pct_change(periods=window).shift(-window)

    # Define the multi-class target based on the thresholds
    target_col_name = f'{ticker}_Target_Multi'
    
    # Create a new column initialized to 'flat' (or 0)
    df[target_col_name] = 0  # 0 for 'flat'

    # Label 'up' movements
    df.loc[df[target_return_col_name] > threshold, target_col_name] = 1 # 1 for 'up'

    # Label 'down' movements
    df.loc[df[target_return_col_name] < -threshold, target_col_name] = -1 # -1 for 'down'

    return df