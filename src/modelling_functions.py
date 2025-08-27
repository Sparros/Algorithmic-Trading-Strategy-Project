from eli5 import show_weights
from eli5.sklearn import PermutationImportance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

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

    if target_return_col_name in df.columns:
        df.drop(columns=[target_return_col_name], inplace=True)
    
    return df


    """
    Performs a detailed error analysis on the best performing model.
    This function re-runs the data preparation and model fitting steps for the single
    best experiment to provide more detailed insights.
    """
    # 1. Get the parameters of the best experiment
    feat_name = best_experiment['Feature_Set']
    model_name = best_experiment['Model']
    target_type = best_experiment['Target_Type']

    # 2. Re-create the data preparation pipeline for the best experiment
    # Find the corresponding target and model configurations
    target_conf = next(item for item in target_configs if item['type'] == target_type)
    model_conf = next(item for item in model_list if item['name'] == model_name)

    # A. Re-create the target variable
    data_target = data.copy()
    if target_conf['type'] == 'binary':
        data_target = create_target_variable(data_target, target_ticker, window=target_conf['window'], threshold=target_conf['threshold'])
        target_col = f'{target_ticker}_Target'
    elif 'multi_class' in target_conf['type']:
        extreme_threshold = target_conf.get('extreme_threshold', None)
        data_target = create_multi_class_target(data_target, target_ticker, window=target_conf['window'], threshold=target_conf['threshold'], extreme_threshold=extreme_threshold)
        target_col = f'{target_ticker}_Target_Multi'
    else: # regression
        data_target[f'{target_ticker}_target_return'] = data_target[f'Close_{target_ticker}'].pct_change(periods=target_conf['window']).shift(-target_conf['window'])
        target_col = f'{target_ticker}_target_return'

    # Drop rows with NaN in the target
    data_target.dropna(subset=[target_col], inplace=True)

    y_full = data_target[target_col]

    # --- FIX FOR DATA LEAKAGE: APPLY THE SAME DROPPING LOGIC ---
    columns_to_drop = [
        f'Open_{target_ticker}',
        f'High_{target_ticker}',
        f'Low_{target_ticker}',
        f'Close_{target_ticker}',
    ]
    # We also need to drop the return column that was created to prevent leakage
    if target_conf['type'] == 'regression':
        columns_to_drop.append(f'{target_ticker}_target_return')
    else:
        columns_to_drop.append(f'{target_ticker}_target_return_{target_conf["window"]}D')

    X_full = data_target.drop(columns=columns_to_drop + [target_col], errors='ignore')
    # --- END OF FIX ---
    
    X_train_full = X_full.loc[:split_date]
    y_train_full = y_full.loc[:split_date]
    X_test_full = X_full.loc[split_date:]
    y_test_full = y_full.loc[split_date:]

    if target_conf['type'] in ['binary', 'multi_class', 'multi_class_extreme']:
        le = LabelEncoder()
        y_train_full = le.fit_transform(y_train_full)
        y_test_full = le.transform(y_test_full)

    # 3. Apply feature engineering from the best experiment
    X_train_transformed = X_train_full.copy()
    X_test_transformed = X_test_full.copy()
    
    # Replace inf with NaN first
    X_train_transformed.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_transformed.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_transformed))
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_transformed))

    X_train_final = X_train_imputed
    X_test_final = X_test_imputed
    
    if feat_name == 'poly_2':
        transformer = PolynomialFeatures(**feature_configs[feat_name]['params'])
        X_train_final = pd.DataFrame(transformer.fit_transform(X_train_imputed))
        X_test_final = pd.DataFrame(transformer.transform(X_test_imputed))

    elif feat_name == 'pca_3':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        transformer = PCA(**feature_configs[feat_name]['params'])
        X_train_final = pd.DataFrame(transformer.fit_transform(X_train_scaled))
        X_test_final = pd.DataFrame(transformer.transform(X_test_scaled))

    elif feat_name == 'pruned_10':
        # Drop constant features
        variances = X_train_imputed.var()
        constant_columns = variances[variances == 0].index
        if not constant_columns.empty:
            X_train_imputed.drop(columns=constant_columns, inplace=True)
            X_test_imputed.drop(columns=constant_columns, inplace=True)
        
        if analysis_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=feature_configs[feat_name]['params']['n_features_to_select'])
        else:
            selector = SelectKBest(score_func=f_classif, k=feature_configs[feat_name]['params']['n_features_to_select'])
        
        X_train_final = pd.DataFrame(selector.fit_transform(X_train_imputed, y_train_full))
        X_test_final = pd.DataFrame(selector.transform(X_test_imputed))

    # 4. Fit the best model on the full training set and evaluate on test set
    model_initial_params = model_conf['initial_params'].copy()
    sample_weight = None

    if analysis_type == 'classification':
        y_train_full_series = pd.Series(y_train_full)
        class_counts = y_train_full_series.value_counts().sort_index()
        total_samples = class_counts.sum()
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        
        if model_name == 'XGBoost':
            sample_weight = np.array([class_weights[y] for y in y_train_full])
            model_initial_params['eval_metric'] = 'mlogloss'
        elif model_name == 'CatBoost':
            model_initial_params['class_weights'] = [class_weights[cls] for cls in sorted(class_counts.keys())]
        elif model_name == 'RandomForest':
            pass
    
    model = model_conf['class'](**model_conf['Best_Params'], **model_initial_params)
    model.fit(X_train_final, y_train_full, sample_weight=sample_weight)
    y_pred = model.predict(X_test_final)

    # 5. Perform analysis based on model type
    print(f"\n--- Detailed Analysis for {model_name} on {feat_name} ({target_type}) ---")

    if analysis_type == 'classification':
        print("\nClassification Report:")
        print(classification_report(y_test_full, y_pred, zero_division=0))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test_full, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test_full), yticklabels=np.unique(y_test_full))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        try:
            # Re-get the feature names, accounting for feature engineering
            if feat_name == 'baseline':
                feature_names = X_train_full.columns.tolist()
            elif feat_name == 'pca_3':
                feature_names = [f'PCA_Component_{i+1}' for i in range(X_train_final.shape[1])]
            elif feat_name == 'pruned_10':
                 feature_names = [str(col) for col in selector.get_support(indices=True)]
            else:
                 feature_names = [f'feature_{i}' for i in range(X_train_final.shape[1])]

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)
                print("\nTop 10 Feature Importances for Classification Model:")
                print(feature_importance_df.head(10).to_string(index=False))
        except Exception as e:
            print(f"Could not compute feature importances. Error: {e}")

    elif analysis_type == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test_full, y_pred))
        print(f"\nTest RMSE: {rmse:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_full.index, y_test_full, label='True Values', alpha=0.7)
        plt.plot(y_test_full.index, y_pred, label='Predicted Values', alpha=0.7)
        plt.title('Regression Model Predictions vs. True Values')
        plt.xlabel('Date')
        plt.ylabel('Target Return')
        plt.legend()
        plt.show()

    """
    Performs detailed error analysis for a single best model.
    """
    print(f"\nAnalyzing best {analysis_type} model: {best_experiment['Model']} with {best_experiment['Feature_Set']} on {best_experiment['Target_Type']} data.")

    # Get the original data split
    target_conf = next(item for item in target_configs if item['type'] == best_experiment['Target_Type'])
    data_target = data.copy()
    
    # Create the target variable
    if analysis_type == 'classification':
        extreme_threshold = target_conf.get('extreme_threshold', None)
        if target_conf['type'] == 'binary':
            data_target = create_target_variable(data_target, target_ticker, window=target_conf['window'], threshold=target_conf['threshold'])
            target_col = f'{target_ticker}_Target'
        elif 'multi_class' in target_conf['type']:
            data_target = create_multi_class_target(data_target, target_ticker, window=target_conf['window'], threshold=target_conf['threshold'], extreme_threshold=extreme_threshold)
            target_col = f'{target_ticker}_Target_Multi'
    else: # regression
        data_target[f'{target_ticker}_target_return'] = data_target[f'Close_{target_ticker}'].pct_change(periods=target_conf['window']).shift(-target_conf['window'])
        target_col = f'{target_ticker}_target_return'

    data_target.dropna(subset=[target_col], inplace=True)
    
    y_full = data_target[target_col]
    columns_to_drop = [
        f'Open_{target_ticker}',
        f'High_{target_ticker}',
        f'Low_{target_ticker}',
        f'Close_{target_ticker}',
    ]

    if target_conf['type'] == 'regression':
        columns_to_drop.append(f'{target_ticker}_target_return')
    else:
        columns_to_drop.append(f'{target_ticker}_target_return_{target_conf["window"]}D')
    
    # Drop columns explicitly identified and the target column
    X_full = data_target.drop(columns=columns_to_drop + [target_col], errors='ignore')

    X_train_full = X_full.loc[:split_date]
    y_train_full = y_full.loc[:split_date]
    X_test_full = X_full.loc[split_date:]
    y_test_full = y_full.loc[split_date:]
    
    if analysis_type == 'classification':
        le = LabelEncoder()
        y_train_full = le.fit_transform(y_train_full)
        y_test_full = le.transform(y_test_full)

    # Apply feature engineering
    X_train_transformed = X_train_full.copy()
    X_test_transformed = X_test_full.copy()
    
    X_train_transformed.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_transformed.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_transformed), columns=X_train_transformed.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_transformed), columns=X_test_transformed.columns)
    
    X_train_final = X_train_imputed
    X_test_final = X_test_imputed

    feat_conf = feature_configs.get(best_experiment['Feature_Set'])
    if feat_conf and feat_conf['name'] == 'PolynomialFeatures':
        transformer = PolynomialFeatures(**feat_conf['params'])
        X_train_final = pd.DataFrame(transformer.fit_transform(X_train_imputed))
        X_test_final = pd.DataFrame(transformer.transform(X_test_imputed))
    elif feat_conf and feat_conf['name'] == 'PCA':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        transformer = PCA(**feat_conf['params'])
        X_train_final = pd.DataFrame(transformer.fit_transform(X_train_scaled))
        X_test_final = pd.DataFrame(transformer.transform(X_test_scaled))
    
    # Re-train the best model
    model_conf = next(item for item in model_list if item['name'] == best_experiment['Model'])
    model = model_conf['class'](**best_experiment['Best_Params'])

    # Handle class weights for the final model training
    if analysis_type == 'classification' and model_conf['name'] == 'CatBoost':
        class_counts = pd.Series(y_train_full).value_counts().sort_index()
        total_samples = class_counts.sum()
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        model.set_params(class_weights=[class_weights[cls] for cls in sorted(class_weights.keys())])
    elif analysis_type == 'classification' and model_conf['name'] == 'XGBoost':
        class_counts = pd.Series(y_train_full).value_counts().sort_index()
        total_samples = class_counts.sum()
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        sample_weight = np.array([class_weights[y] for y in y_train_full])
        model.fit(X_train_final, y_train_full, sample_weight=sample_weight)
    
    model.fit(X_train_final, y_train_full)
    y_pred = model.predict(X_test_final)

    # Feature Importance Analysis
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_full.columns.tolist()
        if best_experiment['Feature_Set'] == 'poly_2':
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly.fit_transform(X_train_imputed)
            feature_names = poly.get_feature_names_out(X_full.columns)
        
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        print(f"\nTop 10 Feature Importances for {analysis_type.capitalize()} Model:")
        print(feature_importance_df.head(10).to_string())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    if analysis_type == 'classification':
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test_full, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for Best Classification Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    else: # Regression analysis
        # Plot Residuals
        residuals = y_test_full - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title('Residuals Distribution for Best Regression Model')
        plt.xlabel('Residuals (True Value - Predicted Value)')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.title('Residuals Plot for Best Regression Model')
        plt.xlabel('Predicted Value')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()