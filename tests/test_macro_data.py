# tests/test_macro_data.py
import pytest
import pandas as pd
from unittest.mock import patch, Mock

from src.macro_data import fetch_fundamental_data, fundamental_metrics, fetch_macro_data_orchestrator

@pytest.fixture
def mock_income_statement_response():
    """Mock API response for INCOME_STATEMENT."""
    return {
        "quarterlyReports": [
            {"fiscalDateEnding": "2025-04-30", "netIncome": "1000", "grossProfit": "5000", "totalRevenue": "10000"},
            {"fiscalDateEnding": "2025-01-31", "netIncome": "900", "grossProfit": "4800", "totalRevenue": "9500"},
        ]
    }

@pytest.fixture
def mock_balance_sheet_response():
    """Mock API response for BALANCE_SHEET."""
    return {
        "quarterlyReports": [
            {"fiscalDateEnding": "2025-04-30", "totalDebt": "5000", "totalEquity": "10000", "weightedAverageSharesOutstanding": "100"},
            {"fiscalDateEnding": "2025-01-31", "totalDebt": "4500", "totalEquity": "9500", "weightedAverageSharesOutstanding": "90"},
        ]
    }

@pytest.fixture
def mock_gdp_response():
    """Mock API response for REAL_GDP."""
    return {
        'data': [
            {'date': '2025-03-31', 'value': '22.0'},
            {'date': '2024-12-31', 'value': '21.5'},
            {'date': '2024-09-30', 'value': '21.2'}
        ]
    }

def test_fetch_fundamental_data_success(mock_income_statement_response):
    """Test if fetch_fundamental_data returns a valid DataFrame on success."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_income_statement_response
        df = fetch_fundamental_data("INCOME_STATEMENT", "TEST")
        assert not df.empty
        assert "netIncome" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df.index)

def test_fundamental_metrics_calculation():
    """Test if fundamental_metrics correctly computes derived columns."""
    # Create a raw DataFrame with columns from both Income Statement and Balance Sheet
    df = pd.DataFrame({
        "fiscalDateEnding": ["2025-04-30", "2025-01-31"],
        "netIncome": ["1000", "900"],
        "totalRevenue": ["10000", "9500"],
        "totalDebt": ["5000", "4500"],
        "totalEquity": ["10000", "9500"],
        "weightedAverageSharesOutstanding": ["100", "90"]
    }).set_index("fiscalDateEnding")
    
    # Convert to numeric first, as the orchestrator would
    cols_to_convert = [col for col in df.columns if col not in ['reportedCurrency', 'fiscalDateEnding']]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    processed_df = fundamental_metrics(df)
    
    assert "eps" in processed_df.columns
    assert "pe_ratio" in processed_df.columns
    assert "debt_to_equity" in processed_df.columns
    assert processed_df.loc["2025-04-30", "eps"] == 10.0
    assert processed_df.loc["2025-04-30", "pe_ratio"] == 1000.0
    assert processed_df.loc["2025-04-30", "debt_to_equity"] == 0.5

@patch("src.macro_data.fetch_general_macro_data")
@patch("src.macro_data.fetch_fundamental_data")
def test_fetch_macro_data_orchestrator(mock_fetch_fundamental, mock_fetch_macro):
    """Test if the orchestrator correctly fetches, processes, and merges all data."""
    # Mock the return values for the sub-functions
    mock_fetch_fundamental.side_effect = [
        pd.DataFrame({
            "fiscalDateEnding": ["2025-04-30", "2025-01-31"],
            "netIncome": ["1000", "900"],
            "totalRevenue": ["10000", "9500"]
        }).set_index("fiscalDateEnding"),
        pd.DataFrame({
            "fiscalDateEnding": ["2025-04-30", "2025-01-31"],
            "totalDebt": ["5000", "4500"],
            "totalEquity": ["10000", "9500"],
            "weightedAverageSharesOutstanding": ["100", "90"]
        }).set_index("fiscalDateEnding")
    ]
    
    mock_fetch_macro.return_value = pd.DataFrame({
        "date": ["2025-03-31", "2024-12-31"],
        "REAL_GDP": [22.0, 21.5]
    }).set_index("date")

    combined_df = fetch_macro_data_orchestrator(
        "TEST",
        fundamental_funcs_to_fetch=['INCOME_STATEMENT', 'BALANCE_SHEET'],
        macro_funcs_to_fetch=['REAL_GDP']
    )
    
    assert not combined_df.empty
    assert "eps" in combined_df.columns
    assert "REAL_GDP" in combined_df.columns
    assert pd.api.types.is_datetime64_any_dtype(combined_df.index)
    
    # Check that the number of rows is greater than the original data points
    # due to resampling, and that the columns are all present.
    assert len(combined_df) > 5