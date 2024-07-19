# Related third party imports
import pandas as pd
import pytest

# Local application/library specific imports
from app.utils import clean_data


# Sample test data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Tax ID": [12345, None, 56789],
        "Employer": ["amzon", "google", "apple"],
        "State": ["NY", "CA", None],
        "City": ["New York", "San Francisco", None],
        "ZIP": [10001, 94105, None],
        "Fiscal Year": ["2021", "2022", "2023"],
        "Initial Approval": [10, -5, 15],
        "Initial Denial": [1, 2, -1],
        "Continuing Approval": [3, 2, -5],
        "Continuing Denial": [2, -8, 5]
    })


# Test case for basic data cleaning
def test_clean_data_basic(sample_data):
    cleaned_data = clean_data(sample_data)
    
    # Assertions to check basic cleaning operations
    assert "tax_id" in cleaned_data.columns
    assert "employer" in cleaned_data.columns
    assert "state" in cleaned_data.columns
    assert "city" in cleaned_data.columns
    assert "zip" in cleaned_data.columns
    assert "fiscal_year" in cleaned_data.columns
    assert "continuing_approval" in cleaned_data.columns
    assert "continuing_denial" in cleaned_data.columns
    
    # Check for no NaN values in specific columns after cleaning
    assert cleaned_data["tax_id"].isnull().sum() == 0
    assert cleaned_data["employer"].isnull().sum() == 0
    assert cleaned_data["state"].isnull().sum() == 0
    assert cleaned_data["city"].isnull().sum() == 0
    assert cleaned_data["zip"].isnull().sum() == 0
    assert cleaned_data["continuing_approval"].isnull().sum() == 0
    assert cleaned_data["continuing_denial"].isnull().sum() == 0
    
    # Check for correct data types
    assert cleaned_data["zip"].dtype == object  # Since we converted ZIP to string
    assert pd.api.types.is_numeric_dtype(cleaned_data["fiscal_year"])  # Check if Fiscal Year is numeric
    
    # Check if rows with negative Initial Approval or Initial Denial are filtered out
    assert (cleaned_data["initial_approval"] < 0).sum() == 0
    assert (cleaned_data["initial_denial"] < 0).sum() == 0
    
    # Check if only valid Fiscal Years are kept
    valid_fiscal_years = [2021, 2022, 2023]
    assert cleaned_data["fiscal_year"].isin(valid_fiscal_years).all()


# Test case for handling missing columns
def test_clean_data_key_error():
    with pytest.raises(ValueError, match="Missing expected columns"):
        clean_data(pd.DataFrame({"Invalid Column": [1, 2, 3]}))

