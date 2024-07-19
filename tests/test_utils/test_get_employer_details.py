# Related third party imports
import pandas as pd
import pytest

# Local application/library specific imports
from app.utils import get_employer_details


# Test case 1: Basic functionality
def test_get_employer_details_basic():
    # Create a DataFrame 'data' with sample H-1B visa application data
    data = pd.DataFrame({
        "employer": ["Company A", "Company A", "Company A", "Company B"],
        "fiscal_year": [2021, 2022, 2023, 2023],
        "initial_approval": [1, 0, 1, 1],
        "initial_denial": [0, 1, 0, 0],
        "state": ["CA", "CA", "NY", "TX"],
        "city": ["Los Angeles", "San Francisco", "New York", "Houston"]
    })
    
    # Define expected output dictionary with aggregated metrics
    expected_output = {
        "total_applications": 3,
        "applications_2021": 1,
        "applications_2022": 1,
        "applications_2023": 1,
        "total_initial_approvals": 2,
        "initial_approvals_2021": 1,
        "initial_approvals_2022": 0,
        "initial_approvals_2023": 1,
        "total_initial_denials": 1,
        "state_highest_applications": "CA",
        "city_highest_applications": "Los Angeles",
        "state_highest_approvals": "CA",
        "city_highest_approvals": "Los Angeles",
    }
    
    # Call function 'get_employer_details' to get actual 
    # output based on 'data' for "Company A"
    actual_output = get_employer_details(data, "Company A")
    # Assert that the actual output matches the expected output
    assert actual_output == expected_output


# Test case 2: Edge case - employer not found
def test_get_employer_details_employer_not_found():
    # Create a DataFrame 'data' with sample H-1B visa 
    # application data for two companies
    data = pd.DataFrame({
        "employer": ["Company A", "Company B"],
        "fiscal_year": [2021, 2022],
        "initial_approval": [1, 0],
        "initial_denial": [0, 1],
        "state": ["CA", "TX"],
        "city": ["Los Angeles", "Houston"]
    })
    
    # Define expected output dictionary with all values 
    # set to 0, indicating no data found for "Company C"
    expected_output = {
        "total_applications": 0,
        "applications_2021": 0,
        "applications_2022": 0,
        "applications_2023": 0,
        "total_initial_approvals": 0,
        "initial_approvals_2021": 0,
        "initial_approvals_2022": 0,
        "initial_approvals_2023": 0,
        "total_initial_denials": 0,
        "state_highest_applications": 0,
        "city_highest_applications": 0,
        "state_highest_approvals": 0,
        "city_highest_approvals": 0,
    }
    
    # Assert that calling 'get_employer_details' 
    # with "Company C" returns the expected output dictionary
    assert get_employer_details(data, "Company C") == expected_output


# Test case 3: Validation error - initial approvals exceed total applications
def test_get_employer_details_validation_error_initial_approvals():
    data = pd.DataFrame({
        "employer": ["Company A", "Company A"],
        "fiscal_year": [2021, 2021],
        "initial_approval": [2, 2],
        "initial_denial": [0, 0],
        "state": ["CA", "CA"],
        "city": ["Los Angeles", "San Francisco"]
    })
    
    with pytest.raises(
        ValueError, 
        match=r"H-1B Visa Details for Company A cannot be retrieved due to insufficient data."):
        get_employer_details(data, "Company A")


# Test case 4: Handling empty Series for mode
def test_get_employer_details_empty_series():
    data = pd.DataFrame({
        "employer": ["Company A", "Company A"],
        "fiscal_year": [2021, 2021],
        "initial_approval": [1, 1],
        "initial_denial": [0, 0],
        "state": ["CA", "CA"],
        "city": ["Los Angeles", "San Francisco"]
    })
    
    expected_output = {
        "total_applications": 2,
        "applications_2021": 2,
        "applications_2022": 0,
        "applications_2023": 0,
        "total_initial_approvals": 2,
        "initial_approvals_2021": 2,
        "initial_approvals_2022": 0,
        "initial_approvals_2023": 0,
        "total_initial_denials": 0,
        "state_highest_applications": "CA",
        "city_highest_applications": "Los Angeles",
        "state_highest_approvals": "CA",
        "city_highest_approvals": "Los Angeles",
    }
    
    actual_output = get_employer_details(data, "Company A")
    
    assert actual_output == expected_output

