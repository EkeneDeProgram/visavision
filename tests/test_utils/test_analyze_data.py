# Related third party imports
import pandas as pd
import pytest

# Local application/library specific imports
from app.utils import analyze_data


@pytest.fixture
def sample_data():
    # Sample data for testing
    return pd.DataFrame({
        "fiscal_year": [2021, 2021, 2022, 2022, 2022, 2023],
        "state": ["NY", "CA", "NY", "CA", "TX", "NY"],
        "city": ["New York", "San Francisco", "New York", 
                 "San Francisco", "Houston", "New York"
        ],
        "employer": ["Company A", "Company B", "Company A", 
                     "Company B", "Company C", "Company A"
        ],
        "initial_approval": [1, 1, 1, 0, 1, 0],
        "initial_denial": [0, 0, 0, 1, 0, 1]
    })

def test_analyze_data_summary(sample_data):
    # Test case to verify the summary structure and basic metrics
    summary = analyze_data(sample_data)
    
    # Check if the summary contains expected keys
    assert "Highest Fiscal Year Application" in summary
    assert "Highest State Application" in summary
    assert "Highest City Application" in summary
    assert "Highest Employer Application" in summary
    assert "Highest Employer Initial Approval" in summary
    assert "Highest Fiscal Year Initial Approval" in summary
    assert "Highest State Initial Approval" in summary
    assert "Highest City Initial Approval" in summary
    assert "Highest Employer Initial Denial" in summary
    assert "Highest Fiscal Year Initial Denial" in summary
    assert "Highest State Initial Denial" in summary
    assert "Highest City Initial Denial" in summary
    assert "Top 20 Cities by Application" in summary
    assert "Top 20 Employers by Application" in summary
    assert "Top 20 Cities by Approval" in summary
    assert "Top 20 Employers by Approval" in summary
    assert "Rate of Application" in summary
    assert "Rate of Initial Approval" in summary


def test_analyze_data_specific_year(sample_data):
    # Test case to check if specific year analysis is correct
    summary = analyze_data(sample_data)
    year = 2022
    
    assert f"Highest Employer Application {year}" in summary
    assert f"Highest State Application {year}" in summary
    assert f"Highest City Application {year}" in summary
    assert f"Highest Employer Initial Approval {year}" in summary
    assert f"Highest State Initial Approval {year}" in summary
    assert f"Highest City Initial Approval {year}" in summary

