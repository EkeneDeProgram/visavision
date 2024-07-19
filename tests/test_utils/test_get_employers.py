# Related third party imports
import pandas as pd
import pytest

# Local application/library specific imports
from app.utils import get_employers  


# Helper function to flatten the list of lists returned by get_employers
def flatten(lst):
    return [item for sublist in lst for item in sublist]


# Test case 1: Basic functionality
def test_get_employers_basic():
    data = pd.DataFrame({
        "employer": ["Company A", "Company B", "Company A", "Company C"]
    })

    # Define the expected list of employer names
    expected_employers = ["Company A", "Company B", "Company C"]
    # Call the function to get the list of unique employers from the data
    unique_employers = get_employers(data)
    # Assert that the sorted list of flattened unique 
    # employers matches the sorted expected list
    assert sorted(flatten(unique_employers)) == sorted(expected_employers)


# Test case 2: Edge case - empty dataframe
def test_get_employers_empty_dataframe():
    data = pd.DataFrame(columns=["employer"])

    # Get unique employers from the data
    unique_employers = get_employers(data)
    # Assert that there are no unique employers found
    assert unique_employers == []


# Test case 3: Edge case - single employer
def test_get_employers_single_employer():
    data = pd.DataFrame({
        "employer": ["Company A"]
    })

    # Define the expected employer(s)
    expected_employer = ["Company A"]
    # Get unique employers from the data
    unique_employers = get_employers(data)
    # Assert that the unique employers match the expected list after flattening
    assert flatten(unique_employers) == expected_employer


# Test case 4: Performance - large dataset
@pytest.mark.parametrize("num_employers", [1000, 10000])
def test_get_employers_large_dataset(num_employers):
    # Generate a list of employer names based on a range of numbers
    employers = [f"Company {i}" for i in range(num_employers)]
    data = pd.DataFrame({
        "employer": employers * 10  # Simulating repetitive data
    })

    # Get unique employers from the data
    unique_employers = get_employers(data)
    # Assert that the flattened unique employers match the 
    # sorted list of expected employers
    assert sorted(flatten(unique_employers)) == sorted(employers)
