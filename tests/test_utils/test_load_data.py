# Related third party imports
import pandas as pd
import pytest
from unittest.mock import patch

# Local application/library specific imports
from app.utils import load_data


# Test successful data loading
def test_load_data_success():
    # Path to dataset
    file_path = "data/h-1b-data-2021-2023.csv"
    try:
        data = load_data(file_path)
        assert not data.empty
        assert len(data) > 0  # Check that data is loaded correctly

    except Exception as e:
        pytest.fail(f"Unexpected error: {str(e)}")


# Test FileNotFoundError
def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")


# Test EmptyDataError
def test_load_data_empty_data():
    with patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError):
        with pytest.raises(pd.errors.EmptyDataError):
            load_data("mock_empty_file.csv")


# Test ParserError
def test_load_data_parser_error():
    with patch("pandas.read_csv", side_effect=pd.errors.ParserError):
        with pytest.raises(pd.errors.ParserError):
            load_data("mock_parser_error_file.csv")


# Test unexpected error
def test_load_data_unexpected_error():
    with patch("pandas.read_csv", side_effect=Exception("Unexpected error")):
        with pytest.raises(Exception) as excinfo:
            load_data("mock_unexpected_error_file.csv")
        assert str(excinfo.value) == "Unexpected error"
