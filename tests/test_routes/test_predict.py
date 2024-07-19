# Standard library imports
from unittest.mock import MagicMock

# Related third party imports
import pytest
import joblib

# Local application/library specific imports
from app import app


@pytest.fixture
def client():
    """Fixture to create a test client for the Flask application."""
    with app.test_client() as client:
        yield client


def test_predict_route_success(client, monkeypatch):
    """Test the /predict route for successful prediction."""
    # Mock input data
    input_data = {
        "feature1": 1,
        "feature2": 2,
        "feature3": 3
    }

    # Mock model components
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.1, 0.2, 0.3]]
    mock_pca = MagicMock()
    mock_pca.transform.return_value = [[0.4, 0.5, 0.6]]
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model_dict = {
        "scaler": mock_scaler,
        "pca": mock_pca,
        "approval_model": mock_model,
        "feature_names": list(input_data.keys())
    }

    # Mock joblib.load to return the mock model components
    monkeypatch.setattr(joblib, "load", lambda f: mock_model_dict)

    # Send a POST request to /predict with the input data
    response = client.post("/predict", json=input_data)

    # Assert the status code is 200 (OK)
    assert response.status_code == 200
    # Assert the response contains the expected JSON data
    response_json = response.get_json()
    assert response_json["approval_prediction"] == "Visa Approved"


def test_predict_route_exception_handling(client, monkeypatch):
    """Test the /predict route for exception handling."""
    # Mock input data
    input_data = {
        "feature1": 1,
        "feature2": 2,
        "feature3": 3
    }

    # Mock joblib.load to raise an exception
    monkeypatch.setattr(
        joblib, "load", 
        lambda f: _mock_joblib_load_raise_error()
    )

    # Send a POST request to /predict with the input data
    response = client.post("/predict", json=input_data)

    # Assert the status code is 500 (Internal Server Error)
    assert response.status_code == 500
    # Assert the response contains the expected error message
    response_json = response.get_json()
    assert "error" in response_json


def _mock_joblib_load_raise_error():
    """Helper function to mock joblib.load raising an error."""
    raise ValueError("Simulated error")
