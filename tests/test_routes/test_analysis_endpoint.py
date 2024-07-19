# Related third party imports
import pytest

# Local application/library specific imports
from app import app  


@pytest.fixture
def client():
    """Fixture to create a test client for the Flask application."""
    # Use a context manager to create a test client
    with app.test_client() as client:
        # Yield the client to be used in tests
        yield client


def test_analysis_endpoint_success(client, monkeypatch):
    """Test the /api/analysis endpoint for successful data retrieval."""
    # Mock analyzed_data function to return a sample data
    sample_data = {"key": "value"}  # Replace with your sample data structure
    monkeypatch.setattr("app.routes.analyzed_data", sample_data)

    # Send a GET request to /api/analysis
    response = client.get("/api/analysis")

    # Assert the status code is 200 (OK)
    assert response.status_code == 200
    # Assert the response contains the expected JSON data
    assert response.json == sample_data

