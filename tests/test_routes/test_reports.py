# Related third party imports
import pytest

# Local application/library specific imports
from app import app


@pytest.fixture
def client():
    """Fixture to create a test client for the Flask application."""
    with app.test_client() as client:
        yield client


def test_reports_route_success(client, monkeypatch):
    """Test the /reports route for successful data retrieval."""
    # Mock analyzed_data to return sample data
    sample_data = {"summary": "Sample summary data"}
    monkeypatch.setattr("app.routes.analyzed_data", sample_data)

    # Send a GET request to /reports
    response = client.get("/reports")

    # Assert the status code is 200 (OK)
    assert response.status_code == 200

