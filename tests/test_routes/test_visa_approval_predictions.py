# Related third party imports
import pytest

# Local application/library specific imports
from app import app


@pytest.fixture
def client():
    """Fixture to create a test client for the Flask application."""
    with app.test_client() as client:
        yield client


def test_predictions_route_success(client):
    """Test the /visa_approval_predictions route for successful rendering."""
    # Send a GET request to /visa_approval_predictions
    response = client.get("/visa_approval_predictions")

    # Assert the status code is 200 (OK)
    assert response.status_code == 200
    # Assert the response contains the expected HTML elements (or text)
    assert b"Visa Approval Predictions" in response.data

