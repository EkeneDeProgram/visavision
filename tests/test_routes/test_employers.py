# Related third party imports
import pytest

# Local application/library specific imports
from app import app as flask_app    


@pytest.fixture
def app():
    # Yield the Flask application instance to be used in tests
    yield flask_app


@pytest.fixture
def client(app):
    # Return a test client for the Flask application instance
    return app.test_client()


def test_employers_route_success(client, monkeypatch):
    """Test successful rendering of /employers route."""
    # Mock get_employers to return sample employers list
    sample_employers = [
        ["Employer A"],
        ["Employer B"],
        ["Employer C"],
    ]
    monkeypatch.setattr(
        # Replace the function `get_employers` in app.routes
        "app.routes.get_employers", 
        # Use a lambda function that returns `sample_employers` for any input
        lambda data: sample_employers
    )

    # Send a GET request to /employers
    response = client.get("/employers")

    # Assert the status code is 200
    assert response.status_code == 200


def test_employers_route_exception_handling(caplog, client, monkeypatch):
    """Test exception handling in /employers route."""
    # Define helper function to raise exception
    def mock_get_employers_raise_error(cleaned_data):
        raise Exception("Simulated error")

    # Mock get_employers to use the helper function
    monkeypatch.setattr(
        "app.routes.get_employers", 
        mock_get_employers_raise_error
    )

    # Send a GET request to /employers
    response = client.get("/employers")

    # Assert the status code is 500 (Internal Server Error)
    assert response.status_code == 500
    # Assert "Internal Server Error" in the response data
    assert b"Internal Server Error" in response.data
    # Check if error is logged correctly
    assert "Error in /employers route: Simulated error" in caplog.text