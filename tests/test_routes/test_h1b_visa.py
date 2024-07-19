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


def test_h1b_visa_route(client, monkeypatch):
    """Test if the /h1b_visa route renders h1b_visa.html with status code 200."""
    # Send a GET request to /h1b_visa
    response = client.get("/h1b_visa")
    # Assert the status code is 200
    assert response.status_code == 200
    # Optionally, assert specific content in the response
    assert b"Overview: H-1B Visa" in response.data

