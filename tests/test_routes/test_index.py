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


def test_index_route_render_template(client):
    """Test if the / route renders index.html correctly."""
    # Send a GET request to /
    response = client.get("/")
    # Assert the status code is 200
    assert response.status_code == 200
    # Optionally, assert specific content in the response
    assert b"Welcome to VisaVision.com" in response.data 

