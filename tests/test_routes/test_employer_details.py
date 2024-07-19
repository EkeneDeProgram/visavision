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


def test_employer_route_success(client, monkeypatch):
    """Test successful rendering of /employer route."""
    # Mock get_employer_details to return sample details
    sample_details = {
        "employer_name": "Sample Employer",
        "total_applications": 100,  
        "applications_2021": 30,
        "applications_2022": 40,
        "applications_2023": 30,
        "total_initial_approvals": 80,
        "initial_approvals_2021": 20,
        "initial_approvals_2022": 30,
        "initial_approvals_2023": 30,
        "total_initial_denials": 20,
        "state_highest_applications": "California",
        "city_highest_applications": "San Francisco",
        "state_highest_approvals": "New York",
        "city_highest_approvals": "New York City",
    }
    monkeypatch.setattr(
        # Replace the function `get_employer_details` in app.routes
        "app.routes.get_employer_details",
        # Use a lambda function that returns `sample_details` for any input
        lambda cleaned_data, name: sample_details
    )

    # Send a GET request to /employer/Sample%20Employer
    response = client.get("/employer/Sample%20Employer")

    # Assert the status code is 200
    assert response.status_code == 200

    
def test_employer_route_value_error_handling(client, monkeypatch, caplog):
    """Test handling of ValueError in /employer route."""
    # Mock get_employer_details to raise ValueError
    monkeypatch.setattr(
        "app.routes.get_employer_details", 
        lambda cleaned_data, name: raise_value_error()
    )

    # Send a GET request to /employer/Invalid%20Employer
    response = client.get("/employer/Invalid%20Employer")

    # Assert the status code is 302 (Redirect to employers route)
    assert response.status_code == 302
    # Adjusted assertion for relative path
    assert response.location == "/employers" 
    # Check if ValueError is logged correctly
    assert "Error retrieving details for Invalid Employer: " in caplog.text


def test_employer_route_unexpected_error_handling(client, monkeypatch, caplog):
    """Test handling of unexpected exceptions in /employer route."""
    # Mock get_employer_details to raise an unexpected exception
    monkeypatch.setattr(
        "app.routes.get_employer_details", 
        lambda cleaned_data, name: raise_unexpected_error()
    )

    # Send a GET request to /employer/Another%20Employer
    response = client.get("/employer/Another%20Employer")

    # Assert the status code is 302 (Redirect to employers route)
    assert response.status_code == 302
    assert response.location == "/employers"  # Adjusted assertion for relative path
    # Check if unexpected error is logged correctly
    assert "Unexpected error in employer route: " in caplog.text


def raise_value_error():
    raise ValueError("Simulated value error")


def raise_unexpected_error():
    raise Exception("Simulated unexpected error")
