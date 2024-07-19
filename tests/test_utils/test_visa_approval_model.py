# Standard library imports
import numpy as np

# Related third party imports
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Local application/library specific imports
from app.utils import visa_approval_model


@pytest.fixture
def sample_data():
    # Create a sample DataFrame with more samples for SMOTE
    data = pd.DataFrame({
        "employer": [
            "Company A", "Company B", "Company C", "Company A", 
            "Company B", "Company D", "Company E", "Company F"
        ],
        "city": ["New York", "San Francisco", "Chicago", "New York", 
                 "San Francisco", "Chicago", "New York", "San Francisco"
        ],
        "state": ["NY", "CA", "IL", "NY", "CA", "IL", "NY", "CA"],
        "fiscal_year": [2024, 2024, 2024, 2025, 2025, 2025, 2024, 2025],
        "initial_approval": [1, 0, 1, 1, 0, 1, 0, 1],
        "initial_denial": [0, 1, 0, 0, 1, 0, 1, 0],
        "continuing_approval": [1, 0, 1, 1, 0, 1, 0, 1],
        "continuing_denial": [0, 1, 0, 0, 1, 0, 1, 0]
    })
    return data


def test_visa_approval_model(sample_data):
    # Call the visa_approval_model function
    results = visa_approval_model(sample_data, n_neighbors=1)

    # Check if the returned dictionary contains the expected keys
    expected_keys = [
        "approval_model", "scaler", "pca", 
        "label_encoder", "feature_names",
        "city_agg", "state_agg", 
        "cross_val_accuracy", "test_accuracy",
        "classification_report", "confusion_matrix"
    ]
    for key in expected_keys:
        assert key in results, f"Missing key {key} in results"

    # Check if the returned model is an instance of RandomForestClassifier
    assert isinstance(
        results["approval_model"],
        RandomForestClassifier), "Model is not a RandomForestClassifier"

    # Check if the scaler is an instance of StandardScaler
    assert isinstance(
        results["scaler"], 
        StandardScaler), "Scaler is not a StandardScaler"

    # Check if the PCA transformer is an instance of PCA
    assert isinstance(results["pca"], PCA), "PCA transformer is not a PCA"

    # Check if the label encoder is an instance of LabelEncoder
    assert isinstance(results["label_encoder"], LabelEncoder), "Label encoder is not a LabelEncoder"

    # Check if the feature names match the expected names
    expected_feature_names = [
        "city", "state", "fiscal_year", 
        "city_approval_rate", "state_approval_rate"
    ]
    assert np.array_equal(
        results["feature_names"], 
        expected_feature_names), "Feature names are incorrect"

    # Check if the cross-validation accuracy is a float
    assert isinstance(
        results["cross_val_accuracy"], 
        float), "Cross-validation accuracy is not a float"

    # Check if the test accuracy is a float
    assert isinstance(
        results["test_accuracy"], 
        float), "Test accuracy is not a float"

    # Check if the classification report is a dictionary
    assert isinstance(
        results["classification_report"], 
        dict), "Classification report is not a dictionary"

    # Check if the confusion matrix is a numpy array
    assert isinstance(
        results["confusion_matrix"], 
        np.ndarray), "Confusion matrix is not a numpy array"

    # Check the length of the confusion matrix (should be 2x2 for binary classification)
    assert results["confusion_matrix"].shape == (2, 2), "Confusion matrix shape is not 2x2"


