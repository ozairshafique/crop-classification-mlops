""" Test cases for the Crop Classification API endpoints. """
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from fastapi import FastAPI
from apis.main import app
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(scope="module")
def client():
    """ Fixture to create a TestClient for the FastAPI app """
    mock_model = MagicMock(spec=RandomForestClassifier)
    mock_model.n_estimators = 100
    mock_model.random_state = 42
    mock_model.min_samples_split = 2
    mock_model.max_depth = None
    mock_model.predict.return_value = [0] # Mock prediction output

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.return_value = ["Rice"] # Mock label encoding

    with patch('joblib.load', side_effect=[mock_model, mock_encoder]):
        from apis.main import app
        with TestClient(app) as c:
            yield c


def test_read_root(client):
    """ Test root endpoint returns API information """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Crop Classification API!",
        "name": "Crop Classification API",
        "version": "0.1.2",
        "github": "https://github.com/ozairshafique/crop-classification-mlops.git"
    }


# Test predict endpoint
def test_predict(client):
    """ Test predict endpoint return valid input data """
    payload = {
        "Nitrogen": 90,
        "Phosphorus": 42,
        "Potassium": 49,
        "Temperature": 20.82,
        "Humidity": 82,
        "pH_Value": 6.5,
        "Rainfall": 202.82
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted" in response.json()
    assert "message" in response.json()
    assert response.json()["input_data"] == payload
    assert response.json()["message"] == "Prediction successful"


# Test Model Performance endpoint
def test_performance(client):
    """ Test performance endpoint to ensure it returns the expected metrics """
    response = client.get("/performance")
    assert response.status_code == 200
    assert "accuracy" in response.json()
    assert "precision" in response.json()
    assert "recall" in response.json()
    assert "f1" in response.json()


# Test Summary endpoint
def test_summary(client):
    """ Test summary endpoint returns dataset statistics """
    response = client.get("/summary")
    assert response.status_code == 200
    assert "Nitrogen" in response.json()
    assert "Phosphorus" in response.json()
    assert "Potassium" in response.json()
    assert "Temperature" in response.json()
    assert "Humidity" in response.json()
    assert "pH_Value" in response.json()
    assert "Rainfall" in response.json()


def test_health_check(client):
    """ Test health check endpoint returns healthy status """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    assert response.json()['model_loaded'] is True
    assert response.json()['label_encoder_loaded'] is True


def test_invalid_predict(client):
    """ Test predict endpoint with invalid input data """
    payload = {
        "Nitrogen": -89,
        "Phosphorus": 42,
        "Potassium": 49,
        "Temperature": 20.82,
        "Humidity": 82,
        "pH_Value": 6.5,
        "Rainfall": 202.82
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Entity for invalid input


def test_model_info(client):
    """ Test model info endpoint returns model information """
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert "Random Forest Classifier" in response.json()["model_name"]
    assert "n_estimators" in response.json()
    assert "random_state" in response.json()
    assert "min_samples_split" in response.json()
    assert "max_depth" in response.json()


def test_model_file_not_found(client):
    """ Test model info endpoint when model file is not found """
    with patch('apis.main.model', None):
        response = client.get("/model-info")
        assert response.status_code == 500
        assert response.json()['detail'] == "Model information not available"


def test_summary_not_available(client):
    """ Test summary endpoint when dataset is not available """
    with patch('apis.main.summary_cache', None):
        response = client.get("/summary")
        assert response.status_code == 404
        assert response.json()['detail'] == "Dataset not available."

def test_dataset_load_success():
    with patch('apis.main.data', return_value=pd.DataFrame({'Nitrogen': [90]})):
        import apis.main
        assert apis.main.data is not None

if __name__ == "__main__":
    pytest.main()
