""" Test cases for the Crop Classification API endpoints. """
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from .main import app


client = TestClient(app)


def test_read_root():
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
def test_predict():
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
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    assert "predicted" in response.json()
    assert "message" in response.json()
    assert response.json()["input_data"] == payload
    assert response.json()["message"] == "Prediction successful"


# Test Model Performance endpoint
def test_performance():
    """ Test performance endpoint to ensure it returns the expected metrics """
    response = client.get("/performance/")
    assert response.status_code == 200
    assert "accuracy" in response.json()
    assert "precision" in response.json()
    assert "recall" in response.json()
    assert "f1" in response.json()


# Test Summary endpoint
def test_summary():
    """ Test summary endpoint returns dataset statistics """
    response = client.get("/summary/")
    assert response.status_code == 200
    assert "Nitrogen" in response.json()
    assert "Phosphorus" in response.json()
    assert "Potassium" in response.json()
    assert "Temperature" in response.json()
    assert "Humidity" in response.json()
    assert "pH_Value" in response.json()
    assert "Rainfall" in response.json()


def test_health_check():
    """ Test health check endpoint returns healthy status """
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    assert response.json()['model_loaded'] is True
    assert response.json()['label_encoder_loaded'] is True


def test_invalid_predict():
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
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422  # Entity for invalid input


if __name__ == "__main__":
    pytest.main()
