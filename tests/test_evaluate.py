import pytest
import os
import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib
from src.models.evaluate import evaluate, setup_mlflow, MLFLOW_TRACKING_URI
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_mlflow_all():
    mock_run = MagicMock()
    mock_run.return_value.__enter__.return_value = MagicMock(return_value=None)
    mock_run.return_value.__exit__.return_value = MagicMock(return_value=False)
    mock_tracker = MagicMock()
    mock_tracker.stop.return_value = 0.0001

    with patch('src.models.evaluate.mlflow.set_tracking_uri'), \
        patch('src.models.evaluate.mlflow.set_experiment'), \
        patch('src.models.evaluate.mlflow.start_run', return_value=mock_run), \
        patch('src.models.evaluate.mlflow.sklearn.log_model'), \
        patch('src.models.evaluate.mlflow.log_params'), \
        patch('src.models.evaluate.DAGSHUB_USERNAME', None), \
        patch('src.models.evaluate.DAGSHUB_REPO', None), \
        patch('src.models.evaluate.mlflow.log_metric'), \
        patch('src.models.evaluate.EmissionsTracker', return_value=mock_tracker), \
        patch('src.models.evaluate.mlflow.end_run'):
        yield

# This is a test function
def test_evaluate():
    # Import the evaluate function

    model = joblib.load('models/model.pkl')

    # Call the evaluate function and store the returned metrics
    test_data = pd.read_csv('data/processed/test.csv')

    # Split the data into features (X) and target (y)
    X_test = test_data.drop('Crop', axis=1)
    y_test = test_data['Crop']

    label_encoder = joblib.load('models/label_encoder.pkl')

    # Make predictions
    y_pred_encoded = model.predict(X_test)

    # Convert predictions back to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    average = 'weighted'
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(
            y_test,
            y_pred,
            average=average,
            zero_division=0),
        'recall': recall_score(y_test, y_pred, average=average),
        'f1': f1_score(y_test, y_pred, average=average)
    }

    # Check if the metrics dictionary is not empty
    assert metrics

    # Check if the metrics dictionary has the required keys
    required_keys = ['accuracy', 'precision', 'recall', 'f1']
    for key in required_keys:
        assert key in metrics

    # Check if the metrics values are floats
    for key, value in metrics.items():
        assert isinstance(value, float)


def test_model_load():
    model = joblib.load('models/model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    assert model is not None
    assert label_encoder is not None
    assert len(label_encoder.classes_) > 10


@pytest.fixture(autouse="True")
def cleanup_mlflow():
    """ Fixture to clean up MLflow runs after each test to prevent interference between tests. """
    mlflow.end_run()  # Ensure any active run is ended before the test
    yield
    mlflow.end_run()  # Ensure any active run is ended after the test

def test_mlflow_with_dagshub():
    with patch('src.models.evaluate.DAGSHUB_USERNAME', 'testuser'), \
         patch('src.models.evaluate.DAGSHUB_REPO', 'testrepo'), \
         patch('src.models.evaluate.mlflow.set_tracking_uri') as mock_set_tracking_uri, \
         patch('src.models.evaluate.mlflow.set_experiment'), \
         patch('src.models.evaluate.mlflow.end_run'):
        import src.models.evaluate as evaluate_module
        evaluate_module.DAGSHUB_USERNAME = 'testuser'
        evaluate_module.DAGSHUB_REPO = 'testrepo'

        setup_mlflow()
        mock_set_tracking_uri.assert_called_once_with('https://dagshub.com/testuser/testrepo.mlflow')
        assert os.environ['MLFLOW_TRACKING_USERNAME'] == 'testuser'


def test_evaluate_model_not_found():
    with patch('joblib.load', side_effect=FileNotFoundError("Model file not found")),\
        patch('mlflow.set_experiment'), \
        patch('mlflow.start_run'), \
        patch('mlflow.end_run'):

        with pytest.raises(FileNotFoundError):
            evaluate()


def test_evaluate_data_not_found():

    mock_model = MagicMock()
    mock_label_encoder = MagicMock()

    with patch('joblib.load', side_effect=[mock_model, mock_label_encoder]), \
        patch('pandas.read_csv', side_effect=FileNotFoundError("data file not found")), \
        patch('mlflow.set_experiment'), \
        patch('mlflow.start_run'), \
        patch('mlflow.end_run'):

        with pytest.raises(FileNotFoundError):
            evaluate()


def test_evaluate_metrics_calculation():
    """Test if the evaluate function calculates metrics correctly with mock data"""

    mock_data = pd.DataFrame({
        'Nitrogen': [90, 85, 80, 29],
        'Phosphorus': [42, 40, 38, 41],
        'Potassium': [49, 45, 40, 42],
        'Temperature': [20.82, 21.0, 19.5, 20.0],
        'Humidity': [82, 80, 78, 85],
        'pH_Value': [6.5, 6.8, 6.2, 6.0],
        'Rainfall': [202.82, 210.0, 190.0, 180.0],
        'Crop': ['Rice', 'Wheat', 'Barley', 'Maize']
    })

    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1, 2, 3]  # Mock predictions
    mock_encoders = MagicMock()
    mock_encoders.inverse_transform.return_value = ['Rice', 'Wheat', 'Barley', 'Maize']
    mock_encoders.classes_ = ['Rice',  'Barley', 'Maize', 'Wheat']

    with patch('joblib.load', side_effect=[mock_model, mock_encoders]), \
         patch('pandas.read_csv', return_value=mock_data):
        metrics = evaluate()

    assert metrics is not None
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics

    for key, value in metrics.items():
        assert isinstance(value, float)

    assert metrics['accuracy'] >= 0.9


if __name__ == "__main__":
    pytest.main()
