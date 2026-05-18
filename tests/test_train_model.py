import os
import sys
import pytest
import pandas as pd
import mlflow
from unittest.mock import patch, MagicMock
from src.data.make_dataset import process_data
from src.models.train_model import train

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(autouse=True)
def mock_mlflow_all():
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)

    mock_tracker = MagicMock()
    mock_tracker.stop.return_value = 0.0001

    with patch('src.models.train_model.mlflow.set_tracking_uri'), \
         patch('src.models.train_model.mlflow.set_experiment'), \
         patch('src.models.train_model.mlflow.sklearn.autolog'), \
         patch('src.models.train_model.mlflow.start_run', return_value=mock_run), \
         patch('src.models.train_model.mlflow.log_params'), \
         patch('src.models.train_model.mlflow.log_metric'), \
         patch('src.models.train_model.mlflow.sklearn.log_model'), \
         patch('src.models.train_model.EmissionsTracker', return_value=mock_tracker), \
         patch('src.models.train_model.mlflow.end_run'), \
         patch('src.models.train_model.DAGSHUB_USERNAME', None), \
         patch('src.models.train_model.DAGSHUB_REPO', None):
        yield


def test_load_data():
    train_data, test_data = process_data()
    assert train_data is not None
    assert test_data is not None
    assert len(train_data) > 0
    assert 'Crop' in train_data.columns


def test_data_not_found():
    with patch('src.data.make_dataset.pd.read_csv',
               side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            process_data()


def test_train_model():
    model, label_encoder = train()
    assert model is not None
    assert label_encoder is not None


def test_data_columns():
    train_data, _ = process_data()
    expected_columns = [
        'Nitrogen', 'Phosphorus', 'Potassium',
        'Temperature', 'Humidity', 'pH_Value',
        'Rainfall', 'Crop'
    ]
    for column in expected_columns:
        assert column in train_data.columns
        assert train_data[column].isnull().sum() == 0


def test_train_data_not_found():
    with patch('src.models.train_model.pd.read_csv',
               side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            train()


if __name__ == "__main__":
    pytest.main()