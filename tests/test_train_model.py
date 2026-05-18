
import os
import sys
import pytest
import pandas as pd
from src.data.make_dataset import process_data
from unittest.mock import patch, MagicMock
from src.models.train_model import train


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_load_data():
    """Test if the data is loaded correctly."""

    train_data, test_data = process_data()
    assert train_data is not None
    assert test_data is not None
    assert len(train_data) > 0  # Check that there are rows in the data
    assert 'Crop' in train_data.columns  # Ensure the 'Crop' column exists


def test_data_not_found():
    """Test if the function raises an error when data file is missing."""

    with patch('src.data.make_dataset.pd.read_csv', side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            process_data()


def test_train_model():
    """Test if the model can be trained."""
    # Load the data
    train_data, _ = process_data()
    X = train_data.drop('Crop', axis=1)
    y = train_data['Crop']
    model, label_encoder = train()

    # Ensure the model and encoder are created
    assert model is not None
    assert label_encoder is not None


def test_data_columns():
    """ Test if the processed data has the expected columns """
    train_data, _ = process_data()
    expected_columns = [
        'Nitrogen',
        'Phosphorus',
        'Potassium',
        'Temperature',
        'Humidity',
        'pH_Value',
        'Rainfall',
        'Crop'
        ]

    for column in expected_columns:
        assert column in train_data.columns
        assert train_data[column].isnull().sum() == 0  # Check for missing values


def test_train_data_not_found():
    """Test if the function raises an error when data file is missing."""
    mock_model = MagicMock()
    mock_label_encoder = MagicMock()
    with patch('joblib.load', side_effect=[mock_model, mock_label_encoder]), \
        patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")), \
        patch('mlflow.set_experiment'), \
        patch('mlflow.start_run'), \
        patch('mlflow.end_run'):
        with pytest.raises(FileNotFoundError):
            train()


if __name__ == "__main__":
    pytest.main()
