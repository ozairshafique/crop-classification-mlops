import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def test_load_data():
    """Test if the data is loaded correctly."""
    data = pd.read_csv('data/raw/Crop_Recommendation.csv')

    assert not data.empty  # Check that the data is not empty
    assert len(data) > 0  # Check that there are rows in the data
    assert 'Crop' in data.columns  # Ensure the 'Crop' column exists


def test_train_model():
    """Test if the model can be trained."""
    # Load the data
    data = pd.read_csv('data/processed/train.csv')
    X = data.drop('Crop', axis=1)
    y = data['Crop']

    # Encode the labels
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        label_encoded,
        test_size=0.2,
        random_state=42)

    # Initialize the model
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Ensure the model is trained
    assert model is not None
    score = model.score(X_test, y_test)
    assert score > 0.5  # Ensure the model has some predictive power


def test_data_columns():
    """ Test if the processed data has the expected columns """
    data = pd.read_csv('data/processed/train.csv')
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
        assert column in data.columns


if __name__ == "__main__":
    pytest.main()
