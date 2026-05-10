import pytest
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib


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


if __name__ == "__main__":
    pytest.main()
