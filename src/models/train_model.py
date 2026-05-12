"""
This module contains functions for training machine learning models.

It includes functions for loading data training data sets,
and training a Random Forest model using the scikit-learn library.
"""
import logging
import os

import dagshub
import dotenv
import joblib
import mlflow.sklearn
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO')


def setup_mlflow():
    if DAGSHUB_USERNAME and DAGSHUB_REPO:
        dagshub.init(
            repo_owner=os.getenv('DAGSHUB_USERNAME'),
            repo_name=os.getenv('DAGSHUB_REPO'),
            mlflow=True
        )
        logger.info("MLflow configured to use DagsHub tracking server")
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)


def train():
    """
    Train a Random Forest model on the given dataset.

    Returns:
        model: Trained Random Forest model.
    """
    setup_mlflow()
    mlflow.sklearn.autolog()

    with mlflow.start_run():

        os.makedirs('models', exist_ok=True)
        output_file = os.path.join('models', 'model.pkl')

        try:
            train_data = pd.read_csv('data/processed/train.csv')
            logger.info(
                "Training data loaded with shape %s", train_data.shape
            )
        except FileNotFoundError as e:
            logger.error("Training data file not found: %s", e)
            raise

        X_train = train_data.drop('Crop', axis=1)
        y_train = train_data['Crop']

        label_encoder = LabelEncoder()
        label_encoded = label_encoder.fit_transform(y_train)

        params = {
            'n_estimators': 100,
            'random_state': 42,
            'min_samples_split': 5,
            'max_depth': 5
        }

        mlflow.log_params(params)

        tracker = EmissionsTracker()
        tracker.start()
        model = RandomForestClassifier(**params)
        model.fit(X_train, label_encoded)
        emissions = tracker.stop()

        logger.info(
            "Estimated Carbon Emission for Model Training: %.5f kg CO2",
            emissions
        )

        mlflow.log_metric("training_emissions", emissions)
        mlflow.log_metric(
            "training_energy_consumption", emissions * 0.000055
        )

        joblib.dump(model, output_file)
        joblib.dump(label_encoder, 'models/label_encoder.pkl')
        logger.info("Model saved to %s", output_file)

        os.makedirs('reports', exist_ok=True)
        report_path = 'reports/train_model_emissions_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(
                f"Estimated Carbon Emission for "
                f"Model Training: {emissions:.5f} kg CO2\n"
                f"Estimated Energy Consumption for "
                f"Model Training: "
                f"{emissions * 0.000055:.5f} kWh\n"
            )

    return model, label_encoder


if __name__ == "__main__":
    train()
