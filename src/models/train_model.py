"""
This module contains functions for training machine learning models.

It includes functions for loading data training data sets,
and training a Random Forest model using the scikit-learn library.
"""
import os
import dotenv
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import mlflow.sklearn
import dagshub
import joblib
from codecarbon import EmissionsTracker

# function to train the model

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
DAGSHUB_REPO =  os.getenv('DAGSHUB_REPO')

def setup_mlflow():
    if "dagshub" in MLFLOW_TRACKING_URI:
        dagshub.init(
            repo_owner=os.getenv('DAGSHUB_USERNAME'),
            repo_name=os.getenv('DAGSHUB_REPO'),
            mlflow=True
        )
        logger.info("MLflow configured to use DagHub tracking server")
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to {MLFLOW_TRACKING_URI}")


def train():

    """
    Train a Random Forest model on the given dataset.

    Returns:
        model: Trained Random Forest model.
    """
    setup_mlflow()
    # applying mlflow autologging
    mlflow.sklearn.autolog()
    with mlflow.start_run():

        # Setup output directory for model artifacts
        os.makedirs('models', exist_ok=True)
        output_file = os.path.join('models', 'model.pkl')

        try:
            # Load the processed data
            train_data = pd.read_csv('data/processed/train.csv')
            logger.info(f"Training data loaded successfully with shape {train_data.shape}")

        except FileNotFoundError as e:
            logger.error(f"Training data file not found: {e}")
            raise

        # Split the data into features (X) and target (y)
        X_train = train_data.drop('Crop', axis=1)
        y_train = train_data['Crop']

        # Create a label encoder object
        label_encoder = LabelEncoder()
        label_encoded = label_encoder.fit_transform(y_train)

        params = {
            'n_estimators': 100,
            'random_state': 42,
            'min_samples_split': 5,
            'max_depth': 5
        }

        # Log hyperparameters to MLflow
        mlflow.log_params(params)

        # Initialize emissions tracker
        tracker = EmissionsTracker()
        tracker.start()
        # Initialize the mode
        model = RandomForestClassifier(**params)
        model.fit(X_train, label_encoded)
        emissions = tracker.stop()
        logger.info(
            f"Estimated Carbon Emission for Model Training: {emissions:.5f} kg CO2"
            )

        mlflow.log_metric("training_emissions", emissions)
        mlflow.log_metric("training_energy_consumption", emissions * 0.000055)

    # Save the model
        joblib.dump(model, output_file)
        joblib.dump(label_encoder, 'models/label_encoder.pkl')
        logger.info(f"Model saved to %s {output_file}")

        # Save the emissions report to in text file
        os.makedirs('reports', exist_ok=True)
        with open('reports/train_model_emissions_report.txt', 'w') as f:
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
