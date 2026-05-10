"""
Evaluation module for crop classification model.

Calculates accuracy, precision, recall, F1-score
and tracks carbon emissions during inference.
"""

import os
import json
import logging
import dotenv
import joblib
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from codecarbon import EmissionsTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI',
    'http://localhost:5000'
)


def setup_mlflow():
    if "dagshub" in MLFLOW_TRACKING_URI:
        dagshub.init(
            repo_owner=os.getenv('DAGSHUB_USERNAME'),
            repo_name=os.getenv('DAGSHUB_REPO'),
            mlflow=True
        )
        logger.info(f"MLflow configured to use DagHub tracking server for repo {os.getenv('DAGSHUB_REPO')}")
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")


def evaluate() -> dict:
    """
    Evaluate the trained crop classification model.

    Loads model and test data, computes metrics,
    tracks emissions, logs to MLflow and DagHub.

    Returns:
        dict: accuracy, precision, recall, f1 scores

    Raises:
        FileNotFoundError: If model or data missing
    """
    setup_mlflow()
    mlflow.set_experiment('Crop Classification Evaluations')
    with mlflow.start_run():

        try:
            model = joblib.load('models/model.pkl')
            label_encoder = joblib.load('models/label_encoder.pkl')
            logger.info("Model and label encoder loaded successfully")
        except FileNotFoundError as e:
            logger.error("Model or label encoder not found: %s", e)
            raise

    # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        # Load test data
        try:
            test_data = pd.read_csv('data/processed/test.csv')
            logger.info(f"Test set loaded: {len(test_data)} rows")
        except FileNotFoundError as e:
            logger.error("Test data not found: %s", e)
            raise

        X_test = test_data.drop('Crop', axis=1)
        y_test = test_data['Crop']

    # Track emissions during inference
        tracker = EmissionsTracker()
        tracker.start()
        y_pred_encoded = model.predict(X_test)
        emissions = tracker.stop()
        logger.info(
            f"Inference emissions: {emissions:.5f} kg CO2"
            )

        # Log emissions to MLflow
        mlflow.log_metric("eval_emissions_kg_co2", emissions)
        mlflow.log_metric("eval_energy_kwh", emissions * 0.000055)

       # Decode predictions
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        # Calculate metrics — all weighted
        average = 'weighted'
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,
                                    average=average,
                                    zero_division=1)
        recall = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        report = classification_report(y_test, y_pred)

        # Log all metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log to console
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-score:  {f1:.4f}")

        # Save metrics dict
        metrics = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }

    # Save metrics to JSON
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to models/metrics.json")
    # Save emissions report
    os.makedirs('reports', exist_ok=True)
    with open('reports/model_report.md', 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write("## Overall Performance\n\n")
        f.write(f"| Metric | Score |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Accuracy** | **{accuracy:.4f}** |\n")
        f.write(f"| **Precision** | **{precision:.4f}** |\n")
        f.write(f"| **Recall** | **{recall:.4f}** |\n")
        f.write(f"| **F1-Score** | **{f1:.4f}** |\n\n")
        f.write("## Carbon Footprint\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Carbon Emissions | {emissions:.5f} kg CO2 |\n")
        f.write(f"| Energy Consumption | {emissions * 0.000055:.8f} kWh |\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("```\n")
        logger.info("Evaluation report saved to reports/report.md")

        mlflow.end_run()
    return metrics


if __name__ == '__main__':
    evaluate()
