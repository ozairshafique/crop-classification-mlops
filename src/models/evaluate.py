"""
Evaluation module for crop classification model.

Calculates accuracy, precision, recall, F1-score
and tracks carbon emissions during inference.
"""

import os
import json
import logging
import joblib
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
from codecarbon import EmissionsTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    dagshub.init(
        repo_owner='ushafique',
        repo_name='CropClassification',
        mlflow=True
    )
    mlflow.set_experiment('Crop Classification Evaluations')
    mlflow.start_run()

    # Load model and encoder
    try:
        model = joblib.load('models/model.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        logger.info("Model and encoder loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # Load test data
    test_data = pd.read_csv('data/processed/test.csv')
    X_test = test_data.drop('Crop', axis=1)
    y_test = test_data['Crop']
    logger.info(f"Test set loaded: {len(test_data)} rows")

    # Track emissions during inference
    tracker = EmissionsTracker()
    tracker.start()
    y_pred_encoded = model.predict(X_test)
    emissions = tracker.stop()
    logger.info(f"Inference emissions: {emissions:.5f} kg CO2")

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
    with open('reports/evaluate_emissions_report.txt', 'w') as f:
        f.write(
            f"Carbon Emissions: {emissions:.5f} kg CO2\n"
            f"Energy Consumption: {emissions * 0.000055:.8f} kWh\n"
        )

    mlflow.end_run()
    return metrics


if __name__ == '__main__':
    evaluate()
