
"""Locust load testing script for the Crop Recommendation API"""
from locust import HttpUser, task, between, events
import random
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Load the crop recommendation dataset
    df = pd.read_csv('data/raw/Crop_Recommendation.csv')
    logger.info("Crop recommendation dataset loaded successfully")
except FileNotFoundError as e:
    logger.error("Dataset not found: %s", e)
    df = None


def get_random_sample():
    """ Get a random sample from the dataset for testing """
    if df is not None:
        row = df.sample().iloc[0]
        return {
            "Nitrogen": row['Nitrogen'],
            "Phosphorus": row['Phosphorus'],
            "Potassium": row['Potassium'],
            "Temperature": row['Temperature'],
            "Humidity": row['Humidity'],
            "pH_Value": row['pH_Value'],
            "Rainfall": row['Rainfall']
        }
    payload = [{
        "Nitrogen": 90,
            "Phosphorus": 42,
            "Potassium": 43,
            "Temperature": 20.87,
            "Humidity": 82.00,
            "pH_Value": 6.50,
            "Rainfall": 202.93
    },
    {
            "Nitrogen": 60,
            "Phosphorus": 55,
            "Potassium": 44,
            "Temperature": 23.00,
            "Humidity": 82.32,
            "pH_Value": 7.84,
            "Rainfall": 263.96
    },
    {
            "Nitrogen": 29,
            "Phosphorus": 55,
            "Potassium": 44,
            "Temperature": 29.00,
            "Humidity": 82.92,
            "pH_Value": 9.84,
            "Rainfall": 261.96
    },
    {
            "Nitrogen": 29,
            "Phosphorus": 89,
            "Potassium": 32,
            "Temperature": 42.89,
            "Humidity": 67.00,
            "pH_Value": 6.50,
            "Rainfall": 201.20
    }]

    return random.choice(payload)


class CropRecommendationUser(HttpUser):
    """ Simulates a user interacting with the Crop Recommendation API """
    wait_time = between(1, 3)  # Wait time between tasks (in seconds)

    def on_start(self):
        """ Called when a simulated user starts """
        with self.client.get(
            "/health",
            catch_response=True,
            name='Health Check'
            ) as response:
            if response.status_code == 200:
                logger.info("Health check passed - starting load test")
            else:
                logger.error("Health check failed with status code: %s", response.status_code)

    @task(1)
    def get_info(self):
        self.client.get('/', name = 'Get_info')

    @task(2)
    def get_model_info(self):
        self.client.get('/model-info', name = 'Get_Model_Info')


    @task(5)
    def get_prediction(self):
        payload = {
            "Nitrogen": 29,
            "Phosphorus": 89,
            "Potassium": 32,
            "Temperature": 42.89,
            "Humidity": 67.0,
            "pH_Value": 6.5,
            "Rainfall": 201.2
        }
        self.client.post("/predict", json=payload, name = 'Get_Prediction')

    @task(1)
    def get_summary(self):
        with self.client.get(
            '/summary',
            name = 'Get_Summary',
            catch_response=True
            ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.faliure(f"Summary endpoint failed with status code: {response.status_code}")

    @task(2)
    def get_model_info(self):
        with self.client.get(
            "/model-info",
            catch_response=True,
            name="GET / model_info"
            ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.faliure(f"Model info endpoint failed with status code: {response.status_code}")
    @task(1)
    def get_performance(self):
        with self.client.get(
            "/performance",
            name="GET / performance",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.faliure(f"Performance endpoint failed with status code: {response.status_code}"
        )

    @events.test_start.add_listener
    def on_test_start(enviorment, **kwargs):
        """ Called when the load test starts """
        logger.info("=" * 50)
        logger.info("Starting load test for Crop Recommendation API")
        logger.info("=" * 50)

    @events.test_stop.add_listener
    def on_test_stop(enviorment, **kwargs):
        """ Called when the load test stops """
        logger.info("=" * 50)
        logger.info("Load test completed for Crop Recommendation API")
        logger.info("=" * 50)

    # @task
    # def get_crop_recommendation(self):
    #     # Randomly select a row from the DataFrame to simulate user input
    #     row = df.sample().iloc[0]

    #     # Prepare the query parameters for the GET request
    #     params = {
    #         "Nitrogen": row['Nitrogen'],
    #         "Phosphorus": row['Phosphorus'],
    #         "Potassium": row['Potassium'],
    #         "Temperature": row['Temperature'],
    #         "Humidity": row['Humidity'],
    #         "pH_Value": row['pH_Value'],
    #         "Rainfall": row['Rainfall']
    #     }

    #     # Make a GET request to the crop recommendation endpoint
    #     self.client.post("/predict", params=params)