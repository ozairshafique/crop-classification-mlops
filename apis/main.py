""" FastAPI application for crop classification """

import json
import logging
from fastapi import FastAPI, HTTPException, status
import joblib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prometheus_client import Counter, Summary
from sklearn.ensemble import RandomForestClassifier
from prometheus_fastapi_instrumentator import Instrumentator
from .schemas import CropFeatures, PerformanceMetrics, PredictionResponse
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor


model = None
label_encoder = None
summary_cache = None
executor = ThreadPoolExecutor(max_workers=8)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ lifespan context manager to handle startup and shutdown events """
    global model, label_encoder, summary_cache
    # Load model and label encoder at startup
    model = joblib.load('models/model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    if data is not None:
        summary_cache = data.describe().to_dict()
    logging.info("Model and label encoder loaded successfully.")
    yield

app = FastAPI(
    title="Crop Classification API",
    version="0.1.2",
    description="API for predicting suitable crops based on soil and weather features.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

REQUEST_TIME = Summary(
    'request_processing_seconds',
    'Time spent processing request'
    )
REQUEST_COUNT = Counter(
    'request_count',
    'Total number of requests received'
    )

try:
    data = pd.read_csv('data/raw/Crop_Recommendation.csv')
    logger.info(f"Dataset loaded with shape {data.shape}.")
except FileNotFoundError:
    logger.warning("Dataset file not found - summary endpoint unavailable.")
    data = None

try:
    model = joblib.load('models/model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    logger.info("Model and label encoder loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model files not found. {e}")
    raise


# Root endpoint
@app.get("/", tags=["Root"])
@REQUEST_TIME.time()
def read_root():

    """
    Root endpoint providing basic information about the API.
    """
    return {"message": "Welcome to the Crop Classification API!",
            "name": "Crop Classification API",
            "version": "0.1.2",
            "github":
            "https://github.com/ozairshafique/crop-classification-mlops.git"}


# Data summary endpoint
@app.get("/summary",
         status_code=status.HTTP_200_OK,
         description="Get summary statistics of the dataset.")
@REQUEST_TIME.time()
async def get_summary():
    """
    Get summary statistics of the dataset.

    """
    REQUEST_COUNT.inc()
    if data is None:
        raise HTTPException(status_code=404, detail="Dataset not available.")
    return summary_cache


# Define performance endpoint
@app.get("/performance",
         status_code=status.HTTP_200_OK,response_model=PerformanceMetrics)
async def get_performance():
    """ Get model performance metrics from the metrics.json file."""
    REQUEST_COUNT.inc()
    try:
        with open('models/metrics.json') as f:
            performance_data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Performance metrics not found.")

    return PerformanceMetrics(**performance_data)


# Define the predict endpoint
@app.post("/predict",
          response_model=PredictionResponse,
          status_code=status.HTTP_200_OK
          )
async def predict(crop_input: CropFeatures):
    """Predict the most suitable crop based on input features """
    REQUEST_COUNT.inc()
    try:
        input_data = pd.DataFrame([crop_input.model_dump()])
        # run in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        prediction = await asyncio.wait_for(
            loop.run_in_executor(executor, model.predict, input_data),
            timeout=10
        )
        predicted_crop = label_encoder.inverse_transform(prediction)[0]
        return PredictionResponse(
            predicted=predicted_crop,
            input_data=crop_input,
            message="Prediction successful"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define model info endpoint
@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    REQUEST_COUNT.inc()
    if isinstance(model, RandomForestClassifier):
        return {
            "model_name": "Random Forest Classifier",
            "n_estimators": model.n_estimators,
            "random_state": model.random_state,
            "min_samples_split": model.min_samples_split,
            "max_depth": model.max_depth
        }
    raise HTTPException(
        status_code=500,
        detail="Model information not available")


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """Health check endpoint to verify that the API is running"""
    REQUEST_COUNT.inc()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
