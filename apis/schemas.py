
""" Model to define request and response body for the API """
# Import the required libraries
from pydantic import BaseModel, Field


class CropFeatures(BaseModel):
    """ Input features for crop classification """
    Nitrogen: float = Field(..., ge=0, le=140,
                            description="Nitrogen content in soil kg/ha")
    Phosphorus: float = Field(..., ge=0, le=145,
                              description="Phosphorus content in soil kg/ha")
    Potassium: float = Field(..., ge=0, le=205,
                             description="Potassium content in soil kg/ha")
    Temperature: float = Field(..., ge=0, le=50,
                               description="Temperature in Celsius")
    Humidity: float = Field(..., ge=0, le=100,
                            description="Humidity percentage")
    pH_Value: float = Field(..., ge=0, le=14,
                            description="pH value of soil")
    Rainfall: float = Field(..., ge=0, le=300,
                            description="Rainfall in mm")

    class Config:
        json_schema_extra = {
            "example": {
                "Nitrogen": 90.0,
                "Phosphorus": 42.0,
                "Potassium": 43.0,
                "Temperature": 20.0,
                "Humidity": 80.0,
                "pH_Value": 6.5,
                "Rainfall": 200.0
            }
        }


# Define the response body
class DataOut(BaseModel):
    """ Basic Prediction Output """
    prediction: str


# Define the response body for model performance
class PerformanceMetrics(BaseModel):
    """ Model Performance Metrics """
    accuracy: float
    precision: float
    recall: float
    f1: float


# Define the Prediction Response
class PredictionResponse(BaseModel):
    """ Prediction Response Model """
    predicted: str
    input_data: CropFeatures
    message: str = "Prediction successful"
