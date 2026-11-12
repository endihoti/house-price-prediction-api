from typing import List, Optional
from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """Input schema for house features (verify field names against your starter model_metadata.json)"""
    MedInc: float = Field(..., gt=0, description="Median income in block group")
    HouseAge: float = Field(..., ge=0, description="Median house age in block group")
    AveRooms: float = Field(..., gt=0, description="Average number of rooms per household")
    AveBedrms: float = Field(..., gt=0, description="Average number of bedrooms per household")
    Population: float = Field(..., gt=0, description="Block group population")
    AveOccup: float = Field(..., gt=0, description="Average number of household members")
    Latitude: float = Field(..., ge=-90, le=90, description="Block group latitude")
    Longitude: float = Field(..., ge=-180, le=180, description="Block group longitude")
    # Add/adjust remaining features to match the 13 features in your starter file


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted house price")
    currency: str = Field(default="USD", description="Currency of the predicted price")
    model_version: str = Field(..., description="Version of the model used for prediction")


class ModelInfoResponse(BaseModel):
    model_type: str = Field(..., description="Algorithm or model architecture name")
    version: str = Field(..., description="Model version identifier")
    features: List[str] = Field(..., description="Ordered list of input feature names")
    training_date: str = Field(..., description="Timestamp or date when the model was trained")
    rmse: float = Field(..., description="Root Mean Squared Error on evaluation data")
    description: Optional[str] = Field(None, description="Summary of the model and dataset")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service health status ('healthy' or 'unhealthy')")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded in memory")
    message: str = Field(..., description="Human-readable status message")
