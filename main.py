from fastapi import FastAPI, HTTPException
import api
from schemas import (
    HealthCheckResponse,
    HouseFeatures,
    ModelInfoResponse,
    PredictionResponse,
)

app = FastAPI(
    title="House Price Prediction API",
    description="Production-ready API service for house price predictions",
    version="1.0.0"
)


@app.on_event("startup")
def startup_event():
    """Load model and metadata once when the application starts."""
    success = api.load_model_and_metadata()
    if not success:
        print("WARNING: Startup completed, but model artifacts failed to load.")


@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """Check service and model health status."""
    return api.check_health()


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Return metadata and feature ordering for the loaded model."""
    try:
        return api.get_model_info()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """Validate input features and return a predicted house price."""
    try:
        return api.make_prediction(features.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required feature: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
