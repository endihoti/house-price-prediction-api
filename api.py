import json
import pickle
from typing import Any, Dict
import joblib
import numpy as np

model = None
metadata: Dict[str, Any] = {}


def load_model_and_metadata(
    model_path: str = "model.pkl",
    metadata_path: str = "model_metadata.json"
) -> bool:
    """Load the trained model and metadata JSON into memory."""
    global model, metadata
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        print("Model and metadata loaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to load model or metadata: {e}")
        model = None
        metadata = {}
        return False


def make_prediction(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features in the exact metadata order, convert to NumPy array, and predict."""
    if model is None or not metadata:
        raise RuntimeError("Model is not loaded.")

    feature_names = metadata.get("features", list(input_data.keys()))
    feature_values = [input_data[feat] for feat in feature_names]
    features_array = np.array([feature_values], dtype=float)

    prediction = float(model.predict(features_array)[0])
    version = str(metadata.get("version", "1.0.0"))

    return {
        "predicted_price": round(prediction, 2),
        "currency": "USD",
        "model_version": version
    }


def get_model_info() -> Dict[str, Any]:
    """Return loaded model metadata or raise an error if unavailable."""
    if model is None or not metadata:
        raise RuntimeError("Model metadata is not available because the model is not loaded.")

    return {
        "model_type": metadata.get("model_type", "Unknown"),
        "version": str(metadata.get("version", "1.0.0")),
        "features": metadata.get("features", []),
        "training_date": metadata.get("training_date", "Unknown"),
        "rmse": float(metadata.get("rmse", 0.0)),
        "description": metadata.get("description", "House price prediction model")
    }


def check_health() -> Dict[str, Any]:
    """Check whether the model and metadata are properly loaded."""
    is_loaded = model is not None and bool(metadata)
    return {
        "status": "healthy" if is_loaded else "unhealthy",
        "model_loaded": is_loaded,
        "message": "Model is loaded and ready for predictions."
        if is_loaded
        else "Model is not loaded."
    }
