# House Price Prediction API

A FastAPI web service that loads a trained house price prediction model at startup and serves predictions with Pydantic schema validation.

## Project Structure
- `main.py`: FastAPI application and route definitions (`/health`, `/model/info`, `/predict`).
- `api.py`: Business logic for loading model artifacts, feature extraction, and running inference.
- `schemas.py`: Pydantic request and response validation models.
- `model.pkl` & `model_metadata.json`: Trained model artifact and feature metadata.

## Setup & Local Execution
```bash
pip install -r requirements.txt
uvicorn main:app --reload
