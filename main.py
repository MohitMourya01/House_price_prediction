import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import os

# --- Configuration ---
MODEL_FILE = 'housing_model.joblib'

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles app startup and shutdown.
    Loads the trained ML model into memory when API starts.
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' not found. "
            "Please run 'python train.py' first to create it."
        )

    # Load model at startup
    model = joblib.load(MODEL_FILE)
    app.state.MODEL = model
    print(f"✅ Model '{MODEL_FILE}' loaded successfully.")

    yield  # Run app

    # Optional cleanup logic
    print("🛑 API shutting down. Cleaning up resources...")

# --- App Creation ---
app = FastAPI(
    title="California House Price Prediction API",
    description="An API to predict house prices in California.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Pydantic Input Model ---
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    class Config:
        schema_extra = {
            "example": {
                "MedInc": 3.8716,
                "HouseAge": 21.0,
                "AveRooms": 5.80,
                "AveBedrms": 1.04,
                "Population": 1425.0,
                "AveOccup": 2.55,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "California House Price Prediction API is running."}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    """Predict house price using trained model."""
    try:
        # Access model from app state
        model = app.state.MODEL
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_value = round(prediction[0], 2)
        
        # Model predicts in 100,000s
        return {"predicted_value_100k": predicted_value}
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return {"error": str(e)}

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
