from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model & scaler
try:
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    logger.info("Model and Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise RuntimeError("Failed to load the model and scaler.")

# Initialize FastAPI app
app = FastAPI(title="Diabetes Prediction API", version="1.0")

# Add Prometheus Monitoring
Instrumentator().instrument(app).expose(app)

# Define the input schema
class DiabetesInput(BaseModel):
    feature_values: list

# Health Check Route
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# ------------------ Prometheus Metrics ------------------ #

# Counter for tracking the number of diabetes predictions
PREDICTION_COUNT = Counter(
    "diabetes_predictions_total", "Total number of diabetes predictions", ["outcome"]
)

# Histogram to measure prediction latency
PREDICTION_LATENCY = Histogram(
    "diabetes_prediction_latency_seconds", "Time taken for each prediction"
)

# ------------------ Prediction Endpoint ------------------ #

@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    start_time = time.time()  # Start time for latency measurement
    try:
        logger.info(f"Received prediction request: {data.feature_values}")

        # Convert input data to NumPy array and reshape
        input_data = np.array(data.feature_values).reshape(1, -1)

        # Standardize the input data
        std_data = scaler.transform(input_data)

        # Predict using the trained model
        prediction = model.predict(std_data)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        # Update Prometheus metrics
        PREDICTION_COUNT.labels(outcome=result).inc()  # Count predictions
        PREDICTION_LATENCY.observe(time.time() - start_time)  # Observe latency

        logger.info(f"Prediction: {result}")

        return {
            "prediction": result,
            "confidence": float(prediction[0])  # Optional confidence score
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Error in prediction. Check input format.")

