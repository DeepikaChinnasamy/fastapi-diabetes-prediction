from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
import logging

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

# Define the input schema
class DiabetesInput(BaseModel):
    feature_values: list

# Health Check Route
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Define the prediction endpoint
@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    try:
        # Convert to NumPy array and reshape
        input_data = np.array(data.feature_values).reshape(1, -1)
        
        # Standardize the input data
        std_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(std_data)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return {
            "prediction": result,
            "confidence": float(prediction[0])  # Add confidence score (optional)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Error in prediction. Check input format.")

