from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Load the trained model & scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define the input schema
class DiabetesInput(BaseModel):
    feature_values: list

# Define the prediction endpoint
@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    try:
        input_data = np.array(data.feature_values).reshape(1, -1)
        std_data = scaler.transform(input_data)
        prediction = model.predict(std_data)

        result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
    
# To run FastAPI
# uvicorn app:app --reload
# To run in different port
# uvicorn app:app --host 127.0.0.1 --port 5000 --reload

# To run in terminal
# curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"feature_values\": [4,110,92,0,0,37.6,0.191,30]}"
