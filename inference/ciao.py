from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
import numpy as np
import sys
import os

# Add parent directory to path to allow importing from model directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import create_model
from sklearn.preprocessing import MinMaxScaler
import uvicorn

app = FastAPI()

class InputData(BaseModel):
    age: float = Field(..., description="Age of the patient")
    gender: str = Field(..., description="Gender of the patient (Female, Male, or Other)")
    hypertension: int = Field(..., description="Hypertension status (0 or 1)")
    heart_disease: int = Field(..., description="Heart disease status (0 or 1)")
    smoking_history: str = Field(..., description="Smoking history (No Info, never, former, not current, current, ever)")
    bmi: float = Field(..., description="Body mass index")
    hba1c_level: float = Field(..., description="HbA1c level")
    blood_glucose_level: float = Field(..., description="Blood glucose level")

class OutputData(BaseModel):
    diabetes: float = Field(..., description="Diabetes prediction")

def preprocess_input(input_data: InputData):
    # Convert smoking_history to numerical value
    smoking_map = {
        'No Info': 0,
        'never': 1,
        'former': 2,
        'not current': 3,
        'current': 4,
        'ever': 5
    }
    smoking_value = smoking_map.get(input_data.smoking_history, 0)
    
    # One-hot encode gender
    gender_female = 1 if input_data.gender == "Female" else 0
    gender_male = 1 if input_data.gender == "Male" else 0
    gender_other = 1 if input_data.gender == "Other" else 0
    
    # Create feature array with the same order as in training
    features = np.array([
        input_data.age,
        input_data.hypertension,
        input_data.heart_disease,
        input_data.bmi,
        input_data.hba1c_level,
        input_data.blood_glucose_level,
        smoking_value,
        gender_female,
        gender_male,
        gender_other
    ], dtype=np.float32)
    
    # Apply scaling to specific columns (age, bmi, hba1c_level, blood_glucose_level, smoking_value)
    # Note: In a production environment, you should load the same scaler used during training
    mm_scaler = MinMaxScaler()
    # We're manually scaling only the continuous variables, similar to the training process
    features[0] = features[0] / 100.0  # age scaling
    features[3] = features[3] / 60.0   # bmi scaling
    features[4] = features[4] / 10.0   # hba1c_level scaling
    features[5] = features[5] / 300.0  # blood_glucose_level scaling
    features[6] = features[6] / 5.0    # smoking_history scaling
    
    return torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    # Load the trained model
    model = create_model(input_size=10)
    model.load_state_dict(torch.load('model/diabetes_model.pth'))
    model.eval()
    
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(processed_input)
    
    # Return the prediction
    return OutputData(diabetes=float(prediction[0][0]))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)