from fastapi import FastAPI
import torch
import numpy as np
import sys
import os
import pickle
from schemas import InputData, OutputData

# Add parent directory to path to allow importing from model directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_training.model import create_model
from sklearn.preprocessing import MinMaxScaler
import uvicorn

app = FastAPI()

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
    ], dtype=np.float32).reshape(1, -1)  # Reshape to 2D for transformer
    
    # Load and use the same transformer with scaler that was used during training
    try:
        with open('model_training/saved/column_transformer.pkl', 'rb') as f:
            column_transformer = pickle.load(f)
        # Apply the same transformation as during training
        scaled_features = column_transformer.transform(features)
        # Convert to 1D array for returning
        return torch.FloatTensor(scaled_features)  # Already has batch dimension
    except FileNotFoundError:
        # Fallback to manual scaling if saved transformer is not found
        print("Warning: Saved scaler not found. Using manual scaling instead.")
        # Extract features from the 2D array back to 1D for manual processing
        features = features.flatten()
        
        # Apply scaling to specific continuous columns using MinMaxScaler
        # Extract continuous variables
        continuous_indices = [0, 3, 4, 5, 6]  # age, bmi, hba1c_level, blood_glucose_level, smoking_value
        continuous_vars = features[continuous_indices].reshape(1, -1)
        
        # Apply MinMaxScaler
        mm_scaler = MinMaxScaler()
        scaled_continuous = mm_scaler.fit_transform(continuous_vars).flatten()
        
        # Put the scaled values back into the features array
        for i, idx in enumerate(continuous_indices):
            features[idx] = scaled_continuous[i]
        
        return torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    # Use fixed threshold
    threshold = 0.35
    
    # Load the trained model
    model = create_model(input_size=10)
    try:
        model.load_state_dict(torch.load('model_training/diabetes_model.pth', map_location=torch.device('cpu'), weights_only=True))
    except:
        print("Warning: Using default model path failed. Trying alternative path.")
        model.load_state_dict(torch.load('model_training/saved/diabetes_model.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    
    # Make prediction
    with torch.no_grad():
        raw_prediction = model(processed_input)
        # Apply sigmoid to get probability between 0 and 1
        probability = torch.sigmoid(raw_prediction)
    
    # Get the probability value as a Python float
    prob_value = float(probability[0][0])
    
    # Return the prediction using the fixed threshold
    return OutputData(
        diabetes="Yes" if prob_value > threshold else "No",
        probability=prob_value
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)