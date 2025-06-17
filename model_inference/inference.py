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

# Percorsi assoluti per i file del modello
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model_training', 'diabetes_model.pth')
TRANSFORMER_PATH = os.path.join(BASE_DIR, 'model_training', 'saved', 'column_transformer.pkl')

# Carica il modello e il transformer una sola volta all'avvio
print(f"Loading model from: {MODEL_PATH}")
print(f"Loading transformer from: {TRANSFORMER_PATH}")

# Carica il modello
model = create_model(input_size=10)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
    print("Model loaded successfully!")
else:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model.eval()

# Carica il transformer
column_transformer = None
if os.path.exists(TRANSFORMER_PATH):
    with open(TRANSFORMER_PATH, 'rb') as f:
        column_transformer = pickle.load(f)
    print("Transformer loaded successfully!")
else:
    print(f"WARNING: Transformer file not found at {TRANSFORMER_PATH}")
    print("Will use fallback scaling method")

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
    # IMPORTANTE: l'ordine deve corrispondere esattamente a quello del dataset originale
    # Ordine CORRETTO delle colonne nel dataset dopo get_dummies:
    # 0: age, 1: hypertension, 2: heart_disease, 3: smoking_history, 4: bmi, 
    # 5: HbA1c_level, 6: blood_glucose_level, 7: gender_Female, 8: gender_Male, 9: gender_Other
    features = np.array([
        input_data.age,                # 0: age
        input_data.hypertension,        # 1: hypertension
        input_data.heart_disease,       # 2: heart_disease
        smoking_value,                  # 3: smoking_history (NOTA: posizione 3, non 6!)
        input_data.bmi,                 # 4: bmi
        input_data.hba1c_level,         # 5: HbA1c_level
        input_data.blood_glucose_level, # 6: blood_glucose_level
        gender_female,                  # 7: gender_Female
        gender_male,                    # 8: gender_Male
        gender_other                    # 9: gender_Other
    ], dtype=np.float32).reshape(1, -1)
    
    print(f"Raw features before scaling: {features}")
    print(f"Column order: age={features[0,0]}, hyper={features[0,1]}, heart={features[0,2]}, bmi={features[0,3]}, hba1c={features[0,4]}, glucose={features[0,5]}, smoking={features[0,6]}")
    
    # Use the transformer if available
    if column_transformer is not None:
        scaled_features = column_transformer.transform(features)
        print(f"Features after transformer: {scaled_features}")
        print(f"Shape after transformer: {scaled_features.shape}")
        
        # IMPORTANTE: Il ColumnTransformer riordina le colonne!
        # Output order: [scaled cols 0,3,4,5,6] + [passthrough cols 1,2,7,8,9]
        # Dobbiamo verificare se questo è il comportamento
        
        return torch.FloatTensor(scaled_features)
    else:
        # Fallback: usa scaling manuale con valori fissi dal dataset di training
        print("WARNING: Using fallback scaling with fixed ranges")
        features_flat = features.flatten()
        
        # Il ColumnTransformer scala solo [0, 3, 4, 5, 6] e riordina l'output
        # Prima applichiamo lo scaling
        scaled_values = features_flat.copy()
        scaled_values[0] = features_flat[0] / 80.0  # age
        scaled_values[3] = features_flat[3] / 5.0  # smoking_history (0-5)
        scaled_values[4] = (features_flat[4] - 10) / 81.82  # bmi (10-91.82)
        scaled_values[5] = (features_flat[5] - 3.5) / 5.5  # hba1c (3.5-9)
        scaled_values[6] = (features_flat[6] - 80) / 220.0  # glucose (80-300)
        
        # Poi riordiniamo come fa il ColumnTransformer
        # Output order: [scaled 0,3,4,5,6] + [passthrough 1,2,7,8,9]
        reordered = np.array([
            scaled_values[0],    # age scalato
            scaled_values[3],    # smoking scalato
            scaled_values[4],    # bmi scalato  
            scaled_values[5],    # hba1c scalato
            scaled_values[6],    # glucose scalato
            features_flat[1],    # hypertension passthrough
            features_flat[2],    # heart_disease passthrough
            features_flat[7],    # gender_Female passthrough
            features_flat[8],    # gender_Male passthrough
            features_flat[9]     # gender_Other passthrough
        ])
        
        print(f"Features after manual scaling and reordering: {reordered}")
        return torch.FloatTensor(reordered).unsqueeze(0)

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    # Log per debug - stampa i dati ricevuti
    print("\n=== Dati ricevuti dall'API ===")
    print(f"Age: {input_data.age}")
    print(f"Gender: {input_data.gender}")
    print(f"Hypertension: {input_data.hypertension}")
    print(f"Heart Disease: {input_data.heart_disease}")
    print(f"Smoking History: {input_data.smoking_history}")
    print(f"BMI: {input_data.bmi}")
    print(f"HbA1c Level: {input_data.hba1c_level}")
    print(f"Blood Glucose Level: {input_data.blood_glucose_level}")
    print("==============================")
    
    # Use fixed threshold - stesso valore usato nel training
    threshold = 0.65
    
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    
    # Log per debug - stampa i dati dopo preprocessing
    print("=== Dati dopo preprocessing ===")
    print(f"Processed tensor shape: {processed_input.shape}")
    print(f"Processed values: {processed_input}")
    print("==============================")
    
    # Make prediction
    with torch.no_grad():
        raw_prediction = model(processed_input)
        # Apply sigmoid to get probability between 0 and 1
        probability = torch.sigmoid(raw_prediction)
        
        # Log per debug - stampa logit e probabilità
        print(f"Raw logit: {raw_prediction[0][0].item()}")
        print(f"Probability after sigmoid: {probability[0][0].item()}")
        print("==============================")
    
    # Get the probability value as a Python float
    prob_value = float(probability[0][0])
    
    # Return the prediction using the fixed threshold
    return OutputData(
        diabetes="Yes" if prob_value > threshold else "No",
        probability=prob_value
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)