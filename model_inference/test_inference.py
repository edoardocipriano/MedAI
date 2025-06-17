import requests
import json

# URL del servizio di inferenza
url = "http://localhost:8000/predict"

# Dati di test - caso 1: paziente a basso rischio
test_data_low_risk = {
    "age": 35,
    "gender": "Male",
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 22.5,
    "hba1c_level": 5.2,
    "blood_glucose_level": 95
}

# Dati di test - caso 2: paziente ad alto rischio
test_data_high_risk = {
    "age": 65,
    "gender": "Female",
    "hypertension": 1,
    "heart_disease": 1,
    "smoking_history": "current",
    "bmi": 32.5,
    "hba1c_level": 8.5,
    "blood_glucose_level": 180
}

# Dati di test - caso 3: paziente con valori medi
test_data_medium = {
    "age": 50,
    "gender": "Male",
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "former",
    "bmi": 27.0,
    "hba1c_level": 6.5,
    "blood_glucose_level": 130
}

def test_inference(test_name, data):
    print(f"\n=== Test: {test_name} ===")
    print(f"Dati inviati: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Risultato: {result}")
            print(f"Predizione: {result['diabetes']}")
            print(f"Probabilità: {result['probability']:.2%}")
        else:
            print(f"Errore: {response.status_code}")
            print(f"Dettagli: {response.text}")
    except Exception as e:
        print(f"Errore di connessione: {e}")
        print("Assicurati che il servizio di inferenza sia in esecuzione su http://localhost:8000")

if __name__ == "__main__":
    print("Test del servizio di inferenza del modello di diabete")
    print("=" * 50)
    
    # Esegui i test
    test_inference("Paziente a basso rischio", test_data_low_risk)
    test_inference("Paziente ad alto rischio", test_data_high_risk)
    test_inference("Paziente con rischio medio", test_data_medium) 