import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os

def load_data():
    df = pd.read_csv('model_training/diabetes_prediction_dataset.csv')

    df = pd.get_dummies(df, columns=['gender'])

    order_map = {
        'No Info': 0,
        'never': 1,
        'former': 2,
        'not current': 3,
        'current': 4,
        'ever': 5
    }

    df['smoking_history'] = df['smoking_history'].map(order_map)

    df['gender_Female'] = df['gender_Female'].astype(int)
    df['gender_Male'] = df['gender_Male'].astype(int)
    df['gender_Other'] = df['gender_Other'].astype(int)

    X = df.drop('diabetes', axis=1).values
    y = df['diabetes'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mm_scaler = MinMaxScaler()

    cf = ColumnTransformer(
        [('scaler', mm_scaler, [0, 3, 4, 5, 6])],
        remainder='passthrough'
    )

    X_train = cf.fit_transform(X_train)
    X_test = cf.transform(X_test)
    
    # Save the fitted ColumnTransformer with scaler using pickle
    os.makedirs('model_training/saved', exist_ok=True)
    with open('model_training/saved/column_transformer.pkl', 'wb') as f:
        pickle.dump(cf, f)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    # Print shapes to verify
    print("X_train tensor shape:", X_train_tensor.shape)
    print("X_test tensor shape:", X_test_tensor.shape)
    print("y_train tensor shape:", y_train_tensor.shape)
    print("y_test tensor shape:", y_test_tensor.shape)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    return train_loader, test_loader