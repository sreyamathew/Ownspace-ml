import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import copy
from feature_augmentation import augment_features

# Configuration
API_URL = "http://127.0.0.1:3001/api/properties"
MODEL_PATH = "price_model.pkl"
ENCODER_PATH = "location_encoder.joblib"

def fetch_data(url):
    """Fetch property data from Node.js backend."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError("No data returned from API")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}. Using fallback mock data.")
        # Expanded mock data for better training
        locations = ["Kochi", "Trivandrum", "Kozhikode", "Thrissur", "Kottayam"]
        mock_data = []
        for loc in locations:
            for _ in range(10): # 10 base samples per location
                size = random.randint(800, 3500)
                bhk = max(1, size // 600)
                bath = max(1, bhk - random.randint(0, 1))
                # Base price influenced by location and size
                loc_multiplier = {"Kochi": 1.2, "Trivandrum": 1.1, "Kozhikode": 1.0, "Thrissur": 0.9, "Kottayam": 0.85}
                price = int(size * 4500 * loc_multiplier[loc] * random.uniform(0.9, 1.1))
                mock_data.append({
                    "location": loc,
                    "size": size,
                    "bhk": bhk,
                    "bath": bath,
                    "price": price
                })
        return mock_data

import random

def expand_dataset(base_data, factor=15):
    """
    Expands the dataset by creating variations of base data with controlled noise.
    """
    expanded = []
    for item in base_data:
        for _ in range(factor):
            new_item = copy.deepcopy(item)
            
            # Add controlled noise to features during augmentation
            # 1. Apply Price-Aware Augmentation first
            new_item = augment_features(new_item)
            
            # 2. Add small noise to existing features to prevent overfitting
            new_item['size'] = int(new_item['size'] * random.uniform(0.98, 1.02))
            new_item['amenitiesScore'] = max(4.0, min(10.0, new_item['amenitiesScore'] + random.uniform(-0.2, 0.2)))
            new_item['propertyAge'] = max(0, new_item['propertyAge'] + random.randint(-1, 1))
            
            expanded.append(new_item)
    return expanded

def train():
    print("Fetching data...")
    raw_data = fetch_data(API_URL)
    
    print(f"Expanding dataset (factor=15x) from {len(raw_data)} samples...")
    # Dataset expansion
    dataset = expand_dataset(raw_data, factor=15)
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    print("Pre-processing data...")
    # Preprocessing
    df = df.dropna(subset=['price', 'location', 'size', 'bhk', 'bath', 'amenitiesScore', 'propertyAge'])
    
    # Encode categorical feature: location using OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    location_encoded = encoder.fit_transform(df[['location']])
    location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['location']))
    
    # Combine features
    X = pd.concat([
        df[['size', 'bhk', 'bath', 'amenitiesScore', 'propertyAge']].reset_index(drop=True),
        location_df.reset_index(drop=True)
    ], axis=1)
    
    y = df['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ML Model: XGBoost Regressor with Tuned Hyperparameters
    print("Training XGBoost Regressor (Tuned)...")
    model = xgb.XGBRegressor(
        n_estimators=500,        # Increased
        learning_rate=0.03,      # Lowered for better convergence
        max_depth=7,             # Increased for complexity
        subsample=0.8,           # Better generalization
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRefined Model Evaluation:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    
    # Sanity check: ensure predictions are not all the same
    print(f"\nSanity Check (Sample Predictions):")
    print(f"Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}")
    
    # Save model and encoder
    print(f"Saving artifacts to {MODEL_PATH} and {ENCODER_PATH}...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print("Refined model saved successfully.")

if __name__ == "__main__":
    train()
