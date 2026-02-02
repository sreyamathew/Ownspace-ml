from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from feature_augmentation import augment_features

app = Flask(__name__)

# Paths
MODEL_PATH = "price_model.pkl"
ENCODER_PATH = "location_encoder.joblib"

# Global variables for model and encoder
model = None
encoder = None

def load_artifacts():
    global model, encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Model and encoder loaded successfully.")
    else:
        print("Warning: Model or encoder files not found. Please run train_model.py first.")

# Initial load
load_artifacts()

@app.route('/predict-price', methods=['POST'])
def predict_price():
    global model, encoder
    
    # Reload if not loaded (lazy loading)
    if model is None or encoder is None:
        load_artifacts()
        if model is None or encoder is None:
            return jsonify({"error": "Model files not found on server."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # 1. Apply Price-Aware Augmentation Logic
        # is_training=False ensures we don't depend on missing 'price' field
        augmented_data = augment_features(data, is_training=False)
        
        # Ensure it's a list for processing
        if not isinstance(augmented_data, list):
            items = [augmented_data]
        else:
            items = augmented_data

        input_df = pd.DataFrame(items)

        # 2. Preprocessing (Same as in training)
        # Required columns
        required_cols = ['size', 'bhk', 'bath', 'amenitiesScore', 'propertyAge', 'location']
        for col in required_cols:
            if col not in input_df.columns:
                 return jsonify({"error": f"Missing required field: {col}"}), 400

        # Encode location
        location_encoded = encoder.transform(input_df[['location']])
        location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['location']))
        
        # Combine features
        X = pd.concat([
            input_df[['size', 'bhk', 'bath', 'amenitiesScore', 'propertyAge']].reset_index(drop=True),
            location_df.reset_index(drop=True)
        ], axis=1)

        # 3. Predict
        predictions = model.predict(X)
        
        # Format response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "predicted_price": float(pred),
                "augmented_features": {
                    "amenitiesScore": float(items[i]['amenitiesScore']),
                    "propertyAge": int(items[i]['propertyAge'])
                }
            })

        return jsonify(results[0] if not isinstance(data, list) else results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Model API is running", "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
