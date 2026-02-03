from flask import Flask, request, jsonify
<<<<<<< HEAD
from flask_cors import CORS
=======
>>>>>>> fc3e6ced5ddc93d7448746c39205db828a73924e
import joblib
import pandas as pd
import numpy as np
import os
<<<<<<< HEAD
import sys
from feature_augmentation import augment_features

# Add current directory to path to ensure modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
# Enable CORS for all routes (important for cross-origin production requests)
CORS(app)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "price_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "location_encoder.joblib")
=======
from feature_augmentation import augment_features

app = Flask(__name__)

# Paths
MODEL_PATH = "price_model.pkl"
ENCODER_PATH = "location_encoder.joblib"
>>>>>>> fc3e6ced5ddc93d7448746c39205db828a73924e

# Global variables for model and encoder
model = None
encoder = None

def load_artifacts():
<<<<<<< HEAD
    """Load the trained model and encoder artifacts."""
    global model, encoder
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            encoder = joblib.load(ENCODER_PATH)
            print("✅ Model and Encoder loaded successfully.")
        else:
            print("⚠️ Warning: Model or Encoder files not found. Please train the model first.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# Load artifacts on startup
=======
    global model, encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Model and encoder loaded successfully.")
    else:
        print("Warning: Model or encoder files not found. Please run train_model.py first.")

# Initial load
>>>>>>> fc3e6ced5ddc93d7448746c39205db828a73924e
load_artifacts()

@app.route('/predict-price', methods=['POST'])
def predict_price():
<<<<<<< HEAD
    """
    Endpoint to predict property price.
    Expects JSON: { "location": "Kochi", "size": 1200, "bhk": 2, "bath": 2 }
    """
    if model is None or encoder is None:
        return jsonify({"error": "Model not loaded on server. Please check logs."}), 503

    try:
        data = request.get_json()
        print(f"Incoming prediction request: {data}")
=======
    global model, encoder
    
    # Reload if not loaded (lazy loading)
    if model is None or encoder is None:
        load_artifacts()
        if model is None or encoder is None:
            return jsonify({"error": "Model files not found on server."}), 500

    try:
        data = request.get_json()
>>>>>>> fc3e6ced5ddc93d7448746c39205db828a73924e
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # 1. Apply Price-Aware Augmentation Logic
        # is_training=False ensures we don't depend on missing 'price' field
        augmented_data = augment_features(data, is_training=False)
        
        # Ensure it's a list for processing
        if not isinstance(augmented_data, list):
<<<<<<< HEAD
            augmented_data = [augmented_data]
            
        df = pd.DataFrame(augmented_data)
        
        # 2. Preprocessing
        # Encode location using the saved encoder
        try:
            loc_encoded = encoder.transform(df[['location']])
            loc_df = pd.DataFrame(loc_encoded, columns=encoder.get_feature_names_out(['location']))
        except Exception as e:
            print(f"Encoding error: {e}")
            return jsonify({"error": f"Invalid location: {df['location'].iloc[0]}"}), 400
            
        # Combine features to match training schema
        X = pd.concat([
            df[['size', 'bhk', 'bath', 'amenitiesScore', 'propertyAge']].reset_index(drop=True),
            loc_df.reset_index(drop=True)
        ], axis=1)
        
        # 3. Predict
        prediction = model.predict(X)[0]
        
        # 4. Risk Assessment Logic (Simulated for Demo)
        # Low Risk: High amenitiesScore, Low propertyAge
        # High Risk: Low amenitiesScore, High propertyAge
        amenities = df['amenitiesScore'].iloc[0]
        age = df['propertyAge'].iloc[0]
        
        risk_score = (amenities / 10.0) - (age / 50.0)
        if risk_score > 0.6:
            risk_category = "Low"
        elif risk_score > 0.3:
            risk_category = "Medium"
        else:
            risk_category = "High"

        # Format response
        return jsonify({
            "predicted_price": int(prediction),
            "risk_category": risk_category,
            "augmented_features": {
                "amenitiesScore": float(amenities),
                "propertyAge": int(age)
            }
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "environment": os.environ.get("NODE_ENV", "production")
    }), 200

if __name__ == "__main__":
    # Use PORT environment variable for Render compatibility
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
=======
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
>>>>>>> fc3e6ced5ddc93d7448746c39205db828a73924e
