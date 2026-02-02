from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
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

# Global variables for model and encoder
model = None
encoder = None

def load_artifacts():
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
load_artifacts()

@app.route('/predict-price', methods=['POST'])
def predict_price():
    """
    Endpoint to predict property price.
    Expects JSON: { "location": "Kochi", "size": 1200, "bhk": 2, "bath": 2 }
    """
    if model is None or encoder is None:
        return jsonify({"error": "Model not loaded on server. Please check logs."}), 503

    try:
        data = request.get_json()
        print(f"Incoming prediction request: {data}")
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # 1. Apply Price-Aware Augmentation Logic
        # is_training=False ensures we don't depend on missing 'price' field
        augmented_data = augment_features(data, is_training=False)
        
        # Ensure it's a list for processing
        if not isinstance(augmented_data, list):
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
        
        # Format response
        return jsonify({
            "predicted_price": int(prediction),
            "augmented_features": {
                "amenitiesScore": df['amenitiesScore'].iloc[0],
                "propertyAge": int(df['propertyAge'].iloc[0])
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
