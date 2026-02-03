from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys

from feature_augmentation import augment_features

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Paths (robust for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "price_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "location_encoder.joblib")

# Global variables
model = None
encoder = None


def load_artifacts():
    """Load ML model and encoder"""
    global model, encoder
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            encoder = joblib.load(ENCODER_PATH)
            print("✅ Model and encoder loaded successfully")
        else:
            print("⚠️ Model or encoder file not found")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")


# Load model on startup
load_artifacts()


@app.route("/predict-price", methods=["POST"])
def predict_price():
    if model is None or encoder is None:
        return jsonify({"error": "Model not loaded on server"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Apply feature augmentation
        augmented = augment_features(data, is_training=False)
        if not isinstance(augmented, list):
            augmented = [augmented]

        df = pd.DataFrame(augmented)

        # Required columns
        required_cols = ['size', 'bhk', 'bath', 'amenitiesScore', 'propertyAge', 'location']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({"error": f"Missing required field: {col}"}), 400

        # Encode location
        try:
            loc_encoded = encoder.transform(df[['location']])
            loc_df = pd.DataFrame(
                loc_encoded,
                columns=encoder.get_feature_names_out(['location'])
            )
        except Exception:
            return jsonify({"error": "Invalid location value"}), 400

        # Combine features
        X = pd.concat([
            df[['size', 'bhk', 'bath', 'amenitiesScore', 'propertyAge']].reset_index(drop=True),
            loc_df.reset_index(drop=True)
        ], axis=1)

        # Predict
        prediction = model.predict(X)[0]

        # Simple risk assessment
        amenities = df['amenitiesScore'].iloc[0]
        age = df['propertyAge'].iloc[0]

        risk_score = (amenities / 10.0) - (age / 50.0)
        if risk_score > 0.6:
            risk = "Low"
        elif risk_score > 0.3:
            risk = "Medium"
        else:
            risk = "High"

        return jsonify({
            "predicted_price": int(prediction),
            "risk_category": risk,
            "augmented_features": {
                "amenitiesScore": float(amenities),
                "propertyAge": int(age)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
