# utils/predictor.py

import numpy as np
import joblib
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # /utils/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))    # /human_activity_recognition/

# Load model artifacts
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ensemble_VC_har_model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scalar.pkl")
SELECTOR_PATH = os.path.join(ROOT_DIR, "models", "feature_selector.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH) if os.path.exists(SELECTOR_PATH) else None
except Exception as e:
    raise RuntimeError(f"Error loading model files: {e}")

# HAR activity labels
activity_labels = [
    'STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS',
     'WALKING_UPSTAIRS'
]

def predict_activity(features):
    """
    Predict activity from a 561-feature array (adds 2 engineered features).

    Args:
        features (np.ndarray): Shape (561,)

    Returns:
        str: Predicted activity label
    """
    # Ensure NumPy array of floats
    features = np.array(features, dtype=float)

    if features.shape != (561,):
        raise ValueError(f"Expected 561 features, got {features.shape[0]}")

    # Feature engineering
    body_acc_mean = np.mean(features[:100])
    body_acc_std = np.std(features[:100])
    features = np.append(features, [body_acc_mean, body_acc_std])  # Now shape = (563,)

    # Preprocess
    features = features.reshape(1, -1)
    features = scaler.transform(features)
    # if selector:
    #     features = selector.transform(features)

    # Predict and return directly (model outputs label strings)
    prediction = model.predict(features)[0]
    
    # Verify the prediction is a valid label
    if prediction not in activity_labels:
        raise ValueError(f"Invalid prediction: {prediction}. Expected one of {activity_labels}")
    
    return prediction