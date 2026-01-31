"""
Inference module for IEEE-CIS Fraud Detection
- Load trained LightGBM model
- Apply identical feature engineering
- Predict fraud probability
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.features import build_features

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_lgb_model.txt")
FREQ_MAP_PATH = os.path.join(MODEL_DIR, "freq_maps.joblib")
FEATURE_NAME_PATH = os.path.join(MODEL_DIR, "feature_names.json")
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

# =========================
# LOAD ARTIFACTS (ONCE)
# =========================
model = None

def load_model():
    global model
    if model is None:
        print("📦 Loading inference artifacts...")
        model = lgb.Booster(model_file=MODEL_PATH)
    return model

freq_maps = joblib.load(FREQ_MAP_PATH)

with open(FEATURE_NAME_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

with open(MODEL_CONFIG_PATH, "r") as f:
    model_config = json.load(f)

THRESHOLD = model_config.get("optimal_threshold", 0.5)

print("✅ Model & artifacts loaded successfully")


# =========================
# CORE PREDICT FUNCTION
# =========================
def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run fraud inference on raw input dataframe

    Args:
        df (pd.DataFrame): raw transaction dataframe

    Returns:
        pd.DataFrame with:
            - fraud_proba
            - fraud_pred
    """

    model = load_model()

    # -------------------------
    # Feature engineering
    # -------------------------
    df_feat, _ = build_features(df.copy(),
                                freq_maps=freq_maps,
                                feature_names=FEATURE_NAMES,
                                is_train=False)

    # -------------------------
    # Prediction
    # -------------------------
    fraud_proba = model.predict(
        df_feat,
        num_iteration=model.best_iteration
    )

    fraud_pred = (fraud_proba >= THRESHOLD).astype(int)

    proba = float(fraud_proba)
    pred = int(fraud_pred)
    return proba, pred



# =========================
# LOCAL TEST
# =========================
if __name__ == "__main__":
    print("🧪 Running local inference test")

    # Dummy example (replace with real transaction row)
    sample = pd.DataFrame([{
        "TransactionAmt": 100.0,
        "ProductCD": "W",
        "card1": 13926,
        "card2": 361.0,
        "addr1": 315.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com"
    }])

    output = predict(sample)
    print(output)
