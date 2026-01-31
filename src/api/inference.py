"""
Inference module for IEEE-CIS Fraud Detection
- Lazy load trained LightGBM model
- Apply identical feature engineering
- Predict fraud probability
- CI safe & production ready
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
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_lgb_model.txt")
FREQ_MAP_PATH = os.path.join(MODEL_DIR, "freq_maps.joblib")
FEATURE_NAME_PATH = os.path.join(MODEL_DIR, "feature_names.json")
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")


# =========================
# GLOBAL ARTIFACT HOLDERS (LAZY LOAD)
# =========================
model = None
freq_maps = None
FEATURE_NAMES = None
model_config = None


# =========================
# LAZY LOAD FUNCTION
# =========================
def load_artifacts():
    """
    Load model & preprocessing artifacts only once.
    Safe for CI & testing (no loading at import time).
    """
    global model, freq_maps, FEATURE_NAMES, model_config

    if model is None:
        print("📦 Loading inference artifacts...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        model = lgb.Booster(model_file=MODEL_PATH)

        if not os.path.exists(FREQ_MAP_PATH):
            raise FileNotFoundError(f"Freq map not found: {FREQ_MAP_PATH}")
        freq_maps = joblib.load(FREQ_MAP_PATH)

        if not os.path.exists(FEATURE_NAME_PATH):
            raise FileNotFoundError(f"Feature names not found: {FEATURE_NAME_PATH}")
        with open(FEATURE_NAME_PATH, "r") as f:
            FEATURE_NAMES = json.load(f)

        if not os.path.exists(MODEL_CONFIG_PATH):
            raise FileNotFoundError(f"Model config not found: {MODEL_CONFIG_PATH}")
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_config = json.load(f)

        print("✅ Model & artifacts loaded successfully")

    return model, freq_maps, FEATURE_NAMES, model_config


# =========================
# CORE PREDICT FUNCTION
# =========================
def predict(df: pd.DataFrame):
    """
    Run fraud inference on raw input dataframe

    Args:
        df (pd.DataFrame): raw transaction dataframe

    Returns:
        tuple:
            - fraud_proba (float)
            - fraud_pred (int)
    """

    model, freq_maps, FEATURE_NAMES, model_config = load_artifacts()
    THRESHOLD = model_config.get("optimal_threshold", 0.5)

    # -------------------------
    # Feature engineering
    # -------------------------
    df_feat, _ = build_features(
        df.copy(),
        freq_maps=freq_maps,
        feature_names=FEATURE_NAMES,
        is_train=False,
    )

    # -------------------------
    # Prediction
    # -------------------------
    fraud_proba = model.predict(
        df_feat,
        num_iteration=model.best_iteration
    )

    fraud_proba = np.asarray(fraud_proba).flatten()
    fraud_pred = (fraud_proba >= THRESHOLD).astype(int)

    # Ensure scalar output
    proba = float(fraud_proba[0])
    pred = int(fraud_pred[0])

    return proba, pred


# =========================
# LOCAL TEST
# =========================
if __name__ == "__main__":
    print("🧪 Running local inference test")

    sample = pd.DataFrame([{
        "TransactionAmt": 100.0,
        "ProductCD": "W",
        "TransactionDT": 86400,
        "card1": 13926,
        "card2": 361.0,
        "addr1": 315.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com"
    }])

    proba, pred = predict(sample)

    print("Fraud probability:", proba)
    print("Fraud prediction:", pred)
