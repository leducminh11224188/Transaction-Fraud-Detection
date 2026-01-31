"""
Training pipeline for IEEE-CIS Fraud Detection
- LightGBM (lgb.train)
- Feature engineering pipeline
- Export deployable artifacts
"""

import os
import json
import gc
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve

from preprocessing import load_and_merge_data, reduce_mem_usage
from features import build_features

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "IEEE-CIS")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# TRAIN CONFIG
# =========================
SEED = 42
TEST_SIZE = 0.2

# =========================
# LightGBM params
# (aligned with notebook logic)
# =========================
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "seed": SEED,
}

NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 100

# =========================
# PREPROCESS
# =========================
def prepare_data(df, is_train=True, freq_maps=None):
    df, freq_maps = build_features(df, freq_maps)

    df = reduce_mem_usage(df)

    if is_train:
        y = df["isFraud"].astype(int)
        X = df.drop(columns=["isFraud"])
    else:
        y = None
        X = df

    # ------------------------
    # Separate column types
    # ------------------------
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # ------------------------
    # Fill categorical only
    # ------------------------
    X[cat_cols] = X[cat_cols].fillna("missing")

    return X, y, freq_maps

# =========================
# TRAIN
# =========================
def train():
    print("🚀 Loading data...")
    train_df, _ = load_and_merge_data(DATA_DIR)

    print("🧠 Feature engineering...")
    X, y, freq_maps = prepare_data(train_df, is_train=True)

    feature_names = X.columns.tolist()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    print(f"Train shape: {X_tr.shape}")
    print(f"Valid shape: {X_val.shape}")
    print(f"Fraud rate train: {y_tr.mean():.4f}")
    print(f"Fraud rate valid: {y_val.mean():.4f}")

    print(X.dtypes.value_counts())

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)

    print("🔥 Training LightGBM...")
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING),
            lgb.log_evaluation(100)
        ]
    )

    # =========================
    # EVALUATION
    # =========================
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = roc_auc_score(y_val, val_pred)

    precision, recall, thresholds = precision_recall_curve(y_val, val_pred)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    best_threshold = thresholds[np.argmax(f1)]

    print(f"✅ Validation AUC: {val_auc:.6f}")
    print(f"🎯 Best threshold (F1-opt): {best_threshold:.4f}")

    # =========================
    # SAVE ARTIFACTS
    # =========================
    model.save_model(os.path.join(MODEL_DIR, "fraud_lgb_model.txt"))

    joblib.dump(freq_maps, os.path.join(MODEL_DIR, "freq_maps.joblib"))

    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump({
            "best_iteration": model.best_iteration,
            "validation_auc": val_auc,
            "optimal_threshold": float(best_threshold),
            "lgb_params": lgb_params
        }, f, indent=2)

    print("📦 All artifacts saved successfully")
    print("🏁 Training completed")

    gc.collect()


if __name__ == "__main__":
    train()
