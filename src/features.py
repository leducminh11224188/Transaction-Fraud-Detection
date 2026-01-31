"""
Feature engineering module for IEEE-CIS Fraud Detection
Production-safe for training & inference
"""

import numpy as np
import pandas as pd
import gc


# =====================================================
# Utils
# =====================================================
def ensure_col(df, col, default):
    if col not in df.columns:
        df[col] = default
    return df


def ensure_numeric(df, col, default=0):
    ensure_col(df, col, default)
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


def ensure_string(df, col, default="unknown"):
    ensure_col(df, col, default)
    df[col] = df[col].astype(str).fillna(default)
    return df


# =====================================================
# 1. Transaction Amount
# =====================================================
def transaction_amount_features(df):
    ensure_numeric(df, "TransactionAmt", 0)

    amt = df["TransactionAmt"].clip(lower=0)
    df["TransactionAmt_Log"] = np.log1p(amt)

    df["TransactionAmt_decimal"] = (
        (df["TransactionAmt"] - df["TransactionAmt"].astype(int)) * 1000
    ).astype("int32")

    df["TransactionAmt_is_round"] = (
        df["TransactionAmt"] == df["TransactionAmt"].astype(int)
    ).astype("int8")

    cents = (df["TransactionAmt"] * 100 % 100).astype("int16")
    df["TransactionAmt_ends_00"] = (cents == 0).astype("int8")
    df["TransactionAmt_ends_95"] = (cents == 95).astype("int8")
    df["TransactionAmt_ends_99"] = (cents == 99).astype("int8")

    df["TransactionAmt_bin"] = pd.cut(
        df["TransactionAmt"],
        bins=[0, 50, 100, 200, 500, 1000, 5000, 10000, np.inf],
        labels=False,
    )

    return df


# =====================================================
# 2. Time
# =====================================================
def time_features(df):
    ensure_numeric(df, "TransactionDT", 0)

    df["Transaction_hour"] = (df["TransactionDT"] // 3600) % 24
    df["Transaction_dow"] = (df["TransactionDT"] // (3600 * 24)) % 7
    df["Transaction_week"] = df["TransactionDT"] // (3600 * 24 * 7)

    df["Transaction_is_weekend"] = (df["Transaction_dow"] >= 5).astype("int8")
    df["Transaction_is_night"] = (df["Transaction_hour"] < 6).astype("int8")
    df["Transaction_is_business"] = (
        (df["Transaction_hour"] >= 9) & (df["Transaction_hour"] < 17)
    ).astype("int8")

    return df


# =====================================================
# 3. Card
# =====================================================
def card_features(df):
    for c in ["card1", "card2", "card3", "card5"]:
        ensure_numeric(df, c, -1)

    for c in ["card4", "card6"]:
        ensure_string(df, c, "unknown")

    ensure_numeric(df, "addr1", -1)

    df["card1_card2"] = df["card1"].astype(int).astype(str) + "_" + df["card2"].astype(int).astype(str)
    df["card4_card6"] = df["card4"] + "_" + df["card6"]
    df["card1_addr1"] = df["card1"].astype(int).astype(str) + "_" + df["addr1"].astype(int).astype(str)

    return df


# =====================================================
# 4. Email
# =====================================================
def email_features(df):
    for col in ["P_emaildomain", "R_emaildomain"]:
        ensure_string(df, col, "unknown")
        df[f"{col}_prefix"] = df[col].str.split(".").str[0]
        df[f"{col}_suffix"] = df[col].str.split(".").str[-1]

    common = ["gmail", "yahoo", "hotmail", "outlook"]
    df["P_email_is_common"] = df["P_emaildomain_prefix"].isin(common).astype("int8")
    df["email_domain_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype("int8")

    return df


# =====================================================
# 5. Device
# =====================================================
def device_features(df):
    ensure_string(df, "DeviceType", "unknown")
    ensure_string(df, "id_31", "unknown")
    ensure_string(df, "id_30", "unknown")

    df["DeviceType_is_mobile"] = (df["DeviceType"] == "mobile").astype("int8")
    df["Browser"] = df["id_31"].str.split().str[0]
    df["OS"] = df["id_30"].str.split().str[0]

    return df


# =====================================================
# 6. Address + Distance
# =====================================================
def address_features(df):
    ensure_numeric(df, "addr1", -1)
    ensure_numeric(df, "addr2", -1)
    ensure_numeric(df, "dist1", 0)
    ensure_numeric(df, "dist2", 0)

    df["addr1_missing"] = (df["addr1"] < 0).astype("int8")
    df["addr2_missing"] = (df["addr2"] < 0).astype("int8")

    df["dist1_log"] = np.log1p(df["dist1"].clip(lower=0))
    df["dist2_log"] = np.log1p(df["dist2"].clip(lower=0))


    return df


# =====================================================
# 7. Aggregates
# =====================================================
def aggregate_features(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return df

    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df[f"{prefix}_sum"] = df[cols].sum(axis=1)
    df[f"{prefix}_mean"] = df[cols].mean(axis=1)
    df[f"{prefix}_std"] = df[cols].std(axis=1).fillna(0)

    return df


def m_features(df):
    for i in range(1, 10):
        col = f"M{i}"
        ensure_string(df, col, "unknown")
        df[f"{col}_isT"] = (df[col] == "T").astype("int8")

    df["M_true_count"] = df[[f"M{i}_isT" for i in range(1, 10)]].sum(axis=1)
    return df


# =====================================================
# 8. FULL PIPELINE
# =====================================================
def build_features(df, freq_maps=None, feature_names=None, is_train=False):
    df = df.copy()

    df = transaction_amount_features(df)
    df = time_features(df)
    df = card_features(df)
    df = email_features(df)
    df = device_features(df)
    df = address_features(df)
    df = aggregate_features(df, "V")
    df = aggregate_features(df, "C")
    df = aggregate_features(df, "D")
    df = m_features(df)

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)


    # ---------- Frequency Encoding ----------
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if is_train:
        freq_maps = {}
        for c in cat_cols:
            freq_maps[c] = df[c].value_counts(dropna=False).to_dict()
            df[c] = df[c].map(freq_maps[c]).fillna(0)
    else:
        for c in cat_cols:
            if freq_maps and c in freq_maps:
                df[c] = df[c].map(freq_maps[c]).fillna(0)
            else:
                df[c] = 0

    df[cat_cols] = df[cat_cols].astype("float32")

    # ---------- ALIGN FEATURE SPACE ----------
    if feature_names is not None:
        df = df.reindex(columns=feature_names, fill_value=0)

    gc.collect()
    return df, freq_maps
