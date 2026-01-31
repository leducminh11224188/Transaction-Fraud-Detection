import pandas as pd
import numpy as np
from src.features import build_features


def sample_df():
    return pd.DataFrame({
        "TransactionAmt": [100.0, 200.5, 0],
        "TransactionDT": [86400, 172800, 259200],
        "ProductCD": ["W", "H", "C"],
        "card1": [1111, 2222, 3333],
    })


def test_build_features_runs():
    df = sample_df()
    df_feat, _ = build_features(df)

    assert isinstance(df_feat, pd.DataFrame)
    assert len(df_feat) == len(df)


def test_feature_columns_created():
    df = sample_df()
    df_feat, _ = build_features(df)

    assert "TransactionAmt_Log" in df_feat.columns
    assert "Transaction_hour" in df_feat.columns


def test_no_all_nan_columns():
    df = sample_df()
    df_feat, _ = build_features(df)

    assert not df_feat.isna().all().any()


def test_missing_columns_safe():
    df = pd.DataFrame({"TransactionAmt": [100]})
    df_feat, _ = build_features(df)

    assert isinstance(df_feat, pd.DataFrame)


def test_feature_alignment():
    df = sample_df()
    feature_names = ["TransactionAmt", "TransactionAmt_Log"]

    df_feat, _ = build_features(df, feature_names=feature_names)

    assert list(df_feat.columns) == feature_names


def test_extreme_values():
    df = pd.DataFrame({
        "TransactionAmt": [0, 1e10, -1],
        "TransactionDT": [0, 999999999, -100],
    })

    df_feat, _ = build_features(df)

    assert len(df_feat) == 3
    assert np.isfinite(df_feat["TransactionAmt_Log"]).all()


EXPECTED_FEATURE_COUNT = 69

def test_feature_count_exact():
    df = sample_df()
    df_feat, _ = build_features(df)

    assert df_feat.shape[1] == EXPECTED_FEATURE_COUNT


def test_feature_count_stable():
    df = sample_df()
    df_feat, _ = build_features(df)

    expected_min_features = 20  # chỉnh theo pipeline bạn
    assert df_feat.shape[1] >= expected_min_features
