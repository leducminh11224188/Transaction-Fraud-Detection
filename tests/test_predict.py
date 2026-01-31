import pandas as pd
from src.api.inference import predict


def sample_input():
    return pd.DataFrame({
        "TransactionAmt": [100.0],
        "TransactionDT": [86400],
        "ProductCD": ["W"],
        "card1": [1111],
    })


def test_predict_output_type():
    df = sample_input()
    proba, pred = predict(df)

    assert isinstance(proba, float)
    assert isinstance(pred, int)


def test_probability_range():
    df = sample_input()
    proba, pred = predict(df)

    assert 0 <= proba <= 1
    assert pred in [0, 1]


def test_predict_deterministic():
    df = sample_input()

    p1, _ = predict(df)
    p2, _ = predict(df)

    assert p1 == p2


def test_missing_columns():
    df = pd.DataFrame({"TransactionAmt": [100]})
    proba, pred = predict(df)

    assert 0 <= proba <= 1


def test_extreme_values():
    df = pd.DataFrame({
        "TransactionAmt": [1e9],
        "TransactionDT": [999999999],
    })

    proba, pred = predict(df)

    assert 0 <= proba <= 1
