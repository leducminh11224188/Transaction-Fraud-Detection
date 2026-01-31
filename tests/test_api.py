from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_predict_success():
    payload = {
        "TransactionAmt": 100.0,
        "TransactionDT": 86400,
        "ProductCD": "W",
        "card1": 1111
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "fraud_proba" in data
    assert "fraud_pred" in data
    assert 0 <= data["fraud_proba"] <= 1


def test_invalid_type():
    payload = {
        "TransactionAmt": "abc"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_required():
    payload = {}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_nested_payload_rejected():
    payload = {
        "TransactionAmt": 100,
        "extra": {"hack": "object"}
    }

    response = client.post("/predict", json=payload)

    assert response.status_code in [200, 422]


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "fraud_api_request_count_total" in response.text

