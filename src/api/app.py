import time
import uuid
import logging
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

from src.api.schema import TransactionInput
from src.api.inference import predict
from src.utils.logger import setup_logger

from prometheus_client import Counter, Histogram, generate_latest

# =========================
# APP
# =========================
app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

#==========================
# PROMETHEUS METRICS
#==========================
REQUEST_COUNT = Counter(
    'fraud-api-request_count',
    'Total number of requests to the fraud detection API',
)

REQUEST_LATENCY = Histogram(
    'fraud-api-request_latency_seconds',
    'Latency of requests to the fraud detection API in seconds',
)

REQUEST_ERRORS = Counter(
    'fraud-api-request_errors',
    'Total number of error responses from the fraud detection API',
)

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "ok"}

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict_fraud(request: TransactionInput):

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        df = pd.DataFrame([request.model_dump()])
        proba, pred = predict(df)

        response = {
        "fraud_proba": float(proba),
        "fraud_pred": int(pred),
        }

        return response

    except Exception:
        REQUEST_ERRORS.inc()
        raise

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
