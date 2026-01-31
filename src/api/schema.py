from typing import Optional
from pydantic import BaseModel, Field


# =========================
# INPUT SCHEMA
# =========================
class TransactionInput(BaseModel):
    # -------- Core transaction --------
    TransactionAmt: float = Field(..., example=100.0)
    ProductCD: Optional[str] = Field(None, example="W")
    TransactionDT: Optional[int] = Field(None, example=86400)

    # -------- Card --------
    card1: Optional[int] = Field(None, example=13926)
    card2: Optional[float] = Field(None, example=361)
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None

    # -------- Address --------
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None

    # -------- Email --------
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None

    # -------- Device --------
    DeviceType: Optional[str] = None
    id_30: Optional[str] = None
    id_31: Optional[str] = None

    # -------- M flags --------
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None


# =========================
# OUTPUT SCHEMA
# =========================
class PredictionOutput(BaseModel):
    fraud_proba: float = Field(..., example=0.23)
    fraud_pred: int = Field(..., example=0)
