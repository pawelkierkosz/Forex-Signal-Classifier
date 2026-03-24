# backend/schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class Candle(BaseModel):
    time: str | None = Field(None, description="Timestamp")
    open: float
    high: float
    low: float
    close: float

    SMA5: Optional[float] = None
    SMA20: Optional[float] = None
    SMA50: Optional[float] = None
    EMA5: Optional[float] = None
    EMA20: Optional[float] = None
    EMA50: Optional[float] = None
    BB_mid: Optional[float] = None
    BB_up: Optional[float] = None
    BB_low: Optional[float] = None
    MACD: Optional[float] = None
    Signal: Optional[float] = None
    RSI: Optional[float] = None
    StochK: Optional[float] = None
    StochD: Optional[float] = None
    ATR: Optional[float] = None
    CCI: Optional[float] = None
    Momentum: Optional[float] = None

    @validator("high")
    def validate_high(cls, v, values):
        if "low" in values and v < values["low"]:
            return values["low"]
        return v

    @validator("low")
    def validate_low(cls, v, values):
        if "high" in values and v > values["high"]:
            return values["high"]
        return v

class LSTMPredictionResponse(BaseModel):
    last_close_price: float
    predicted_next_close: float
    predicted_movement_pips: float
    direction: str
    used_sequence_length: int

class PredictionRequest(BaseModel):
    candles: List[Candle] = Field(...)

class PredictionResponse(BaseModel):
    probability: float
    recommendation: str
    avg_pivot_height_pips: Optional[int] = None
    avg_pivot_width_bars: Optional[int] = None

class SampleResponse(BaseModel):
    candles: List[Candle]

class HistoryResponse(BaseModel):
    candles: List[Candle]
    total: int

class RetrainResponse(BaseModel):
    success: bool
    message: str
    logs: str
