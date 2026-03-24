# backend/main.py
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .schemas import (
    HistoryResponse,
    PredictionRequest,
    PredictionResponse,
    SampleResponse,
    LSTMPredictionResponse,
)
from .service import SignalService

import pandas as pd
from .model_files.config import MT5_CSV, ILE_ZZ, MIN_ZP, BACKSTEP
from .model_files.data_pipeline import full_ZZ
PIVOTS_CSV = MT5_CSV.with_name("pivots_ZZ.csv")
app = FastAPI(title="Trade Signal API", version="0.3.2-streaming")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ROOT = Path(__file__).resolve().parent / "model_files"
print(f"DEBUG: MODEL_ROOT is set to: {MODEL_ROOT}")
service = SignalService(model_root=MODEL_ROOT)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/sample", response_model=SampleResponse)
def sample(limit: int = 12) -> SampleResponse:
    candles = service.load_sample_candles(limit)
    if not candles:
        raise HTTPException(status_code=404, detail="Sample data unavailable")
    return SampleResponse(candles=candles)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if len(request.candles) < 5:
        raise HTTPException(status_code=400, detail="At least 5 candles are required")
    try:
        result = service.predict_direction(request.candles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/history", response_model=HistoryResponse)
def history(limit: int = 10000) -> HistoryResponse:
    candles = service.load_history(limit)
    if not candles:
        raise HTTPException(status_code=404, detail="History unavailable")
    return HistoryResponse(candles=candles, total=len(candles))


@app.get("/predict/from-file", response_model=PredictionResponse)
def predict_from_file() -> PredictionResponse:
    try:
        return service.predict_from_history()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/history/upload", response_model=HistoryResponse)
async def history_upload(limit: int = 10000, file: UploadFile = File(...)) -> HistoryResponse:
    try:
        content = await file.read()
        candles = service.load_history_from_bytes(content, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return HistoryResponse(candles=candles, total=len(candles))


@app.post("/predict/from-upload", response_model=PredictionResponse)
async def predict_from_upload(file: UploadFile = File(...)) -> PredictionResponse:
    try:
        content = await file.read()
        return service.predict_from_upload(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/retrain")
def retrain():
    return StreamingResponse(service.retrain_model_stream(), media_type="text/plain")


@app.post("/retrain/mlp")
def retrain_mlp():
    return StreamingResponse(service.retrain_model_stream(), media_type="text/plain")


@app.post("/retrain/lstm")
def retrain_lstm():
    return StreamingResponse(service.retrain_lstm_stream(), media_type="text/plain")


@app.post("/retrain/both")
def retrain_both():
    return StreamingResponse(service.retrain_both_stream(), media_type="text/plain")



def load_pivots_from_csv():
    pivots_path = service.sample_csv.parent / "pivots_ZZ.csv"
    if not pivots_path.exists():
        raise HTTPException(status_code=404, detail="Brak pliku pivots_ZZ.csv (jeszcze nie wygenerowany).")

    df_piv = pd.read_csv(pivots_path)

    text = service.sample_csv.read_text(encoding="utf-8", errors="ignore")
    candles_all = service._parse_csv_text(text)
    n_candles = len(candles_all)

    pivots_high = []
    pivots_low = []
    zigzag = []

    for _, row in df_piv.iterrows():
        bar = int(row["bar"])
        price = float(row["price"])
        typ = int(row["type"])

        if bar < 1 or bar > n_candles:
            continue

        time_str = candles_all[bar - 1].time
        pivot_dict = {
            "index": bar,
            "time": time_str,
            "price": price,
            "type": "high" if typ == 1 else "low",
        }

        zigzag.append(pivot_dict)
        if typ == 1:
            pivots_high.append(pivot_dict)
        else:
            pivots_low.append(pivot_dict)

    zigzag.sort(key=lambda p: p["index"])

    return pivots_high, pivots_low, zigzag


@app.get("/zigzag-csv")
def zigzag_from_csv(depth: int = 23, deviation: float = 0.00554, backstep: int = 4):
    candles = service.load_history(limit=0)
    if not candles:
        raise HTTPException(status_code=404, detail="Brak danych historycznych do ZigZag.")

    n_candles = len(candles)

    df = service._df_from_candles(candles)

    df_piv = full_ZZ(
        df_swieczki=df,
        ileZZ=ILE_ZZ,
        minzmnpkt=MIN_ZP,
        backstep=BACKSTEP
    )

    pivots_high = []
    pivots_low = []
    zigzag = []

    for _, row in df_piv.iterrows():
        bar = int(row["bar"])
        price = float(row["price"])
        typ = int(row["type"])

        if not (1 <= bar <= n_candles):
            continue

        c = candles[bar - 1]
        typ_str = "high" if typ == 1 else "low"

        p_dict = {
            "index": bar - 1,
            "time": c.time,
            "price": price,
            "type": typ_str,
        }

        zigzag.append(p_dict)
        if typ_str == "high":
            pivots_high.append(p_dict)
        else:
            pivots_low.append(p_dict)

    zigzag.sort(key=lambda x: x["index"])

    return {
        "total_candles": n_candles,
        "pivot_highs": pivots_high,
        "pivot_lows": pivots_low,
        "count_highs": len(pivots_high),
        "count_lows": len(pivots_low),
        "zigzag": zigzag,
    }


@app.get("/predict/lstm", response_model=LSTMPredictionResponse)
def predict_lstm_check():
    candles = service.load_history(limit=200)
    if not candles:
        raise HTTPException(status_code=404, detail="Brak danych historycznych")
    try:
        result = service.run_lstm_prediction(candles)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
