# backend/service.py
import csv
import io
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Iterator
from collections import deque
import numpy as np
import pandas as pd
import torch
import MetaTrader5 as mt5

from .schemas import (
    Candle,
    PredictionResponse,
    LSTMPredictionResponse
)

from .model_files.config import (
    HYPERPARAMS, 
    DL_ZZ, LICZBA_SW, MIN_ZP, ILE_ZZ, BACKSTEP,
    MODEL_PATH, STATS_PATH, MT5_CSV
)

from .model_files import data_pipeline as dp
from .model_files_lstm.lstm_section import update_zigzag, normalize_features, calculate_technical_indicators
from .model_files_lstm.lstm_model import LSTMRegressor
from .model_files_lstm.lstm_config import (
    MODEL_PATH as LSTM_MODEL_PATH,
    STATS_PATH as LSTM_STATS_PATH,
    HYPERPARAMS as LSTM_HYPERPARAMS,
    WSP_DELTA,
    DL_ZZ as LSTM_DL_ZZ,
    LICZBA_SW as LSTM_LICZBA_SW,
    MIN_ZP as LSTM_MIN_ZP,
    ILE_ZZ as LSTM_ILE_ZZ,
    BACKSTEP as LSTM_BACKSTEP
)

@dataclass
class SignalService:
    model_root: Path

    def __post_init__(self):
        default_csv = self.model_root / "ohlc_EURUSD_H1.csv"
        self.sample_csv = MT5_CSV if MT5_CSV.exists() else default_csv

        dp.PIVOTS_CSV_PATH = MT5_CSV.parent / "pivots_ZZ.csv"

        self.model: torch.nn.Module | None = None
        self._input_dim: int | None = None
        self.norm_stats = self._load_norm_stats()

        self.lstm_model: LSTMRegressor | None = None
        self.lstm_stats: dict | None = None
        self._load_lstm_resources()

    def _load_lstm_resources(self):
        print("\n>>> ŁADOWANIE LSTM <<<")
        if not LSTM_STATS_PATH.exists() or not LSTM_MODEL_PATH.exists():
            print(f"Brak plików modelu LSTM: {LSTM_MODEL_PATH} lub {LSTM_STATS_PATH}")
            return

        try:
            with open(LSTM_STATS_PATH, "rb") as f:
                self.lstm_stats = pickle.load(f)

            hp = LSTM_HYPERPARAMS
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.load(LSTM_MODEL_PATH, map_location=device)

            weight = state.get("lstm.weight_ih_l0")
            input_dim = weight.shape[1] if weight is not None else 33 

            self.lstm_model = LSTMRegressor(
                input_dim=input_dim,
                hidden_size=hp["lstm_hidden_size"],
                num_layers=hp["lstm_num_layers"],
                dropout_p=hp["dropout_p"],
                bidirectional=hp["bidirectional"]
            )
            self.lstm_model.load_state_dict(state)
            self.lstm_model.to(device)
            self.lstm_model.eval()
            print(f"LSTM załadowany poprawnie. Input dim: {input_dim}, Device: {device}")

        except Exception as e:
            print(f"Krytyczny błąd ładowania LSTM: {e}")
            self.lstm_model = None

    def run_lstm_prediction(self, candles: List[Candle]) -> LSTMPredictionResponse:
        """
        Uruchamia predykcję LSTM, odtwarzając dokładnie ten sam pipeline danych co w treningu.
        """
        if self.lstm_model is None:
            self._load_lstm_resources()
            if self.lstm_model is None:
                raise ValueError("Model LSTM nie jest gotowy (brak plików wagi/statystyk).")

        seq_len = LSTM_HYPERPARAMS["seq_len"]
        
        required_history = seq_len + LSTM_ILE_ZZ + 60
        
        if len(candles) < required_history:
             raise ValueError(f"Za mało świec do analizy LSTM. Wymagane: {required_history}, Dostępne: {len(candles)}")

        df_raw = self._df_from_candles(candles)
        
        try:
            X_tensor, last_close = self._prepare_lstm_input_sequence(df_raw, seq_len)
        except Exception as e:
            raise ValueError(f"Błąd przygotowania danych dla LSTM: {e}")

        device = next(self.lstm_model.parameters()).device
        X_tensor = X_tensor.to(device)

        with torch.no_grad():
            out_delta_norm = self.lstm_model(X_tensor).item()

        predicted_delta = out_delta_norm / WSP_DELTA
        predicted_price = last_close + predicted_delta
        
        diff_pips = predicted_delta * 10000.0
        direction = "UP" if predicted_delta > 0 else "DOWN"

        return LSTMPredictionResponse(
            last_close_price=last_close,
            predicted_next_close=predicted_price,
            predicted_movement_pips=round(diff_pips, 2),
            direction=direction,
            used_sequence_length=seq_len
        )

    def _prepare_lstm_input_sequence(self, df: pd.DataFrame, seq_len: int):
        df_ind = calculate_technical_indicators(df, len(df))

        df_ind.index = df.index 
        
        H = df["H"].values
        L = df["L"].values
        C = df["C"]
        O = df["O"]
        
        last_row, last_count = None, 0
        piv_deque = deque(maxlen=LSTM_DL_ZZ + 1)
        
        rows_input = []
        indices_for_open = [] 
        
        start_idx = 1
        end_idx = len(df)
        
        for sw in range(start_idx, end_idx + 1):
            new_row, new_count, flag = update_zigzag(
                sw, last_row, last_count, H, L, 
                LSTM_ILE_ZZ, LSTM_MIN_ZP, LSTM_BACKSTEP
            )
            if flag == 1 and last_row is not None:
                piv_deque.append(last_row)
            last_row, last_count = new_row, new_count
            
            if sw > (end_idx - seq_len):
                
                all_dyn = list(piv_deque)
                if last_row is not None:
                    all_dyn.append(last_row)
                
                needed = LSTM_DL_ZZ + 1
                slice_piv = all_dyn[-needed:] if len(all_dyn) >= needed else all_dyn
                prices = [p[1] for p in slice_piv]
                
                if len(prices) < needed:
                    prices = [np.nan] * (needed - len(prices)) + prices
                
                closes = []
                for j in range(LSTM_LICZBA_SW - 1, -1, -1):
                    idx = sw - j
                    val = C.loc[idx] if (idx >= 1 and idx <= end_idx) else np.nan
                    closes.append(val)
                
                rows_input.append(prices + closes)
                indices_for_open.append(sw)

        cols_input = [f"ZZ_price_{k + 1}" for k in range(LSTM_DL_ZZ + 1)] + [f"C_close_{k + 1}" for k in range(LSTM_LICZBA_SW)]
        df_input = pd.DataFrame(rows_input, columns=cols_input)
        
        selected_inds = df_ind.loc[indices_for_open].reset_index(drop=True)
        
        df_full = pd.concat([df_input, selected_inds], axis=1)
        
        open_vals = O.loc[indices_for_open].reset_index(drop=True)
        
        X_norm, _ = normalize_features(df_full, open_vals, LSTM_DL_ZZ, LSTM_LICZBA_SW, stats=self.lstm_stats)
        
        X_norm = X_norm.fillna(0.0)

        X_np = X_norm.to_numpy(dtype=np.float32)
        X_tensor = torch.from_numpy(X_np).unsqueeze(0) 
        
        last_close_price = C.iloc[-1]
        
        return X_tensor, last_close_price

    def _calculate_user_indicators(self, candles: List[Candle]) -> List[Candle]:
        if not candles: return []
        data = [{"C": c.close, "H": c.high, "L": c.low, "O": c.open, "original_obj": c} for c in candles]
        df_range = pd.DataFrame(data)

        Close = df_range["C"].to_numpy()

        df_range["SMA5"] = pd.Series(Close).rolling(5).mean()
        df_range["SMA20"] = pd.Series(Close).rolling(20).mean()
        df_range["SMA50"] = pd.Series(Close).rolling(50).mean()

        for p in [5, 20, 50, 12, 26]:
            df_range[f"EMA{p}"] = pd.Series(Close).ewm(span=p, adjust=False).mean()

        # MACD
        df_range["MACD"] = df_range["EMA12"] - df_range["EMA26"]
        df_range["Signal"] = df_range["MACD"].ewm(span=9, adjust=False).mean()

        # BB
        r20 = pd.Series(Close).rolling(20)
        df_range["BB_mid"] = r20.mean()
        std = r20.std()
        df_range["BB_up"] = df_range["BB_mid"] + 2*std
        df_range["BB_low"] = df_range["BB_mid"] - 2*std

        # RSI
        delta = df_range["C"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df_range["RSI"] = 100 - (100 / (1 + rs))

        # Stoch
        l14 = df_range["L"].rolling(14).min()
        h14 = df_range["H"].rolling(14).max()
        df_range["StochK"] = 100 * ((df_range["C"] - l14) / (h14 - l14))
        df_range["StochD"] = df_range["StochK"].rolling(3).mean()
        
        # ATR
        h_l = df_range["H"] - df_range["L"]
        h_pc = (df_range["H"] - df_range["C"].shift(1)).abs()
        l_pc = (df_range["L"] - df_range["C"].shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        df_range["ATR"] = tr.rolling(14).mean()
        
        # CCI
        tp = (df_range["H"] + df_range["L"] + df_range["C"]) / 3
        sma_tp = tp.rolling(20).mean()
        mad = (tp - sma_tp).abs().rolling(20).mean()
        df_range["CCI"] = (tp - sma_tp) / (0.015 * mad)

        df_range["Momentum"] = df_range["C"].diff(10)
        
        df_range = df_range.fillna(0.0)

        updated = []
        for i, row in df_range.iterrows():
            c = row["original_obj"]
            for col in ["SMA5","SMA20","SMA50","EMA5","EMA20","EMA50","BB_mid","BB_up","BB_low","MACD","Signal","RSI","StochK","StochD","ATR","CCI","Momentum"]:
                val = float(row[col])
                setattr(c, col, val)
            updated.append(c)
        return updated

    def _fetch_live_candle(self) -> Optional[Candle]:
        try:
            if not mt5.initialize(): return None
            rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1)
            mt5.shutdown()
            if rates is None or len(rates) == 0: return None
            r = rates[0]
            dt = datetime.fromtimestamp(r['time'], tz=timezone.utc)
            return Candle(time=dt.isoformat(), open=float(r['open']), high=float(r['high']), low=float(r['low']), close=float(r['close']))
        except: return None

    def _parse_csv_text(self, text: str) -> List[Candle]:
        candles = []
        reader = csv.reader(io.StringIO(text))
        for row in reader:
            if len(row) < 6: continue
            try: ts = datetime.fromisoformat(row[0]).isoformat()
            except: ts = row[0]
            try: o,h,l,c = map(float, row[2:6])
            except: continue
            candles.append(Candle(time=ts, open=o, high=h, low=l, close=c))
        candles.reverse()
        return candles

    def load_history(self, limit: int = 10000) -> List[Candle]:
        history = []
        if self.sample_csv.exists():
            text = self.sample_csv.read_text(encoding="utf-8", errors="ignore")
            history = self._parse_csv_text(text)

        live = self._fetch_live_candle()
        if history: history.pop()
        if live: history.append(live)
        
        full = self._calculate_user_indicators(history)
        return full[-limit:] if limit else full

    def load_sample_candles(self, limit: int = 12) -> List[Candle]: return self.load_history(limit)
    
    def load_history_from_bytes(self, content: bytes, limit: int = 10000) -> List[Candle]:
        text = content.decode("utf-8", errors="ignore")
        hist = self._parse_csv_text(text)
        full = self._calculate_user_indicators(hist)
        return full[-limit:] if limit else full

    class MLP(torch.nn.Module):
        def __init__(self, input_dim, hidden_sizes, output_dim, dropout_p=0.5):
            super().__init__()
            layers=[]
            prev=input_dim
            for h in hidden_sizes:
                layers+=[torch.nn.Linear(prev,h), torch.nn.ReLU(), torch.nn.Dropout(dropout_p)]
                prev=h
            layers+=[torch.nn.Linear(prev, output_dim)]
            self.net=torch.nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    def _load_norm_stats(self):
        if not STATS_PATH.exists(): return {}
        with STATS_PATH.open("rb") as f: return pickle.load(f)

    def _ensure_model(self, input_dim: int):
        if self.model is None or self._input_dim != input_dim:
            self.model = self.MLP(input_dim, HYPERPARAMS["hidden_sizes"], HYPERPARAMS["output_dim"], HYPERPARAMS["dropout_p"])
            if MODEL_PATH.exists():
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
                self.model.eval()
            self._input_dim=input_dim

    # MLP Prediction
    def _predict_from_df(self, df: pd.DataFrame) -> PredictionResponse:
        if len(df)<2: raise ValueError("Za mało danych")
        if not self.norm_stats: self.norm_stats=self._load_norm_stats()
        X, _, h, w = dp.features_last_bar(df, MIN_ZP, ILE_ZZ, BACKSTEP, DL_ZZ, LICZBA_SW, self.norm_stats)
        if X is None: raise ValueError("Brak features")
        self._ensure_model(X.shape[1])
        with torch.no_grad(): prob = torch.sigmoid(self.model(torch.from_numpy(X))).item()
        return PredictionResponse(probability=prob, recommendation="buy" if prob>=0.5 else "sell", avg_pivot_height_pips=h, avg_pivot_width_bars=w)

    def _df_from_candles(self, candles: List[Candle]) -> pd.DataFrame:
        if len(candles)<2: raise ValueError("Za mało danych")
        return pd.DataFrame([(c.open,c.high,c.low,c.close) for c in candles], columns=["O","H","L","C"], index=range(1, len(candles)+1))

    def _df_from_bytes_mt5(self, content: bytes) -> pd.DataFrame:
        from io import StringIO
        raw = pd.read_csv(StringIO(content.decode("utf-8", errors="ignore")), header=None)
        df = raw.iloc[:, 2:6].copy(); df.columns=["O","H","L","C"]
        df = df.iloc[::-1].reset_index(drop=True); df.index=range(1,len(df)+1)
        return df

    def predict_direction(self, candles): return self._predict_from_df(self._df_from_candles(candles))
    def predict_from_history(self): return self.predict_direction(self.load_history(0))
    def predict_from_upload(self, c): return self._predict_from_df(self._df_from_bytes_mt5(c))

    def _stream_training_process(self, script_path, cwd, on_success=None, prefix=""):
        if not script_path.exists(): yield f"{prefix}BŁĄD: Brak {script_path}\n"; return
        env = os.environ.copy(); env["PYTHONIOENCODING"]="utf-8"
        try:
            p = subprocess.Popen([sys.executable, "-u", str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd, bufsize=1, encoding="utf-8", errors="replace", env=env)
            yield f"{prefix}Start...\n"
            for line in p.stdout:
                yield line
            p.wait()
            if p.returncode==0:
                if on_success: on_success()
                yield f"{prefix}Gotowe.\n"
            else: yield f"{prefix}Błąd (kod {p.returncode}).\n"
        except Exception as e: yield f"{prefix}Exception: {e}\n"

    def retrain_model_stream(self) -> Iterator[str]:
        def ok(): self.model=None; self.norm_stats=self._load_norm_stats()
        yield from self._stream_training_process(self.model_root/"train_model.py", self.model_root, ok)

    def retrain_lstm_stream(self) -> Iterator[str]:
        base = self.model_root.parent/"model_files_lstm"
        script = base/"train_lstm.py"
        def ok(): self.lstm_model=None; self.lstm_stats=None; self._load_lstm_resources()
        yield from self._stream_training_process(script, base, ok, "[LSTM] ")

    def retrain_both_stream(self) -> Iterator[str]:
        yield from self.retrain_model_stream()
        yield from self.retrain_lstm_stream()