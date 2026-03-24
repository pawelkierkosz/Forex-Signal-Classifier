# lstm_section.py
import numpy as np
import pandas as pd

try:
    from .lstm_config import WSP_DELTA
except ImportError:
    from lstm_config import WSP_DELTA


def update_zigzag(bar: int, last_row: tuple, last_count: int, H: np.ndarray, L: np.ndarray, ileZZ: int, minzp: float,
                  backstep: int):
    idx = bar - 1
    start = max(0, idx - ileZZ)
    segH, segL = H[start:idx + 1], L[start:idx + 1]
    is_peak = int(H[idx] >= segH.max())
    is_trough = int(L[idx] <= segL.min())

    if is_peak + is_trough != 1:
        return last_row, last_count, 0

    typ = 1 if is_peak else -1
    price = H[idx] if is_peak else L[idx]
    new_row = (bar, price, typ)

    if last_count == 0:
        return new_row, 1, 1

    prev_bar, prev_price, prev_typ = last_row

    if prev_typ * typ == -1:
        if abs(price - prev_price) > minzp and (bar - prev_bar) >= backstep:
            return new_row, last_count + 1, 1
        else:
            return last_row, last_count, 0
    else:
        if (price - prev_price) * typ > 0:
            return new_row, last_count, 0
        else:
            return last_row, last_count, 0


def calculate_technical_indicators(df_swieczki, m):
    od = 1
    do = len(df_swieczki)
    sma_short, sma_mid, sma_long = 5, 20, 50
    macd_fast, macd_slow, macd_sig = 12, 26, 9
    rsi_period = 14
    stoch_k, stoch_d = 14, 3
    atr_period = 14
    cci_period = 20
    mom_period = 10

    df_range = df_swieczki.copy()
    Close = df_range["C"].to_numpy();
    High = df_range["H"].to_numpy();
    Low = df_range["L"].to_numpy()
    N = len(df_range)

    # SMA
    SMA5 = pd.Series(Close).rolling(sma_short, min_periods=sma_short).mean().to_numpy()
    SMA20 = pd.Series(Close).rolling(sma_mid, min_periods=sma_mid).mean().to_numpy()
    SMA50 = pd.Series(Close).rolling(sma_long, min_periods=sma_long).mean().to_numpy()

    # EMA
    def ema(arr, period):
        a = 2 / (period + 1)
        out = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            out[period - 1] = arr[:period].mean()
            for t in range(period, len(arr)):
                out[t] = a * arr[t] + (1 - a) * out[t - 1]
        return out

    EMA5, EMA20, EMA50 = ema(Close, 5), ema(Close, 20), ema(Close, 50)
    EMA12, EMA26 = ema(Close, macd_fast), ema(Close, macd_slow)
    MACD = EMA12 - EMA26

    # Signal
    Signal = np.full_like(Close, np.nan, dtype=float)
    valid = ~np.isnan(MACD)
    if valid.sum() >= macd_sig:
        idxs = np.flatnonzero(valid)
        start = idxs[0] + macd_sig - 1
        Signal[start] = MACD[idxs[0]:idxs[0] + macd_sig].mean()
        for t in range(start + 1, len(MACD)):
            Signal[t] = 2 / (macd_sig + 1) * MACD[t] + (1 - 2 / (macd_sig + 1)) * Signal[t - 1]

    # Bollinger Bands
    rolling20 = pd.Series(Close).rolling(sma_mid, min_periods=sma_mid)
    BB_mid = rolling20.mean().to_numpy();
    BB_std = rolling20.std().to_numpy()
    BB_up = BB_mid + 2 * BB_std;
    BB_low = BB_mid - 2 * BB_std

    # RSI
    delta = np.diff(Close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0);
    loss = np.where(delta < 0, -delta, 0)
    RSI = np.full_like(Close, np.nan, dtype=float)
    if N > rsi_period:
        avg_gain = gain[1:rsi_period + 1].mean();
        avg_loss = loss[1:rsi_period + 1].mean()
        RS = avg_gain / avg_loss if avg_loss != 0 else np.inf
        RSI[rsi_period] = 100 - 100 / (1 + RS)
        for t in range(rsi_period + 1, N):
            avg_gain = (avg_gain * (rsi_period - 1) + gain[t]) / rsi_period
            avg_loss = (avg_loss * (rsi_period - 1) + loss[t]) / rsi_period
            RS = avg_gain / avg_loss if avg_loss != 0 else np.inf
            RSI[t] = 100 - 100 / (1 + RS)

    # Stochastic
    StochK = np.full_like(Close, np.nan, dtype=float)
    if N >= stoch_k:
        low_min = pd.Series(Low).rolling(stoch_k, min_periods=stoch_k).min().to_numpy()
        high_max = pd.Series(High).rolling(stoch_k, min_periods=stoch_k).max().to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            rawK = 100 * (Close - low_min) / (high_max - low_min)
        StochK[stoch_k - 1:] = rawK[stoch_k - 1:]
        StochD = pd.Series(StochK).rolling(stoch_d, min_periods=stoch_d).mean().to_numpy()
    else:
        StochD = np.full_like(Close, np.nan)

    # ATR
    TR = np.maximum.reduce([High - Low, np.abs(High - np.concatenate([[np.nan], Close[:-1]])),
                            np.abs(Low - np.concatenate([[np.nan], Close[:-1]]))])
    ATR = pd.Series(TR).rolling(atr_period, min_periods=1).mean().to_numpy()

    # CCI
    TP = (High + Low + Close) / 3.0
    SMA_TP = pd.Series(TP).rolling(cci_period, min_periods=1).mean().to_numpy()
    MD = pd.Series(np.abs(TP - SMA_TP)).rolling(cci_period, min_periods=1).mean().to_numpy()
    Den = 0.015 * MD
    with np.errstate(divide='ignore', invalid='ignore'):
        CCI = np.divide(TP - SMA_TP, Den, out=np.zeros_like(TP, dtype=float), where=Den != 0)

    # Momentum
    Momentum = np.full_like(Close, np.nan, dtype=float)
    if N > mom_period: Momentum[mom_period:] = Close[mom_period:] - Close[:-mom_period]

    Indicators = np.column_stack(
        [SMA5, SMA20, SMA50, EMA5, EMA20, EMA50, BB_mid, BB_up, BB_low, MACD, Signal, RSI, StochK, StochD, ATR, CCI,
         Momentum])
    df_ind = pd.DataFrame(Indicators, index=df_range.index,
                          columns=["SMA5", "SMA20", "SMA50", "EMA5", "EMA20", "EMA50", "BB_mid", "BB_up", "BB_low",
                                   "MACD", "Signal", "RSI", "StochK", "StochD", "ATR", "CCI", "Momentum"])

    df_ind_f = df_ind.tail(m).reset_index(drop=True).fillna(0.0)
    return df_ind_f


def normalize_features(W: pd.DataFrame, O: pd.Series, dl_zz: int, liczba_sw: int, stats: dict | None = None):
    W = W.copy().reset_index(drop=True).astype(float)
    O = O.reset_index(drop=True).astype(float)

    piv = [f"ZZ_price_{i + 1}" for i in range(dl_zz + 1)]
    clo = [f"C_close_{i + 1}" for i in range(liczba_sw)]
    ind = ["SMA5", "SMA20", "SMA50", "EMA5", "EMA20", "EMA50", "BB_mid", "BB_up", "BB_low", "MACD", "Signal", "RSI",
           "StochK", "StochD", "ATR", "CCI", "Momentum"]
    W.columns = piv + clo + ind

    # Normalizacja cen względem Open
    W.iloc[:, :dl_zz + 1 + liczba_sw] = (W.iloc[:, :dl_zz + 1 + liczba_sw].values - O.values.reshape(-1, 1)) * WSP_DELTA

    if stats is None:
        stats = {}
        for col in ["MACD", "Signal", "Momentum"]:
            mu, sd = W[col].mean(), W[col].std(ddof=0) or 1.0
            stats[col] = (mu, sd)
        mu, sd = (W["ATR"] / O).mean(), (W["ATR"] / O).std(ddof=0) or 1.0
        stats["ATR_ratio"] = (mu, sd)

    W["MACD"] = (W["MACD"] - stats["MACD"][0]) / stats["MACD"][1]
    W["Signal"] = (W["Signal"] - stats["Signal"][0]) / stats["Signal"][1]
    W["Momentum"] = (W["Momentum"] - stats["Momentum"][0]) / stats["Momentum"][1]
    W["RSI"] = W["RSI"] / 100.0
    for col in ["StochK", "StochD"]:
        W[col] = (W[col] - 50.0) / 100.0

    W["ATR"] = ((W["ATR"] / O) - stats["ATR_ratio"][0]) / stats["ATR_ratio"][1]
    W["CCI"] = W["CCI"] / 100.0

    return W, stats