# data_pipeline.py
import sys
import os
import numpy as np
import pandas as pd
from collections import deque
from lstm_config import (MIN_ZP, ILE_ZZ, BACKSTEP,
                         DL_ZZ, LICZBA_SW, WSP_DELTA)
from lstm_section import update_zigzag, calculate_technical_indicators, normalize_features


# -- Wczytanie danych --
def load_and_clean_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"BŁĄD: Brak pliku {csv_path}")
        sys.exit(1)

    try:
        df_raw = pd.read_csv(
            csv_path,
            header=None,
            usecols=[2, 3, 4, 5],
            names=["O", "H", "L", "C"]
        )
    except Exception as e:
        print(f"Błąd krytyczny przy odczycie CSV: {e}")
        sys.exit(1)

    df_swieczki = df_raw.iloc[::-1].copy()
    df_swieczki = df_swieczki.iloc[:-1].copy()
    df_swieczki.reset_index(drop=True, inplace=True)
    df_swieczki.index = range(1, len(df_swieczki) + 1)

    for col in ["O", "H", "L", "C"]:
        df_swieczki[col] = pd.to_numeric(df_swieczki[col], errors='coerce')

    if df_swieczki.isnull().values.any():
        print("Uwaga: Znaleziono wartości NaN. Usuwanie...")
        df_swieczki.dropna(inplace=True)
        df_swieczki.reset_index(drop=True, inplace=True)
        df_swieczki.index = range(1, len(df_swieczki) + 1)

    return df_swieczki


def distance_to_next_pivot(sw: int, df_zz_all: pd.DataFrame, df_swieczki: pd.DataFrame):
    C = df_swieczki["C"]
    try:
        _ = C.loc[sw]
    except KeyError:
        return np.nan, np.nan
    mask = df_zz_all["bar"] > sw
    if not mask.any():
        return np.nan, np.nan
    piv = df_zz_all.loc[mask, :].iloc[0]
    pivot_bar = float(piv["bar"])
    pivot_price = float(piv["price"])
    delta = (pivot_price - C.loc[sw]) * WSP_DELTA
    return float(delta), pivot_bar


# -- Zigzag / etykiety --
def full_ZZ(df_swieczki, ileZZ, minzmnpkt, backstep):
    H = df_swieczki["H"].values
    L = df_swieczki["L"].values
    prevH, prevL = deque(), deque()
    pivots = []
    last_pivot = None
    last_n = 0
    for i, (h, l) in enumerate(zip(H, L)):
        bar = i + 1
        if len(prevH) == ileZZ:
            local_max = max(prevH)
            local_min = min(prevL)
            isPeak = (h >= local_max)
            isTrough = (l <= local_min)
            if isPeak ^ isTrough:
                typ = 1 if isPeak else -1
                price = h if isPeak else l
                if last_n == 0:
                    pivots.append((bar, price, typ))
                    last_pivot = (bar, price, typ)
                    last_n = 1
                else:
                    lb, lp_price, lp_typ = last_pivot
                    if lp_typ * typ == -1:
                        if abs(price - lp_price) > minzmnpkt and (bar - lb) >= backstep:
                            pivots.append((bar, price, typ))
                            last_pivot = (bar, price, typ)
                            last_n += 1
                    else:
                        if (price - lp_price) * typ > 0:
                            pivots[-1] = (bar, price, typ)
                            last_pivot = (bar, price, typ)
        prevH.append(h);
        prevL.append(l)
        if len(prevH) > ileZZ:
            prevH.popleft();
            prevL.popleft()
    return pd.DataFrame(pivots, columns=["bar", "price", "type"])


def prepare_data(df_swieczki):
    df_ZZ = full_ZZ(df_swieczki, ILE_ZZ, MIN_ZP, BACKSTEP)

    od = 1
    do = len(df_swieczki)
    H = df_swieczki["H"].values
    L = df_swieczki["L"].values
    C = df_swieczki["C"]

    rows_input, rows_output = [], []
    piv_deque = deque(maxlen=DL_ZZ + 1)
    last_row, last_count = None, 0

    for sw in range(od, do + 1):
        new_row, new_count, flag = update_zigzag(sw, last_row, last_count, H, L, ILE_ZZ, MIN_ZP, BACKSTEP)
        if flag == 1 and last_row is not None:
            piv_deque.append(last_row)
        last_row, last_count = new_row, new_count

        all_dyn = list(piv_deque)
        if last_row is not None:
            all_dyn.append(last_row)

        needed = DL_ZZ + 1
        slice_piv = all_dyn[-needed:] if len(all_dyn) >= needed else all_dyn
        prices = [p[1] for p in slice_piv]
        if len(prices) < needed:
            prices = [np.nan] * (needed - len(prices)) + prices

        closes = []
        for j in range(LICZBA_SW - 1, -1, -1):
            idx = sw - j
            closes.append(C.loc[idx] if idx >= 1 else np.nan)

        rows_input.append(prices + closes)
        delta, piv_bar = distance_to_next_pivot(sw, df_ZZ, df_swieczki)
        rows_output.append([delta, piv_bar])

    cols_input = [f"ZZ_price_{k + 1}" for k in range(DL_ZZ + 1)] + [f"C_close_{k + 1}" for k in range(LICZBA_SW)]
    df_input = pd.DataFrame(rows_input, index=range(od, do + 1), columns=cols_input)
    df_output = pd.DataFrame(rows_output, index=range(od, do + 1), columns=["delta_ZZ", "pivot_bar"])

    valid = df_input.notna().all(axis=1) & df_output["delta_ZZ"].notna()
    df_wej_f = df_input.loc[valid].copy()
    df_wyj_f = df_output.loc[valid].copy()

    df_wej_f.reset_index(drop=True, inplace=True)
    df_wyj_f.reset_index(drop=True, inplace=True)

    n_before = len(df_input)
    n_after = len(df_wej_f)
    removed_filter = n_before - n_after
    pierwsza_sw = od + removed_filter

    df_ind = calculate_technical_indicators(df_swieczki, len(df_wej_f))
    df_full = pd.concat([df_wej_f, df_ind], axis=1, ignore_index=True)

    return df_full, df_wyj_f, pierwsza_sw


def get_full_dataset(df_full, df_wyj, df_swieczki, pierwsza_sw):
    total_len = len(df_full)
    real_bar_indices = np.arange(pierwsza_sw, pierwsza_sw + total_len)

    max_bar_available = len(df_swieczki)

    # Usuwamy tylko te wiersze, gdzie pivot_bar > ostatnia dostępna świeczka w pliku
    mask_valid = df_wyj["pivot_bar"] <= max_bar_available

    X_raw = df_full.loc[mask_valid].copy()
    y_df = df_wyj.loc[mask_valid].copy()

    y_full = y_df["delta_ZZ"].values.astype(np.float32).ravel()

    # Mapowanie indeksów, aby pobrać ceny Open do normalizacji
    indices = real_bar_indices[X_raw.index]
    O_vals = df_swieczki.loc[indices, "O"]

    X_norm, stats = normalize_features(X_raw, O_vals, DL_ZZ, LICZBA_SW, stats=None)

    return X_norm.to_numpy(np.float32), y_full, stats


def make_lstm_sequences(X, y, seq_len):
    if len(X) < seq_len:
        return np.array([]), np.array([])
    X_seq, y_seq = [], []
    for i in range(seq_len - 1, len(X)):
        X_seq.append(X[i - seq_len + 1:i + 1, :])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)