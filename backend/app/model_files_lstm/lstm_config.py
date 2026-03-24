# lstm_config.py
from pathlib import Path

# -- Ścieżki --
DATA_DIR = Path(r"C:\Users\rudow\AppData\Roaming\MetaQuotes\Terminal\Common\Files\ai_forex")
MT5_CSV  = DATA_DIR / "ohlc_EURUSD_H1.csv"

# -- Pliki wyjściowe --
MODEL_PATH = DATA_DIR / "lstm_model.pt"
STATS_PATH = DATA_DIR / "lstm_norm_stats.pkl"

# -- Parametry ZigZag / datasetu --
MIN_ZP     = 0.005 # Deviation
ILE_ZZ     = 30 # Depth
BACKSTEP   = 3
DL_ZZ      = 6
LICZBA_SW  = 11

WSP_DELTA = 100.0
PIP_VALUE = 0.0001

# -- Hiperparametry LSTM --
HYPERPARAMS = {
    "seq_len":          27,
    "lstm_hidden_size": 20,
    "lstm_num_layers":  2,
    "bidirectional":    True,
    "dropout_p":        0.32021717208852246,
    "learning_rate":    1.858164642731971e-05,
    "batch_size":       128,
    "epochs":           50,
    "weight_decay":     1.0117142985674184e-05,
}

SEED = 12
