from pathlib import Path

# -- Ścieżki --
DATA_DIR = Path(r"C:\Users\rudow\AppData\Roaming\MetaQuotes\Terminal\Common\Files\ai_forex")
MT5_CSV  = DATA_DIR / "ohlc_EURUSD_H1.csv"

# -- Parametry ZigZag / datasetu --
MIN_ZP     = 0.0050  # Deviation
ILE_ZZ     = 30  # Depth
LICZBA_SW  = 15
DL_ZZ      = 10
BACKSTEP   = 3

# -- Hiperparametry MLP --
HYPERPARAMS = {
    "hidden_sizes": [64, 32],
    "output_dim": 1,
    "learning_rate": 0.00004,
    "batch_size": 32,
    "epochs": 50,
    "dropout_p": 0.6,
    "weight_decay": 0.0002
}

SEED = 12
MODEL_PATH = DATA_DIR / "model.pt"
STATS_PATH = DATA_DIR / "norm_stats.pkl"
