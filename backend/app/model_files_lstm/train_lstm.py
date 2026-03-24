# train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import lstm_config as cfg
from lstm_model import LSTMRegressor
import data_pipeline as dp

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)


def main():
    print(f"--- START TRENINGU LSTM (FULL DATASET 100%) ---")
    print(f"Wczytywanie danych z: {cfg.MT5_CSV}")

    df_swieczki = dp.load_and_clean_data(cfg.MT5_CSV)
    print(f"Liczba świec całkowita: {len(df_swieczki)}")

    df_full, df_wyj, p_sw = dp.prepare_data(df_swieczki)
    print(f"Dane przygotowane. Features shape: {df_full.shape}")

    X_full_raw, y_full_raw, stats = dp.get_full_dataset(
        df_full, df_wyj, df_swieczki, p_sw
    )

    print(f"Zapisywanie statystyk normalizacji do {cfg.STATS_PATH}...")
    with open(cfg.STATS_PATH, "wb") as f:
        pickle.dump(stats, f)

    seq_len = cfg.HYPERPARAMS["seq_len"]
    X_train, y_train = dp.make_lstm_sequences(X_full_raw, y_full_raw, seq_len)

    print(f"Wielkość zbioru treningowego (sekwencje): {X_train.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=cfg.HYPERPARAMS["batch_size"], shuffle=True)

    model = LSTMRegressor(
        input_dim=X_train.shape[2],
        hidden_size=cfg.HYPERPARAMS["lstm_hidden_size"],
        num_layers=cfg.HYPERPARAMS["lstm_num_layers"],
        dropout_p=cfg.HYPERPARAMS["dropout_p"],
        bidirectional=cfg.HYPERPARAMS["bidirectional"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.HYPERPARAMS["learning_rate"],
        weight_decay=cfg.HYPERPARAMS["weight_decay"]
    )

    epochs = cfg.HYPERPARAMS["epochs"]

    for epoch in range(epochs):
        model.train()
        train_loss_acc = 0.0
        count = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            train_loss_acc += loss.item()
            count += 1

        avg_train_loss = train_loss_acc / count
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_train_loss:.6f}")

    print(f"\nZapisywanie modelu do: {cfg.MODEL_PATH}")
    torch.save(model.state_dict(), cfg.MODEL_PATH)
    print("GOTOWE.")


if __name__ == "__main__":
    main()