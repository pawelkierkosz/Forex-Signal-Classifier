import pickle, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from config import *
from data_pipeline import *

torch.manual_seed(SEED); np.random.seed(SEED)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, dropout_p=0.5):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_p)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def accuracy(y_logits, y_true):
    preds = torch.sigmoid(y_logits)
    preds_class = (preds >= 0.5).float()
    return (preds_class.squeeze() == y_true.squeeze()).float().mean().item()

def evaluate(model, X, y, crit, bs):
    X = torch.from_numpy(X); y = torch.from_numpy(y).unsqueeze(1)
    ld = DataLoader(TensorDataset(X,y), batch_size=bs, shuffle=False)
    model.eval(); loss_sum=0; acc_sum=0; n=0
    with torch.no_grad():
        for xb, yb in ld:
            out = model(xb); loss = crit(out, yb)
            k = xb.size(0); loss_sum += loss.item()*k; acc_sum += accuracy(out, yb)*k; n += k
    return loss_sum/n, acc_sum/n

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_mt5_csv(MT5_CSV)
    N = len(df); od, do = 1, N

    df_ZZ = full_ZZ(df, ILE_ZZ, MIN_ZP, BACKSTEP)
    df_in, df_out, first_bar = prepare_features(df, df_ZZ, od, do, DL_ZZ, LICZBA_SW, ILE_ZZ, MIN_ZP, BACKSTEP)
    df_ind = calculate_technical_indicators(df, od, do, len(df_in))
    df_all = combine_features(df_in, df_ind)

    sw_index = np.arange(first_bar, do+1)
    df_all.index = sw_index; df_out.index = sw_index

    # -- TRENING NA CAŁYM ZAKRESIE --
    m = len(df_all)

    X_fit_raw = df_all.copy()
    y_fit_df  = df_out.copy()

    if "pivot_bar" in y_fit_df.columns:
        fit_end_bar = X_fit_raw.index[-1]
        mask = (y_fit_df["pivot_bar"] <= fit_end_bar)
        X_fit_raw = X_fit_raw.loc[mask]; y_fit_df = y_fit_df.loc[mask]

    O_next_series = df["O"].shift(-1)
    fit_idx_valid = X_fit_raw.index.intersection(O_next_series.dropna().index)

    X_fit_raw = X_fit_raw.loc[fit_idx_valid]
    y_fit_df  = y_fit_df.loc[fit_idx_valid]
    O_fit     = O_next_series.loc[fit_idx_valid]

    X_fit, stats = normalize_features(X_fit_raw, O_fit, DL_ZZ, LICZBA_SW, stats=None)

    X_fit = X_fit.to_numpy(np.float32)
    y_fit = y_fit_df["label"].values.astype(np.float32)

    model = MLP(X_fit.shape[1], HYPERPARAMS["hidden_sizes"], HYPERPARAMS["output_dim"], HYPERPARAMS["dropout_p"])
    crit = nn.BCEWithLogitsLoss()
    opt  = optim.Adam(model.parameters(), lr=HYPERPARAMS["learning_rate"], weight_decay=HYPERPARAMS["weight_decay"])

    ds = TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit).unsqueeze(1))
    ld = DataLoader(ds, batch_size=HYPERPARAMS["batch_size"], shuffle=True)

    for epoch in range(HYPERPARAMS["epochs"]):
        model.train(); runl=runacc=n=0
        for xb,yb in ld:
            opt.zero_grad(); out=model(xb); loss=crit(out,yb); loss.backward(); opt.step()
            k=xb.size(0); runl+=loss.item()*k; n+=k
            with torch.no_grad(): runacc += accuracy(out,yb)*k
        tr_loss=runl/n; tr_acc=runacc/n
        print(f"[{epoch+1}/{HYPERPARAMS['epochs']}] Train loss {tr_loss:.4f}, acc {tr_acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    with open(STATS_PATH, "wb") as f:
        pickle.dump(stats, f)
    print(f"Zapisano: {MODEL_PATH} oraz {STATS_PATH}")