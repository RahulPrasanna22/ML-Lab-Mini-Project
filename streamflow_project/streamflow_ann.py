"""
Streamflow forecasting with ANNs: feed-forward network vs LSTM.
Replicates and extends Xiao & You (Stanford CS229, 2020).

Input: a daily CSV with columns  date, rain_in, flow_cfs
       (rainfall in inches, discharge in cubic feet per second)

Run:   python streamflow_ann.py
       QUICK=0 python streamflow_ann.py     # full grid (slow), bash
       $env:QUICK="0"; python streamflow_ann.py   # same, PowerShell
Outputs (in ./outputs): grid results CSVs, summary.json, hydrograph.png
"""
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tensorflow import keras

# ------------------------- configuration -------------------------
DATA_CSV = os.environ.get("DATA_CSV", "data/leaf_river_daily.csv")
OUT_DIR = "outputs"

TRAIN_YEARS = (2003, 2007)   # same as the paper
VAL_YEARS = (2008, 2010)     # used for model selection
TEST_YEARS = (2011, 2012)    # untouched until the final evaluation
# To reproduce the paper's protocol exactly:
#   VAL_YEARS = (2008, 2012); TEST_YEARS = None

PEAK_THRESHOLD_CFS = 1500.0  # peak / non-peak split used in the paper

QUICK = os.environ.get("QUICK", "1") == "1"
LAGS = [2, 3, 5, 7] if QUICK else [2, 3, 4, 5, 6, 7]
HIDDEN_SIZES = [8, 12, 16] if QUICK else [4, 8, 12, 16, 24, 32]
EPOCHS = int(os.environ.get("EPOCHS", 200))
PATIENCE = 30
BATCH_SIZE = 32
LEARNING_RATE = {"mlp": 0.009, "lstm": 0.001}  # values reported in the paper
FINAL_SEEDS = 5

SPLITS = {"train": TRAIN_YEARS, "val": VAL_YEARS, "test": TEST_YEARS}
SPLITS = {k: v for k, v in SPLITS.items() if v is not None}
EVAL_SPLITS = [s for s in ("val", "test") if s in SPLITS]


# ------------------------- data -------------------------
def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated()]
    df = df.asfreq("D")  # any missing day becomes NaN
    df.loc[df["flow_cfs"] < 0, "flow_cfs"] = np.nan
    df.loc[df["rain_in"] < 0, "rain_in"] = np.nan
    return df


def make_samples(df, lag_rain, lag_flow, q_scale, r_scale, sequence):
    """Build (X, y, dates). A sample is dropped if any input or the target is NaN.

    Standard NN features: [R(t-1)..R(t-lag_rain), Q(t-1)..Q(t-lag_flow)]
    LSTM features: sequence of shape (L, 2) = [R, Q] per step, oldest first.
    """
    rain = (df["rain_in"] / r_scale).to_numpy()
    flow = (df["flow_cfs"] / q_scale).to_numpy()
    if sequence and lag_rain != lag_flow:
        raise ValueError("LSTM needs lag_rain == lag_flow")
    L = max(lag_rain, lag_flow)
    X, y, idx = [], [], []
    for t in range(L, len(df)):
        if sequence:
            feats = np.stack([rain[t - L:t], flow[t - L:t]], axis=1)
        else:
            feats = np.concatenate([rain[t - lag_rain:t][::-1], flow[t - lag_flow:t][::-1]])
        if np.isnan(feats).any() or np.isnan(flow[t]):
            continue
        X.append(feats)
        y.append(flow[t])
        idx.append(df.index[t])
    return np.asarray(X, np.float32), np.asarray(y, np.float32), pd.DatetimeIndex(idx)


def prepare(df, kind, lag_rain, lag_flow, scales):
    q_scale, r_scale = scales
    X, y, dates = make_samples(df, lag_rain, lag_flow, q_scale, r_scale, sequence=(kind == "lstm"))
    out = {}
    for name, (lo, hi) in SPLITS.items():
        mask = (dates.year >= lo) & (dates.year <= hi)
        out[name] = (X[mask], y[mask], dates[mask])
    return out


# ------------------------- models -------------------------
def build_model(kind, hidden, input_shape, lr):
    inp = keras.Input(shape=input_shape)
    if kind == "mlp":
        x = keras.layers.Dense(hidden, activation="sigmoid")(inp)  # sigmoid hidden layer
    else:
        x = keras.layers.LSTM(hidden)(inp)
    out = keras.layers.Dense(1)(x)  # linear output
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model


def fit_model(kind, hidden, data, seed):
    keras.utils.set_random_seed(seed)
    X_tr, y_tr, _ = data["train"]
    X_va, y_va, _ = data["val"]
    model = build_model(kind, hidden, X_tr.shape[1:], LEARNING_RATE[kind])
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[stop], verbose=0,
    )
    return model


def predict_cfs(model, X, q_scale):
    return np.maximum(model.predict(X, verbose=0).ravel() * q_scale, 0.0)


# ------------------------- metrics -------------------------
def compute_metrics(y_true, y_pred, q_max):
    """RMSE values are % of max flow, as in the paper's Table 1. NSE is added."""
    err = y_pred - y_true
    peak = y_true > PEAK_THRESHOLD_CFS

    def rel_rmse(e):
        return 100.0 * float(np.sqrt(np.mean(e ** 2))) / q_max if len(e) else float("nan")

    nse = 1.0 - float(np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    peak_bias = 100.0 * float(np.mean(err[peak] / y_true[peak])) if peak.any() else float("nan")
    return {
        "rmse_pct": rel_rmse(err),
        "peak_rmse_pct": rel_rmse(err[peak]),
        "nonpeak_rmse_pct": rel_rmse(err[~peak]),
        "nse": nse,
        "peak_bias_pct": peak_bias,  # negative = underestimates peaks
    }


# ------------------------- experiments -------------------------
def grid_search(df, kind, scales):
    q_scale = scales[0]
    if kind == "lstm":
        lag_pairs = [(L, L) for L in LAGS]
    else:
        lag_pairs = list(itertools.product(LAGS, LAGS))
    rows = []
    for lag_rain, lag_flow in lag_pairs:
        data = prepare(df, kind, lag_rain, lag_flow, scales)
        for hidden in HIDDEN_SIZES:
            model = fit_model(kind, hidden, data, seed=0)
            pred = predict_cfs(model, data["val"][0], q_scale)
            m = compute_metrics(data["val"][1] * q_scale, pred, q_scale)
            rows.append({
                "kind": kind, "lag_rain": lag_rain, "lag_flow": lag_flow, "hidden": hidden,
                **{f"val_{k}": v for k, v in m.items()},
            })
            print(f"[{kind}] rain_lag={lag_rain} flow_lag={lag_flow} hidden={hidden} "
                  f"val RMSE={m['rmse_pct']:.3f}%  NSE={m['nse']:.3f}")
            keras.backend.clear_session()
    return pd.DataFrame(rows)


def final_eval(df, kind, best, scales):
    """Retrain the best config with several seeds; report mean/std and the seed-averaged curve."""
    q_scale = scales[0]
    lag_rain, lag_flow, hidden = int(best["lag_rain"]), int(best["lag_flow"]), int(best["hidden"])
    data = prepare(df, kind, lag_rain, lag_flow, scales)
    preds = {s: [] for s in EVAL_SPLITS}
    for seed in range(FINAL_SEEDS):
        model = fit_model(kind, hidden, data, seed)
        for s in EVAL_SPLITS:
            preds[s].append(predict_cfs(model, data[s][0], q_scale))
        keras.backend.clear_session()
    summary, curves = {}, {}
    for s in EVAL_SPLITS:
        y_true = data[s][1] * q_scale
        per_seed = pd.DataFrame([compute_metrics(y_true, p, q_scale) for p in preds[s]])
        summary[s] = {"mean": per_seed.mean().to_dict(), "std": per_seed.std().to_dict()}
        curves[s] = (data[s][2], y_true, np.mean(preds[s], axis=0))
    return summary, curves


def run_baselines(df, best_mlp, scales):
    """Persistence (Q(t) = Q(t-1)) and linear regression on the same inputs as the best MLP."""
    q_scale = scales[0]
    lag_rain, lag_flow = int(best_mlp["lag_rain"]), int(best_mlp["lag_flow"])
    data = prepare(df, "mlp", lag_rain, lag_flow, scales)
    lin = LinearRegression().fit(data["train"][0], data["train"][1])
    out = {}
    for s in EVAL_SPLITS:
        X, y, _ = data[s]
        y_cfs = y * q_scale
        persistence = X[:, lag_rain] * q_scale  # first flow feature is Q(t-1)
        linear = np.maximum(lin.predict(X) * q_scale, 0.0)
        out[s] = {
            "persistence": compute_metrics(y_cfs, persistence, q_scale),
            "linear_regression": compute_metrics(y_cfs, linear, q_scale),
        }
    return out


# ------------------------- plots -------------------------
def make_plots(curves, split):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    colors = {"mlp": "tab:blue", "lstm": "tab:red"}
    dates, obs, _ = curves["mlp"][split]
    axes[0].plot(dates, obs, "k", lw=1.2, label="observed")
    for kind in ("mlp", "lstm"):
        d, y_true, pred = curves[kind][split]
        axes[0].plot(d, pred, color=colors[kind], lw=0.9, alpha=0.8, label=kind)
        axes[1].scatter(np.maximum(y_true, 1), np.maximum(pred, 1), s=6, alpha=0.5,
                        color=colors[kind], label=kind)
    axes[0].set(title=f"Hydrograph ({split} set)", ylabel="Flow (cfs)")
    axes[0].legend()
    lim = [1, max(obs.max(), 1) * 1.2]
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set(xscale="log", yscale="log", xlabel="Observed (cfs)", ylabel="Predicted (cfs)",
                title="Observed vs predicted (log-log)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "hydrograph.png"), dpi=150)


# ------------------------- main -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data(DATA_CSV)
    lo, hi = TRAIN_YEARS
    train = df[(df.index.year >= lo) & (df.index.year <= hi)]
    scales = (float(train["flow_cfs"].max()), float(train["rain_in"].max()))  # scale by TRAIN max only
    print(f"Days: {len(df)} | missing flow: {df['flow_cfs'].isna().sum()} "
          f"| missing rain: {df['rain_in'].isna().sum()}")
    print(f"Train max flow = {scales[0]:.0f} cfs, train max rain = {scales[1]:.2f} in "
          f"(paper: 29700 cfs, 6.8 in)")

    report, curves, best_rows = {}, {}, {}
    for kind in ("mlp", "lstm"):
        grid = grid_search(df, kind, scales)
        grid.to_csv(os.path.join(OUT_DIR, f"grid_{kind}.csv"), index=False)
        best = grid.sort_values("val_rmse_pct").iloc[0]
        best_rows[kind] = best
        summary, kind_curves = final_eval(df, kind, best, scales)
        report[kind] = {
            "best_config": {"lag_rain": int(best["lag_rain"]), "lag_flow": int(best["lag_flow"]),
                            "hidden": int(best["hidden"]), "lr": LEARNING_RATE[kind]},
            **summary,
        }
        curves[kind] = kind_curves

    report["baselines"] = run_baselines(df, best_rows["mlp"], scales)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    split = "test" if "test" in EVAL_SPLITS else "val"
    make_plots(curves, split)

    print("\n=== Results (mean over seeds; RMSE as % of max flow) ===")
    for s in EVAL_SPLITS:
        print(f"\n-- {s} set --")
        for kind in ("mlp", "lstm"):
            m = report[kind][s]["mean"]
            print(f"{kind:>18}: RMSE {m['rmse_pct']:.3f}% | peak {m['peak_rmse_pct']:.3f}% | "
                  f"non-peak {m['nonpeak_rmse_pct']:.3f}% | NSE {m['nse']:.3f} | "
                  f"peak bias {m['peak_bias_pct']:+.1f}%")
        for name, m in report["baselines"][s].items():
            print(f"{name:>18}: RMSE {m['rmse_pct']:.3f}% | peak {m['peak_rmse_pct']:.3f}% | "
                  f"non-peak {m['nonpeak_rmse_pct']:.3f}% | NSE {m['nse']:.3f} | "
                  f"peak bias {m['peak_bias_pct']:+.1f}%")


if __name__ == "__main__":
    main()
