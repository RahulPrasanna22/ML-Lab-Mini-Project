"""Local Streamflow Forecasting GUI. Run with: streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from tensorflow import keras

import streamflow_ann as experiment

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "leaf_river_daily.csv"
SUMMARY_FILE = ROOT / "outputs" / "summary.json"
PEAK_THRESHOLD_CFS = 1500.0

st.set_page_config(page_title="Streamflow Forecasting", page_icon="🌊", layout="wide")


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return experiment.load_data(DATA_FILE)


def load_uploaded_data(file) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(file)
    required = {"date", "rain_in", "flow_cfs"}
    missing_cols = required - set(raw.columns)
    if missing_cols:
        raise ValueError("CSV must contain columns date, rain_in, flow_cfs. Missing: "
                         + ", ".join(sorted(missing_cols)))
    original_rows = len(raw)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["rain_in"] = pd.to_numeric(raw["rain_in"], errors="coerce")
    raw["flow_cfs"] = pd.to_numeric(raw["flow_cfs"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    if raw.empty:
        raise ValueError("No valid dates were found in the CSV.")
    if (raw["rain_in"].dropna() < 0).any() or (raw["flow_cfs"].dropna() < 0).any():
        raise ValueError("Rainfall and flow must be non-negative. Use blank cells for missing values.")
    daily = raw.groupby("date")[["rain_in", "flow_cfs"]].mean().sort_index().asfreq("D")
    return daily, original_rows - len(raw)


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, q_scale: float) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    peak = y_true > PEAK_THRESHOLD_CFS

    def relative_rmse(values):
        return float(100 * np.sqrt(np.mean(values**2)) / q_scale) if values.size else float("nan")

    total = float(np.sum((y_true - y_true.mean()) ** 2))
    nse = 1 - float(np.sum(err**2)) / total if total > 0 else float("nan")
    peak_bias = float(100 * np.mean(err[peak] / y_true[peak])) if peak.any() else float("nan")
    return {
        "RMSE (%)": relative_rmse(err),
        "Peak RMSE (%)": relative_rmse(err[peak]),
        "Non-peak RMSE (%)": relative_rmse(err[~peak]),
        "NSE": nse,
        "Peak bias (%)": peak_bias,
    }


def load_project_config(kind: str) -> dict:
    try:
        with SUMMARY_FILE.open(encoding="utf-8") as file:
            return json.load(file)[kind]["best_config"]
    except (OSError, ValueError, KeyError):
        return {"lag_rain": 5, "lag_flow": 5, "hidden": 8} if kind == "mlp" else {
            "lag_rain": 2, "lag_flow": 2, "hidden": 12
        }


def make_samples(df, kind, lag_rain, lag_flow, q_scale, r_scale):
    return experiment.make_samples(
        df, lag_rain, lag_flow, q_scale, r_scale, sequence=(kind == "lstm")
    )


def train_one_model(kind, config, X, y, train_end, val_end, epochs):
    keras.utils.set_random_seed(42)
    model = experiment.build_model(
        kind, int(config["hidden"]), X.shape[1:], experiment.LEARNING_RATE[kind]
    )
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    model.fit(X[:train_end], y[:train_end],
              validation_data=(X[train_end:val_end], y[train_end:val_end]),
              epochs=epochs, batch_size=32, callbacks=[stop], verbose=0)
    return model


def chronological_indices(n, train_fraction, val_fraction):
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    if train_end < 30 or val_end - train_end < 5 or n - val_end < 5:
        raise ValueError("Too few usable daily samples for a chronological split. Provide more observed data.")
    return train_end, val_end


def run_backtest(df, kind, epochs):
    config = load_project_config(kind)
    lag_rain, lag_flow = int(config["lag_rain"]), int(config["lag_flow"])
    _, y0, dates = make_samples(df, kind, lag_rain, lag_flow, 1.0, 1.0)
    n = len(y0)
    train_end, val_end = chronological_indices(n, 0.70, 0.15)
    train_period = df.loc[:dates[train_end - 1]]
    q_scale = float(train_period["flow_cfs"].max())
    r_scale = float(train_period["rain_in"].max())
    if not np.isfinite(q_scale) or q_scale <= 0:
        raise ValueError("Training data needs positive observed flow values.")
    if not np.isfinite(r_scale) or r_scale <= 0:
        r_scale = 1.0
    X, y, dates = make_samples(df, kind, lag_rain, lag_flow, q_scale, r_scale)
    if len(y) != n:
        raise ValueError("Could not construct consistent samples from this CSV.")
    with st.spinner(f"Training the {kind.upper()} on uploaded history..."):
        model = train_one_model(kind, config, X, y, train_end, val_end, epochs)
        X_test, y_test = X[val_end:], y[val_end:] * q_scale
        predicted = np.maximum(model.predict(X_test, verbose=0).ravel() * q_scale, 0)
        linear_model = LinearRegression().fit(X[:train_end].reshape(train_end, -1), y[:train_end])
        linear = np.maximum(linear_model.predict(X_test.reshape(len(X_test), -1)) * q_scale, 0)
        persistence = (X_test[:, lag_rain] if kind == "mlp" else X_test[:, -1, 1]) * q_scale
    results = pd.DataFrame({
        "date": dates[val_end:], "observed_flow_cfs": y_test,
        f"{kind.upper()}_prediction_cfs": predicted,
        "persistence_cfs": persistence, "linear_regression_cfs": linear,
    })
    metrics = {
        kind.upper(): safe_metrics(y_test, predicted, q_scale),
        "Persistence": safe_metrics(y_test, persistence, q_scale),
        "Linear regression": safe_metrics(y_test, linear, q_scale),
    }
    return results, metrics, config, (train_end, val_end, n)


def run_forecast(df, kind, epochs):
    config = load_project_config(kind)
    lag_rain, lag_flow = int(config["lag_rain"]), int(config["lag_flow"])
    known = df["flow_cfs"].notna()
    if not known.any():
        raise ValueError("The CSV needs historical flow values to train the model.")
    missing_dates = df.index[~known]
    last_observed = df.index[known][-1]
    future_dates = missing_dates[missing_dates > last_observed]
    if len(future_dates) != len(missing_dates):
        raise ValueError("Forecast mode expects blank flow only after the last observed date. "
                         "Use backtest mode for gaps within the historical record.")
    if not len(future_dates):
        raise ValueError("No future blank flow rows found. Add future dates with blank flow_cfs.")

    _, y0, sample_dates = make_samples(df, kind, lag_rain, lag_flow, 1.0, 1.0)
    train_end, val_end = chronological_indices(len(y0), 0.80, 0.20)
    train_period = df.loc[:sample_dates[train_end - 1]]
    q_scale = float(train_period["flow_cfs"].max())
    r_scale = float(train_period["rain_in"].max())
    if not np.isfinite(q_scale) or q_scale <= 0:
        raise ValueError("Training data needs positive observed flow values.")
    if not np.isfinite(r_scale) or r_scale <= 0:
        r_scale = 1.0
    X, y, _ = make_samples(df, kind, lag_rain, lag_flow, q_scale, r_scale)
    with st.spinner(f"Training the {kind.upper()} on observed history..."):
        model = train_one_model(kind, config, X, y, train_end, val_end, epochs)

    flow_filled = df["flow_cfs"].to_numpy(dtype=float).copy()
    rain = df["rain_in"].to_numpy(dtype=float) / r_scale
    q = flow_filled / q_scale
    dates = list(df.index)
    predictions = []
    for target_date in future_dates:
        t = dates.index(target_date)
        length = max(lag_rain, lag_flow)
        if t < length:
            raise ValueError("Not enough history for the model's look-back window.")
        if kind == "lstm":
            features = np.stack([rain[t - length:t], q[t - length:t]], axis=1)
        else:
            features = np.concatenate([rain[t - lag_rain:t][::-1], q[t - lag_flow:t][::-1]])
        if np.isnan(features).any():
            raise ValueError(f"Missing input history before {target_date.date()}. Fill historical inputs first.")
        pred = max(float(model.predict(features[np.newaxis, ...], verbose=0).ravel()[0]) * q_scale, 0.0)
        flow_filled[t] = pred
        q[t] = pred / q_scale
        predictions.append(pred)
    return pd.DataFrame({"date": future_dates, "predicted_flow_cfs": predictions}), config, q_scale


def show_dataset_stats(df, dropped_dates):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily rows", f"{len(df):,}")
    c2.metric("Date range", f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
    c3.metric("Missing rainfall days", f"{df['rain_in'].isna().sum():,}")
    c4.metric("Missing flow days", f"{df['flow_cfs'].isna().sum():,}")
    flow, rain = df["flow_cfs"].dropna(), df["rain_in"].dropna()
    stats = pd.DataFrame([
        {"Statistic": "Mean", "Rain (in)": rain.mean(), "Flow (cfs)": flow.mean()},
        {"Statistic": "Median", "Rain (in)": rain.median(), "Flow (cfs)": flow.median()},
        {"Statistic": "Maximum", "Rain (in)": rain.max(), "Flow (cfs)": flow.max()},
    ])
    st.dataframe(stats, hide_index=True, width="stretch")
    if dropped_dates:
        st.caption(f"Skipped {dropped_dates} rows with invalid dates.")


def main():
    st.title("🌊 Streamflow Forecasting")
    st.write("Upload daily measurements to explore flow predictions and statistics. Required columns: "
             "**date**, **rain_in**, and **flow_cfs** (rain in inches; flow in cfs).")
    uploaded = st.file_uploader("Choose a daily CSV", type=["csv"])
    st.download_button("Download included example data", DATA_FILE.read_bytes(),
                       file_name="leaf_river_daily.csv", mime="text/csv")
    try:
        if uploaded is None:
            df, dropped_dates = load_default_data(), 0
            st.info("Using the included Leaf River data. Upload a CSV to use your own.")
        else:
            df, dropped_dates = load_uploaded_data(uploaded)
        if len(df) < 60:
            raise ValueError("The dataset is too short. At least about one year of daily records is recommended.")
    except Exception as error:
        st.error(str(error))
        st.stop()

    with st.expander("Dataset statistics", expanded=True):
        show_dataset_stats(df, dropped_dates)

    left, right = st.columns(2)
    with left:
        mode = st.radio("What should the app do?",
                        ["Backtest observed data", "Forecast blank future flow values"],
                        help="Backtest scores predictions against known flow. Forecast fills blank flow rows after the last observed date.")
    with right:
        kind = st.selectbox("Neural network", ["MLP", "LSTM"]).lower()
    config = load_project_config(kind)
    st.caption(f"Starting settings from the quick run: rain lag {config['lag_rain']}, "
               f"flow lag {config['lag_flow']}, {config['hidden']} hidden units. The model retrains on the loaded data.")
    epochs = st.slider("Maximum training epochs", 30, 200, 80, 10)
    st.caption("The interactive app uses one seed; these exploratory metrics can differ from the five-seed project summary.")

    if st.button("Run predictions", type="primary", width="stretch"):
        try:
            if mode == "Backtest observed data":
                results, metrics, config, split = run_backtest(df, kind, epochs)
                st.subheader("Backtest results")
                st.caption(f"Chronological split: {split[0]:,} training, {split[1]-split[0]:,} validation, "
                           f"{split[2]-split[1]:,} test samples.")
                rows = [{"Model": name, **values} for name, values in metrics.items()]
                st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")
                st.caption("Lower RMSE and higher NSE are better. Peak days are observed flows above 1,500 cfs.")
                graph = results.set_index("date")[[
                    "observed_flow_cfs", f"{kind.upper()}_prediction_cfs",
                    "persistence_cfs", "linear_regression_cfs",
                ]]
                st.line_chart(graph, height=320)
                st.dataframe(results, hide_index=True, width="stretch")
            else:
                results, config, q_scale = run_forecast(df, kind, epochs)
                st.subheader("Future flow predictions")
                st.write(f"Generated {len(results)} prediction(s); training flow scale: {q_scale:,.1f} cfs.")
                st.dataframe(results, hide_index=True, width="stretch")
                st.line_chart(results.set_index("date"), height=280)
                st.warning("For consecutive future days, each prediction becomes an input to the next day, so errors can accumulate. "
                           "The model uses previous days' rainfall and flow, not rainfall on the forecast date.")
            st.download_button("Download predictions as CSV", results.to_csv(index=False).encode("utf-8"),
                               file_name="streamflow_predictions.csv", mime="text/csv")
        except Exception as error:
            st.error(str(error))

    st.divider()
    st.caption("Educational project only; not an operational flood-warning system.")


if __name__ == "__main__":
    main()
