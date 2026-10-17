
# chronos2_inference.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from chronos import Chronos2Pipeline

CSV_PATH = "./assets/ETTm2.csv"
TIME_COL = "date"
TARGET_COL = "OT"
HORIZON = 96
SEQ_LEN = 672

df = pd.read_csv(CSV_PATH, parse_dates=[TIME_COL])
df = df.sort_values(TIME_COL).set_index(TIME_COL)
df = df[[TARGET_COL]].apply(pd.to_numeric, errors="coerce")
df = df.asfreq("15min")
df[TARGET_COL] = df[TARGET_COL].interpolate(
    method="time", limit=4, limit_area="inside"
)
df = df.dropna()

# Hold out the last 15% as test data.
values = df[TARGET_COL].to_numpy(dtype=np.float32)
test_start = int(len(values) * 0.85)

context = values[max(0, test_start - SEQ_LEN):test_start]
actual = values[test_start:test_start + HORIZON]

if len(actual) != HORIZON:
    raise ValueError("Not enough held-out data for the horizon.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map=device,
)

# API: forecast from the historical context.
start = time.perf_counter()
forecast_df = pd.DataFrame({
    "item_id": ["series_1"] * len(context),
    "timestamp": df.index[max(0, test_start - SEQ_LEN):test_start],
    "target": context,
})
prediction = model.predict_df(
    forecast_df,
    prediction_length=HORIZON,
    quantile_levels=[0.1, 0.5, 0.9],
)
elapsed = time.perf_counter() - start

# Inspect returned columns in case the installed version
# uses a different output naming convention.
print(prediction.head())
print("Output columns:", prediction.columns.tolist())

# In the standard Chronos-2 DataFrame API, the output
# includes the median forecast and requested quantiles.
median_col = "0.5"
if median_col not in prediction.columns:
    raise KeyError(
        f"Median forecast column not found: {prediction.columns}"
    )

pred = prediction[median_col].to_numpy()[:HORIZON]

mae = np.mean(np.abs(actual - pred))
rmse = np.sqrt(np.mean((actual - pred) ** 2))
smape = np.mean(
    200 * np.abs(actual - pred)
    / np.maximum(np.abs(actual) + np.abs(pred), 1e-8)
)

print({
    "model": "Chronos-2 zero-shot",
    "MAE": float(mae),
    "RMSE": float(rmse),
    "sMAPE_pct": float(smape),
    "inference_seconds": elapsed,
})

plt.figure(figsize=(12, 4))
plt.plot(actual, label="Actual")
plt.plot(pred, label="Chronos-2 forecast")
plt.title("Chronos-2: 24-hour forecast")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("results/chronos2_forecast.png", dpi=150)
plt.show()
