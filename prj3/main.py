def analysis_dataset():
    from pathlib import Path
    import numpy as np
    import pandas as pd

    ROOT = Path("assets/SKAB_0.9.0/valve1")
    OUT = Path("outputs/eda")
    OUT.mkdir(parents=True, exist_ok=True)
    SENSORS = [
        "Accelerometer1RMS",
        "Accelerometer2RMS",
        "Current",
        "Pressure",
        "Temperature",
        "Thermocouple",
        "Voltage",
        "Volume Flow RateRMS",
    ]
    LABELS = ["anomaly", "changepoint"]
    REQUIRED = ["datetime", *SENSORS, *LABELS]
    file_reports = []
    sensor_reports = []
    frames = []

    paths = sorted(ROOT.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files under {ROOT}")

    for path in paths:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = df.columns.str.strip()
        missing_cols = set(REQUIRED) - set(df.columns)
    
        if missing_cols:
            raise ValueError(f"{path}: missing {missing_cols}")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        for col in SENSORS + LABELS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            name = path.relative_to(ROOT).as_posix()
            original_time = df["datetime"]
            valid_time = original_time.dropna()
            delta = valid_time.diff().dt.total_seconds()
            delta = delta.dropna()
            invalid_labels = {}

        for label in LABELS:
            invalid_labels[label] = int((~df[label].isin([0, 1])).sum())


        # Count consecutive anomalous runs.
        a = df["anomaly"].eq(1).to_numpy()
        starts = a & ~np.r_[False, a[:-1]]
        ends = a & ~np.r_[a[1:], False]
        file_reports.append({
            "file": name,
            "rows": len(df),
            "start": valid_time.min(),
            "end": valid_time.max(),
            "missing_datetime": int(original_time.isna().sum()),
            "missing_sensor_cells": int(
            df[SENSORS].isna().sum().sum()
            ),
            "duplicate_timestamps": int(
            valid_time.duplicated().sum()
            ),
            "nonpositive_time_steps": int((delta <= 0).sum()),
            "median_interval_s": delta.median(),
            "max_interval_s": delta.max(),
            "anomaly_count": int(a.sum()),
            "anomaly_ratio": float(a.mean()),
            "anomaly_segments": int(starts.sum()),
            "changepoint_count": int(
            df["changepoint"].eq(1).sum()
            ),
            "invalid_anomaly_labels": invalid_labels["anomaly"],
            "invalid_changepoint_labels": invalid_labels["changepoint"],
        })
        for sensor in SENSORS:
            for label_value in (0, 1):
                values = df.loc[
                df["anomaly"].eq(label_value), sensor
                ].dropna()
                sensor_reports.append({
                    "file": name,
                    "sensor": sensor,
                    "anomaly": label_value,
                    "count": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "median": values.median(),
                    "max": values.max(),
                    "missing_total": int(df[sensor].isna().sum()),
                })
            df["source_file"] = name
            frames.append(df)

    files = pd.DataFrame(file_reports)
    stats = pd.DataFrame(sensor_reports)
    all_data = pd.concat(frames, ignore_index=True)
    
    # Correlations are descriptive only; not fitted preprocessing.
    normal = all_data.loc[all_data["anomaly"].eq(0)]
    abnormal = all_data.loc[all_data["anomaly"].eq(1)]
    
    files.to_csv(OUT / "file_summary.csv", index=False)
    stats.to_csv(OUT / "sensor_summary.csv", index=False)
    
    normal[SENSORS].corr().to_csv(
    OUT / "correlation_normal.csv"
    )
    
    abnormal[SENSORS].corr().to_csv(
    OUT / "correlation_anomaly.csv"
    )
    
    print("CSV files:", len(paths))
    print("Total rows:", len(all_data))
    print("Anomaly rows:", len(abnormal))
    print("Changepoint rows:",
    int(all_data["changepoint"].eq(1).sum()))
    print("Output:", OUT.resolve())

if __name__ == "__main__":
    analysis_dataset()