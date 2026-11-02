"""Download the UCI Bike Sharing dataset, train a demand model, and save it with Joblib."""
from pathlib import Path
import json
import zipfile
import requests
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
ZIP_PATH = DATA_DIR / "bike_sharing_dataset.zip"

def load_data():
    if not ZIP_PATH.exists():
        print("Downloading UCI Bike Sharing dataset...")
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        with archive.open("hour.csv") as csv_file:
            return pd.read_csv(csv_file)

def main():
    df = load_data()
    # cnt is total rentals. casual and registered are excluded to avoid target leakage.
    features = ["season", "yr", "mnth", "hr", "holiday", "weekday",
                "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
    target = "cnt"
    X = df[features]
    y = df[target]

    categorical = ["season", "mnth", "hr", "weekday", "weathersit"]
    numeric = [c for c in features if c not in categorical]

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=250, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    joblib.dump({"model": model, "features": features}, MODEL_DIR / "ecoride_model.joblib")
    metrics = {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2),
               "r2": round(float(r2), 4), "training_rows": int(len(X_train)),
               "testing_rows": int(len(X_test))}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Training complete. Model saved to model/ecoride_model.joblib")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
