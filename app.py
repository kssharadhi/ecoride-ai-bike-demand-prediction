from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "ecoride_model.joblib"

app = Flask(__name__)
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

def predict_demand(values):
    if bundle is None:
        raise RuntimeError("Model not found. Run python train_model.py first.")
    row = pd.DataFrame([values], columns=bundle["features"])
    prediction = max(0, float(bundle["model"].predict(row)[0]))
    return round(prediction)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    values = {
        "season": 2, "yr": 1, "mnth": 6, "hr": 17, "holiday": 0,
        "weekday": 2, "workingday": 1, "weathersit": 1,
        "temp": 0.60, "atemp": 0.58, "hum": 0.55, "windspeed": 0.15
    }
    if request.method == "POST":
        try:
            values = {
                "season": int(request.form["season"]),
                "yr": int(request.form["yr"]),
                "mnth": int(request.form["mnth"]),
                "hr": int(request.form["hr"]),
                "holiday": int(request.form["holiday"]),
                "weekday": int(request.form["weekday"]),
                "workingday": int(request.form["workingday"]),
                "weathersit": int(request.form["weathersit"]),
                "temp": float(request.form["temp"]),
                "atemp": float(request.form["atemp"]),
                "hum": float(request.form["hum"]),
                "windspeed": float(request.form["windspeed"])
            }
            prediction = predict_demand(values)
        except Exception as exc:
            error = str(exc)
    return render_template("index.html", prediction=prediction, error=error, values=values)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
        required = bundle["features"]
        missing = [key for key in required if key not in payload]
        if missing:
            return jsonify({"error": "Missing fields", "fields": missing}), 400
        values = {key: payload[key] for key in required}
        result = predict_demand(values)
        return jsonify({"predicted_bike_rentals": result, "unit": "bikes per hour"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": bundle is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
