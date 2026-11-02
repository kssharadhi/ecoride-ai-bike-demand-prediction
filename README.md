# EcoRide AI – Smart Bike Demand Prediction

A complete machine-learning capstone that predicts hourly bicycle rental demand from calendar and weather inputs.

## Features
- Public UCI Bike Sharing dataset (`hour.csv`)
- Data preprocessing with missing-value handling, scaling, and one-hot encoding
- Random Forest Regression model
- Train/test split and MAE, RMSE, and R² evaluation
- Model serialization using Joblib
- Flask web interface and JSON prediction API
- Health-check endpoint for deployment

## Project structure
```text
EcoRide_AI/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── render.yaml
├── .gitignore
├── data/                 # downloaded dataset is stored here
├── model/                # generated model and metrics
└── templates/
    └── index.html
```

## Run locally
Use Python 3.10 or later.

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python train_model.py
python app.py
```
Open `http://127.0.0.1:5000`.

The first training run downloads the UCI Bike Sharing dataset and saves it under `data/`. Training creates `model/ecoride_model.joblib` and `model/metrics.json`. Commit the generated model folder to GitHub before deploying.

## JSON API
Send a POST request to `/api/predict` with JSON:
```json
{
  "season": 2, "yr": 1, "mnth": 6, "hr": 17,
  "holiday": 0, "weekday": 2, "workingday": 1,
  "weathersit": 1, "temp": 0.60, "atemp": 0.58,
  "hum": 0.55, "windspeed": 0.15
}
```
The response contains `predicted_bike_rentals`.

## Deployment on Render
1. Push the project to a GitHub repository.
2. Run `python train_model.py` locally and ensure `model/ecoride_model.joblib` exists.
3. Commit and push the generated `model/` folder.
4. In Render, create a **Web Service** connected to the repository.
5. Build command: `pip install -r requirements.txt`
6. Start command: `gunicorn app:app`
7. After deployment, test `/health` and the home page.

Render free-instance availability and terms can change. The app can also be run locally or deployed on another Python-compatible host.

## Notes
- The dataset target is `cnt` (total rentals). `casual` and `registered` are deliberately excluded from model inputs to prevent target leakage.
- Temperature, feels-like temperature, humidity, and wind speed use the normalized 0–1 values supplied by the dataset.
- This is an educational forecasting tool, not a guarantee of future demand.
- Dataset: UCI Machine Learning Repository, Bike Sharing Dataset, DOI: 10.24432/C5W894.
