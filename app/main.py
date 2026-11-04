
# import os
# import joblib
# import pandas as pd

# from fastapi import FastAPI, HTTPException
# from src.schemas import CustomerInput


# MODEL_PATH = "artifacts/churn_pipeline.joblib"

# app = FastAPI(
#     title="RetailMind AI - Churn Prediction API"
# )

# model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


# @app.get("/")
# def home():
#     return {
#         "message": "RetailMind AI Churn Prediction API",
#         "model_loaded": model is not None
#     }


# @app.post("/predict")
# def predict(customer: CustomerInput):
#     if model is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Model not trained. Run: python -m src.train"
#         )

#     data = pd.DataFrame([customer.model_dump()])

#     prediction = model.predict(data)[0]
#     probability = float(model.predict_proba(data)[0][1])

#     return {
#         "churn_prediction": 1 if prediction == "Yes" else 0,
#         "churn_label": prediction,
#         "churn_probability": round(probability, 4)
#     }

import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from src.schemas import CustomerInput


MODEL_PATH = "artifacts/churn_model_v4.joblib"

app = FastAPI(
    title="RetailMind AI - Churn Prediction API"
)

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@app.get("/")
def home():
    return {
        "message": "RetailMind AI Churn Prediction API",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Day 4 model not found. Run: python -m src.train_day4"
        )

    data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(data)[0]

    if prediction in [1, "1", "Yes"]:
        prediction_value = 1
        prediction_label = "Yes"
    else:
        prediction_value = 0
        prediction_label = "No"

    response = {
        "churn_prediction": prediction_value,
        "churn_label": prediction_label
    }

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(data)[0][1]
        response["churn_probability"] = round(float(probability), 4)

    return response