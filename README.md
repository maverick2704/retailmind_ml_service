# RetailMind AI - Churn Prediction API

## Windows CMD

1. Open CMD in this project folder.
2. Create a virtual environment:
   py -m venv .venv
3. Activate it:
   .venv\Scripts\activate
4. Install dependencies:
   python -m pip install -r requirements.txt
5. Put the Telco Customer Churn dataset in `data` and rename it to `telco_churn.csv`.
6. Train and save the pipeline:
   python -m src.train
7. Start the API:
   python -m uvicorn app.main:app --reload
8. Open http://127.0.0.1:8000/docs

## Sample JSON for POST /predict

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1026.0
}
