import os
import joblib
import pandas as pd
# import numpy as np

# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.feature_engineering import FeatureEngineer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


DATA_PATH = "data/telco_churn.csv"

DAY3_MODEL_PATH = "artifacts/churn_model_v3.joblib"

DAY4_MODEL_PATH = "artifacts/churn_model_v4.joblib"

RESULTS_PATH = "artifacts/day4_model_comparison.csv"


# class FeatureEngineer(BaseEstimator, TransformerMixin):

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = X.copy()

#         X["tenure_bucket"] = pd.cut(
#             X["tenure"],
#             bins=[-1, 12, 24, 48, 60, np.inf],
#             labels=[
#                 "0-12",
#                 "13-24",
#                 "25-48",
#                 "49-60",
#                 "60+"
#             ]
#         ).astype(str)

#         X["charge_ratio"] = (
#             X["TotalCharges"] /
#             (X["MonthlyCharges"] + 1)
#         )

#         X["monthly_per_tenure"] = (
#             X["MonthlyCharges"] /
#             (X["tenure"] + 1)
#         )

#         X["is_month_to_month"] = (
#             X["Contract"] == "Month-to-month"
#         ).astype(int)

#         X["is_two_year_contract"] = (
#             X["Contract"] == "Two year"
#         ).astype(int)

#         X["has_online_security"] = (
#             X["OnlineSecurity"] == "Yes"
#         ).astype(int)

#         X["has_tech_support"] = (
#             X["TechSupport"] == "Yes"
#         ).astype(int)

#         X["uses_fiber"] = (
#             X["InternetService"] == "Fiber optic"
#         ).astype(int)

#         X["electronic_check"] = (
#             X["PaymentMethod"] == "Electronic check"
#         ).astype(int)

#         return X


def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    return df


def apply_feature_engineering(X):

    engineer = FeatureEngineer()

    return engineer.fit_transform(X)


def build_preprocessor(X):

    engineered = apply_feature_engineering(X)

    numeric_columns = engineered.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = engineered.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ])


def evaluate_model(
    model_name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    return {
        "Model": model_name,

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }


def main():

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    df = load_data()

    X = df.drop(
        columns=["Churn"]
    )

    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .map({
            "No": 0,
            "Yes": 1
        })
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    preprocessor = build_preprocessor(
        X_train
    )

    random_forest_pipeline = Pipeline([
        (
            "feature_engineering",
            FeatureEngineer()
        ),

        (
            "preprocessor",
            preprocessor
        ),

        (
            "pca",
            PCA(
                n_components=0.95,
                random_state=42
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            )
        )
    ])

    xgboost_pipeline = Pipeline([
        (
            "feature_engineering",
            FeatureEngineer()
        ),

        (
            "preprocessor",
            preprocessor
        ),

        (
            "pca",
            PCA(
                n_components=0.95,
                random_state=42
            )
        ),

        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            )
        )
    ])

    models = {
        "Random Forest Day 4":
            random_forest_pipeline,

        "XGBoost Day 4":
            xgboost_pipeline
    }

    results = []

    trained_models = {}

    for name, model in models.items():

        print()
        print("=" * 60)
        print("Training", name)
        print("=" * 60)

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

        result = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results.append(result)

        print(
            "Accuracy:",
            round(result["Accuracy"], 4)
        )

        print(
            "Precision:",
            round(result["Precision"], 4)
        )

        print(
            "Recall:",
            round(result["Recall"], 4)
        )

        print(
            "F1:",
            round(result["F1 Score"], 4)
        )

    if os.path.exists(DAY3_MODEL_PATH):

        print()
        print("=" * 60)
        print("Evaluating Day 3 Champion")
        print("=" * 60)

        day3_model = joblib.load(
            DAY3_MODEL_PATH
        )

        day3_result = evaluate_model(
            "Day 3 Champion",
            day3_model,
            X_test,
            y_test
        )

        results.append(
            day3_result
        )

    else:

        print()
        print(
            "Day 3 model not found. "
            "Skipping Day 3 comparison."
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by=[
            "Recall",
            "F1 Score"
        ],
        ascending=False
    ).reset_index(drop=True)

    print()
    print("=" * 60)
    print("DAY 4 MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    day4_results = results_df[
        results_df["Model"] !=
        "Day 3 Champion"
    ]

    champion_name = (
        day4_results.iloc[0]["Model"]
    )

    champion_model = (
        trained_models[
            champion_name
        ]
    )

    joblib.dump(
        champion_model,
        DAY4_MODEL_PATH
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print()
    print("=" * 60)
    print("DAY 4 FINAL MODEL")
    print("=" * 60)

    print(
        "Production model:",
        champion_name
    )

    print(
        "Saved to:",
        DAY4_MODEL_PATH
    )

    print(
        "Comparison saved to:",
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()