import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "artifacts/champion_model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    return df


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return preprocessor


def main():
    df = load_data()

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

        "SVM": SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Recall",
        ascending=False
    )

    print("\nMODEL COMPARISON\n")
    print(results_df.to_string(index=False))

    champion_name = results_df.iloc[0]["Model"]

    champion_model = trained_models[champion_name]

    print("\nChampion Model:", champion_name)

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(
        champion_model,
        MODEL_PATH
    )

    print("Champion model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
