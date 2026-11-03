import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    learning_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "artifacts/churn_model_v3.joblib"
RESULTS_PATH = "artifacts/day3_model_comparison.csv"
LEARNING_CURVE_PATH = "artifacts/day3_learning_curves.png"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    if "Churn" not in df.columns:
        raise ValueError(
            "Dataset must contain the Churn column"
        )

    return df


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
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
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
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


def get_model_configurations():
    return {
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            ),
            "parameters": {
                "classifier__max_depth": [
                    3,
                    5,
                    8,
                    None
                ],
                "classifier__min_samples_split": [
                    2,
                    5,
                    10
                ],
                "classifier__min_samples_leaf": [
                    1,
                    2,
                    4
                ],
                "classifier__criterion": [
                    "gini",
                    "entropy"
                ]
            }
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=1
            ),
            "parameters": {
                "classifier__n_estimators": [
                    100,
                    200
                ],
                "classifier__max_depth": [
                    5,
                    10,
                    None
                ],
                "classifier__min_samples_split": [
                    2,
                    5
                ],
                "classifier__min_samples_leaf": [
                    1,
                    2
                ]
            }
        },

        "SVM": {
            "model": SVC(
                class_weight="balanced",
                random_state=42
            ),
            "parameters": {
                "classifier__C": [
                    0.1,
                    1,
                    10
                ],
                "classifier__kernel": [
                    "linear",
                    "rbf"
                ],
                "classifier__gamma": [
                    "scale",
                    "auto"
                ]
            }
        }
    }


def create_learning_curves(
    trained_models,
    X_train,
    y_train,
    cv
):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    train_sizes = [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ]

    for axis, (model_name, model) in zip(
        axes,
        trained_models.items()
    ):
        sizes, training_scores, validation_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="recall",
            train_sizes=train_sizes,
            n_jobs=1
        )

        training_mean = training_scores.mean(axis=1)
        validation_mean = validation_scores.mean(axis=1)

        axis.plot(
            sizes,
            training_mean,
            marker="o",
            label="Training Recall"
        )

        axis.plot(
            sizes,
            validation_mean,
            marker="o",
            label="Validation Recall"
        )

        axis.set_title(model_name)
        axis.set_xlabel("Training Examples")
        axis.set_ylabel("Recall")
        axis.set_ylim(0, 1.05)
        axis.grid(True)
        axis.legend()

    figure.suptitle(
        "Day 3 Learning Curves",
        fontsize=16
    )

    figure.tight_layout()

    figure.savefig(
        LEARNING_CURVE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


def main():
    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    df = load_data()

    X = df.drop(columns=["Churn"])

    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .map({
            "No": 0,
            "Yes": 1
        })
    )

    if y.isna().any():
        raise ValueError(
            "Churn values must contain only Yes and No"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    model_configurations = get_model_configurations()

    results = []
    tuned_models = {}

    for model_name, configuration in model_configurations.items():
        print()
        print("=" * 60)
        print(f"Tuning {model_name}")
        print("=" * 60)

        pipeline = Pipeline([
            (
                "preprocessor",
                build_preprocessor(X_train)
            ),
            (
                "classifier",
                configuration["model"]
            )
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=configuration["parameters"],
            scoring="recall",
            cv=cross_validation,
            n_jobs=1,
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(
            X_train,
            y_train
        )

        tuned_model = grid_search.best_estimator_

        predictions = tuned_model.predict(X_test)

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
            "Model": model_name,
            "Best CV Recall": grid_search.best_score_,
            "Test Accuracy": accuracy,
            "Test Precision": precision,
            "Test Recall": recall,
            "Test F1 Score": f1,
            "Best Parameters": str(
                grid_search.best_params_
            )
        })

        tuned_models[model_name] = tuned_model

        print()
        print("Best parameters:")
        print(grid_search.best_params_)

        print(
            "Best CV recall:",
            round(grid_search.best_score_, 4)
        )

        print(
            "Test recall:",
            round(recall, 4)
        )

        print()
        print(
            classification_report(
                y_test,
                predictions,
                target_names=[
                    "No Churn",
                    "Churn"
                ],
                zero_division=0
            )
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "Best CV Recall",
            "Test Recall",
            "Test F1 Score"
        ],
        ascending=False
    ).reset_index(drop=True)

    print()
    print("=" * 60)
    print("DAY 3 MODEL COMPARISON")
    print("=" * 60)

    display_columns = [
        "Model",
        "Best CV Recall",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1 Score"
    ]

    print(
        results_df[display_columns].to_string(
            index=False
        )
    )

    champion_name = results_df.iloc[0]["Model"]
    champion_model = tuned_models[champion_name]

    joblib.dump(
        champion_model,
        MODEL_PATH
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print()
    print("=" * 60)
    print("DAY 3 CHAMPION")
    print("=" * 60)

    print(
        "Tuned champion:",
        champion_name
    )

    print(
        "Champion saved to:",
        MODEL_PATH
    )

    print(
        "Comparison saved to:",
        RESULTS_PATH
    )

    print()
    print("Creating learning curves...")

    create_learning_curves(
        tuned_models,
        X_train,
        y_train,
        cross_validation
    )

    print(
        "Learning curves saved to:",
        LEARNING_CURVE_PATH
    )


if __name__ == "__main__":
    main()