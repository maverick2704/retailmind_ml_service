import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = "data/telco_churn.csv"

DAY4_MODEL = "artifacts/churn_model_v4.joblib"
ANN_MODEL = "artifacts/churn_ann_v5.keras"
PREPROCESSOR = "artifacts/churn_ann_preprocessor_v5.joblib"
RESULTS = "artifacts/day5_model_comparison.csv"


# -------------------------
# Load data
# -------------------------

df = pd.read_csv(DATA_PATH)

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# -------------------------
# Train/test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# Preprocessing
# -------------------------

numeric_cols = X.select_dtypes(
    include=["number"]
).columns

categorical_cols = X.select_dtypes(
    include=["object", "str", "category"]
).columns


preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric_cols
    ),

    (
        "cat",
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]),
        categorical_cols
    )
])


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# -------------------------
# ANN
# -------------------------

model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(X_train_processed.shape[1],)
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -------------------------
# Early stopping
# -------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# -------------------------
# Train ANN
# -------------------------

print("\nTraining Day 5 ANN...\n")

model.fit(
    X_train_processed,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# -------------------------
# ANN prediction
# -------------------------

probabilities = model.predict(
    X_test_processed,
    verbose=0
).flatten()

ann_predictions = (
    probabilities >= 0.5
).astype(int)


def metrics(name, y_true, predictions):

    return {
        "Model": name,

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions
        ),

        "Recall": recall_score(
            y_true,
            predictions
        ),

        "F1": f1_score(
            y_true,
            predictions
        )
    }


results = []

results.append(
    metrics(
        "ANN Day 5",
        y_test,
        ann_predictions
    )
)


# -------------------------
# Compare with Day 4
# -------------------------

if os.path.exists(DAY4_MODEL):

    day4_model = joblib.load(
        DAY4_MODEL
    )

    day4_predictions = day4_model.predict(
        X_test
    )

    results.append(
        metrics(
            "Day 4 Champion",
            y_test,
            day4_predictions
        )
    )


# -------------------------
# Show results
# -------------------------

results_df = pd.DataFrame(results)

print("\nDAY 5 MODEL COMPARISON\n")

print(
    results_df.to_string(
        index=False
    )
)


# -------------------------
# Save Day 5
# -------------------------

os.makedirs(
    "artifacts",
    exist_ok=True
)

model.save(
    ANN_MODEL
)

joblib.dump(
    preprocessor,
    PREPROCESSOR
)

results_df.to_csv(
    RESULTS,
    index=False
)


print("\nSaved:")
print(ANN_MODEL)
print(PREPROCESSOR)
print(RESULTS)
