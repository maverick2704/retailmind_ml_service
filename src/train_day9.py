import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/train.csv"
MODEL_PATH = "artifacts/demand_lstm_v9.keras"
PLOT_PATH = "artifacts/day9_forecast.png"

LOOKBACK = 30


# 1. Load data
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")


# Use one store-item series for the forecasting prototype
df = df[
    (df["store"] == 1) &
    (df["item"] == 1)
].copy()

sales = df[["sales"]].values


# 2. Normalize sales
scaler = MinMaxScaler()

sales_scaled = scaler.fit_transform(sales)


# 3. Create 30-day -> next-day windows
X = []
y = []

for i in range(LOOKBACK, len(sales_scaled)):

    X.append(
        sales_scaled[i - LOOKBACK:i]
    )

    y.append(
        sales_scaled[i]
    )


X = np.array(X)
y = np.array(y)


# 4. Train/test split
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


# 5. Build LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(LOOKBACK, 1)
    ),

    tf.keras.layers.LSTM(
        50
    ),

    tf.keras.layers.Dense(
        1
    )
])


model.compile(
    optimizer="adam",
    loss="mse"
)


# 6. Train
print("\nTraining Day 9 LSTM...\n")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# 7. Predict
predictions = model.predict(
    X_test,
    verbose=0
)


# Convert normalized values back to actual sales
predictions = scaler.inverse_transform(
    predictions
)

actual = scaler.inverse_transform(
    y_test
)


# 8. Evaluate
mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)


print("\nDAY 9 RESULTS")

print(
    "MAE:",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)


# 9. Plot actual vs predicted
plt.figure(
    figsize=(10, 5)
)

plt.plot(
    actual,
    label="Actual Sales"
)

plt.plot(
    predictions,
    label="Predicted Sales"
)

plt.title(
    "Day 9 Demand Forecast"
)

plt.xlabel(
    "Day"
)

plt.ylabel(
    "Sales"
)

plt.legend()

os.makedirs(
    "artifacts",
    exist_ok=True
)

plt.savefig(
    PLOT_PATH
)

plt.close()


# 10. Save model
model.save(
    MODEL_PATH
)


print(
    "\nModel saved to:",
    MODEL_PATH
)

print(
    "Forecast plot saved to:",
    PLOT_PATH
)