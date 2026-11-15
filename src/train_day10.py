import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_DAYS = 30
OUTPUT_DAYS = 7


# Load data
df = pd.read_csv("data/train.csv")
df["date"] = pd.to_datetime(df["date"])

# Use one store-item time series
df = df[
    (df["store"] == 1) &
    (df["item"] == 1)
].sort_values("date")

sales = df["sales"].values.reshape(-1, 1)


# Scale sales
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales)


# Past 30 days -> next 7 days
X, y = [], []

for i in range(
    INPUT_DAYS,
    len(sales_scaled) - OUTPUT_DAYS + 1
):
    X.append(
        sales_scaled[i - INPUT_DAYS:i]
    )

    y.append(
        sales_scaled[i:i + OUTPUT_DAYS]
    )

X = np.array(X)
y = np.array(y)


# Train-test split
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


# =========================
# Seq2Seq + Attention
# =========================

inputs = tf.keras.Input(
    shape=(INPUT_DAYS, 1)
)


# Encoder
encoder_output, state_h, state_c = tf.keras.layers.LSTM(
    64,
    return_sequences=True,
    return_state=True
)(inputs)


# Decoder input
decoder_input = tf.keras.layers.RepeatVector(
    OUTPUT_DAYS
)(state_h)


# Decoder
decoder_output = tf.keras.layers.LSTM(
    64,
    return_sequences=True
)(
    decoder_input,
    initial_state=[state_h, state_c]
)


# Attention
context = tf.keras.layers.Attention()([
    decoder_output,
    encoder_output
])


# Combine decoder + attention
combined = tf.keras.layers.Concatenate()([
    decoder_output,
    context
])


# Predict 7 future sales values
outputs = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(1)
)(combined)


model = tf.keras.Model(
    inputs,
    outputs
)


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# Train
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32
)


# Predict
prediction = model.predict(
    X_test[:1],
    verbose=0
)


actual = scaler.inverse_transform(
    y_test[0]
).flatten()

predicted = scaler.inverse_transform(
    prediction[0]
).flatten()


# Evaluate whole test set
all_predictions = model.predict(
    X_test,
    verbose=0
)

actual_all = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

predicted_all = scaler.inverse_transform(
    all_predictions.reshape(-1, 1)
)


mae = mean_absolute_error(
    actual_all,
    predicted_all
)

rmse = np.sqrt(
    mean_squared_error(
        actual_all,
        predicted_all
    )
)


print("\nMAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

print("\nActual 7 days:")
print(np.round(actual, 2))

print("\nPredicted 7 days:")
print(np.round(predicted, 2))


# Plot
days = np.arange(
    1,
    OUTPUT_DAYS + 1
)

plt.plot(
    days,
    actual,
    marker="o",
    label="Actual"
)

plt.plot(
    days,
    predicted,
    marker="o",
    label="Forecast"
)

plt.xlabel("Future Day")
plt.ylabel("Sales")
plt.title("7-Day Demand Forecast")
plt.legend()

os.makedirs(
    "artifacts",
    exist_ok=True
)

plt.savefig(
    "artifacts/day10_forecast.png"
)

plt.close()


# Save model
model.save(
    "artifacts/demand_seq2seq_v10.keras"
)

print("\nModel saved successfully.")