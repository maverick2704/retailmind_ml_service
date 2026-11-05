import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


MODEL_PATH = "artifacts/fashion_cnn_v6.keras"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)


# Normalize pixels: 0-255 -> 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# Add channel dimension
# (28, 28) -> (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)


# Build CNN
model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        10,
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Stop training when validation loss stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


print("\nTraining Day 6 CNN...\n")

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)


# Evaluate
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    "\nTest Accuracy:",
    round(test_accuracy, 4)
)


# Predictions
probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# Confusion matrix
print("\nCONFUSION MATRIX\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# Per-class metrics
print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES
    )
)


# Save CNN
os.makedirs(
    "artifacts",
    exist_ok=True
)

model.save(
    MODEL_PATH
)

print(
    "\nCNN saved to:",
    MODEL_PATH
)