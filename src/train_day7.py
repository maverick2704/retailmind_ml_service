import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


CLASSIFICATION_DATASET = "data/fashion_products"
DETECTION_IMAGES = "data/detection_images"
MODEL_PATH = "artifacts/fashion_classifier_v7.keras"

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70


# =========================================================
# 1. LOAD TRAINING IMAGES
# =========================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    CLASSIFICATION_DATASET,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=32
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    CLASSIFICATION_DATASET,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=32
)

class_names = train_data.class_names

print("\nClasses:", class_names)


# =========================================================
# 2. PRETRAINED RESNET50 BACKBONE
# =========================================================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


model = Sequential([
    Input(shape=(224, 224, 3)),

    tf.keras.layers.Lambda(
        preprocess_input
    ),

    base_model,

    GlobalAveragePooling2D(),

    Dense(
        len(class_names),
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 3. TRAIN CLASSIFICATION HEAD
# =========================================================

print("\nTraining classification head...\n")

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5
)


# =========================================================
# 4. FINE-TUNE TOP RESNET LAYERS
# =========================================================

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nFine-tuning model...\n")

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=3
)


# =========================================================
# 5. SAVE TRAINED MODEL
# =========================================================

os.makedirs(
    "artifacts",
    exist_ok=True
)

model.save(
    MODEL_PATH
)

print("\nModel saved to:", MODEL_PATH)


# =========================================================
# 6. TEST ON DETECTION_IMAGES
# =========================================================

print("\n======================================")
print("TESTING NEW IMAGES")
print("======================================\n")


if not os.path.exists(DETECTION_IMAGES):
    print("Folder not found:", DETECTION_IMAGES)

else:

    for filename in os.listdir(DETECTION_IMAGES):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            DETECTION_IMAGES,
            filename
        )

        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        image_array = tf.keras.utils.img_to_array(
            image
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        prediction = model.predict(
            image_array,
            verbose=0
        )[0]

        class_index = np.argmax(
            prediction
        )

        confidence = float(
            prediction[class_index]
        )

        predicted_class = class_names[
            class_index
        ]

        print("Image:", filename)

        if confidence >= CONFIDENCE_THRESHOLD:

            print(
                "Prediction:",
                predicted_class
            )

            print(
                "Confidence:",
                round(
                    confidence * 100,
                    2
                ),
                "%"
            )

        else:

            print("Prediction: NO DETECTION")

            print(
                "Highest confidence:",
                round(
                    confidence * 100,
                    2
                ),
                "%"
            )

        print("--------------------------------------")