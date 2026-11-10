import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input


MODEL_PATH = "artifacts/fashion_classifier_v7.keras"
IMAGE_FOLDER = "data/detection_images"
OUTPUT_FOLDER = "artifacts/day8_outputs"

CLASS_NAMES = ["Bags", "Jeans", "Shirt"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False
)



# -----------------------------------------
# 2. Remove background using OpenCV
# -----------------------------------------

def remove_background(image_path):

    image = cv2.imread(image_path)

    mask = np.zeros(
        image.shape[:2],
        np.uint8
    )

    bg_model = np.zeros(
        (1, 65),
        np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64
    )

    height, width = image.shape[:2]

    rectangle = (
        5,
        5,
        width - 10,
        height - 10
    )

    cv2.grabCut(
        image,
        mask,
        rectangle,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    mask = np.where(
        (mask == 2) | (mask == 0),
        0,
        1
    ).astype("uint8")

    product = image * mask[:, :, np.newaxis]

    white_background = np.ones_like(image) * 255

    clean_image = np.where(
        mask[:, :, np.newaxis] == 1,
        product,
        white_background
    )

    return clean_image


# -----------------------------------------
# 3. Classify cleaned image
# -----------------------------------------

def classify(image):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        rgb_image,
        (224, 224)
    )

    input_image = np.expand_dims(
        resized,
        axis=0
    )

    prediction = model.predict(
        input_image,
        verbose=0
    )[0]

    index = np.argmax(prediction)

    return (
        CLASS_NAMES[index],
        float(prediction[index])
    )


# -----------------------------------------
# 4. Process images
# -----------------------------------------

for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    clean_image = remove_background(
        image_path
    )

    predicted_class, confidence = classify(
        clean_image
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "clean_" + filename
    )

    cv2.imwrite(
        output_path,
        clean_image
    )

    print()
    print("Image:", filename)

    print(
        "Prediction:",
        predicted_class
    )

    print(
        "Confidence:",
        round(confidence * 100, 2),
        "%"
    )

    print(
        "Clean image saved:",
        output_path
    )


print(
    "\nDay 8 processing completed."
)