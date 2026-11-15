import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model, Sequential


DATA_PATH = "data/fashion_products/Bags"
OUTPUT_PATH = "artifacts/day11_generated"

IMG_SIZE = 64
LATENT_DIM = 100
EPOCHS = 30

os.makedirs(OUTPUT_PATH, exist_ok=True)


# -------------------------
# 1. Load minority class
# -------------------------

dataset = tf.keras.utils.image_dataset_from_directory(
    "data/fashion_products",
    labels="inferred",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=8
)

class_names = dataset.class_names

print("Classes:", class_names)

# Normalize images to [-1, 1]
dataset = dataset.map(
    lambda x, y: (
        tf.cast(x, tf.float32) / 127.5 - 1.0,
        y
    )
)


# Use Bags as minority class
bag_index = class_names.index("Bags")

bag_images = []

for images, labels in dataset:

    mask = labels == bag_index

    selected = tf.boolean_mask(
        images,
        mask
    )

    if len(selected) > 0:
        bag_images.append(selected)


bag_images = tf.concat(
    bag_images,
    axis=0
)

bag_dataset = tf.data.Dataset.from_tensor_slices(
    bag_images
).shuffle(
    len(bag_images)
).batch(4)


# -------------------------
# 2. Generator
# -------------------------

generator = Sequential([
    layers.Input(shape=(LATENT_DIM,)),

    layers.Dense(
        8 * 8 * 128
    ),

    layers.Reshape(
        (8, 8, 128)
    ),

    layers.Conv2DTranspose(
        64,
        4,
        strides=2,
        padding="same",
        activation="relu"
    ),

    layers.Conv2DTranspose(
        32,
        4,
        strides=2,
        padding="same",
        activation="relu"
    ),

    layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding="same",
        activation="tanh"
    )
])


# -------------------------
# 3. Discriminator
# -------------------------

discriminator = Sequential([
    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    layers.Conv2D(
        32,
        4,
        strides=2,
        padding="same"
    ),

    layers.LeakyReLU(0.2),

    layers.Conv2D(
        64,
        4,
        strides=2,
        padding="same"
    ),

    layers.LeakyReLU(0.2),

    layers.Flatten(),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


discriminator.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)


# -------------------------
# 4. GAN
# -------------------------

discriminator.trainable = False

noise = layers.Input(
    shape=(LATENT_DIM,)
)

fake_image = generator(noise)

decision = discriminator(
    fake_image
)

gan = Model(
    noise,
    decision
)

gan.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)


# -------------------------
# 5. Train GAN
# -------------------------

print("\nTraining GAN...\n")

for epoch in range(EPOCHS):

    for real_images in bag_dataset:

        batch_size = tf.shape(real_images)[0]

        random_noise = tf.random.normal(
            (batch_size, LATENT_DIM)
        )

        fake_images = generator.predict(
            random_noise,
            verbose=0
        )

        discriminator.trainable = True

        discriminator.train_on_batch(
            real_images,
            np.ones((batch_size, 1))
        )

        discriminator.train_on_batch(
            fake_images,
            np.zeros((batch_size, 1))
        )

        discriminator.trainable = False

        random_noise = tf.random.normal(
            (batch_size, LATENT_DIM)
        )

        gan.train_on_batch(
            random_noise,
            np.ones((batch_size, 1))
        )

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )


# -------------------------
# 6. Generate images
# -------------------------

noise = tf.random.normal(
    (5, LATENT_DIM)
)

generated = generator.predict(
    noise,
    verbose=0
)

generated = (
    generated + 1
) / 2


for i, image in enumerate(generated):

    plt.imshow(image)

    plt.axis("off")

    plt.savefig(
        f"{OUTPUT_PATH}/synthetic_bag_{i + 1}.png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


generator.save(
    "artifacts/day11_generator.keras"
)

print("\nGenerated images saved to:")
print(OUTPUT_PATH)

print("\nGenerator saved to:")
print("artifacts/day11_generator.keras")