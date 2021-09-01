# pulls images from the dataset folder and creates a tensorflow dataset
import math
import os
import pathlib

import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.script_ops import numpy_function


"""
IMAGE AUGMENTATION
"""


def get_random_bool():
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)


def randomly_augment_image(img):
    img = tf.cond(
        tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5),
        lambda: flip_horizontally(img),
        lambda: img
    )
    # img = tf.cond(
    #     tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5),
    #     lambda: tf.numpy_function(radial_transform, [img], tf.float32),
    #     lambda: img
    # )
    return img


# described in the paper https://arxiv.org/pdf/1708.04347.pdf
def radial_transform(X):
    M, N, _ = X.shape
    K = np.zeros(X.shape)

    u, v = np.random.randint(0, M - 1), np.random.randint(0, N - 1)
    for r in range(N):
        for m in range(M):
            theta = 2 * math.pi * m / M
            x = round(r * math.cos(theta))
            y = round(r * math.sin(theta))
            if 0 <= u + x < M and 0 <= v + y < N:
                K[m, r, :] = X[u + x, v + y, :]
    return np.ndarray.astype(K, np.float32)


def flip_horizontally(img):
    flipped_img = tf.image.flip_left_right(img)
    return flipped_img


"""
LOADING IMAGES
"""


def get_images(path='./images', batch_size=64, img_height=128, img_width=128, seed=None):
    if seed is None:
        seed = np.random.randint(0, 1e6)

    if type(path) == str:
        data_dir = pathlib.Path(path)
    else:
        data_dir = path

    data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode=None,
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return data


def get_preprocessed_images(path='./images', batch_size=64, img_height=128, img_width=128, seed=None, color_range=255):
    normalize = tf.keras.layers.experimental.preprocessing.Rescaling(1./(color_range/2.), offset=-1)

    data = get_images(path, batch_size, img_height, img_width, seed)
    data = data.map(lambda x: tf.map_fn(randomly_augment_image, x))
    data = data.map(normalize)
    return data


if __name__ == '__main__':
    if False:
        print(get_random_bool())

    elif True:
        DATA_DIR = './images'
        COLOR_RANGE = 255
        HEIGHT = 128
        WIDTH = 128
        BATCH_SIZE = 64
        SAMPLES = 9

        g = math.ceil(SAMPLES ** 0.5)

        data = get_preprocessed_images(DATA_DIR, BATCH_SIZE, HEIGHT, WIDTH, color_range=COLOR_RANGE)
        image_batch = next(iter(data))
        first_image = image_batch[0]
        print('Dataset pixel range: ', np.min(first_image), np.max(first_image))

        for images in data.take(1):
            for i in range(SAMPLES):
                ax = plt.subplot(g, g, i + 1)
                # un-normalize the image to be displayed
                image = ((images[i].numpy() + 1) * (COLOR_RANGE / 2))
                plt.imshow(image.astype("uint8"))
                plt.axis("off")
        plt.show()
