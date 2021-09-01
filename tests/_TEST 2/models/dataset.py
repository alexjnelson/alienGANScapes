# pulls images from the dataset folder and creates a tensorflow dataset
import math
import os
import pathlib

import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt


def get_dataset(path='./images', batch_size=256, img_height=128, img_width=128, seed=None):
    if seed is None:
        seed = np.random.randint(0, 1e6)

    if type(path) == str:
        data_dir = pathlib.Path(path)
    else:
        data_dir = path

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_data, val_data


def get_normalized_dataset(path='./images', batch_size=256, img_height=128, img_width=128, seed=None, color_range=255):
    '''
    Wrapper for get_dataset taking same arguments, with the addition of:
    color_range: the maximum color value of a pixel (default 255)
    '''

    # scales between [-1, 1] as the range of the Generator output is the tanh activation function which
    # is also bound to this range
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./(color_range/2.), offset=-1)

    train_data, val_data = get_dataset(path, batch_size, img_height, img_width, seed=seed)
    class_names = train_data.class_names

    normalized_train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

    return normalized_train_data, normalized_val_data, class_names


def get_normalized_images(path='./images', batch_size=256, img_height=128, img_width=128, seed=None, color_range=255):
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
        batch_size=batch_size)
    
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./(color_range/2.), offset=-1)
    normalized_data = data.map(lambda x: (normalization_layer(x)))
    return normalized_data


if __name__ == '__main__' and False:
    DATA_DIR = './images'
    COLOR_RANGE = 255
    HEIGHT = 128
    WIDTH = 128
    BATCH_SIZE = 256

    train_data, val_data, class_names = get_normalized_dataset(DATA_DIR, BATCH_SIZE, HEIGHT, WIDTH, color_range=COLOR_RANGE)
    image_batch, labels_batch = next(iter(train_data))
    first_image = image_batch[0]
    print('Dataset pixel range: ', np.min(first_image), np.max(first_image))

    # the next-higher number that is a perfect square
    gridsize = math.floor(math.sqrt(len(class_names))) + 1
    used_classes = []
    used_images = []
    for images, labels in train_data.take(1):
        for i in range(len(images)):
            if class_names[labels[i]] not in used_classes:
                used_images.append(images[i])
                used_classes.append(class_names[labels[i]])
            if len(used_classes) >= len(class_names):
                break

        for i, im in enumerate(zip(used_images, used_classes)):
            ax = plt.subplot(gridsize, gridsize, i + 1)
            # un-normalize the image to be displayed
            image = ((im[0].numpy() + 1) * (COLOR_RANGE / 2))
            plt.imshow(image.astype("uint8"))
            plt.title(im[1])
            plt.axis("off")
    plt.show()


if __name__ == '__main__' and True:
    DATA_DIR = './images'
    COLOR_RANGE = 255
    HEIGHT = 128
    WIDTH = 128
    BATCH_SIZE = 256

    data = get_normalized_images(DATA_DIR, BATCH_SIZE, HEIGHT, WIDTH, color_range=COLOR_RANGE)
    image_batch = next(iter(data))
    first_image = image_batch[0]
    print('Dataset pixel range: ', np.min(first_image), np.max(first_image))

    for images in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            # un-normalize the image to be displayed
            image = ((images[i].numpy() + 1) * (COLOR_RANGE / 2))
            plt.imshow(image.astype("uint8"))
            plt.axis("off")
    plt.show()
