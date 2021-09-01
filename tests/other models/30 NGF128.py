# model framework is based on the tensorflow official tutorial, https://www.tensorflow.org/tutorials/generative/dcgan
# layer parameters were made based on https://github.com/4ndrewparr/DCGAN-landscapes/blob/master/PyTorch%20DCGAN%20Landscapes.ipynb
# also see the original DCGAN paper, https://arxiv.org/pdf/1511.06434.pdf

# MODIFICATIONS:
# the generator uses 128 features while the discriminator still uses 64. this is meant to give the generator an edge late in training

import argparse
import glob
import math
import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from IPython import display
from tensorflow import keras
from tensorflow.keras import layers

from dataset import get_normalized_images


EPOCHS = 50
BATCH_SIZE = 128

# dimensions of noise vector
noise_dim = 100
# how many unique seeds to use, to better visualize progress for each seed
num_examples_to_generate = 16

# colour channels of the generated image
n_channels = 3

# number of generator feature maps
NGF = 128
# number of discriminator feature maps
NDF = 64

# learning rate
LR = 2e-4
# betas
BETA1 = 0.5
BETA2 = 0.999


# define loss functions and optimizers
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(LR, beta_1=BETA1, beta_2=BETA2)
discriminator_optimizer = tf.keras.optimizers.Adam(LR, beta_1=BETA1, beta_2=BETA2)


# reuse this seed over time (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def make_generator_model() -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(layers.Dense(4 * 4 * 16 * NGF, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Reshape((4, 4, 16 * NGF)))
    assert model.output_shape == (None, 4, 4, 16 * NGF)

    model.add(layers.Conv2DTranspose(8 * NGF, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 8 * NGF)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(4 * NGF, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 4 * NGF)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(2 * NGF, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 2 * NGF)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(NGF, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, NGF)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=True, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 3)

    return model


def make_discriminator_model():
    # NOTE CONSIDERATIONS:
    # should we use dropout or not? it was in the TF tut but not the art gen Github
    # also, TF tut does not use batchnorm (but it is recommended in the original paper)
    model = tf.keras.Sequential()
    # NOTE in the github, they did NOT use bias, but a comment in their code says that it would make more sense if they did
    model.add(layers.Conv2D(NDF, (4, 4), strides=(2, 2), padding='same', use_bias=True,
                            input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(NDF * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(NDF * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(NDF * 8, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(NDF * 16, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def generator_loss(fake_output):
    '''
    Compare loss as the difference between the discriminator's output and
    an array of ones. The discriminator would output an array of ones if
    the generator tricked it 100% of the time, so an array of ones would
    be the best possible performance that the generator optimizes to achieve.
    '''
    return loss_fn(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    '''
    Discriminator's loss is based on its ability to classify all real images as real (array
    of ones) and all fake images as fake (array of zeroes)
    '''
    real_loss = loss_fn(tf.ones_like(real_output), real_output)
    fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generate_images(model, test_input):
    predictions = model(test_input, training=False)
    # sets grid size to the square root of the num_examples_to_generate if it is a perfect square; if not,
    # it sets the grid size to the floor of the square root + 1
    n = math.floor(math.sqrt(num_examples_to_generate))
    gridsize = n if math.sqrt(num_examples_to_generate) == n else n + 1

    fig = plt.figure(figsize=(gridsize, gridsize))

    for i in range(predictions.shape[0]):
        im = tf.cast(predictions[i, :, :, :] * 127.5 + 127.5, tf.int16)

        plt.subplot(gridsize, gridsize, i+1)
        plt.imshow(im)
        plt.axis('off')

    return predictions


def get_model(trainable=True, load_checkpoint=False, start_epoch=0, checkpoint_dir='./checkpoints', sample_dir='./samples'):
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # initialize checkpointing
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    if load_checkpoint:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    if trainable:
        def generate_and_save_images(model, epoch, test_input, sampledir=sample_dir):
            generate_images(model, test_input)

            plt.savefig(os.path.join(sampledir, f'image_at_epoch_{epoch}.png'))
            plt.clf()

        def make_gifs():
            anim_file = os.path.join(sample_dir, 'dcgan.gif')
            img_files = os.path.join(sample_dir, 'image*.png')

            with imageio.get_writer(anim_file, mode='I') as writer:
                filenames = glob.glob(img_files)
                filenames = sorted(filenames)
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    image = imageio.imread(filename)
                    writer.append_data(image)

        @tf.function
        def train_step(images):
            noise = tf.random.normal([BATCH_SIZE, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(images, training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            
        def train(dataset, epochs):
            img_files = os.path.join(sample_dir, 'image*.png')
            filenames = glob.glob(img_files)
            start_epoch = len(filenames)

            for epoch in range(epochs):
                start = time.time()

                for image_batch in dataset:
                    train_step(image_batch)

                # Produce images for the GIF as you go
                display.clear_output(wait=True)
                generate_and_save_images(generator,
                                         epoch + 1 + start_epoch,
                                         seed)

                # Save the model every 15 epochs
                if (epoch + 1 + start_epoch) % 15 == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)

                print('Time for epoch {} is {} sec'.format(epoch + 1 + start_epoch, time.time()-start))

            # Generate samples and training GIF after the final epoch
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epochs + start_epoch,
                                     seed)
            make_gifs()

        return generator, discriminator, train

    else:
        return generator, discriminator


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-m", "--mode", type=int, default=0,
                    help="""
                    mode 0: model summary\n
                    mode 1: train from scratch\n
                    mode 2: load checkpoint and continue training\n
                    mode 3: load checkpoint and generate image\n
                    """)

    ap.add_argument("-c", "--checkpoint", default="./checkpoints",
                    help="path to checkpoint directory")

    ap.add_argument("-s", "--samples", default="./samples",
                    help="path to output sample directory")

    ap.add_argument("-d", "--dataset", default="./images",
                    help="path to dataset directory")

    args = vars(ap.parse_args())

    if args['mode'] == 0:
        generator, discriminator = get_model(
            trainable=False,
            load_checkpoint=False,
            checkpoint_dir=args['checkpoint'],
            sample_dir=args['samples'],
        )
        generator.summary()
        discriminator.summary()

    elif args['mode'] == 1:
        generator, discriminator, train = get_model(
            trainable=True,
            load_checkpoint=False,
            checkpoint_dir=args['checkpoint'],
            sample_dir=args['samples'],
        )
        data = get_normalized_images(args['dataset'], BATCH_SIZE, 128, 128, color_range=255)
        train(data, EPOCHS)

    elif args['mode'] == 2:
        generator, discriminator, train = get_model(
            trainable=True,
            load_checkpoint=True,
            checkpoint_dir=args['checkpoint'],
            sample_dir=args['samples'],
        )
        data = get_normalized_images(args['dataset'], BATCH_SIZE, 128, 128, color_range=255)
        train(data, EPOCHS)

    elif args['mode'] == 3:
        generator, discriminator = get_model(
            trainable=False,
            load_checkpoint=True,
            checkpoint_dir=args['checkpoint'],
            sample_dir=args['samples'],
        )
        generate_images(generator, seed)
        plt.show()
