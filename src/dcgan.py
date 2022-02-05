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

from dataset import get_preprocessed_images
from utils import get_params


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-m", "--mode", type=int, default=0,
                    help="""
                    mode 0: model summary\n
                    mode 1: train from scratch\n
                    mode 2: load checkpoint and continue training\n
                    mode 3: load checkpoint and generate image\n
                    mode 4: load specified checkpoint and save output
                    """)

    ap.add_argument("-c", "--checkpoint", default="./checkpoints",
                    help="path to checkpoint directory")

    ap.add_argument("-s", "--samples", default="./samples",
                    help="path to output sample directory")

    ap.add_argument("-d", "--dataset", default="./images",
                    help="path to dataset directory")

    ap.add_argument("-lc", "--load_checkpoint", default="ckpt-13",
                    help="which checkpoint to load (only valid if mode 4 is specified)")

    ap.add_argument("-o", "--output", default="",
                    help="where to save output from loaded model (only valid if mode 4 is specified)")

    args = vars(ap.parse_args())


EPOCHS = 200
BATCH_SIZE = 64
EPOCHS_TO_SAVE = 10

DIM = 512

# dimensions of noise vector
noise_dim = 100
# how many unique seeds to use, to better visualize progress for each seed
num_examples_to_generate = 9

# colour channels of the generated image
n_channels = 3

# number of generator feature maps
NGF = 64
# number of discriminator feature maps
NDF = 64

N_CONV_LAYERS = 4

# learning rate
LR = 2e-4
# betas
BETA1 = 0.5
BETA2 = 0.999

LR_SCHEDULE = tf.keras.optimizers.schedules.ExponentialDecay(
    LR,
    decay_steps=100000,
    decay_rate=0.99,
    staircase=True)


# define loss functions and optimizers
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_SCHEDULE, beta_1=BETA1, beta_2=BETA2)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_SCHEDULE, beta_1=BETA1, beta_2=BETA2)

# reuse this seed over time (so it's easier
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

assert (DIM & (DIM - 1)) == 0  # check that DIM is a power of 2
if not ((DIM & (DIM - 1)) == 0):
    raise ValueError('Output DIM must be a power of 2')


def make_generator_model() -> tf.keras.Model:
    m = 2 ** N_CONV_LAYERS
    start_dim, upscale = get_params(DIM, N_CONV_LAYERS)

    model = tf.keras.Sequential()
    model.add(layers.Dense(start_dim * start_dim * 16 * NGF, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((start_dim, start_dim, 16 * NGF)))
    assert model.output_shape == (None, start_dim, start_dim, 16 * NGF)

    for i in range(N_CONV_LAYERS - 1, 0, -1):
        m = 2 ** i
        model.add(layers.Conv2DTranspose(m / 2 * NGF, (4, 4), strides=(upscale, upscale), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(upscale, upscale), padding='same', use_bias=True, activation='tanh'))
    assert model.output_shape == (None, DIM, DIM, 3)
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(NDF, (4, 4), strides=(2, 2), padding='same', use_bias=True,
                            input_shape=[DIM, DIM, 3]))
    model.add(layers.LeakyReLU(0.2))

    for i in range(1, N_CONV_LAYERS):
        m = 2 ** i
        model.add(layers.Conv2D(NDF * m, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

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

    if type(load_checkpoint) == str:
        checkpoint.restore(os.path.join(checkpoint_dir, load_checkpoint))
    elif load_checkpoint:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    if trainable:
        def generate_and_save_images(model, epoch, test_input, sampledir=sample_dir):
            generate_images(model, test_input)

            plt.savefig(os.path.join(sampledir, f'image_at_epoch_{epoch}.png'))
            plt.close('all')

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
            try:
                for epoch in range(epochs):
                    start = time.time()

                    for image_batch in dataset:
                        train_step(image_batch)

                    # Produce images for the GIF as you go
                    display.clear_output(wait=True)
                    generate_and_save_images(generator,
                                             epoch + 1 + start_epoch,
                                             seed)

                    # Save the model every few epochs (default 25)
                    if (epoch + 1 + start_epoch) % EPOCHS_TO_SAVE == 0:
                        print('Checkpointing at epoch ' + str(epoch + 1 + start_epoch))
                        checkpoint.save(file_prefix=checkpoint_prefix)

                    print('Time for epoch {} is {} sec'.format(epoch + 1 + start_epoch, time.time()-start))
            finally:
                print('Final checkpoint at epoch ' + str(epoch + 1 + start_epoch))
                checkpoint.save(file_prefix=checkpoint_prefix)
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
        data = get_preprocessed_images(args['dataset'], BATCH_SIZE, DIM, DIM, color_range=255)
        train(data, EPOCHS)

    elif args['mode'] == 2:
        generator, discriminator, train = get_model(
            trainable=True,
            load_checkpoint=True,
            checkpoint_dir=args['checkpoint'],
            sample_dir=args['samples'],
        )
        data = get_preprocessed_images(args['dataset'], BATCH_SIZE, DIM, DIM, color_range=255)
        train(data, EPOCHS)

    elif args['mode'] == 3:
        generator, discriminator = get_model(
            trainable=False,
            load_checkpoint=True,
            checkpoint_dir=args['checkpoint'],
            sample_dir=args['samples'],
        )
        generate_images(generator, tf.random.normal([num_examples_to_generate, noise_dim]))
        plt.show()

    elif args['mode'] == 4:
        num_examples_to_generate = 1
        if args['output'] == '':
            args['output'] = args['load_checkpoint'] + ' output'

        generator, discriminator = get_model(
            trainable=False,
            load_checkpoint=args['load_checkpoint'],
            checkpoint_dir=args['checkpoint'],
            sample_dir=args['samples'],
        )

        try:
            os.mkdir(args['output'])
        except FileExistsError:
            pass

        for i in range(64):
            generate_images(generator, tf.random.normal([num_examples_to_generate, noise_dim]))
            plt.savefig(os.path.join(args['output'], f'image_{i}.png'))
            plt.close('all')
