import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.models import Model
import os
import histo_preprocessing

def build_generator(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (4, 4), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    outputs = Conv2D(3, (7, 7), padding='same', activation='tanh')(x)
    return Model(inputs, outputs)

def build_discriminator(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    outputs = Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, outputs)

def train_cyclegan(generator_AtoB, generator_BtoA, discriminator_A, discriminator_B,
                   dataset_A, dataset_B, epochs=100, lambda_cycle=10.0, lambda_identity=5.0):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for real_A, real_B in zip(dataset_A, dataset_B):
            with tf.GradientTape(persistent=True) as tape:
                fake_B = generator_AtoB(real_A, training=True)
                fake_A = generator_BtoA(real_B, training=True)
                cycled_A = generator_BtoA(fake_B, training=True)
                cycled_B = generator_AtoB(fake_A, training=True)
                same_A = generator_BtoA(real_A, training=True)
                same_B = generator_AtoB(real_B, training=True)

                disc_real_A = discriminator_A(real_A, training=True)
                disc_fake_A = discriminator_A(fake_A, training=True)
                disc_real_B = discriminator_B(real_B, training=True)
                disc_fake_B = discriminator_B(fake_B, training=True)

                gen_AtoB_loss = mse_loss(tf.ones_like(disc_fake_B), disc_fake_B)
                gen_BtoA_loss = mse_loss(tf.ones_like(disc_fake_A), disc_fake_A)
                total_cycle_loss = lambda_cycle * (mae_loss(real_A, cycled_A) + mae_loss(real_B, cycled_B))
                id_loss_A = lambda_identity * mae_loss(real_A, same_A)
                id_loss_B = lambda_identity * mae_loss(real_B, same_B)
                total_gen_AtoB_loss = gen_AtoB_loss + total_cycle_loss + id_loss_B
                total_gen_BtoA_loss = gen_BtoA_loss + total_cycle_loss + id_loss_A
                disc_A_loss = mse_loss(tf.ones_like(disc_real_A), disc_real_A) + mse_loss(tf.zeros_like(disc_fake_A), disc_fake_A)
                disc_B_loss = mse_loss(tf.ones_like(disc_real_B), disc_real_B) + mse_loss(tf.zeros_like(disc_fake_B), disc_fake_B)

            generator_AtoB_grads = tape.gradient(total_gen_AtoB_loss, generator_AtoB.trainable_variables)
            generator_BtoA_grads = tape.gradient(total_gen_BtoA_loss, generator_BtoA.trainable_variables)
            discriminator_A_grads = tape.gradient(disc_A_loss, discriminator_A.trainable_variables)
            discriminator_B_grads = tape.gradient(disc_B_loss, discriminator_B.trainable_variables)

            generator_optimizer.apply_gradients(zip(generator_AtoB_grads, generator_AtoB.trainable_variables))
            generator_optimizer.apply_gradients(zip(generator_BtoA_grads, generator_BtoA.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_A_grads, discriminator_A.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_B_grads, discriminator_B.trainable_variables))

            print(f"Gen A->B Loss: {total_gen_AtoB_loss:.4f}, Gen B->A Loss: {total_gen_BtoA_loss:.4f}, "
                  f"Disc A Loss: {disc_A_loss:.4f}, Disc B Loss: {disc_B_loss:.4f}")

# Eg
input_directory = 'path_to_histology_images'
output_directory = 'path_to_output_images'
histo_preprocessing.define_paths(input_directory, output_directory)

generator_AtoB = build_generator()
generator_BtoA = build_generator()
discriminator_A = build_discriminator()
discriminator_B = build_discriminator()

dataset_A = tf.keras.preprocessing.image_dataset_from_directory(directory=output_directory, label_mode=None, image_size=(256, 256), batch_size=1)
dataset_B = dataset_A  # Change to real domain B
train_cyclegan(generator_AtoB, generator_BtoA, discriminator_A, discriminator_B, dataset_A, dataset_B, epochs=100)