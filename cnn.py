import numpy as np
import matplotlib.pyplot as plt
import os
import json
import utils

from sklearn.model_selection import train_test_split

import tensorflow as tf

def spectogram_calc(input_shape, num_genres, lambda_reg=0.02):
    # Load in spectogram data
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # First conv block with residual connection
    x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    # Second conv block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    
    # Squeeze-excitation block
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(64 // 16, activation='leaky_relu')(se)
    se = tf.keras.layers.Dense(64, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, 64))(se)
    x = tf.keras.layers.Multiply()([x, se])
    
    # Third conv block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    # Fourth conv block
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    # Fifth conv block 
    x = tf.keras.layers.Conv2D(256, kernel_size=(3,5), padding='same')(x)  # Wider in time dimension
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.SpatialDropout2D(0.3)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(256, activation='leaky_relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Dense(128, activation='leaky_relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_genres, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def tabular_calc(input_shape):

    # model to handle tabular data
    tabular_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu')
    ])
    # tabular_output = tabular_model(tracks)
    return tabular_model

def final_calc(num_genres):
    # final few layers to run concatenated outputs through, end w softmax
    final_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_genres, activation='softmax')
    ])
    return final_model


def build_full_model(spectogram_shape, tabular_shape, num_genres):

    # tabular input from metadata
    # spectograms = utils.load("INSERT FILE PATH FROM PREPROCESSING")
    # tracks = utils.load("data/fma_metadata/tracks.csv")

    # run both inputs through model
    spectogram_model = spectogram_calc(spectogram_shape, num_genres)
    # spectogram_output = spectogram_output(spectograms)
    tabular_model = tabular_calc(tabular_shape)

    spectogram_input = tf.keras.Input(shape=spectogram_shape)
    tabular_input = tf.keras.Input(shape=tabular_shape)

    spectogram_output = spectogram_model(spectogram_input)
    tabular_output = tabular_model(tabular_input)

    # concatenate both outputs
    combined = tf.keras.layers.Concatenate()
    output = combined([spectogram_output, tabular_output])

    # find final output
    final_model = final_calc(num_genres)
    final_output = final_model(output)

    model = tf.keras.Model(inputs=[spectogram_input, tabular_input], outputs=final_output)
    return model

def build_spectogram_model(spectogram_shape, num_genres):
    spectogram_model = spectogram_calc(spectogram_shape)
    spectogram_input = tf.keras.Input(shape=spectogram_shape)
    spectogram_output = spectogram_model(spectogram_input)
    final_model = final_calc(num_genres)
    final_output = final_model(spectogram_output)
    model = tf.keras.Model(inputs=[spectogram_input], outputs=final_output)
    return model



