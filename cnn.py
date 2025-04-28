import numpy as np
import matplotlib.pyplot as plt
import os
import json
import utils

from sklearn.model_selection import train_test_split

import tensorflow as tf

def spectogram_calc(input_shape):
    # Load in spectogram data

    # spectogram model to handle visual data
    spectogram_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
    ])
    # spectogram_output = spectogram_model(spectogram_data)
    return spectogram_model

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
        tf.keras.Dense(64, activation='relu'),
        tf.keras.Dropout(0.3),
        tf.keras.Dense(num_genres, activation='softmax')
    ])
    return final_model


def build_full_model(spectogram_shape, tabular_shape, num_genres):

    # tabular input from metadata
    # spectograms = utils.load("INSERT FILE PATH FROM PREPROCESSING")
    # tracks = utils.load("data/fma_metadata/tracks.csv")

    # run both inputs through model
    spectogram_model = spectogram_calc(spectogram_shape)
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



