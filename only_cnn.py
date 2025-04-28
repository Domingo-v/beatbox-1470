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
        tf.keras.layers.Conv2D(a64, kernel_size=(3,3), activation='relu'),
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


def build_full_model(spectogram_shape, tabular_shape, num_genres):

    # tabular input from metadata
    # spectograms = utils.load("INSERT FILE PATH FROM PREPROCESSING")
    # tracks = utils.load("data/fma_metadata/tracks.csv")

    spectogram_model = spectogram_calc(spectogram_shape)

    spectogram_input = tf.keras.Input(shape=spectogram_shape)

    spectogram_output = spectogram_model(spectogram_input)

    return spectogram_output



