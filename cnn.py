import numpy as np
import matplotlib.pyplot as plt
import os
import json
import utils

from sklearn.model_selection import train_test_split

import tensorflow as tf

def spectogram_calc(file_path):
    # Load in spectogram data
    spectogram_data = utils.load(file_path)

    # spectogram model to handle visual data
    spectogram_model = tf.keras.Sequential([
        tf.keras.Conv2D(32, kernel_size=(3,3), activation='relu'),
        tf.keras.MaxPooling2D(pool_size=(2,2)),
        tf.keras.BatchNormalization(),
        tf.keras.Conv2D(64, kernel_size=(3,3), activation='relu'),
        tf.keras.MaxPooling2D(pool_size=(2,2)),
        tf.keras.BatchNormalization(),
        tf.keras.Conv2D(128, kernel_size=(3,3), activation='relu'),
        tf.keras.MaxPooling2D(pool_size=(2,2)),
        tf.keras.BatchNormalization(),
        tf.keras.Flatten(),
        tf.keras.Dense(128, activation='relu'),
        tf.keras.Dropout(0.3),
    ])
    spectogram_output = spectogram_model(spectogram_data)
    return spectogram_output

def tabular_calc(file_path):
    # tabular input from metadata
    tracks = utils.load(file_path)

    # model to handle tabular data
    tabular_model = tf.keras.Sequential([
        tf.keras.Dense(64, activation='relu'),
        tf.BatchNormalization(),
        tf.Dropout(0.3),
        tf.Dense(32, activation='relu')
    ])
    tabular_output = tabular_model(tracks)
    return tabular_output

def final_calc(concatenated_output, num_genres):
    # final few layers to run concatenated outputs through, end w softmax
    final_model = tf.keras.Sequential([
        tf.keras.Dense(64, activation='relu'),
        tf.keras.Dropout(0.3),
        tf.keras.Dense(num_genres, activation='softmax')
    ])
    return final_model(concatenated_output)


def main():
    # run both inputs through model
    spectogram_output = spectogram_calc("INSERT PATH FROM PREPROCESSING")
    tabular_output = tabular_calc('data/fma_metadata/tracks.csv')

    num_genres = 8
    spectogram_shape = (96, 1293, 1)

    # concatenate both outputs
    combined = tf.Concatenate()
    output = combined([spectogram_output, tabular_output])

    # find final output
    final_output = final_calc(output, num_genres)
    return final_output



