import numpy as np
import matplotlib.pyplot as plt
import os
import json
import utils

from sklearn.model_selection import train_test_split

import tensorflow as tf

def spectogram_calc(input_shape, num_genres):
    # Load in spectogram data

    # spectogram model to handle visual data
    spectogram_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # First conv block
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='leaky_relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.SpatialDropout2D(0.1),
        
        # Second conv block
        tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='leaky_relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.SpatialDropout2D(0.2),

        #Third conv block
         tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='leaky_relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.SpatialDropout2D(0.3),


        #Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_genres, activation='softmax')
    ])
    # spectogram_output = spectogram_model(spectogram_data)
    return spectogram_model


def build_full_model(spectogram_shape, num_genres):
    """
    Build a model that takes spectogram input and outputs genre predictions
    
    Args:
        spectogram_shape: Shape of the input spectrograms (height, width, channels)
        num_genres: Number of genre classes to predict
        
    Returns:
        A compiled Keras model
    """
    # Create input layer
    spectogram_input = tf.keras.Input(shape=spectogram_shape)
    
    # Get the spectogram model
    spectogram_model = spectogram_calc(spectogram_shape, num_genres)
    
    # Apply the model to the input
    spectogram_output = spectogram_model(spectogram_input)
    
    # Create and return a proper model
    model = tf.keras.Model(inputs=spectogram_input, outputs=spectogram_output)
    
    return model



