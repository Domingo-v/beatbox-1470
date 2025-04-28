import numpy as np
import matplotlib.pyplot as plt
import os
import json
import utils

from sklearn.model_selection import train_test_split

import tensorflow as tf

def spectogram_calc(input_shape, num_genres):
    # Load in spectogram data

    # # spectogram model to handle visual data
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
    spectogram_model = spectogram_calc_improved(spectogram_shape, num_genres)
    
    # Apply the model to the input
    spectogram_output = spectogram_model(spectogram_input)
    
    # Create and return a proper model
    model = tf.keras.Model(inputs=spectogram_input, outputs=spectogram_output)
    
    return model

def spectogram_calc_improved(input_shape, num_genres, lambda_reg=0.01):
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # First conv block with residual connection
    x=tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    # Second conv block
    x=tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64 // 16, activation='leaky_relu')(x)
    x = tf.keras.layers.Dense(64, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((1, 1, 64))(x)
    x = tf.keras.layers.Multiply()([x, x])
    
    # Third conv block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    # Fourthy conv block
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='leaky_relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))
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
    x = tf.keras.layers.Dense(256, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_reg))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Dense(128, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_genres, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
