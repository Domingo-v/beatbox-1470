import numpy as np
import matplotlib.pyplot as plt
import os
import json
import utils

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dense, Dropout

# Load in audio data
audio_data = "INSERT PATH FROM PREPROCESSING"

tracks = utils.load('data/fma_metadata/tracks.csv')
genres = utils.load('data/fma_metadata/genres.csv')
features = utils.load('data/fma_metadata/features.csv')