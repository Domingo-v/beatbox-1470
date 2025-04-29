import librosa
import numpy as np
import librosa.display
import sys
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cnn
import only_cnn
import pandas as pd
from sklearn.model_selection import train_test_split
import utils

from_path = 'data/fma_processed'
csv_path = 'data/fma/fma_metadata/echonest.csv'

def load_data(test_size=0.2, from_path='data/fma_processed', csv_path='data/fma/fma_metadata/echonest.csv'):
    """
    Returns: (X_spec_train, X_tab_train, y_train), (X_spec_val,   X_tab_val,   y_val)
    Shapes:
    X_spec: (n,96,1293,1) one for the chanel dim
    X_tab:  (n,8)
    y:(n, num_classes) in one hot 
    """
    df_meta = utils.load(csv_path)
    # if it’s multi-indexed, grab the 'track' section
    if isinstance(df_meta.columns, pd.MultiIndex):
        df_meta = df_meta['echonest', 'audio_features']

    print("[DEBUG] DataFrame columns:", df_meta.columns.tolist())
    print(df_meta.head(100))
    print("CHECK1", 2 in df_meta.index)
    print("CHECK2", 134 in df_meta.index)
    print("CHECK3", 694 in df_meta.index)

        
    df_meta = df_meta.iloc[:, :8] #only get the audio tabular data features 
    df_meta.index = df_meta.index.astype(int) #formatting from the weird table

    #folder processing
    genres = sorted([d for d in os.listdir(from_path) if os.path.isdir(os.path.join(from_path, d))])
    genre_to_index_map = {g:i for i,g in enumerate(genres)}
    num_classes = len(genres)

    x_spec, x_tab, ys = [], [], []
    count = 0

    #load the preprocessed .npy and match to their respective tabular row with the id 
    for genre in genres :
        folder = os.path.join(from_path, genre)
        for fn in os.listdir(folder):
            if not fn.endswith('.npy'): #avoid any other files
                continue
            tab_id = int(os.path.splitex_tab(fn)[0]) #get tab id from name
            spec = np.load(os.path.join(folder, fn)).astype(np.float32)
            spec = spec[..., np.newaxis]  # add dim for channel for spectogram data

            if tab_id not in df_meta.index: #T ODO for some reason it sometimes only recognizes 3 of thousands of table ids -> was a problem with table format
                # raise KeyError(f"Track ID {tab_id} not in Echonest CSV")
                if tab_id == 2:
                    print("missing in dataset", tab_id, "   ", count)
                    count += 1
                continue
            
            tab = df_meta.loc[tab_id].values.astype(np.float32)#tabular data valiues from the loaded in table data 

            x_spec.append(spec)
            x_tab.append(tab)
            ys.append(genre_to_index_map[genre])

    X_spec = np.stack(x_spec, axis=0)
    X_tab = np.stack(x_tab, axis=0)
    y_int = np.array(ys, dtype=np.int32)
    y_cat = tf.keras.utils.to_categorical(y_int, num_classes) #https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
    
    # train validate split
    X_spec_train,  X_spec_temp, X_tab_train,   X_tab_temp, y_train, y_temp = train_test_split(
        X_spec, X_tab, y_cat, test_size=test_size, stratify=y_int)

    # divide test into validate
    # stratify on the original class integers extracted from y_temp
    y_temp_int = np.argmax(y_temp, axis=1)
    X_spec_val, X_spec_test, X_tab_val, X_tab_test, y_val, y_test = train_test_split(
        X_spec_temp, X_tab_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp_int)


    print(f"Data loaded and split:")
    print(f" Training: {len(X_spec_train)} samples ({len(X_spec_train)/len(X_spec)*100:.1f}%)")
    print(f" Validation: {len(X_spec_val)} samples ({len(X_spec_val)/len(X_spec)*100:.1f}%)")
    print(f" Test: {len(X_spec_test)} samples ({len(X_spec_test)/len(X_spec)*100:.1f}%)")
    print(f"Spectrogram shape: {X_spec_train[0].shape}")
    


    # return (X_spec_train, X_tab_train, y_train), (X_spec_val, X_tab_val, y_val)
    return ((X_spec_train, X_tab_train, y_train),(X_spec_val,   X_tab_val,   y_val),(X_spec_test,  X_tab_test,  y_test))




# main.py
def main():
    spectogram_shape = (96,1293,1)
    tabular_shape = (8)
    num_classes = 8
    
    # if len(sys.argv) < 3: #DEPRECATED FROM PREPROCESS 
    #     print("Usage: python main.py /path/to/genres_original /path/to/output_folder [--spectogram-only]")
    #     sys.exit(1)
        
    # input_dir = sys.argv[1] 
    # output_dir = sys.argv[2]
    
    # Check if we should use spectogram-only model
    use_spectogram_only = len(sys.argv) > 3 and sys.argv[3] == "--spectogram-only"
    
    # preprocess_dataset(input_dir, output_dir)
    # print("PREPROCESSED ALL FILES")

    # (X_spec_train, X_tab_train, y_train), (X_spec_val, X_tab_val, y_val) = load_data() # insert preprocessed data
    (X_spec_train, X_tab_train, y_train), (X_spec_val, X_tab_val, y_val), (X_spec_test, X_tab_test, y_test) = load_data(from_path=from_path,csv_path=csv_path)    
    # Choose the appropriate model based on the flag
    if use_spectogram_only:
        print("Using spectogram-only model")
        model = only_cnn.build_full_model(spectogram_shape, num_classes)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            X_spec_train,
            y_train,
            validation_data=(X_spec_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
        )
    else:
        print("Using combined spectrogram and tabular model")
        tabular_shape = (X_tab_train.shape[1],)  # Determine shape from actual data
        model = cnn.build_full_model(spectogram_shape, tabular_shape, num_classes)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        #both tab and spec data have the same y vals 
        model.fit(
            [X_spec_train, X_tab_train],
            y_train,
            validation_data=([X_spec_val, X_tab_val], y_val),
            epochs=30,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(restore_best_weights=True)
                # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
        test_loss, test_acc = model.evaluate([X_spec_test,X_tab_test], y_test, verbose=2)
        print(f"\nTest accuracy: {test_acc:.4f}")

    os.makedirs('models', exist_ok=True)
    model.save('models/genre_classifier.keras')
    print("Model saved to models/genre_classifier.keras")



if __name__ == "__main__":
    main()

