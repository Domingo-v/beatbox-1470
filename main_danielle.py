import librosa
import numpy as np
import librosa.display

import sys
import matplotlib.pyplot as plt
import librosa.display
import os
import tensorflow as tf
import cnn
import only_cnn

from sklearn.model_selection import train_test_split
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

# # 1a) Load and resample
# y, sr = librosa.load(path, sr=22050)      # downsample to 22 050 Hz

# # 1b) (Optional) Trim leading/trailing silence
# y, _ = librosa.effects.trim(y, top_db=20)

# #boosting high frequencies
# pre_emphasis = 0.97
# y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])


# #divide into parts
# clip_length = 3 * sr # 3 sec = 66150 samples
# stride      = clip_length # non‑overlapping; use smaller stride for overlap
# clips = [
#     y[i:i+clip_length]
#     for i in range(0, len(y)-clip_length+1, stride)
# ]


# import librosa.display

# # STFT parameters (Bahuleyan et al. use these)
# n_fft      = 2048
# hop_length = 512
# n_mels     = 96
# fmax       = sr // 2

# S = librosa.feature.melspectrogram(
#         y=clip, sr=sr,
#         n_fft=n_fft, hop_length=hop_length,
#         n_mels=n_mels, fmax=fmax
#     )

# # Power → log-scale (dB)
# S_dB = librosa.power_to_db(S, ref=np.max)


TARGET_FRAMES = 1293

#not all data is the exact same size (around 55 are not 96x1293) so it will either pad or trim those
def pad_or_trim(S: np.ndarray, target: int = TARGET_FRAMES) -> np.ndarray:
    # S = shape (n_mels, n_frames)
    n_mels, n_frames = S.shape
    if n_frames < target:
        # if width smaller pad with zeros on the right
        pad_width = target - n_frames
        return np.pad(S, ((0,0),(0,pad_width)), mode='constant')
    else:
        # else truncate extra rightside frames 
        return S[:, :target]


def preprocess_clip(path, sr=22050, pre_emph=0.97, n_fft=2048, hop_length=512, n_mels=96):
    """
    Grabs a clip and turns it from audio waveform to melspectogram which 
    can be treated similar to an image in a CNN
    """
    y, _ = librosa.load(path, sr=sr) #load in data
    # y, _ = librosa.effects.trim(y) #trim trailing silence NOTE not anymore because was causing shape issues BUT CAN BE ADDED IN LATERR
    y = np.append(y[0], y[1:] - pre_emph * y[:-1]) #a little frequency boosting for clearer patterns #source says Eq. 1 in Bahuleyan
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=sr // 2,
        power=2.0
    )
    S_dB = librosa.power_to_db(S, ref=np.max) # spectogram to decibel info units
    # standardize
    S_std = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6) #dataset minmaxed
    # potentially have to resize to  to 216×216 if

    S_fixed = pad_or_trim(S_std, TARGET_FRAMES)

    # return S_std
    return S_fixed



def preprocess_dataset(input_dir, output_dir):
    """
    Goes through a dataset preprocessing each .wav file
    outputs a .npy file with genre folders -> chose .npy for easy manipulation of data later
    through this function can be tweaked to run directly in a data processing -> model pipeline
    """
    for root, _, files in os.walk(input_dir):
        # filter files
        wavs = [f for f in files if f.lower().endswith('.wav')]
        if not wavs:
            continue

        # output folder
        rel_path = os.path.relpath(root, input_dir)
        out_folder = os.path.join(output_dir, rel_path)
        os.makedirs(out_folder, exist_ok=True)

        for fn in wavs:
            in_path  = os.path.join(root, fn)
            out_name = os.path.splitext(fn)[0] + '.npy'
            out_path = os.path.join(out_folder, out_name)

            try:
                S_std = preprocess_clip(in_path)
                np.save(out_path, S_std)
                print(f"[OK] {in_path} → {out_path}  ({S_std.shape})")
            except Exception as e:
                print(f"[ERR] {in_path}: {e}")

def load_data(processed_data_dir):
    """
    Load preprocessed spectrograms from genre folders and split into train/test/validation sets
    with an 80/10/10 split
    
    Args:
        processed_data_dir: Directory containing processed data with genre subfolders
        
    Returns:
        ((X_spec_train, X_tab_train, y_train), (X_spec_val, X_tab_val, y_val), (X_spec_test, X_tab_test, y_test))
        For spectogram-only mode, tabular data will be None
    """
    use_spectogram_only = len(sys.argv) > 3 and sys.argv[3] == "--spectogram-only"
    
    # Dictionary to map genre names to indices
    genres = sorted(os.listdir(processed_data_dir))
    genres = [g for g in genres if os.path.isdir(os.path.join(processed_data_dir, g))]
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    
    print(f"Found {len(genres)} genres: {genres}")
    
    # Load all spectrograms and their labels
    spectrograms = []
    labels = []
    
    for genre in genres:
        genre_dir = os.path.join(processed_data_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        
        genre_idx = genre_to_idx[genre]
        print(f"Loading spectrograms for genre '{genre}' (index {genre_idx})...")
        
        for file_name in os.listdir(genre_dir):
            if file_name.endswith('.npy'):
                # Load spectrogram
                spec_path = os.path.join(genre_dir, file_name)
                spectrogram = np.load(spec_path)
                
                # Add channel dimension for CNN
                spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1], 1)
                
                spectrograms.append(spectrogram)
                labels.append(genre_idx)
    
    # Convert to numpy arrays
    X_spec = np.array(spectrograms)
    y = np.array(labels)
    
    # Create one-hot encoded labels
    num_classes = len(genres)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    
    # Implement 80/10/10 split
    # First, split data into 80% train and 20% temp
    X_spec_train, X_spec_temp, y_train, y_temp = train_test_split(
        X_spec, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    # Then split the temp data into test and validation (50% each, which is 10% of original data)
    X_spec_val, X_spec_test, y_val, y_test = train_test_split(
        X_spec_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Data loaded and split:")
    print(f"  Training: {len(X_spec_train)} samples ({len(X_spec_train)/len(X_spec)*100:.1f}%)")
    print(f"  Validation: {len(X_spec_val)} samples ({len(X_spec_val)/len(X_spec)*100:.1f}%)")
    print(f"  Test: {len(X_spec_test)} samples ({len(X_spec_test)/len(X_spec)*100:.1f}%)")
    print(f"Spectrogram shape: {X_spec_train[0].shape}")
    
    if use_spectogram_only:
        # Return None for tabular data in spectogram-only mode
        X_tab_train = None
        X_tab_val = None
        X_tab_test = None
        return (X_spec_train, X_tab_train, y_train), (X_spec_val, X_tab_val, y_val), (X_spec_test, X_tab_test, y_test)
    else:
        raise NotImplementedError("Tabular data not implemented yet")


# main.py
def main():
    spectogram_shape = (96,1293,1)
   # tabular_shape = ???
    num_classes = 8
    
    if len(sys.argv) < 3:
        print("Usage: python main.py /path/to/genres_original /path/to/output_folder [--spectogram-only]")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Check if we should use spectogram-only model
    use_spectogram_only = len(sys.argv) > 3 and sys.argv[3] == "--spectogram-only"
    
    preprocess_dataset(input_dir, output_dir)
    print("PREPROCESSED ALL FILES")

    # Now with three sets: train, validation, test
    (X_spec_train, X_tab_train, y_train), (X_spec_val, X_tab_val, y_val), (X_spec_test, X_tab_test, y_test) = load_data(output_dir)
    
    # Choose the appropriate model based on the flag
    if use_spectogram_only:
        print("Using spectogram-only model")
        model = only_cnn.build_full_model(spectogram_shape, num_classes)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_spec_train,
            y_train,
            validation_data=(X_spec_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
        )
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_spec_test, y_test, verbose=2)
        print(f"\nTest accuracy: {test_acc:.4f}")
    else:
        print("Using combined spectrogram and tabular model")
        tabular_shape = (X_tab_train.shape[1],)  # Determine shape from actual data
        model = cnn.build_full_model(spectogram_shape, tabular_shape, num_classes)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            [X_spec_train, X_tab_train],
            y_train,
            validation_data=([X_spec_val, X_tab_val], y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
        )
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate([X_spec_test, X_tab_test], y_test, verbose=2)
        print(f"\nTest accuracy: {test_acc:.4f}")

    # Save the model
    os.makedirs('models', exist_ok=True)
    # Option 1: Save in the newer, more efficient Keras format
    model.save('models/genre_classifier.keras')
    print("Model saved to models/genre_classifier.keras")

    # Adding LIME Explainer for Model Interpretation
    def explain_prediction_with_lime(model, X_test, y_test, class_names, sample_idx=0):
        """
        Explains a model prediction using LIME
        
        Args:
            model: Trained Keras model
            X_test: Test data
            y_test: Test labels (one-hot encoded)
            class_names: List of class names
            sample_idx: Index of the sample to explain
        """
  
        
        # Get a sample to explain
        sample = X_test[sample_idx]
        true_class = np.argmax(y_test[sample_idx])
        
        # Get the prediction for this sample
        pred = model.predict(sample[np.newaxis, ...])
        pred_class = np.argmax(pred[0])
        
        print(f"Explaining prediction for sample {sample_idx}")
        print(f"True class: {class_names[true_class]}")
        print(f"Predicted class: {class_names[pred_class]} (confidence: {pred[0][pred_class]:.4f})")
        
        # Create a custom segmentation function for spectrograms
        def spectrogram_segmentation(image):
            """
            Creates segments for a spectrogram image using a grid-based approach
            Returns a 2D array of segment labels
            """
            segments = np.zeros(image.shape[:2], dtype=np.int32)
            
            # Create a grid-based segmentation (10x10 grid = 100 segments)
            height, width = segments.shape
            h_step = height // 10
            w_step = width // 20
            
            segment_id = 1
            for i in range(0, height, h_step):
                for j in range(0, width, w_step):
                    segments[i:min(i+h_step, height), j:min(j+w_step, width)] = segment_id
                    segment_id += 1
            
            return segments
        
        # Create the LIME explainer
        explainer = lime_image.LimeImageExplainer(verbose=False)
        
        # Function to get model predictions for perturbed images
        def model_predict_fn(images):
            return model.predict(images)
        
        # Ensure the sample has the right shape and type for LIME
        lime_sample = sample.copy()
        
        # Generate the explanation with the custom segmentation function
        explanation = explainer.explain_instance(
            lime_sample.astype('double'), 
            model_predict_fn,
            top_labels=5, 
            hide_color=0, 
            num_samples=1000,
            segmentation_fn=spectrogram_segmentation
        )
        
        # Get the original spectrogram
        spectrogram = sample[:, :, 0]  # Remove channel dimension
        
        # Generate explanation for the top predicted class
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=True, 
            num_features=10, 
            hide_rest=False
        )
        
        # Display the explanation
        plt.figure(figsize=(12, 6))
        
        # Plot the original spectrogram
        plt.subplot(1, 2, 1)
        plt.imshow(spectrogram, cmap='viridis')
        plt.title(f'Original Spectrogram\nTrue: {class_names[true_class]}')
        plt.colorbar(format='%+2.0f dB')
        
        # Plot the explanation
        plt.subplot(1, 2, 2)
        plt.imshow(mark_boundaries(temp[:, :, 0], mask), cmap='viridis')
        plt.title(f'Explanation for {class_names[pred_class]}\nConfidence: {pred[0][pred_class]:.2f}')
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'lime_explanation_sample{sample_idx}_{class_names[true_class]}_as_{class_names[pred_class]}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        return explanation

    # Get class names from the genre folders
    genres = sorted(os.listdir(output_dir))
    genres = [g for g in genres if os.path.isdir(os.path.join(output_dir, g))]
    
    # Explain a few predictions using LIME
    print("\nGenerating LIME explanations...")
    
    # Explain 3 random samples from the test set
    import random
    for i in range(3):
        random_idx = random.randint(0, len(X_spec_test) - 1)
        explain_prediction_with_lime(model, X_spec_test, y_test, genres, sample_idx=random_idx)




if __name__ == "__main__":
    main()


#DEPRACATED MAIN DISPLAYING
# def main(audio_path w):
#     # preprocess
#     S_std = preprocess_clip(audio_path)

#     # shape
#     print(f"Processed mel‑spec shape: {S_std.shape}")
#     #(96, 128) for a 3 secs and should be (96, ~1296) for 30 

#     # plot as figure
#     plt.figure(figsize=(6, 4))
#     librosa.display.specshow(
#         S_std,
#         sr=22050,                
#         hop_length=512,
#         x_axis='time',
#         y_axis='mel',
#     )
#     plt.colorbar(label='Standardized dB')
#     plt.title("Pre‑emphasized, log‑mel spectrogram")
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python main.py path/to/audio.wav")
#         sys.exit(1)
#     audio_file = sys.argv[1]
#     main(audio_file)

