import librosa
import numpy as np
import librosa.display

import sys
import matplotlib.pyplot as plt
import librosa.display
import os
import tensorflow as tf
import cnn


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



# main.py
def main():
    spectogram_shape = (96,1293,1)
    tabular_shape = ???
    num_classes = 8
    if len(sys.argv) != 3:
        print("Usage: python main.py /path/to/genres_original /path/to/output_folder")
        sys.exit(1)
    input_dir  = sys.argv[1]
    output_dir = sys.argv[2]
    preprocess_dataset(input_dir, output_dir)
    print("PREPROCESSED ALL FILES")

    (X_spec_train, X_tab_train, y_train), (X_spec_val, X_tab_val, y_val) = load_data() # insert preprocessed data
    model = cnn.build_full_model(spectogram_shape, tabular_shape, num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        [X_spec_train, X_tab_train],
        y_train,
        val_data=([X_spec_val, X_tab_val], y_val),
        epochs=30,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
    )

    model.save('models/genre_classifier.h5')
    print("Model saved to models/genre_classifier.h5")



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

