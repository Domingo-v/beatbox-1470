import os, sys
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from random import sample

def validate_dataset(path, expected_shape=None, n_examples=3):
    all_means, all_stds = [], []
    bad_files = []
    file_paths = []

    # loop over processed directory
    for root, _, files in os.walk(path):
        for fn in files:
            if not fn.lower().endswith('.npy'):
                continue
            fp = os.path.join(root, fn)
            file_paths.append(fp)

            arr = np.load(fp)

            # validate shape
            if expected_shape and arr.shape != expected_shape:
                bad_files.append((fp, f"shape {arr.shape} != {expected_shape}"))
            # account for NAN and INF specifically
            if not np.isfinite(arr).all():
                bad_files.append((fp, "contains non-finite values"))
            # account for fully empty data
            if np.allclose(arr, 0):
                bad_files.append((fp, "all zeros"))

            all_means.append(arr.mean())
            all_stds.append(arr.std())

    # print a summary
    total = len(file_paths)
    print(f"→ Files checked: {total}")
    print(f"→ Problematic files: {len(bad_files)}")
    for fp, reason in bad_files[:5]:
        print(f"   - {reason}: {fp}")
    if bad_files:
        print("   …")

    print(f"\nMean of means: {np.mean(all_means):.3f}  Std of means: {np.std(all_means):.3f}")
    print(f"Mean of stds:  {np.mean(all_stds):.3f}  Std of stds:  {np.std(all_stds):.3f}")

    # show visualization of data
    if total > 0:
        print(f"\nShowing {n_examples} random samples:")
        for fp in sample(file_paths, min(n_examples, total)):
            arr = np.load(fp)
            plt.figure(figsize=(4, 3))
            librosa.display.specshow(arr, sr=22050, hop_length=512,
                                     x_axis='time', y_axis='mel')
            plt.title(os.path.basename(fp))
            plt.colorbar()
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py /path/to/mel_specs [expected_height expected_width]")
        sys.exit(1)

    spec_dir = sys.argv[1]
    if len(sys.argv) == 3:
        # If you know your CNN wants, say, 216×216 inputs
        h, w = map(int, sys.argv[2].split('x'))
        validate_dataset(spec_dir, expected_shape=(h, w))
    else:
        validate_dataset(spec_dir)
