import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import random
import sys

# Disable operation fusion (needed for M1/M2 Macs)
tf.config.optimizer.set_jit(False)  # Disable XLA
os.environ['TF_DISABLE_FUSED_OPS'] = '1'  # Disable fused operations

def load_test_data(processed_data_dir):
    """Load only the test data portion"""
    from sklearn.model_selection import train_test_split
    
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
    
    # Split to get the test data (using same random seed for consistency)
    X_spec_train, X_spec_temp, y_train, y_temp = train_test_split(
        X_spec, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    X_spec_val, X_spec_test, y_val, y_test = train_test_split(
        X_spec_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return X_spec_test, y_test, genres

def explain_prediction_with_lime(model, X_test, y_test, class_names, sample_idx=0):
    """
    Explains a model prediction using LIME with a custom segmentation for spectrograms
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
        # Handle RGB images from LIME
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:,:,0]
        
        segments = np.zeros(image.shape, dtype=np.int32)
        
        # Create a grid-based segmentation 
        height, width = segments.shape
        h_step = max(1, height // 6)  # Use 6 segments vertically (frequency bands)
        w_step = max(1, width // 10)  # Use 10 segments horizontally (time segments)
        
        segment_id = 1
        for i in range(0, height, h_step):
            for j in range(0, width, w_step):
                segments[i:min(i+h_step, height), j:min(j+w_step, width)] = segment_id
                segment_id += 1
        
        return segments
    
    # Create a direct prediction function without reshape tricks
    def predict_fn(images):
        print(f"LIME sent {len(images)} images, first shape: {images[0].shape}")
        processed_images = []
        
        for img in images:
            # Check if LIME converted to RGB (3 channels)
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Take only the first channel (they're all the same in grayscale->RGB conversion)
                img = img[:,:,0]
            
            # Reshape to model's expected input shape
            if img.size == 96*1293:  # If it's the right size but wrong shape
                img_3d = img.reshape(96, 1293, 1)
            else:
                print(f"Warning: unexpected image shape {img.shape}, size {img.size}")
                # This is a fallback that shouldn't be needed if the above logic works
                img_3d = np.zeros((96, 1293, 1))
            
            processed_images.append(img_3d)
        
        # Convert to batch
        batch = np.array(processed_images)
        return model.predict(batch, verbose=0)
    
    # Get the 2D spectrogram (no channel dimension)
    spectrogram_2d = sample[:, :, 0]
    
    # Debug information
    print(f"Sample shape: {sample.shape}")
    print(f"Spectrogram 2D shape: {spectrogram_2d.shape}")
    print(f"Expected reshape size: {spectrogram_2d.size}")
    
    # Create the LIME explainer
    explainer = lime_image.LimeImageExplainer(verbose=False)
    
    # Generate the explanation
    print("Generating LIME explanation... (this might take a minute)")
    explanation = explainer.explain_instance(
        spectrogram_2d, 
        predict_fn,
        top_labels=3,
        hide_color=0,
        num_samples=300,  # Reduced for faster processing
        segmentation_fn=spectrogram_segmentation,
        random_seed=42
    )
    
    # Get the original spectrogram for display
    spectrogram = sample[:, :, 0]
    
    # Create visualization with only 2 plots (original and important regions)
    plt.figure(figsize=(12, 6))
    
    # Plot the original spectrogram
    plt.subplot(1, 2, 1)
    plt.imshow(spectrogram, cmap='viridis')
    plt.title(f'Original Spectrogram\nTrue: {class_names[true_class]}')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    
    # Plot a heatmap of the important regions
    plt.subplot(1, 2, 2)
    
    # Create a heatmap of the important regions
    heatmap = np.zeros_like(spectrogram)
    
    # Extract the segments with positive influence
    dict_heatmap = dict(explanation.local_exp[pred_class])
    for k, v in dict_heatmap.items():
        if v > 0:  # Only positive contributions
            heatmap[explanation.segments == k] = v
    
    # Normalize the heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Display the heatmap on top of the spectrogram
    plt.imshow(spectrogram, cmap='gray', alpha=0.6)
    plt.imshow(heatmap, cmap='hot', alpha=0.4)
    plt.colorbar(label='Feature Importance')
    plt.title(f'Important Regions for\n{class_names[pred_class]} (Conf: {pred[0][pred_class]:.2f})')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    
    plt.tight_layout()
    
    # Save a higher quality figure
    output_filename = f'lime_explanation_{class_names[true_class]}_as_{class_names[pred_class]}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved explanation to {output_filename}")
    plt.show()
    
    return explanation

def main():
    if len(sys.argv) < 3:
        print("Usage: python lime_explainer.py /path/to/processed_data /path/to/model.keras [num_explanations]")
        sys.exit(1)
    
    processed_data_dir = sys.argv[1]
    model_path = sys.argv[2]
    num_explanations = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    print(f"Loading test data from {processed_data_dir}...")
    X_spec_test, y_test, genres = load_test_data(processed_data_dir)
    
    # Generate LIME explanations
    print(f"\nGenerating {num_explanations} LIME explanations...")
    
    # Instead of random samples, try with a few fixed indices first to debug
    sample_indices = [0, 10, 20]  # Start with known indices
    
    for i in range(min(num_explanations, len(sample_indices))):
        sample_idx = sample_indices[i]
        try:
            explain_prediction_with_lime(model, X_spec_test, y_test, genres, sample_idx=sample_idx)
        except Exception as e:
            print(f"Error explaining sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 