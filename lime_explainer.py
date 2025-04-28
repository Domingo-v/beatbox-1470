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
        # Handle RGB images (LIME converts to RGB internally)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:,:,0]  # Just use first channel
            
        segments = np.zeros(image.shape, dtype=np.int32)
        
        # Create a larger grid-based segmentation
        height, width = segments.shape
        h_step = max(1, height // 6)
        w_step = max(1, width // 10)
        
        segment_id = 1
        for i in range(0, height, h_step):
            for j in range(0, width, w_step):
                segments[i:min(i+h_step, height), j:min(j+w_step, width)] = segment_id
                segment_id += 1
                
        return segments
    
    # Define a smarter prediction function that handles various input shapes
    def predict_fn(images):
        # Convert inputs to the right format
        batch = []
        
        for img in images:
            # LIME might send RGB images, even though our model expects grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Take just the first channel if it's RGB
                img = img[:,:,0]
                
            # Check if dimensions match our expected input
            if img.size != sample.shape[0] * sample.shape[1]:
                print(f"Warning: Unexpected image size {img.shape}, expected {sample.shape[0:2]}")
                # Try to resize to match expected dimensions
                from skimage.transform import resize
                img = resize(img, (sample.shape[0], sample.shape[1]), anti_aliasing=True)
                
            # Add channel dimension for model
            img = img.reshape(sample.shape[0], sample.shape[1], 1)
            batch.append(img)
            
        return model.predict(np.array(batch), verbose=0)
    
    # Get the 2D spectrogram (no channel dimension)
    spectrogram_2d = sample[:, :, 0]
    
    # Create the LIME explainer with debugging
    explainer = lime_image.LimeImageExplainer(verbose=True)
    
    # Try a simpler approach with fewer segments and samples
    print("Generating LIME explanation...")
    try:
        explanation = explainer.explain_instance(
            spectrogram_2d, 
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=200,
            batch_size=10,  # Smaller batch size
            segmentation_fn=spectrogram_segmentation,
            random_seed=42
        )
        
        # Get the original spectrogram for display
        spectrogram = sample[:, :, 0]
        
        # Generate explanation for the predicted class
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=True, 
            num_features=5,
            hide_rest=False
        )
        
        # Plot visualization
        plt.figure(figsize=(15, 5))
        
        # Plot original spectrogram
        plt.subplot(1, 2, 1)
        plt.imshow(spectrogram, cmap='viridis')
        plt.title(f'Original: {class_names[true_class]}\nPredicted: {class_names[pred_class]}')
        plt.colorbar(format='%+2.0f dB')
        
        # Plot heatmap
        plt.subplot(1, 2, 2)
        
        # Create heatmap of important regions
        heatmap = np.zeros_like(spectrogram)
        dict_heatmap = dict(explanation.local_exp[pred_class])
        for k, v in dict_heatmap.items():
            if v > 0:  # Only positive contributions
                heatmap[explanation.segments == k] = v
        
        # Normalize the heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Display heatmap on spectrogram
        plt.imshow(spectrogram, cmap='gray', alpha=0.6)
        plt.imshow(heatmap, cmap='hot', alpha=0.4)
        plt.colorbar(label='Feature Importance')
        plt.title(f'Important regions for {class_names[pred_class]}')
        
        plt.tight_layout()
        output_filename = f'lime_simple_{class_names[true_class]}_as_{class_names[pred_class]}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved explanation to {output_filename}")
        plt.show()
        
        return explanation
        
    except Exception as e:
        print(f"LIME explanation failed: {e}")
        print("Trying fallback approach with superpixel segmentation...")
        
        # Fallback: Use a very simple grid approach
        def simple_grid(img):
            segments = np.zeros(img.shape[:2], dtype=np.int32)
            h, w = segments.shape
            
            # Create a 5x5 grid
            for i in range(5):
                for j in range(5):
                    segments[i*h//5:(i+1)*h//5, j*w//5:(j+1)*w//5] = i*5+j+1
                    
            return segments
        
        try:
            explanation = explainer.explain_instance(
                spectrogram_2d, 
                predict_fn,
                top_labels=1,
                hide_color=0, 
                num_samples=100,
                segmentation_fn=simple_grid
            )
            
            # Simple visualization of just the heatmap
            plt.figure(figsize=(10, 6))
            
            # Get important features
            ind = explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            
            # Create heatmap
            heatmap = np.zeros(explanation.segments.shape)
            for k, v in dict_heatmap.items():
                if v > 0:
                    heatmap[explanation.segments == k] = v
            
            # Show importance heatmap on grayscale spectrogram 
            plt.imshow(spectrogram_2d, cmap='gray')
            plt.imshow(heatmap, cmap='hot', alpha=0.4)
            plt.colorbar(label='Feature Importance')
            plt.title(f'Simple grid explanation for {class_names[pred_class]}\nTrue: {class_names[true_class]}')
            
            # Save and show
            plt.tight_layout()
            plt.savefig(f'lime_fallback_{class_names[true_class]}_as_{class_names[pred_class]}.png', dpi=300)
            plt.show()
            
        except Exception as e2:
            print(f"Fallback explanation also failed: {e2}")
            # Last resort: just show a message
            print("Could not generate LIME explanation. Consider using other explanation methods.")

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
    
    # Try specific samples or generate random ones
    successes = 0
    attempts = 0
    
    while successes < num_explanations and attempts < 10:
        sample_idx = random.randint(0, len(X_spec_test) - 1)
        attempts += 1
        
        print(f"\nAttempt {attempts}: Explaining sample {sample_idx}")
        try:
            explain_prediction_with_lime(model, X_spec_test, y_test, genres, sample_idx=sample_idx)
            successes += 1
        except Exception as e:
            print(f"Error with sample {sample_idx}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 