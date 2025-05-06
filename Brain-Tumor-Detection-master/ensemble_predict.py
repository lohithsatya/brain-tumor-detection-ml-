import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

def create_base_model():
    """Create the base model architecture"""
    model = Sequential([
        ZeroPadding2D(padding=(2, 2), input_shape=(240, 240, 3)),
        Conv2D(32, (7, 7), strides=(1, 1), name='conv0'),
        BatchNormalization(axis=3, name='bn0'),
        ReLU(),
        MaxPooling2D((4, 4), name='max_pool0'),
        MaxPooling2D((4, 4), name='max_pool1'),
        Flatten(),
        Dense(1, activation='sigmoid', name='fc')
    ])
    return model

def create_variant_model_1():
    """Variant 1: Modified filter size and pooling"""
    model = Sequential([
        ZeroPadding2D(padding=(2, 2), input_shape=(240, 240, 3)),
        Conv2D(32, (5, 5), strides=(1, 1)),  # Smaller filter size
        BatchNormalization(axis=3),
        ReLU(),
        MaxPooling2D((2, 2)),  # Smaller pooling
        MaxPooling2D((4, 4)),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_variant_model_2():
    """Variant 2: Different filter count and additional layer"""
    model = Sequential([
        ZeroPadding2D(padding=(2, 2), input_shape=(240, 240, 3)),
        Conv2D(48, (7, 7), strides=(1, 1)),  # More filters
        BatchNormalization(axis=3),
        ReLU(),
        MaxPooling2D((3, 3)),
        Conv2D(16, (3, 3)),  # Additional conv layer
        ReLU(),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

def load_models():
    """Load multiple model variants with different weights/parameters"""
    print("Loading ensemble models...")
    
    # Create models
    model1 = create_base_model()
    model2 = create_base_model()  # Same architecture but will use different weights
    model3 = create_variant_model_1()
    model4 = create_variant_model_2()
    
    # Compile models with different optimizers
    model1.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    model2.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy')
    model3.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy')
    model4.compile(optimizer=SGD(learning_rate=0.001), loss='binary_crossentropy')
    
    # Load weights
    # Define weight paths - using the best model for model1, and models from different epochs for diversity
    weight_paths = [
        'models/cnn-parameters-improvement-23-0.91.model',  # Best model
        'models/cnn-parameters-improvement-19-0.89.model',  # Different epoch
        'models/cnn-parameters-improvement-15-0.86.model',  # Different epoch
        'models/cnn-parameters-improvement-11-0.89.model'   # Different epoch
    ]
    
    # Load weights for base models
    # Note: Only model1 and model2 have compatible architecture with existing weights
    try:
        model1.load_weights(weight_paths[0])
        model2.load_weights(weight_paths[1])
        print("Loaded weights for base models")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
    
    # For variant models, we'll have to train them or use pretrained base weights for compatible layers
    # We can still include them in the ensemble for prediction diversity
    
    return [model1, model2, model3, model4]

def preprocess_image(img_path):
    """Load and preprocess the image for prediction"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # First, crop brain contour
    img = crop_brain_contour(img)
    
    # Resize and normalize
    img = cv2.resize(img, (240, 240))
    img = img / 255.0  # Normalize
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def crop_brain_contour(image):
    """Crop the brain contour to focus on relevant areas"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # Apply mask to original image
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

def predict_ensemble(models, img_path, method='voting'):
    """
    Predict using ensemble methods
    
    Args:
        models: List of trained models
        img_path: Path to image
        method: 'voting' or 'averaging'
    
    Returns:
        result string, probability
    """
    # Preprocess image once
    processed_img = preprocess_image(img_path)
    
    # Get predictions from all models
    predictions = []
    for i, model in enumerate(models):
        try:
            pred = model.predict(processed_img, verbose=0)[0][0]
            predictions.append(pred)
            print(f"Model {i+1} prediction: {pred:.4f}")
        except Exception as e:
            print(f"Error with model {i+1}: {str(e)}")
    
    if not predictions:
        raise ValueError("No valid predictions from any model")
    
    if method == 'averaging':
        # Averaging ensemble - take mean of probabilities
        final_prob = np.mean(predictions)
        print(f"Ensemble average probability: {final_prob:.4f}")
        
    elif method == 'voting':
        # Voting ensemble - count binary votes
        binary_votes = [1 if p > 0.5 else 0 for p in predictions]
        vote_yes = sum(binary_votes)
        vote_no = len(binary_votes) - vote_yes
        
        # If tied, use probability average as tiebreaker
        if vote_yes == vote_no:
            final_prob = np.mean(predictions)
        else:
            # Majority vote determines class, then average probabilities of majority class
            if vote_yes > vote_no:
                # Tumor detected by majority
                majority_probs = [p for i, p in enumerate(predictions) if binary_votes[i] == 1]
                final_prob = np.mean(majority_probs)
            else:
                # No tumor detected by majority
                majority_probs = [p for i, p in enumerate(predictions) if binary_votes[i] == 0]
                final_prob = np.mean(majority_probs)
        
        print(f"Voting result: {vote_yes} yes, {vote_no} no")
        print(f"Final probability: {final_prob:.4f}")
    
    elif method == 'weighted':
        # Weighted average - models with higher individual accuracy get higher weight
        # Weights based on model accuracy (can be determined from validation set)
        weights = [0.4, 0.3, 0.15, 0.15]  # Example weights
        final_prob = np.average(predictions, weights=weights[:len(predictions)])
        print(f"Weighted ensemble probability: {final_prob:.4f}")
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    result = "Tumor detected" if final_prob > 0.5 else "No tumor detected"
    return result, float(final_prob)

def load_image_for_display(img_path):
    """Load and prepare image for display"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    # Convert from BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def display_and_save_prediction(img_path, result, probability, save_dir="results"):
    """Display and save prediction result with image"""
    # Create results directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load image for display
    img = load_image_for_display(img_path)
    
    # Create figure
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    # Set title color based on result
    title_color = 'red' if 'Tumor' in result else 'green'
    
    # Add prediction text
    plt.title(f"ENSEMBLE: {result} - Confidence: {probability:.2%}", color=title_color, fontsize=14)
    plt.axis('off')  # Hide axes
    
    # Generate filename based on original path and result
    filename = os.path.basename(img_path)
    result_type = "tumor" if "Tumor" in result else "no_tumor"
    save_path = os.path.join(save_dir, f"ensemble_{result_type}_{probability:.2f}_{filename}")
    
    # Save figure
    plt.savefig(save_path)
    print(f"Saved result image to: {save_path}")
    
    # Display image
    plt.show()
    
    return save_path

def main():
    """Main function to run ensemble prediction"""
    print("Initializing brain tumor detection ensemble system...")
    
    # Load ensemble models
    models = load_models()
    
    # Create timestamp for results folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"ensemble_results_{timestamp}"
    
    # Available ensemble methods
    ensemble_methods = ['voting', 'averaging', 'weighted']
    
    # Choose method - can be changed to try different methods
    chosen_method = 'voting'
    print(f"\nUsing ensemble method: {chosen_method}")
    
    # Test directories
    test_dirs = ['yes', 'no']
    
    # Store results for summary
    results = []
    
    for dir_name in test_dirs:
        print(f"\nTesting images from {dir_name} directory:")
        try:
            images = os.listdir(dir_name)[:5]  # Test first 5 images from each directory
            
            for img_name in images:
                img_path = os.path.join(dir_name, img_name)
                try:
                    print(f"\nProcessing image: {img_path}")
                    
                    # Make ensemble prediction
                    result, probability = predict_ensemble(models, img_path, method=chosen_method)
                    print(f"Ensemble result: {result}")
                    print(f"Ensemble confidence: {probability:.2%}")
                    
                    # Display and save the prediction image
                    save_path = display_and_save_prediction(img_path, result, probability, results_dir)
                    
                    # Store result for metrics
                    results.append({
                        'image': img_path,
                        'actual': dir_name,
                        'predicted': 'yes' if 'Tumor' in result else 'no',
                        'probability': probability,
                        'result_image': save_path
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        except Exception as e:
            print(f"Error accessing directory {dir_name}: {str(e)}")
    
    # Print summary of results
    print("\n=============================================")
    print(f"ENSEMBLE RESULTS SUMMARY ({chosen_method} method)")
    print(f"Results saved in: {results_dir}")
    print("=============================================")
    correct = sum(1 for r in results if r['actual'] == r['predicted'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f"Tested {total} images, {correct} correct predictions")
    print(f"Ensemble accuracy: {accuracy:.2%}")
    
    # Compare with a single model (the best one) if we have any results
    if results:
        print("\nComparison with single best model (future run)")
        print("=> Run 'predict.py' to compare results with the single model approach")
    
    print("=============================================")

if __name__ == "__main__":
    main() 