import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import datetime


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=(2, 2), input_shape=(240, 240, 3)),
        tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), name='conv0'),
        tf.keras.layers.BatchNormalization(axis=3, name='bn0'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((4, 4), name='max_pool0'),
        tf.keras.layers.MaxPooling2D((4, 4), name='max_pool1'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid', name='fc')
    ])
    return model

def load_image_for_display(img_path):
    # Load the image for display (without preprocessing)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    # Convert from BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_image(img_path):
    # Load and preprocess the image for prediction
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.resize(img, (240, 240))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_tumor(model, img_path):
    # Preprocess the image
    processed_img = preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(processed_img, verbose=0)
    probability = prediction[0][0]
    
    result = "Tumor detected" if probability > 0.5 else "No tumor detected"
    return result, float(probability)

def display_and_save_prediction(img_path, result, probability, save_dir="results"):
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
    plt.title(f"{result} - Confidence: {probability:.2%}", color=title_color, fontsize=14)
    plt.axis('off')  # Hide axes
    
    # Generate filename based on original path and result
    filename = os.path.basename(img_path)
    result_type = "tumor" if "Tumor" in result else "no_tumor"
    save_path = os.path.join(save_dir, f"{result_type}_{probability:.2f}_{filename}")
    
    # Save figure
    plt.savefig(save_path)
    print(f"Saved result image to: {save_path}")
    
    # Display image
    plt.show()
    
    return save_path

def main():
    # Create and load the model
    print("Creating model architecture...")
    model = create_model()
    
    # Load the best weights
    model_path = 'models/cnn-parameters-improvement-23-0.91.model'
    print("Loading pre-trained weights...")
    try:
        model.load_weights(model_path)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return

    # Create timestamp for results folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"prediction_results_{timestamp}"
    
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
                    result, probability = predict_tumor(model, img_path)
                    print(f"\nImage: {img_path}")
                    print(f"Result: {result}")
                    print(f"Confidence: {probability:.2%}")
                    
                    # Display and save the prediction image
                    save_path = display_and_save_prediction(img_path, result, probability, results_dir)
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
    
    # Print summary
    print("\n=============================================")
    print(f"Results summary (saved in {results_dir}):")
    print("=============================================")
    correct = sum(1 for r in results if r['actual'] == r['predicted'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f"Tested {total} images, {correct} correct predictions")
    print(f"Accuracy: {accuracy:.2%}")
    print("=============================================")

if __name__ == "__main__":
    main()
