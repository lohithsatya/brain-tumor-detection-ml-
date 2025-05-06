import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

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

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.resize(img, (240, 240))
    return img / 255.0

def prepare_dataset():
    # Load images from yes (tumor) and no (no tumor) directories
    X = []
    y = []
    
    # Load tumor images
    print("Loading tumor images...")
    for img_name in os.listdir('yes')[:20]:  # Load first 20 images
        try:
            img_path = os.path.join('yes', img_name)
            img = load_and_preprocess_image(img_path)
            X.append(img)
            y.append(1)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
    
    # Load non-tumor images
    print("Loading non-tumor images...")
    for img_name in os.listdir('no')[:20]:  # Load first 20 images
        try:
            img_path = os.path.join('no', img_name)
            img = load_and_preprocess_image(img_path)
            X.append(img)
            y.append(0)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def predict_tumor(model, img_path):
    # Load and preprocess the image
    img = load_and_preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img, verbose=0)
    probability = prediction[0][0]
    
    result = "Tumor detected" if probability > 0.5 else "No tumor detected"
    return result, float(probability)

def main():
    # Prepare the dataset
    print("Preparing dataset...")
    X_train, X_test, y_train, y_test = prepare_dataset()
    
    # Create and compile the model
    print("\nCreating and compiling model...")
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    print("\nTraining model...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model in the new format
    print("\nSaving model...")
    model.save('quick_test_model')
    
    # Test the model on a few images
    print("\nTesting model on sample images...")
    test_dirs = ['yes', 'no']
    
    for dir_name in test_dirs:
        print(f"\nTesting images from {dir_name} directory:")
        try:
            images = os.listdir(dir_name)[:3]
            
            for img_name in images:
                img_path = os.path.join(dir_name, img_name)
                try:
                    result, probability = predict_tumor(model, img_path)
                    print(f"\nImage: {img_path}")
                    print(f"Result: {result}")
                    print(f"Confidence: {probability:.2%}")
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        except Exception as e:
            print(f"Error accessing directory {dir_name}: {str(e)}")

if __name__ == "__main__":
    main()
