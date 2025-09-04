# train.py

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import os
import pickle


# Set a consistent image size
IMAGE_SIZE = (200, 200)

# Function to extract features from an image
def extract_features(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        return None

    # Step 1: Resize the image to a consistent size
    image = cv2.resize(image, IMAGE_SIZE)

    # Step 2: Apply a Gaussian blur for noise reduction (Image Filtering)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 3: Canny Edge Detection
    edges = cv2.Canny(blurred_image, 100, 200)

    # Step 4: Flatten the image into a 1D array of pixel values
    features = edges.flatten()

    return features

# Paths to the dataset
BASE_DIR = 'dataset'
SPIRAL_PATH = os.path.join(BASE_DIR, 'spiral', 'training')
WAVES_PATH = os.path.join(BASE_DIR, 'waves', 'training')

# Initialize data and labels lists
data = []
labels = []

# Load spiral images
for class_name in ['healthy', 'parkinson']:
    class_path = os.path.join(SPIRAL_PATH, class_name)
    for filename in os.listdir(class_path):
        if filename.endswith(".png"):
            img_path = os.path.join(class_path, filename)
            features = extract_features(img_path)
            if features is not None:
                data.append(features)
                labels.append(1 if class_name == 'parkinson' else 0)

# Load wave images
for class_name in ['healthy', 'parkinson']:
    class_path = os.path.join(WAVES_PATH, class_name)
    for filename in os.listdir(class_path):
        if filename.endswith(".png"):
            img_path = os.path.join(class_path, filename)
            features = extract_features(img_path)
            if features is not None:
                data.append(features)
                labels.append(1 if class_name == 'parkinson' else 0)

# Convert lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

if data.size == 0:
    print("No images found or features could not be extracted. Please check the dataset paths.")
else:
    # Dimensionality Reduction with PCA
    pca = PCA(n_components=50) # Reduce to 50 principal components
    data_pca = pca.fit_transform(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_pca, labels, test_size=0.2, random_state=42)

    # Train a Support Vector Machine (SVM) classifier
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Save the trained model and PCA object
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    
    # Evaluate the model (accuracy doesn't matter, but it's good practice)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully! Accuracy: {accuracy:.2f}")