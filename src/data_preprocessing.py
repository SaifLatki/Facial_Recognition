import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
from collections import namedtuple

# Dataset structure
LFWDataset = namedtuple('LFWDataset', ['images', 'names', 'labels'])

# Get the absolute path to the dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
lfw_path = os.path.join(current_dir, '../dataset/lfw-deepfunneled/lfw-deepfunneled')

def load_lfw_dataset(balanced=True, min_faces_balance=10):
    """
    Load LFW dataset from directory structure.
    Returns an object with images and names.
    """
    images = []
    names = []
    
    if not os.path.exists(lfw_path):
        raise FileNotFoundError(f"Dataset not found at {lfw_path}")
    
    # Load images from person directories
    person_dirs = [d for d in os.listdir(lfw_path) if os.path.isdir(os.path.join(lfw_path, d))]
    person_dirs.sort()
    
    person_counts = {}
    
    for person_name in person_dirs:
        person_dir = os.path.join(lfw_path, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))]
        
        # Apply minimum faces requirement
        if balanced and len(image_files) < min_faces_balance:
            continue
        
        person_counts[person_name] = len(image_files)
        
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(img)
                images.append(img_array)
                names.append(person_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    images = np.array(images)
    print(f"Loaded {len(images)} images from {len(person_counts)} people")
    
    return LFWDataset(images=images, names=names, labels=None)

def create_binary_labels(lfw_data):
    """
    Create binary labels: 1 for Arnold Schwarzenegger, 0 for others.
    """
    y = np.array([1 if name == 'Arnold_Schwarzenegger' else 0 for name in lfw_data.names])
    print(f"Class distribution: Arnold={np.sum(y)}, Others={len(y)-np.sum(y)}")
    return y

def prepare_features(lfw_data):
    """
    Prepare features by flattening images.
    """
    n_samples = lfw_data.images.shape[0]
    X = lfw_data.images.reshape(n_samples, -1).astype(np.float32)
    return X

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load the data
    lfw_data = load_lfw_dataset(balanced=True, min_faces_balance=10)
    
    # Create labels
    y = create_binary_labels(lfw_data)
    
    # Prepare features
    X = prepare_features(lfw_data)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Data preprocessing complete.")
    print(f"Feature shape: {X.shape}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
