# File: Preprocess_Pytorch.py

import os
import cv2
import numpy as np
import pywt
import torch
from sklearn.utils.class_weight import compute_class_weight

# Function to apply Gaussian Notch Filter
def apply_gaussian_notch_filter(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)

    # Gaussian notch filter
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0

    fft_shift *= mask
    fft_ishift = np.fft.ifftshift(fft_shift)
    img_filtered = np.abs(np.fft.ifft2(fft_ishift))
    return img_filtered

# Function to apply Wavelet Transform
def wavelet_transform(img, wavelet='haar'):
    coeffs = pywt.dwt2(img, wavelet)
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH

# Preprocessing and Feature Extraction
def preprocess_and_extract(dataset_dir, output_file, weight_adjustment=None):
    images, labels = [], []

    for label, folder in enumerate(['0', '1']):  # 0 for Benign, 1 for Malignant
        folder_path = os.path.join(dataset_dir, folder)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, 0)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img_filtered = apply_gaussian_notch_filter(img)
                LL, LH, HL, HH = wavelet_transform(img_filtered)
                features = np.hstack([LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()])
                images.append(features)
                labels.append(label)

    # Reshape features to be compatible with CNN (batch_size, channels, height, width)
    images = np.array(images).reshape(-1, 1, 64, 64)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=np.array(labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Adjust class weights if needed
    if weight_adjustment:
        class_weights *= torch.tensor(weight_adjustment, dtype=torch.float32)

    # Save preprocessed data as a PyTorch tensor
    torch.save({
        "features": torch.tensor(images, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "class_weights": class_weights
    }, output_file)
    print(f"Preprocessed data saved to {output_file}")
    print(f"Class weights used: {class_weights.tolist()}")

# Main function
if __name__ == "__main__":
    dataset_dir = "datasets"  # Change to the path of your dataset
    output_file = "processed_features.pt"
    weight_adjustment = [1.0, 1.2]  # Adjust weights: e.g., prioritize class 1 slightly
    preprocess_and_extract(dataset_dir, output_file, weight_adjustment=weight_adjustment)
