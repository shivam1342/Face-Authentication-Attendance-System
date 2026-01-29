"""
Contains image preprocessing utilities such as grayscale conversion and lighting normalization.
Improves robustness under varying camera and lighting conditions.
"""

import cv2
import numpy as np


def to_grayscale(image):
    """
    Convert image to grayscale.
    
    Args:
        image: BGR image
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_lighting(image):
    """
    Normalize lighting using histogram equalization.
    Improves robustness under varying lighting conditions.
    
    Args:
        image: Input image (grayscale or BGR)
        
    Returns:
        Image with normalized lighting
    """
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Apply CLAHE directly to grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def preprocess_face(image, bbox, target_size=(128, 128)):
    """
    Preprocess face region for encoding.
    Extracts face region, normalizes lighting, and resizes.
    
    Args:
        image: Input image
        bbox: Face bounding box (x, y, w, h)
        target_size: Target size for face image
        
    Returns:
        Preprocessed face image
    """
    x, y, w, h = bbox
    
    # Extract face region
    face = image[y:y+h, x:x+w]
    
    # Normalize lighting
    face = normalize_lighting(face)
    
    # Resize to target size
    face = cv2.resize(face, target_size)
    
    return face


def adjust_brightness(image, value=30):
    """
    Adjust image brightness.
    
    Args:
        image: Input image
        value: Brightness adjustment (-255 to 255)
        
    Returns:
        Brightness-adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255).astype('uint8')
    
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

