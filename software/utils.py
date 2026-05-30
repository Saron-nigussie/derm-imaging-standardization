# utils.py
# Helper functions for the dermatological imaging pipeline
# Project: Smartphone Dermatological Imaging Standardization System
# Author: Saron Nigussie — Florida International University

import numpy as np
import cv2
from skimage import color

def load_image(filepath):
    """
    Load an image from filepath and return as RGB numpy array
    with values normalized to 0-1 range.
    
    Parameters:
        filepath: string path to image file
    
    Returns:
        img_rgb: numpy array shape (H, W, 3) with values 0 to 1
    """
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        raise ValueError(f"Could not load image from {filepath}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(float) / 255.0
    return img_rgb


def srgb_to_linear(img_srgb):
    """
    Convert sRGB image to linear light space.
    Removes gamma encoding so pixel values are proportional
    to physical light intensity.
    Implements Uthoff et al. 2020 Equation 1.
    
    Parameters:
        img_srgb: numpy array with values 0 to 1
    
    Returns:
        img_linear: numpy array with values 0 to 1
    """
    img_linear = np.where(
        img_srgb <= 0.04045,
        img_srgb / 12.92,
        ((img_srgb + 0.055) / 1.055) ** 2.4
    )
    return img_linear


def linear_to_lab(img_linear):
    """
    Convert linear RGB image to CIELAB color space.
    L* channel = melanin proxy (lower = more melanin)
    a* channel = erythema proxy (higher = more redness)
    
    Parameters:
        img_linear: numpy array linear RGB values 0 to 1
    
    Returns:
        img_lab: numpy array shape (H, W, 3) in CIELAB
        L: 2D array of L* values
        a: 2D array of a* values
        b: 2D array of b* values
    """
    # Convert linear to sRGB for skimage (it expects sRGB input)
    # Apply gamma encoding before passing to rgb2lab
    img_srgb_back = np.where(
        img_linear <= 0.0031308,
        img_linear * 12.92,
        1.055 * (img_linear ** (1/2.4)) - 0.055
    )
    img_srgb_uint8 = (np.clip(img_srgb_back, 0, 1) * 255).astype(np.uint8)
    img_lab = color.rgb2lab(img_srgb_uint8)
    
    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]
    
    return img_lab, L, a, b


def apply_flatfield_correction(img_linear, U):
    """
    Apply flatfield correction to remove illumination non-uniformity.
    Implements Uthoff et al. 2020 Equation 6.
    
    Parameters:
        img_linear: numpy array linear RGB image 0 to 1
        U: numpy array flatfield reference matrix (same spatial size)
    
    Returns:
        img_corrected: flatfield corrected image
    """
    img_corrected = np.zeros_like(img_linear)
    for c in range(3):
        img_corrected[:, :, c] = img_linear[:, :, c] / (U + 1e-6)
    img_corrected = np.clip(img_corrected, 0, 1)
    return img_corrected
