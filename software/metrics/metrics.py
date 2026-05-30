# metrics.py
# Image quality metrics for standardization validation
# Project: Smartphone Dermatological Imaging Standardization System
# Author: Saron Nigussie — Florida International University

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def compute_blur_metric(img_rgb):
    """
    Compute Laplacian variance as focus/blur metric.
    Higher value = sharper image.
    Threshold for acceptance: > 100 (calibrate empirically).
    
    Parameters:
        img_rgb: numpy array RGB image values 0 to 1
    
    Returns:
        blur_metric: float — higher is sharper
    """
    img_uint8 = (img_rgb * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_metric = laplacian.var()
    return blur_metric


def compute_illumination_uniformity(img_linear):
    """
    Compute illumination uniformity using coefficient of variation.
    Implements Uthoff et al. 2020 Equation 21.
    Uniformity = 1 - (std / mean)
    Target: > 0.960
    
    Parameters:
        img_linear: numpy array linear RGB image 0 to 1
    
    Returns:
        uniformity: float between 0 and 1 (higher is more uniform)
        U: flatfield reference matrix for correction
    """
    # Convert to grayscale luminance
    gray = np.mean(img_linear, axis=2)
    
    # Compute uniformity reference matrix
    U = gray / (np.max(gray) + 1e-6)
    
    # Compute coefficient of variation
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    if mean_val < 1e-6:
        return 0.0, U
    
    cv = std_val / mean_val
    uniformity = 1.0 - cv
    
    return uniformity, U


def compute_ssim(img1_rgb, img2_rgb):
    """
    Compute Structural Similarity Index between two images.
    Implements Wang et al. 2004.
    Range: -1 to 1. Higher is more similar.
    Target inter-session SSIM > 0.85.
    
    Parameters:
        img1_rgb: numpy array RGB first image values 0 to 1
        img2_rgb: numpy array RGB second image values 0 to 1
    
    Returns:
        ssim_value: float
    """
    # Resize img2 to match img1 if needed
    if img1_rgb.shape != img2_rgb.shape:
        h, w = img1_rgb.shape[:2]
        img2_rgb = cv2.resize(img2_rgb, (w, h))
    
    img1_uint8 = (img1_rgb * 255).astype(np.uint8)
    img2_uint8 = (img2_rgb * 255).astype(np.uint8)
    
    ssim_value = ssim(
        img1_uint8,
        img2_uint8,
        multichannel=True,
        channel_axis=2,
        data_range=255
    )
    return ssim_value


def compute_delta_e(L_measured, a_measured, b_measured,
                    L_reference, a_reference, b_reference):
    """
    Compute CIE76 Delta E color difference between measured
    and reference CIELAB values.
    Delta E < 2.0 = clinically acceptable color accuracy.
    Delta E < 3.0 = acceptable for V1 research prototype.
    
    Parameters:
        L_measured, a_measured, b_measured: measured CIELAB values
        L_reference, a_reference, b_reference: ground truth values
    
    Returns:
        delta_e: float color difference
    """
    delta_e = np.sqrt(
        (L_measured - L_reference) ** 2 +
        (a_measured - a_reference) ** 2 +
        (b_measured - b_reference) ** 2
    )
    return delta_e


def quality_gate(blur_metric, mean_luminance,
                 blur_threshold=100, min_lum=0.05, max_lum=0.95):
    """
    Accept or reject an image based on quality criteria.
    
    Parameters:
        blur_metric: float from compute_blur_metric
        mean_luminance: float mean pixel value 0 to 1
        blur_threshold: minimum acceptable blur metric
        min_lum: minimum acceptable mean luminance
        max_lum: maximum acceptable mean luminance
    
    Returns:
        accepted: boolean
        reason: string explanation
    """
    if blur_metric < blur_threshold:
        return False, f"Image too blurry (metric={blur_metric:.1f}, threshold={blur_threshold})"
    
    if mean_luminance < min_lum:
        return False, f"Image underexposed (luminance={mean_luminance:.3f})"
    
    if mean_luminance > max_lum:
        return False, f"Image overexposed (luminance={mean_luminance:.3f})"
    
    return True, "Accepted"
