# pipeline.py
# Main pipeline — process a folder of images and save results
# Project: Smartphone Dermatological Imaging Standardization System
# Author: Saron Nigussie — Florida International University
#
# Usage:
#   python pipeline.py --input test_images/bare_phone --output results/bare_phone
#   python pipeline.py --input test_images/isic --output results/isic

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from utils import load_image, srgb_to_linear, linear_to_lab
from metrics import (compute_blur_metric, compute_illumination_uniformity,
                     compute_ssim, quality_gate)
from chromophore_maps import generate_chromophore_maps


def process_image_folder(input_folder, output_folder, 
                          reference_image_path=None):
    """
    Process all images in a folder through the full pipeline.
    
    Parameters:
        input_folder: path to folder containing images
        output_folder: path to save results
        reference_image_path: path to reference image for SSIM 
                              (uses first accepted image if None)
    
    Returns:
        results_df: pandas DataFrame with all metrics
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'maps'), exist_ok=True)
    
    # Get all image files
    extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(extensions)
    ]
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"No image files found in {input_folder}")
        return None
    
    print(f"\nProcessing {len(image_files)} images from {input_folder}")
    print(f"Results will be saved to {output_folder}\n")
    
    # Load reference image for SSIM
    reference_img = None
    if reference_image_path and os.path.exists(reference_image_path):
        reference_img = load_image(reference_image_path)
        print(f"Reference image loaded: {reference_image_path}")
    
    # Process each image
    all_results = []
    
    for i, filename in enumerate(image_files):
        filepath = os.path.join(input_folder, filename)
        print(f"[{i+1}/{len(image_files)}] Processing {filename}...")
        
        try:
            # Load image
            img_rgb = load_image(filepath)
            img_linear = srgb_to_linear(img_rgb)
            
            # Quality metrics
            blur_metric = compute_blur_metric(img_rgb)
            mean_luminance = np.mean(img_rgb)
            accepted, reason = quality_gate(blur_metric, mean_luminance)
            
            # Illumination uniformity
            uniformity, U = compute_illumination_uniformity(img_linear)
            
            # CIELAB conversion
            img_lab, L, a, b = linear_to_lab(img_linear)
            
            # SSIM vs reference
            ssim_value = None
            if reference_img is not None:
                ssim_value = compute_ssim(img_rgb, reference_img)
            elif i == 0 and accepted:
                # Use first accepted image as reference
                reference_img = img_rgb.copy()
                ssim_value = 1.0
                print(f"  Using {filename} as SSIM reference")
            elif reference_img is not None:
                ssim_value = compute_ssim(img_rgb, reference_img)
            
            # Generate chromophore map and save
            map_output = os.path.join(
                output_folder, 'maps',
                f"{os.path.splitext(filename)[0]}_maps.png"
            )
            generate_chromophore_maps(
                filepath, 
                output_path=map_output, 
                show=False
            )
            
            # Collect results
            result = {
                'filename': filename,
                'accepted': accepted,
                'quality_reason': reason,
                'blur_metric': round(blur_metric, 2),
                'mean_luminance': round(mean_luminance, 4),
                'illumination_uniformity': round(uniformity, 4),
                'L_mean': round(np.mean(L), 3),
                'L_std': round(np.std(L), 3),
                'a_mean': round(np.mean(a), 3),
                'a_std': round(np.std(a), 3),
                'b_mean': round(np.mean(b), 3),
                'b_std': round(np.std(b), 3),
                'ssim_vs_reference': round(ssim_value, 4) if ssim_value else None,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            all_results.append(result)
            
            status = "✓ ACCEPTED" if accepted else f"✗ REJECTED ({reason})"
            print(f"  {status} | Blur: {blur_metric:.1f} | "
                  f"Lum: {mean_luminance:.3f} | "
                  f"Uniformity: {uniformity:.3f} | "
                  f"a* mean: {np.mean(a):.2f}")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            all_results.append({
                'filename': filename,
                'accepted': False,
                'quality_reason': f'Processing error: {str(e)}',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_folder, 'results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Print summary statistics
    accepted_df = results_df[results_df['accepted'] == True]
    print(f"\n{'='*50}")
    print(f"SUMMARY — {input_folder}")
    print(f"{'='*50}")
    print(f"Total images:    {len(results_df)}")
    print(f"Accepted:        {len(accepted_df)}")
    print(f"Rejected:        {len(results_df) - len(accepted_df)}")
    
    if len(accepted_df) > 0:
        print(f"\nILLUMINATION UNIFORMITY")
        print(f"  Mean: {accepted_df['illumination_uniformity'].mean():.4f}")
        print(f"  Std:  {accepted_df['illumination_uniformity'].std():.4f}")
        
        print(f"\nERYTHEMA (a* channel)")
        print(f"  Mean a*: {accepted_df['a_mean'].mean():.3f}")
        print(f"  Std a*:  {accepted_df['a_mean'].std():.3f}")
        print(f"  CoV a*:  {accepted_df['a_mean'].std() / accepted_df['a_mean'].mean():.4f}")
        
        print(f"\nMELANIN (L* channel)")
        print(f"  Mean L*: {accepted_df['L_mean'].mean():.3f}")
        print(f"  Std L*:  {accepted_df['L_mean'].std():.3f}")
        
        if 'ssim_vs_reference' in accepted_df.columns:
            ssim_vals = accepted_df['ssim_vs_reference'].dropna()
            if len(ssim_vals) > 1:
                print(f"\nSSIM vs REFERENCE")
                print(f"  Mean: {ssim_vals.mean():.4f}")
                print(f"  Std:  {ssim_vals.std():.4f}")
    
    # Generate summary plot
    generate_summary_plot(accepted_df, output_folder)
    
    return results_df


def generate_summary_plot(results_df, output_folder):
    """Generate a summary visualization of all results."""
    
    if len(results_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1 — blur metrics
    axes[0, 0].plot(results_df['blur_metric'].values, 'b-o', markersize=4)
    axes[0, 0].axhline(y=100, color='r', linestyle='--', label='Threshold (100)')
    axes[0, 0].set_title('Blur Metric Across Images')
    axes[0, 0].set_xlabel('Image index')
    axes[0, 0].set_ylabel('Laplacian variance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2 — illumination uniformity
    axes[0, 1].plot(results_df['illumination_uniformity'].values, 
                    'g-o', markersize=4)
    axes[0, 1].axhline(y=0.960, color='r', linestyle='--', label='Target (0.960)')
    axes[0, 1].axhline(y=0.954, color='orange', linestyle='--', 
                        label='Uthoff baseline (0.954)')
    axes[0, 1].set_title('Illumination Uniformity')
    axes[0, 1].set_xlabel('Image index')
    axes[0, 1].set_ylabel('Uniformity score')
    axes[0, 1].set_ylim([0.7, 1.0])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3 — mean luminance
    axes[0, 2].plot(results_df['mean_luminance'].values, 'm-o', markersize=4)
    axes[0, 2].axhline(y=0.05, color='r', linestyle='--', label='Min acceptable')
    axes[0, 2].axhline(y=0.95, color='r', linestyle='--', label='Max acceptable')
    axes[0, 2].set_title('Mean Luminance')
    axes[0, 2].set_xlabel('Image index')
    axes[0, 2].set_ylabel('Mean pixel value (0-1)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4 — a* erythema across images
    axes[1, 0].plot(results_df['a_mean'].values, 'r-o', markersize=4)
    axes[1, 0].fill_between(
        range(len(results_df)),
        results_df['a_mean'] - results_df['a_std'],
        results_df['a_mean'] + results_df['a_std'],
        alpha=0.2, color='red'
    )
    axes[1, 0].set_title('Erythema (a* mean ± std)')
    axes[1, 0].set_xlabel('Image index')
    axes[1, 0].set_ylabel('a* value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5 — L* melanin across images
    axes[1, 1].plot(results_df['L_mean'].values, 'k-o', markersize=4)
    axes[1, 1].fill_between(
        range(len(results_df)),
        results_df['L_mean'] - results_df['L_std'],
        results_df['L_mean'] + results_df['L_std'],
        alpha=0.2, color='gray'
    )
    axes[1, 1].set_title('Melanin proxy (L* mean ± std)')
    axes[1, 1].set_xlabel('Image index')
    axes[1, 1].set_ylabel('L* value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6 — SSIM if available
    if 'ssim_vs_reference' in results_df.columns:
        ssim_vals = results_df['ssim_vs_reference'].dropna()
        if len(ssim_vals) > 1:
            axes[1, 2].plot(ssim_vals.values, 'c-o', markersize=4)
            axes[1, 2].axhline(y=0.85, color='r', linestyle='--',
                               label='Target (0.85)')
            axes[1, 2].set_title('SSIM vs Reference')
            axes[1, 2].set_xlabel('Image index')
            axes[1, 2].set_ylabel('SSIM')
            axes[1, 2].set_ylim([0, 1])
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'SSIM requires\n2+ images',
                           ha='center', va='center',
                           transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')
    
    plt.suptitle('Pipeline Summary Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_folder, 'summary_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dermatological imaging standardization pipeline'
    )
    parser.add_argument('--input', required=True,
                       help='Input folder containing images')
    parser.add_argument('--output', required=True,
                       help='Output folder for results')
    parser.add_argument('--reference', default=None,
                       help='Reference image path for SSIM comparison')
    
    args = parser.parse_args()
    
    results = process_image_folder(
        args.input,
        args.output,
        args.reference
    )
