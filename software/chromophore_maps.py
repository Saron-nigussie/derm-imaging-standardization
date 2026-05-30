# chromophore_maps.py
# Generate erythema and melanin maps from skin images
# Project: Smartphone Dermatological Imaging Standardization System
# Author: Saron Nigussie — Florida International University

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import load_image, srgb_to_linear, linear_to_lab
from metrics import compute_blur_metric, quality_gate


def generate_chromophore_maps(image_path, output_path=None, show=True):
    """
    Load a skin image and generate L* melanin map and a* erythema map.
    
    Parameters:
        image_path: string path to input image
        output_path: string path to save output figure (optional)
        show: boolean whether to display the figure
    
    Returns:
        results: dictionary with L_mean, a_mean, L_std, a_std,
                 blur_metric, accepted, reason
    """
    # Load and check image
    img_rgb = load_image(image_path)
    
    # Quality gate
    blur_metric = compute_blur_metric(img_rgb)
    mean_luminance = np.mean(img_rgb)
    accepted, reason = quality_gate(blur_metric, mean_luminance)
    
    # Convert to CIELAB
    img_linear = srgb_to_linear(img_rgb)
    img_lab, L, a, b = linear_to_lab(img_linear)
    
    # Build results dictionary
    results = {
        'image_path': image_path,
        'blur_metric': blur_metric,
        'mean_luminance': mean_luminance,
        'accepted': accepted,
        'quality_reason': reason,
        'L_mean': np.mean(L),
        'L_std': np.std(L),
        'a_mean': np.mean(a),
        'a_std': np.std(a),
        'b_mean': np.mean(b),
        'b_std': np.std(b),
        'L_map': L,
        'a_map': a
    }
    
    # Visualization
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig)
    
    # Panel 1 — original image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    status = "ACCEPTED" if accepted else f"REJECTED\n{reason}"
    color = 'green' if accepted else 'red'
    ax1.text(0.5, -0.08, status, transform=ax1.transAxes,
             ha='center', color=color, fontsize=9)
    
    # Panel 2 — L* melanin map
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(L, cmap='gray', vmin=0, vmax=100)
    ax2.set_title(f'L* Melanin Map\n(mean={np.mean(L):.1f})', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='L* (0=dark, 100=light)')
    
    # Panel 3 — a* erythema map
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(a, cmap='RdYlGn_r', vmin=-10, vmax=30)
    ax3.set_title(f'a* Erythema Map\n(mean={np.mean(a):.1f})', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, label='a* (positive = more red)')
    
    # Panel 4 — statistics summary
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    stats_text = (
        f"IMAGE QUALITY\n"
        f"{'─' * 28}\n"
        f"Blur metric:   {blur_metric:.1f}\n"
        f"Mean luminance: {mean_luminance:.3f}\n"
        f"Status: {status}\n\n"
        f"MELANIN (L* channel)\n"
        f"{'─' * 28}\n"
        f"Mean L*:  {np.mean(L):.2f}\n"
        f"Std L*:   {np.std(L):.2f}\n"
        f"Min L*:   {np.min(L):.2f}\n"
        f"Max L*:   {np.max(L):.2f}\n\n"
        f"ERYTHEMA (a* channel)\n"
        f"{'─' * 28}\n"
        f"Mean a*:  {np.mean(a):.2f}\n"
        f"Std a*:   {np.std(a):.2f}\n"
        f"Min a*:   {np.min(a):.2f}\n"
        f"Max a*:   {np.max(a):.2f}"
    )
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.set_title('Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle('Chromophore Map Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()
    
    return results
