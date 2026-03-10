import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Visual comparison of temporal images to check for VISIBLE color changes.
This helps diagnose whether the issue is biological or analytical.
"""

def create_visual_comparison():
    timepoints = [0, 5, 10, 15, 20, 25]
    images = []

    print("Loading images...")
    for t in timepoints:
        img_path = f'samples/{t}th_min.jpeg'
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append((t, img_rgb))
        else:
            print(f"Warning: Could not load {img_path}")

    if len(images) < 2:
        print("Error: Need at least 2 images for comparison")
        return

    # Create comparison figure
    n_images = len(images)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (t, img) in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].set_title(f'Time = {t} minutes', fontsize=14, fontweight='bold')
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Visual Timeline Comparison\nLook for color changes in sample wells',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visual_timeline_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: visual_timeline_comparison.png")

    # Create side-by-side comparison of t=0 vs t=25
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))

    img_t0 = images[0][1]
    img_t25 = images[-1][1]

    axes2[0].imshow(img_t0)
    axes2[0].set_title('t = 0 min (Baseline)', fontsize=14, fontweight='bold')
    axes2[0].axis('off')

    axes2[1].imshow(img_t25)
    axes2[1].set_title('t = 25 min (Final)', fontsize=14, fontweight='bold')
    axes2[1].axis('off')

    plt.suptitle('Direct Comparison: Baseline vs Final\nCan you see any color difference?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visual_t0_vs_t25.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: visual_t0_vs_t25.png")

    # Create difference image (amplified)
    print("\nGenerating difference visualization...")

    # Ensure images are the same size for comparison
    h1, w1 = images[0][1].shape[:2]
    h2, w2 = images[-1][1].shape[:2]

    if (h1, w1) != (h2, w2):
        print(f"  Note: Resizing images for comparison (t0: {w1}x{h1}, t25: {w2}x{h2})")
        # Resize t25 to match t0
        img_t25_resized = cv2.resize(images[-1][1], (w1, h1))
        img_t0_for_diff = images[0][1]
        img_t25_for_diff = img_t25_resized
    else:
        img_t0_for_diff = images[0][1]
        img_t25_for_diff = images[-1][1]

    img_t0_gray = cv2.cvtColor(img_t0_for_diff, cv2.COLOR_RGB2GRAY)
    img_t25_gray = cv2.cvtColor(img_t25_for_diff, cv2.COLOR_RGB2GRAY)

    # Calculate absolute difference and amplify
    diff = np.abs(img_t25_gray.astype(float) - img_t0_gray.astype(float))
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)  # Amplify by 5x

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

    axes3[0].imshow(img_t0_for_diff)
    axes3[0].set_title('t = 0 min', fontsize=12)
    axes3[0].axis('off')

    axes3[1].imshow(img_t25_for_diff)
    axes3[1].set_title('t = 25 min', fontsize=12)
    axes3[1].axis('off')

    im = axes3[2].imshow(diff_amplified, cmap='hot')
    axes3[2].set_title('Difference Image (5x amplified)', fontsize=12)
    axes3[2].axis('off')
    plt.colorbar(im, ax=axes3[2], label='Intensity difference')

    plt.suptitle('Change Detection: Brighter areas = more change',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visual_difference_map.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: visual_difference_map.png")

    # Statistical summary of differences
    print("\n" + "="*60)
    print("VISUAL CHANGE STATISTICS")
    print("="*60)
    print(f"Mean absolute difference: {diff.mean():.2f} intensity units")
    print(f"Max difference: {diff.max():.2f} intensity units")
    print(f"Pixels with >10 unit change: {np.sum(diff > 10)}")
    print(f"Pixels with >20 unit change: {np.sum(diff > 20)}")
    print(f"Pixels with >30 unit change: {np.sum(diff > 30)}")

    if diff.mean() < 5:
        print("\n⚠️  WARNING: Very small differences detected!")
        print("   → Suggests minimal biological activity or imaging artifacts")
    elif diff.mean() < 10:
        print("\n⚡ BORDERLINE: Small but detectable differences")
        print("   → May need better indicators or longer incubation")
    else:
        print("\n✓ GOOD: Clear differences detected")
        print("   → Issue may be with color space selection or ROI extraction")

    print("\n" + "="*60)
    print("NEXT: Open the saved images and inspect visually!")
    print("="*60)

if __name__ == "__main__":
    create_visual_comparison()
