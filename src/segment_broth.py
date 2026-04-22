import os
import cv2
import numpy as np
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
from glob import glob
from scipy.signal import find_peaks

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "broth_samples")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "broth")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Grid size
N_ROWS = 2
N_COLS = 10

def detect_fluid_centers(image_rgb):
    """
    Detects 2x10 grid of fluid centers.
    Uses vertical and horizontal projections of color intensity.
    """
    # Use inverted grayscale as primary signal. 
    # Works well for dark fluid on light background (white or cloth).
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    fluid_signal = 255 - gray
    
    h, w = fluid_signal.shape
    
    # 1. Row detection
    row_proj = np.mean(fluid_signal, axis=1)
    # Smooth to avoid local noise peaks
    row_proj = cv2.GaussianBlur(row_proj.reshape(-1, 1), (1, 101), 0).flatten()
    
    # Expected rows are roughly at 1/3 and 2/3 of height
    r1_peak = np.argmax(row_proj[h//10 : h//2]) + h//10
    r2_peak = np.argmax(row_proj[h//2 : 9*h//10]) + h//2
    row_centers = [r1_peak, r2_peak]
    
    # 2. Column detection
    # Sum across the rows found to get better signal
    margin = h // 40
    col_proj1 = np.mean(fluid_signal[max(0, r1_peak-margin):min(h, r1_peak+margin), :], axis=0)
    col_proj2 = np.mean(fluid_signal[max(0, r2_peak-margin):min(h, r2_peak+margin), :], axis=0)
    col_proj = col_proj1 + col_proj2
    
    # Find 10 peaks, ignoring the far edges (first and last 5%)
    edge_margin = w // 20
    peaks, _ = find_peaks(col_proj[edge_margin : w - edge_margin], distance=w//15, prominence=10)
    peaks = peaks + edge_margin # Offset correction
    
    if len(peaks) < N_COLS:
        print(f"Warning: Only found {len(peaks)} peaks, falling back to grid.")
        col_centers = np.linspace(w//15, w*14//15, N_COLS).astype(int)
    else:
        # Take the top 10 peaks and sort them
        peaks = sorted(peaks, key=lambda p: col_proj[p], reverse=True)[:N_COLS]
        col_centers = sorted(peaks)
        
    centers = []
    # Row by row, col by col
    for r in row_centers:
        for c in col_centers:
            centers.append((c, r))
    
    return centers

def segment_broth(image_path, model, processor, device):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    centers = detect_fluid_centers(image_rgb)
    
    # SAM Inference
    masks = []
    h_img, w_img = image_rgb.shape[:2]
    
    # Estimate box size based on image dimensions
    box_w = w_img // 15
    box_h = h_img // 6
    
    for pt in centers:
        # Prompt: Bounding box + Center point + Negative point above
        # The box helps SAM narrow down the object, center point confirms it,
        # and negative point prevents leaking up the tube.
        box = [
            max(0, pt[0] - box_w // 2),
            max(0, pt[1] - box_h // 2),
            min(w_img, pt[0] + box_w // 2),
            min(h_img, pt[1] + box_h // 2)
        ]
        
        shifted_pt = [pt[0], pt[1] + h_img // 100] # Slightly down
        neg_pt = [pt[0], pt[1] - h_img // 15] # Safe distance above
        
        input_points = [[[shifted_pt, neg_pt]]]
        input_labels = [[[1, 0]]]
        input_boxes = [[box]]
        
        inputs = processor(
            pil_image, 
            input_points=input_points, 
            input_labels=input_labels, 
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process
        mask = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )[0][0]
        
        # Select mask with highest IoU
        best_mask_idx = torch.argmax(outputs.iou_scores.cpu().squeeze()).item()
        mask = mask[best_mask_idx].numpy().astype(bool)
        masks.append(mask)
        
    return image_rgb, centers, np.array(masks)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading SAM model...")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    
    image_paths = sorted(glob(os.path.join(SAMPLES_DIR, "*.png")))
    if not image_paths:
        print(f"No images found in {SAMPLES_DIR}")
        return
    
    for img_path in image_paths:
        base_name = os.path.basename(img_path).replace(".png", "")
        print(f"Processing {base_name}...")
        
        try:
            img_rgb, centers, masks = segment_broth(img_path, model, processor, device)
            
            # Save masks
            np.save(os.path.join(RESULTS_DIR, f"{base_name}_masks.npy"), masks)
            
            # Save overlay
            overlay = img_rgb.copy()
            for idx, mask in enumerate(masks):
                # Row 1 (0-9): Reddish, Row 2 (10-19): Bluish
                color = (255, 50, 50) if idx < 10 else (50, 50, 255)
                overlay[mask] = (overlay[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
                # Draw center point for reference
                cv2.circle(overlay, tuple(centers[idx]), 3, (255, 255, 255), -1)
                
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(RESULTS_DIR, f"{base_name}_overlay.png"), overlay_bgr)
            print(f"Successfully processed {base_name}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

if __name__ == "__main__":
    main()
