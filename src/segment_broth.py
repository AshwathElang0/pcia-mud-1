import os
import cv2
import numpy as np
from glob import glob
from scipy.signal import find_peaks

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "broth_samples")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "broth")
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
from transformers import SamModel, SamProcessor
from PIL import Image

# Grid size
N_ROWS = 2
N_COLS = 10

def detect_fluid_centers(image_rgb):
    """
    Detects 2x10 grid of fluid centers using the Saturation channel.
    This is much more robust to shadows and background texture than intensity.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    h, w = saturation.shape
    
    # 1. Row detection
    row_proj = np.mean(saturation, axis=1)
    row_proj = cv2.GaussianBlur(row_proj.reshape(-1, 1), (1, 101), 0).flatten()
    
    # Find two main peaks for the rows
    peaks_r, props_r = find_peaks(row_proj, distance=h//4, prominence=10)
    if len(peaks_r) < 2:
        # Fallback
        row_centers = [h//3, 2*h//3]
    else:
        # Take the two most prominent peaks and sort by Y
        top_idx = np.argsort(props_r['prominences'])[-2:]
        row_centers = sorted(peaks_r[top_idx])
    
    # 2. Column detection
    margin = h // 20
    col_projs = []
    for r in row_centers:
        col_projs.append(np.mean(saturation[max(0, r-margin):min(h, r+margin), :], axis=0))
    col_proj = np.mean(col_projs, axis=0)
    
    # Find 10 peaks
    edge_margin = w // 20
    peaks_c, props_c = find_peaks(col_proj[edge_margin : w - edge_margin], distance=w//15, prominence=10)
    peaks_c = peaks_c + edge_margin 
    
    if len(peaks_c) < N_COLS:
        print(f"Warning: Only found {len(peaks_c)} column peaks, falling back to uniform grid.")
        col_centers = np.linspace(w//10, w*9//10, N_COLS).astype(int)
    else:
        # Take the top 10 most prominent peaks and sort by X
        top_idx = np.argsort(props_c['prominences'])[-N_COLS:]
        col_centers = sorted(peaks_c[top_idx])
        
    centers = []
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
    
    masks = []
    h_img, w_img = image_rgb.shape[:2]
    
    # SAM Inference for each center
    for pt in centers:
        # Prompt: Point prompt (center) + Bounding box
        box = [
            max(0, pt[0] - w_img // 30),
            max(0, pt[1] - h_img // 15),
            min(w_img, pt[0] + w_img // 30),
            min(h_img, pt[1] + h_img // 15)
        ]
        
        inputs = processor(
            pil_image, 
            input_points=[[[pt]]], 
            input_labels=[[[1]]], 
            input_boxes=[[box]],
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        mask = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )[0][0]
        
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
        print(f"Processing {base_name} with Signal-Guided ML...")
        
        try:
            img_rgb, centers, masks = segment_broth(img_path, model, processor, device)
            
            # Save masks
            np.save(os.path.join(RESULTS_DIR, f"{base_name}_masks.npy"), masks)
            
            # Save overlay
            overlay = img_rgb.copy()
            for idx, mask in enumerate(masks):
                # Row 1 (0-9): Reddish, Row 2 (10-19): Bluish
                color = (255, 50, 50) if idx < 10 else (50, 50, 255)
                overlay[mask] = (overlay[mask] * 0.7 + np.array(color) * 0.3).astype(np.uint8)
                # Draw center point for reference
                cv2.circle(overlay, tuple(centers[idx]), 3, (255, 255, 255), -1)
                
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(RESULTS_DIR, f"{base_name}_overlay.png"), overlay_bgr)
            print(f"Successfully processed {base_name}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
