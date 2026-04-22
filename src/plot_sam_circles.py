import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_DIR = os.path.join(ROOT_DIR, "samples", "well_plate")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "statistical", "well_plate", "sam_circle_overlays")


def extract_time(path: str) -> int:
    """Extract minute prefix from names like 10th_min.png."""
    name = os.path.basename(path)
    try:
        return int(name.split("th")[0])
    except (ValueError, IndexError):
        return 10**9


def mask_to_circle(mask: np.ndarray) -> tuple[int, int, int] | None:
    """Fit a circle to a binary mask using its largest contour."""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if radius <= 0:
        return None

    return int(round(cx)), int(round(cy)), int(round(radius))


def draw_mask_circles(image_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Draw one fitted circle for each SAM mask in the (rows, cols, H, W) array."""
    overlay = image_rgb.copy()
    flat_masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])

    for idx, mask in enumerate(flat_masks):
        circle = mask_to_circle(mask)
        if circle is None:
            continue

        x, y, r = circle
        cv2.circle(overlay, (x, y), r, (255, 80, 0), 2)
        cv2.circle(overlay, (x, y), 2, (0, 255, 255), -1)
        cv2.putText(
            overlay,
            str(idx + 1),
            (x - 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mask_paths = sorted(glob(os.path.join(CONTROL_DIR, "*_sam_masks.npy")), key=extract_time)
    if not mask_paths:
        raise FileNotFoundError(f"No SAM mask files found in: {CONTROL_DIR}")

    for mask_path in mask_paths:
        image_path = mask_path.replace("_sam_masks.npy", ".png")
        if not os.path.exists(image_path):
            print(f"Skipping (image missing): {os.path.basename(mask_path)}")
            continue

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Skipping (could not read image): {os.path.basename(image_path)}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = np.load(mask_path)

        overlay = draw_mask_circles(image_rgb, masks)

        base = os.path.basename(image_path).replace(".png", "")
        out_path = os.path.join(OUTPUT_DIR, f"{base}_sam_circles.png")

        plt.figure(figsize=(12, 6))
        plt.imshow(overlay)
        plt.title(f"SAM circles overlay: {base}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
