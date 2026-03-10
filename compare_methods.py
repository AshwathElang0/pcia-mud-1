#!/usr/bin/env python
"""
Quick comparison: Baseline vs SAM segmentation
Run this to see if SAM captures better color trends
"""
import subprocess
import sys

print("="*60)
print("RUNNING COMPARISON: Baseline vs SAM Segmentation")
print("="*60)

print("\n[1/2] Running baseline analysis...")
result1 = subprocess.run([sys.executable, 'baseline_color.py'], capture_output=True, text=True)
if result1.returncode == 0:
    print("  ✓ Baseline complete")
else:
    print(f"  ✗ Error: {result1.stderr}")

print("\n[2/2] Running SAM analysis...")
print("  (This may take a few minutes...)")
result2 = subprocess.run([sys.executable, 'test_sam.py'], capture_output=True, text=True)
if result2.returncode == 0:
    print("  ✓ SAM complete")
else:
    print(f"  ✗ Error: {result2.stderr}")

print("\n" + "="*60)
print("COMPARISON COMPLETE")
print("="*60)
print("\nGenerated files:")
print("  - baseline_color_analysis.png")
print("  - sam_color_analysis.png")
print("  - sam_segmentation_viz.png (shows mask quality)")
print("\nCompare the two analysis plots to see if SAM gives better trends.")
print("Look for more consistent slope patterns across columns.")
