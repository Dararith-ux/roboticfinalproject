"""
Generate a combined evaluation summary image
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.gridspec import GridSpec

# Paths
OUTPUT_DIR = r"C:\Users\Dararith\Desktop\Fall_2025\Robotic_submission\training\eval"
PR_CURVES_PATH = f"{OUTPUT_DIR}/pr_curves.png"
CONFUSION_MATRIX_PATH = f"{OUTPUT_DIR}/confusion_matrix.png"
SUMMARY_PATH = f"{OUTPUT_DIR}/evaluation_summary.png"

# Metrics data
metrics = {
    "mAP@0.50": 1.0000,
    "mAP@[.50:.95]": 0.7990,
}

per_class_ap = {
    "Expired_Cosmetic": 1.0000,
    "Syringe": 1.0000,
    "Tablet": 1.0000,
    "Used_Battery": 1.0000,
}

# Create figure
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.3, wspace=0.2)

# Title
fig.suptitle('Model Evaluation Summary\nSSD MobileNet V2 FPNLite - Hazardous Waste Detection',
             fontsize=16, fontweight='bold', y=0.98)

# 1. Metrics Summary (top left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')

# Create metrics table
metrics_text = """
╔══════════════════════════════════════════════╗
║           EVALUATION METRICS                 ║
╠══════════════════════════════════════════════╣
║                                              ║
║   mAP@0.50:        {:.2%}                 ║
║   mAP@[.50:.95]:   {:.2%}                 ║
║                                              ║
╠══════════════════════════════════════════════╣
║           PER-CLASS AP @0.50                 ║
╠══════════════════════════════════════════════╣
║                                              ║
║   Expired_Cosmetic:   {:.2%}              ║
║   Syringe:            {:.2%}              ║
║   Tablet:             {:.2%}              ║
║   Used_Battery:       {:.2%}              ║
║                                              ║
╚══════════════════════════════════════════════╝
""".format(
    metrics["mAP@0.50"],
    metrics["mAP@[.50:.95]"],
    per_class_ap["Expired_Cosmetic"],
    per_class_ap["Syringe"],
    per_class_ap["Tablet"],
    per_class_ap["Used_Battery"],
)

ax1.text(0.5, 0.5, metrics_text, transform=ax1.transAxes, fontsize=12,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax1.set_title('Metrics Summary', fontsize=14, fontweight='bold', pad=10)

# 2. Bar chart for per-class AP (top right)
ax2 = fig.add_subplot(gs[0, 1])
classes = list(per_class_ap.keys())
ap_values = list(per_class_ap.values())
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

bars = ax2.bar(classes, ap_values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylim(0, 1.1)
ax2.set_ylabel('Average Precision', fontsize=12)
ax2.set_title('Per-Class AP @0.50 IoU', fontsize=14, fontweight='bold')
ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect AP')

# Add value labels on bars
for bar, val in zip(bars, ap_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.tick_params(axis='x', rotation=15)
ax2.legend()

# 3. PR Curves (bottom left)
ax3 = fig.add_subplot(gs[1, 0])
pr_img = mpimg.imread(PR_CURVES_PATH)
ax3.imshow(pr_img)
ax3.axis('off')
ax3.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold', pad=10)

# 4. Confusion Matrix (bottom right)
ax4 = fig.add_subplot(gs[1, 1])
cm_img = mpimg.imread(CONFUSION_MATRIX_PATH)
ax4.imshow(cm_img)
ax4.axis('off')
ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=10)

# Add footer
fig.text(0.5, 0.01, 'Test Dataset: 48 images | 4 Classes | IoU Threshold: 0.5',
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(SUMMARY_PATH, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Evaluation summary saved to: {SUMMARY_PATH}")
