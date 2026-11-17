# Detailed Qualitative Error Analysis: Baseline vs. Augmented U-Net

This document investigates the visual predictions, failure modes, boundary behaviors, and error maps generated across the 26 untouched test photographs from the Penn-Fudan Pedestrian Dataset.

---

## 1. Visual Prediction Inspection Framework

For each test photograph, error maps were generated using the following color-coding convention:
- **Green (`#22C55E`)**: True Positives (TP) — correctly identified pedestrian pixels.
- **Red (`#EF4444`)**: False Positives (FP) — background pixels incorrectly predicted as pedestrian.
- **Blue (`#3B82F6`)**: False Negatives (FN) — ground-truth pedestrian pixels missed by the model.
- **Dark Gray (`#19191E`)**: True Negatives (TN) — correctly classified background.

Visual prediction artifacts are stored in [`results/test_visualizations/`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/).

---

## 2. Key Error Categories & Empirical Findings

### A. Background Clutter & False Positive Hallucinations
- **Representative Samples**: `FudanPed00023.png`, `FudanPed00066.png`, `PennPed00009.png`
- **Visual Observations**:
  - The unaugmented **Baseline U-Net** frequently hallucinations pedestrian presence on vertical, person-like background structures such as tree trunks, door frames, lamp posts, and textured foliage. In `FudanPed00023.png`, the Baseline model misclassified a dark vertical background column as a pedestrian, incurring **3,177 false positive pixels**.
  - The **Augmented U-Net** demonstrates markedly superior spatial discrimination against background artifacts. In `FudanPed00023.png`, false positive pixels dropped from **3,177 to 247** (a $92.2\%$ reduction), lifting sample IoU from **0.6356 to 0.8920**.
  - Similarly, in `FudanPed00066.png`, the Baseline model hallucinated a large phantom region in the upper foliage (IoU: **0.3971**), while the Augmented model completely suppressed this artifact (IoU: **0.8611**, $\Delta = +0.4640$).
- **Mechanism**: The scale jittering (zoom-in and zoom-out with zero-fill background padding) forced the Augmented model to learn invariant representations of actual human anatomies rather than relying on vertical context in the background.

### B. Multi-Person Scenes & Inter-Pedestrian Boundaries
- **Representative Samples**: `FudanPed00025.png`, `FudanPed00057.png`
- **Visual Observations**:
  - In `FudanPed00025.png`, six pedestrians are walking at varying distances across a wide campus plaza.
  - The Baseline model severely blurred boundaries, merged distant pedestrians into amorphous blobs, and generated **5,802 false positive pixels** in the surrounding ground texture (IoU: **0.5034**).
  - The Augmented model achieved **0.6746 IoU** ($\Delta = +0.1712$), successfully separating discrete pedestrian silhouettes and slashing false positives to **1,732 pixels**.
  - However, where pedestrians directly overlap in the line of sight (e.g. `FudanPed00057.png`), both models merge the two silhouettes into a single contiguous foreground mask. Because this is semantic segmentation rather than instance segmentation, this merging is structurally correct according to ground-truth labels.

### C. Boundary Bleed vs. Boundary Erosion
- **Representative Samples**: `PennPed00017.png`, `PennPed00038.png`
- **Visual Observations**:
  - In high-confidence cases like `PennPed00017.png`, both models achieve strong performance (Baseline: **0.8235**, Augmented: **0.8647**).
  - Inspecting the error maps reveals that the Baseline model exhibits "boundary bleed" — a 2-to-4 pixel thick halo of False Positives (red) radiating around the pedestrian's arms, legs, and shoes.
  - The Augmented model produces significantly tighter, crisper contours with minimal haloing. The precision gain on the global test set ($+12.56\%$, from $0.7016$ to $0.7897$) directly corroborates this tighter contour adherence.

### D. Hard Cases & Failure Modes (Under-Segmentation & Regressions)
- **Representative Samples**: `PennPed00012.png`, `PennPed00077.png`
- **Visual Observations**:
  - `PennPed00012.png` represents the only significant regression for the Augmented model: Baseline IoU was **0.4319**, while Augmented IoU dropped to **0.2179** ($\Delta = -0.2139$).
  - **Visual Cause**: The photograph contains a distant pedestrian under high shadow contrast wearing dark clothing that blends into the background pavement. The Augmented model's higher precision threshold caused it to under-segment (producing blue False Negatives on the lower torso and legs), whereas the Baseline model's higher tendency to predict positive pixels captured more of the silhouette despite being noisier overall.
  - In `PennPed00077.png`, both models struggled (Baseline IoU: **0.3991**, Augmented: **0.4270**). The pedestrian carries large, dark bags and is partially occluded by a handrail. Both models exhibit ambiguous predictions around the carried luggage, which is partially labeled as person in the ground truth.

---

## 3. Summary of Error Metrics

| Error Mode | Baseline U-Net Behavior | Augmented U-Net Behavior | Primary Driver |
|---|---|---|---|
| **Background Hallucination** | Frequent false positives on trees, posts, and foliage | Dramatically reduced ($92\%$ FP reduction on key samples) | Random zoom and canvas translation |
| **Contour Haloing** | 2–4 pixel red halo around extremities | Clean, tight contours matching ground-truth edges | Horizontal flip & zoom regularization |
| **Crowded Scenes** | Merged distant pedestrians with ground texture | Discrete silhouettes retained | Multi-scale feature invariance |
| **Low-Contrast / Shadows** | Over-predicts but captures parts of limbs | Conservative; prone to false negative limb dropouts | Precision-recall trade-off |
| **Carried Objects / Luggage** | Inconsistent bag inclusion | Inconsistent bag inclusion | Inherent annotation ambiguity in raw dataset |

---

## 4. Visual Prediction Showcase Links

Detailed 5-panel predictions and error maps are accessible directly:
- [`pred_FudanPed00066.png`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/pred_FudanPed00066.png) — Major improvement showcase ($\Delta = +0.4640$)
- [`pred_FudanPed00025.png`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/pred_FudanPed00025.png) — 6-person crowded plaza scene ($\Delta = +0.1712$)
- [`pred_FudanPed00023.png`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/pred_FudanPed00023.png) — Background artifact elimination ($\Delta = +0.2564$)
- [`pred_PennPed00017.png`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/pred_PennPed00017.png) — Clear high-confidence pedestrian ($0.8647$ IoU)
- [`pred_PennPed00012.png`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/pred_PennPed00012.png) — Hard case / shadow regression ($0.2179$ IoU)
- [`pred_PennPed00009.png`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/pred_PennPed00009.png) — High background texture complexity
- [`pred_PennPed00038.png`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/pred_PennPed00038.png) — Severe under-segmentation fix ($\Delta = +0.3525$)
- [`pred_PennPed00077.png`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/pred_PennPed00077.png) — Carried luggage ambiguity
- [`00_test_predictions_overview_grid.png`](file:///c:/Users/rohit/OneDrive/Desktop/ML%20Project/person-segmentation/results/test_visualizations/00_test_predictions_overview_grid.png) — Full 8-sample comparative overview grid
