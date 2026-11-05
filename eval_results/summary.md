# Evaluation summary

**Test set:** 1600 images
- Accuracy: 95.19%
- Macro F1: 0.9506

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 0.9970 | 0.8225 | 0.9014 |
| meningioma | 0.9144 | 0.9875 | 0.9495 |
| notumor | 0.9238 | 1.0000 | 0.9604 |
| pituitary | 0.9852 | 0.9975 | 0.9913 |

**Calibration** (T fitted on 800 images, evaluated on the other 800)
- Fitted temperature T = 1.827
- ECE: 0.0228 -> 0.0288
- NLL: 0.1992 -> 0.1695
- Brier: 0.0671 -> 0.0669
- Mean confidence: 97.87% -> 93.65% (accuracy on this half: 96.00%)

**Train/test leakage check**
- Near-duplicate test images (Hamming <= 4/64): 852 (53.2%)
- Accuracy on the 748 clean test images: 90.64% (macro F1 0.8540)
