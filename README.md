---
title: Explainable Brain Tumor MRI Classifier
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Explainable Brain Tumor MRI Classification with Uncertainty-Aware Decision Support

An explainable decision support prototype for brain tumor MRI classification, developed as part of research
in mathematics and medical AI at Bahauddin Zakariya University, Multan.

**Pipeline**
1. **Classification:** MobileNetV2 (ImageNet pre-trained) fine-tuned in PyTorch on four classes:
   glioma, meningioma, pituitary tumor, no tumor.
2. **Explainability:** Grad-CAM heatmaps on the last convolutional block show which regions drove each prediction.
3. **Escalation:** predictions below an adjustable confidence threshold are flagged for human review.
4. **Retrieval:** MiniLM sentence embeddings and a FAISS index retrieve background facts for the predicted class.
5. **Reporting:** an LLM (via Groq) writes a short educational report using only the retrieved facts.

**Evaluation** (1,600 held-out test images)
| Metric | Full test set | Clean subset (1,018 images, duplicates removed) |
|---|---|---|
| Accuracy | 95.2% | 92.8% |
| Macro F1 | 0.951 | 0.892 |
| Glioma recall | 82.3% | 81.1% |
| ECE | 0.028 | 0.045 |

A perceptual-hash and pixel-correlation check found 582 test images (36.4%) that duplicate training images, including 88% of the no-tumor test class, so the clean subset gives a more honest estimate. Temperature scaling (T = 1.83) reduces negative log-likelihood by 15%. Evaluation code: `evaluate_model.py`, `check_duplicates.py`.

**Known limitations**
- Works on single 2D slices, not 3D volumes.
- On out-of-distribution scans the model can give low-confidence misclassifications and skewed attention maps.
- Softmax confidence is not calibrated.

**Setup:** add a `GROQ_API_KEY` secret in the Space settings to enable LLM reports and follow-up questions.
Without it, the app still runs and produces a template report. Put a few sample scans in an `examples/` folder
to show them as one-click examples.

> For research and educational use only. Not for clinical diagnosis or patient care.

Author: Hafza Noor
