"""
Evaluation of the Brain MRI MobileNetV2 classifier.

Produces:
  1. Test accuracy, macro F1, per-class precision/recall/F1, confusion matrix
  2. Calibration: Expected Calibration Error (ECE), NLL, Brier score, reliability diagrams,
     before and after temperature scaling (T fitted on a held-out half of the test set)
  3. Train/test leakage check: near-duplicate test images (perceptual hash) and accuracy on the clean subset
  4. Optional out-of-distribution (OOD) test on a second dataset

Usage (from the "MRI Agent Gradio" folder):
  python evaluate_model.py --train_dir "PATH/Training" --test_dir "PATH/Testing"
  python evaluate_model.py --train_dir "PATH/Training" --test_dir "PATH/Testing" --ood_dir "PATH/OtherDataset"

Folders must contain one sub-folder per class named like models/class_names.json
(glioma, meningioma, notumor, pituitary). Results are written to ./eval_results/.
"""
import os
import json
import argparse

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------- data
def list_images(root, class_names):
    paths, labels = [], []
    for idx, name in enumerate(class_names):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(folder, f))
                labels.append(idx)
    return paths, np.array(labels, dtype=int)


# ---------------------------------------------------------------- model
def load_model(model_path, n_classes):
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v2

    model = mobilenet_v2(weights=None)
    nf = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(nf, n_classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def get_logits(model, paths, batch_size=32):
    import torch
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
            out.append(model(batch.to(device)).cpu().numpy())
            print(f"\r  {min(i + batch_size, len(paths))}/{len(paths)} images", end="", flush=True)
    print()
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------- calibration math
def ece_score(probs, labels, n_bins=15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def nll_score(logits, labels, T=1.0):
    return float(-log_softmax(logits / T, axis=1)[np.arange(len(labels)), labels].mean())


def brier_score(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def fit_temperature(logits, labels):
    """Temperature T > 0 minimizing the negative log-likelihood (a 1-D convex problem in T)."""
    res = minimize_scalar(lambda logT: nll_score(logits, labels, np.exp(logT)),
                          bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def stratified_halves(labels, seed=42):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        h = len(idx) // 2
        a += idx[:h].tolist()
        b += idx[h:].tolist()
    return np.array(sorted(a)), np.array(sorted(b))


def calib_metrics(logits, labels, T=1.0):
    p = softmax(logits / T, axis=1)
    return {"ECE": ece_score(p, labels), "NLL": nll_score(logits, labels, T),
            "Brier": brier_score(p, labels), "mean_confidence": float(p.max(1).mean()),
            "accuracy": float((p.argmax(1) == labels).mean())}


# ---------------------------------------------------------------- leakage check
def dhash(path, size=8):
    from PIL import Image
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0]


def min_hamming(test_h, train_h, chunk=256):
    train_b = np.unpackbits(train_h.astype(">u8").view(np.uint8).reshape(-1, 8), axis=1).astype(bool)
    out = np.empty(len(test_h), dtype=int)
    for i in range(0, len(test_h), chunk):
        tb = np.unpackbits(test_h[i:i + chunk].astype(">u8").view(np.uint8).reshape(-1, 8), axis=1).astype(bool)
        d = (tb[:, None, :] ^ train_b[None, :, :]).sum(2)
        out[i:i + chunk] = d.min(1)
    return out


# ---------------------------------------------------------------- plots
def plot_confusion(cm, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_reliability(items, path, n_bins=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4.6))
    axes = np.atleast_1d(axes)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for ax, (title, probs, labels) in zip(axes, items):
        conf = probs.max(1)
        correct = probs.argmax(1) == labels
        accs = [correct[(conf > lo) & (conf <= hi)].mean() if ((conf > lo) & (conf <= hi)).any() else 0
                for lo, hi in zip(edges[:-1], edges[1:])]
        ax.bar(centers, accs, width=1 / n_bins, edgecolor="black", alpha=0.8, label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE = {ece_score(probs, labels):.4f}")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--train_dir", default=None, help="enables the train/test duplicate check")
    ap.add_argument("--ood_dir", default=None, help="optional external dataset with the same class folder names")
    ap.add_argument("--model", default=os.path.join("models", "mobilenet_v2_brain_mri.pth"))
    ap.add_argument("--classes", default=os.path.join("models", "class_names.json"))
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--dup_threshold", type=int, default=4, help="max Hamming distance (of 64) counted as near-duplicate")
    args = ap.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    os.makedirs(args.out, exist_ok=True)
    with open(args.classes, encoding="utf-8") as f:
        names = json.load(f)
    results = {}

    # 1. Test performance
    test_paths, y = list_images(args.test_dir, names)
    if len(test_paths) == 0:
        raise SystemExit(f"No images found in {args.test_dir} (expected sub-folders {names}).")
    print(f"Test images: {len(test_paths)}")
    model = load_model(args.model, len(names))
    logits = get_logits(model, test_paths)
    np.save(os.path.join(args.out, "test_logits.npy"), logits)
    pred = logits.argmax(1)
    results["test"] = {
        "n_images": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class": classification_report(y, pred, target_names=names, output_dict=True, digits=4),
    }
    cm = confusion_matrix(y, pred, labels=range(len(names)))
    plot_confusion(cm, names, os.path.join(args.out, "confusion_matrix.png"))
    with open(os.path.join(args.out, "classification_report.txt"), "w") as f:
        f.write(classification_report(y, pred, target_names=names, digits=4))

    # 2. Calibration with temperature scaling
    cal_idx, ev_idx = stratified_halves(y)
    T = fit_temperature(logits[cal_idx], y[cal_idx])
    before = calib_metrics(logits[ev_idx], y[ev_idx], 1.0)
    after = calib_metrics(logits[ev_idx], y[ev_idx], T)
    results["calibration"] = {"temperature": T, "n_fit": int(len(cal_idx)), "n_eval": int(len(ev_idx)),
                              "full_test_ECE_uncalibrated": ece_score(softmax(logits, 1), y),
                              "before": before, "after": after}
    plot_reliability([("Before temperature scaling", softmax(logits[ev_idx], 1), y[ev_idx]),
                      (f"After temperature scaling (T = {T:.2f})", softmax(logits[ev_idx] / T, 1), y[ev_idx])],
                     os.path.join(args.out, "reliability_diagram.png"))

    # 3. Leakage check
    if args.train_dir:
        train_paths, _ = list_images(args.train_dir, names)
        print(f"Hashing {len(train_paths)} training and {len(test_paths)} test images for duplicate check...")
        train_h = np.array([dhash(p) for p in train_paths], dtype=">u8")
        test_h = np.array([dhash(p) for p in test_paths], dtype=">u8")
        d = min_hamming(test_h, train_h)
        dup = d <= args.dup_threshold
        clean = ~dup
        results["leakage"] = {
            "threshold_hamming": args.dup_threshold,
            "n_test_near_duplicates_of_train": int(dup.sum()),
            "fraction_near_duplicates": float(dup.mean()),
            "accuracy_on_duplicates": float((pred[dup] == y[dup]).mean()) if dup.any() else None,
            "accuracy_on_clean_subset": float((pred[clean] == y[clean]).mean()) if clean.any() else None,
            "macro_f1_on_clean_subset": float(f1_score(y[clean], pred[clean], average="macro")) if clean.any() else None,
            "n_clean": int(clean.sum()),
        }
        with open(os.path.join(args.out, "near_duplicates.txt"), "w") as f:
            for p, dist in zip(np.array(test_paths)[dup], d[dup]):
                f.write(f"{dist}\t{p}\n")

    # 4. Out-of-distribution test
    if args.ood_dir:
        ood_paths, y_ood = list_images(args.ood_dir, names)
        if ood_paths:
            print(f"OOD images: {len(ood_paths)}")
            lo = get_logits(model, ood_paths)
            p_ood = softmax(lo, 1)
            results["ood"] = {
                "n_images": int(len(y_ood)),
                "classes_present": sorted({names[i] for i in y_ood}),
                "accuracy": float((lo.argmax(1) == y_ood).mean()),
                "macro_f1": float(f1_score(y_ood, lo.argmax(1), average="macro")),
                "mean_confidence": float(p_ood.max(1).mean()),
                "ECE_uncalibrated": ece_score(p_ood, y_ood),
                "ECE_temperature_scaled": ece_score(softmax(lo / T, 1), y_ood),
                "in_distribution_mean_confidence": float(softmax(logits, 1).max(1).mean()),
            }
            plot_reliability([("OOD, uncalibrated", p_ood, y_ood),
                              (f"OOD, temperature scaled (T = {T:.2f})", softmax(lo / T, 1), y_ood)],
                             os.path.join(args.out, "ood_reliability_diagram.png"))
        else:
            print(f"No OOD images found in {args.ood_dir} with class folders {names}; skipped.")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_summary(results, names, os.path.join(args.out, "summary.md"))
    print(open(os.path.join(args.out, "summary.md"), encoding="utf-8").read())
    print(f"All results saved in: {os.path.abspath(args.out)}")


def write_summary(r, names, path):
    t, c = r["test"], r["calibration"]
    L = ["# Evaluation summary", "",
         f"**Test set:** {t['n_images']} images", f"- Accuracy: {t['accuracy']*100:.2f}%",
         f"- Macro F1: {t['macro_f1']:.4f}", "", "| Class | Precision | Recall | F1 |", "|---|---|---|---|"]
    for n in names:
        if n in t["per_class"]:
            pc = t["per_class"][n]
            L.append(f"| {n} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1-score']:.4f} |")
    b, a = c["before"], c["after"]
    L += ["", f"**Calibration** (T fitted on {c['n_fit']} images, evaluated on the other {c['n_eval']})",
          f"- Fitted temperature T = {c['temperature']:.3f}",
          f"- ECE: {b['ECE']:.4f} -> {a['ECE']:.4f}",
          f"- NLL: {b['NLL']:.4f} -> {a['NLL']:.4f}",
          f"- Brier: {b['Brier']:.4f} -> {a['Brier']:.4f}",
          f"- Mean confidence: {b['mean_confidence']*100:.2f}% -> {a['mean_confidence']*100:.2f}% "
          f"(accuracy on this half: {b['accuracy']*100:.2f}%)"]
    if "leakage" in r:
        k = r["leakage"]
        L += ["", "**Train/test leakage check**",
              f"- Near-duplicate test images (Hamming <= {k['threshold_hamming']}/64): "
              f"{k['n_test_near_duplicates_of_train']} ({k['fraction_near_duplicates']*100:.1f}%)"]
        if k["accuracy_on_clean_subset"] is not None:
            L.append(f"- Accuracy on the {k['n_clean']} clean test images: {k['accuracy_on_clean_subset']*100:.2f}% "
                     f"(macro F1 {k['macro_f1_on_clean_subset']:.4f})")
    if "ood" in r:
        o = r["ood"]
        L += ["", f"**Out-of-distribution test:** {o['n_images']} images ({', '.join(o['classes_present'])})",
              f"- Accuracy: {o['accuracy']*100:.2f}% (vs {t['accuracy']*100:.2f}% in-distribution)",
              f"- Mean confidence: {o['mean_confidence']*100:.2f}% (vs {o['in_distribution_mean_confidence']*100:.2f}% in-distribution)",
              f"- ECE: {o['ECE_uncalibrated']:.4f} uncalibrated, {o['ECE_temperature_scaled']:.4f} temperature scaled"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
