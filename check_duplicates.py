"""
Follow-up leakage analysis (no PyTorch needed; reuses eval_results/test_logits.npy).

- Finds the closest training image for every test image (perceptual dHash)
- Confirms matches at the pixel level (correlation of 64x64 grayscale versions)
- Reports accuracy, macro F1, glioma recall and ECE on the clean subset for several thresholds
- Saves image montages of matched pairs so you can check them by eye

Usage (same folders as evaluate_model.py):
  python check_duplicates.py --train_dir "PATH/Training" --test_dir "PATH/Testing"
"""
import os
import json
import argparse

import numpy as np
from PIL import Image
from scipy.special import softmax

from evaluate_model import list_images, dhash, ece_score


def bits(h):
    return np.unpackbits(h.astype(">u8").view(np.uint8).reshape(-1, 8), axis=1).astype(bool)


def nearest(test_h, train_h, chunk=256):
    tb_all, trb = bits(test_h), bits(train_h)
    dist = np.empty(len(test_h), int)
    idx = np.empty(len(test_h), int)
    for i in range(0, len(test_h), chunk):
        d = (tb_all[i:i + chunk, None, :] ^ trb[None]).sum(2)
        idx[i:i + chunk] = d.argmin(1)
        dist[i:i + chunk] = d.min(1)
    return dist, idx


def small(path, s=64):
    return np.asarray(Image.open(path).convert("L").resize((s, s), Image.BILINEAR), dtype=np.float64).ravel()


def pixel_corr(a, b):
    a, b = small(a), small(b)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 1.0


def montage(pairs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(pairs)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(4.2, 2.1 * n))
    axes = np.atleast_2d(axes)
    for r, (tp, trp, d, c) in enumerate(pairs):
        for k, (p, lab) in enumerate(((tp, "TEST"), (trp, "TRAIN"))):
            axes[r, k].imshow(Image.open(p).convert("L"), cmap="gray")
            axes[r, k].set_title(f"{lab}: {os.path.basename(p)}", fontsize=7)
            axes[r, k].axis("off")
        axes[r, 0].text(0, -0.25, f"hash dist {d}, pixel corr {c:.3f}", transform=axes[r, 0].transAxes, fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--classes", default=os.path.join("models", "class_names.json"))
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--corr", type=float, default=0.95, help="pixel correlation needed to confirm a duplicate")
    args = ap.parse_args()

    from sklearn.metrics import f1_score, recall_score

    names = json.load(open(args.classes, encoding="utf-8"))
    test_paths, y = list_images(args.test_dir, names)
    train_paths, _ = list_images(args.train_dir, names)
    logits = np.load(os.path.join(args.out, "test_logits.npy"))
    assert len(logits) == len(test_paths), "test_logits.npy does not match the test folder; rerun evaluate_model.py"
    pred = logits.argmax(1)
    probs = softmax(logits, 1)
    g = names.index("glioma") if "glioma" in names else 0

    print(f"Hashing {len(train_paths)} training and {len(test_paths)} test images...")
    train_h = np.array([dhash(p) for p in train_paths], dtype=">u8")
    test_h = np.array([dhash(p) for p in test_paths], dtype=">u8")
    dist, idx = nearest(test_h, train_h)

    print("Checking candidate pairs at the pixel level...")
    corr = np.full(len(test_paths), np.nan)
    for i in np.where(dist <= 8)[0]:
        corr[i] = pixel_corr(test_paths[i], train_paths[idx[i]])

    def stats(mask):
        m = ~mask
        return {
            "n_flagged": int(mask.sum()),
            "n_clean": int(m.sum()),
            "clean_accuracy": float((pred[m] == y[m]).mean()),
            "clean_macro_f1": float(f1_score(y[m], pred[m], average="macro")),
            "clean_glioma_recall": float(recall_score(y[m] == g, pred[m] == g)),
            "clean_ECE": ece_score(probs[m], y[m]),
            "clean_class_counts": {n: int((y[m] == i).sum()) for i, n in enumerate(names)},
            "flagged_accuracy": float((pred[mask] == y[mask]).mean()) if mask.any() else None,
        }

    report = {"full_test": {"accuracy": float((pred == y).mean()),
                            "glioma_recall": float(recall_score(y == g, pred == g)),
                            "ECE": ece_score(probs, y)},
              "hash_thresholds": {}, "confirmed": {}}
    for t in (0, 2, 4, 6, 8):
        report["hash_thresholds"][str(t)] = stats(dist <= t)
    confirmed = (dist <= 8) & (corr >= args.corr)
    report["confirmed"] = {"rule": f"hash distance <= 8 and pixel correlation >= {args.corr}", **stats(confirmed)}
    cross = confirmed & (np.array([names.index(os.path.basename(os.path.dirname(train_paths[j]))) for j in idx]) != y)
    report["confirmed"]["pairs_with_different_labels"] = int(cross.sum())

    with open(os.path.join(args.out, "duplicates_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "duplicate_pairs.tsv"), "w", encoding="utf-8") as f:
        f.write("hash_dist\tpixel_corr\ttest\ttrain\n")
        for i in np.argsort(dist):
            if dist[i] <= 8:
                f.write(f"{dist[i]}\t{corr[i]:.4f}\t{test_paths[i]}\t{train_paths[idx[i]]}\n")

    rng = np.random.default_rng(0)
    for lo, hi, name in ((0, 0, "pairs_dist0.png"), (3, 4, "pairs_dist3-4.png"), (5, 8, "pairs_dist5-8.png")):
        cand = np.where((dist >= lo) & (dist <= hi))[0]
        pick = rng.choice(cand, size=min(8, len(cand)), replace=False) if len(cand) else []
        montage([(test_paths[i], train_paths[idx[i]], dist[i], corr[i]) for i in pick],
                os.path.join(args.out, name), f"Random test/train pairs, hash distance {lo}-{hi}")

    c = report["confirmed"]
    print(f"\nFull test set: accuracy {report['full_test']['accuracy']*100:.2f}%, "
          f"glioma recall {report['full_test']['glioma_recall']*100:.2f}%, ECE {report['full_test']['ECE']:.4f}")
    print("\nHash threshold | flagged | clean acc | clean macro F1 | clean glioma recall | clean ECE")
    for t, s in report["hash_thresholds"].items():
        print(f"  <= {t:<10} | {s['n_flagged']:7d} | {s['clean_accuracy']*100:8.2f}% | {s['clean_macro_f1']:.4f}"
              f"         | {s['clean_glioma_recall']*100:6.2f}%             | {s['clean_ECE']:.4f}")
    print(f"\nConfirmed duplicates ({c['rule']}): {c['n_flagged']} "
          f"({c['n_flagged']/len(y)*100:.1f}% of test), {c['pairs_with_different_labels']} with a different label")
    print(f"Clean subset ({c['n_clean']} images): accuracy {c['clean_accuracy']*100:.2f}%, "
          f"macro F1 {c['clean_macro_f1']:.4f}, glioma recall {c['clean_glioma_recall']*100:.2f}%, ECE {c['clean_ECE']:.4f}")
    print(f"Clean class counts: {c['clean_class_counts']}")
    print(f"\nSaved duplicates_report.json, duplicate_pairs.tsv and pair images in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
