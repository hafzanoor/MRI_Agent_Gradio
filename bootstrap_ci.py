"""
Bootstrap 95% confidence intervals for the brain MRI classifier's test metrics.

Uses the saved test-set logits (eval_results/test_logits.npy), so the model does not need to be re-run.

  Full test set only (no data folder needed):
      python bootstrap_ci.py
  Full test set and the clean (de-duplicated) subset:
      python bootstrap_ci.py --test_dir "PATH/Testing"

Metrics: accuracy, macro F1, glioma recall and Expected Calibration Error (ECE).
Results are written to eval_results/bootstrap_ci.json and printed as a Markdown table.
"""
import os
import csv
import json
import argparse

import numpy as np
from scipy.special import softmax

from evaluate_model import ece_score, list_images


def macro_f1(y, p, k):
    f1 = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        denom = 2 * tp + fp + fn
        f1.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1))


def metrics(y, probs, glioma_idx):
    p = probs.argmax(1)
    g = y == glioma_idx
    return {
        "accuracy": float((p == y).mean()),
        "macro_f1": macro_f1(y, p, probs.shape[1]),
        "glioma_recall": float((p[g] == glioma_idx).mean()) if g.any() else float("nan"),
        "ece": ece_score(probs, y),
    }


def bootstrap(y, probs, glioma_idx, n_boot=2000, seed=0):
    """Point estimate and 95% percentile interval for each metric (resampling images with replacement)."""
    rng = np.random.default_rng(seed)
    point = metrics(y, probs, glioma_idx)
    samples = {k: [] for k in point}
    n = len(y)
    for _ in range(n_boot):
        b = rng.integers(0, n, size=n)
        for k, v in metrics(y[b], probs[b], glioma_idx).items():
            samples[k].append(v)
    return {k: {"value": point[k],
                "ci_low": float(np.percentile(samples[k], 2.5)),
                "ci_high": float(np.percentile(samples[k], 97.5))} for k in point}


def labels_from_metrics(results_dir, names):
    """Rebuild test labels from class supports: evaluate_model.py lists images class by class."""
    rep = json.load(open(os.path.join(results_dir, "metrics.json")))["test"]["per_class"]
    return np.concatenate([np.full(int(rep[c]["support"]), i) for i, c in enumerate(names)])


def clean_mask(test_paths, pairs_file, max_dist=8, min_corr=0.95):
    """True for test images that are NOT confirmed duplicates of a training image."""
    dup = set()
    with open(pairs_file, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if int(r["hash_dist"]) <= max_dist and float(r["pixel_corr"]) >= min_corr:
                parts = r["test"].replace("\\", "/").split("/")
                dup.add("/".join(parts[-2:]))
    keys = ["/".join(p.replace("\\", "/").split("/")[-2:]) for p in test_paths]
    return np.array([k not in dup for k in keys])


def fmt(m, k, pct):
    v = m[k]
    if pct:
        return f"{v['value']*100:.1f}% ({v['ci_low']*100:.1f}–{v['ci_high']*100:.1f})"
    return f"{v['value']:.3f} ({v['ci_low']:.3f}–{v['ci_high']:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="eval_results")
    ap.add_argument("--classes", default=os.path.join("models", "class_names.json"))
    ap.add_argument("--test_dir", default=None, help="needed only for the clean-subset intervals")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    names = json.load(open(args.classes, encoding="utf-8"))
    logits = np.load(os.path.join(args.results_dir, "test_logits.npy"))
    probs = softmax(logits, axis=1)
    g = names.index("glioma")

    if args.test_dir:
        paths, y = list_images(args.test_dir, names)
    else:
        paths, y = None, labels_from_metrics(args.results_dir, names)
    assert len(y) == len(probs), "labels and logits do not match; check --test_dir"

    out = {"n_boot": args.n_boot, "full": bootstrap(y, probs, g, args.n_boot)}
    out["full"]["n"] = int(len(y))
    if paths is not None:
        m = clean_mask(paths, os.path.join(args.results_dir, "duplicate_pairs.tsv"))
        out["clean"] = bootstrap(y[m], probs[m], g, args.n_boot)
        out["clean"]["n"] = int(m.sum())

    with open(os.path.join(args.results_dir, "bootstrap_ci.json"), "w") as f:
        json.dump(out, f, indent=2)

    cols = [("full", "Full test set")] + ([("clean", "Clean subset")] if "clean" in out else [])
    print("| Metric | " + " | ".join(f"{t} (n = {out[k]['n']:,})" for k, t in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for key, label, pct in [("accuracy", "Accuracy", True), ("macro_f1", "Macro F1", False),
                            ("glioma_recall", "Glioma recall", True), ("ece", "ECE", False)]:
        print(f"| {label} | " + " | ".join(fmt(out[k], key, pct) for k, _ in cols) + " |")
    print(f"\nValues are point estimates with 95% bootstrap intervals ({args.n_boot} resamples).")


if __name__ == "__main__":
    main()
