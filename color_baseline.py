import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from dataset import build_splits

HIST_BINS = (8, 8, 8)
HIST_RANGES = [0, 180, 0, 256, 0, 256]


def extract_histogram(path):
    """Compute a normalized HSV color histogram feature vector for one frame."""
    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HIST_BINS, HIST_RANGES)
    cv2.normalize(hist, hist)
    return hist.flatten()


def build_features(samples):
    """Extract histogram features and labels for a list of (path, label) samples."""
    features = np.stack([extract_histogram(path) for path, _ in samples])
    labels = np.array([label for _, label in samples])
    return features, labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Color-histogram sanity baseline (no CNN)."
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation_runs/color_baseline")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_samples, val_samples, test_samples = build_splits()

    print(
        f"extracting features: train={len(train_samples)} "
        f"val={len(val_samples)} test={len(test_samples)}",
        flush=True,
    )
    train_x, train_y = build_features(train_samples)
    val_x, val_y = build_features(val_samples)
    test_x, test_y = build_features(test_samples)

    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(train_x, train_y)

    val_accuracy = model.score(val_x, val_y)
    test_accuracy = model.score(test_x, test_y)
    print(f"val_acc={val_accuracy:.3f} test_acc={test_accuracy:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "test_samples": len(test_samples),
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
