import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import CLASSES, ScreenshotDataset, build_splits
from train_baseline import BATCH_SIZE, IMAGE_SIZE, TinyCNN


def build_test_loader():
    """Build the untouched test loader with baseline preprocessing."""
    _, _, test_samples = build_splits()
    transform = v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    dataset = ScreenshotDataset(test_samples, transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE), test_samples


def predict(model, loader, loss_fn, device):
    """Return test loss plus true and predicted labels."""
    total_loss = 0.0
    y_true = []
    y_pred = []

    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return total_loss / len(y_true), y_true, y_pred


def save_confusion_matrix(y_true, y_pred, output_path):
    """Save a confusion matrix containing every movie class."""
    labels = list(range(len(CLASSES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(14, 12))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASSES)
    display.plot(ax=axis, xticks_rotation=45, colorbar=False)
    axis.set_title("Baseline test confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the baseline on the test split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("baseline_epoch_search_best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_runs/baseline_epoch_search_best"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.accelerator.current_accelerator() or torch.device("cpu")
    test_loader, test_samples = build_test_loader()

    model = TinyCNN(len(CLASSES)).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    test_loss, y_true, y_pred = predict(
        model,
        test_loader,
        nn.CrossEntropyLoss(),
        device,
    )
    test_accuracy = sum(true == pred for true, pred in zip(y_true, y_pred)) / len(
        y_true
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "confusion_matrix.png"
    metrics_path = args.output_dir / "metrics.json"
    save_confusion_matrix(y_true, y_pred, matrix_path)
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "test_samples": len(test_samples),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"device={device} test_samples={len(test_samples)}")
    print(f"test_loss={test_loss:.3f} test_acc={test_accuracy:.3f}")
    print(f"saved {matrix_path}")
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
