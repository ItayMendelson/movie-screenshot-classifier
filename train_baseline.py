import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import CLASSES, ScreenshotDataset, build_splits

BATCH_SIZE = 32
IMAGE_SIZE = 128


class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features.flatten(1))


def build_dataloaders():
    """Build training and validation loaders while leaving test data untouched."""
    train_samples, val_samples, _ = build_splits()
    train_transform = v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    val_transform = v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    train_loader = DataLoader(
        ScreenshotDataset(train_samples, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        ScreenshotDataset(val_samples, val_transform),
        batch_size=BATCH_SIZE,
    )
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, loss_fn, device, max_batches=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            if max_batches is not None and batch_index >= max_batches:
                break
    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train the tiny CNN baseline.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Train without saving baseline_model.pt.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one training batch and one validation batch without saving.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.accelerator.current_accelerator() or torch.device("cpu")
    train_loader, val_loader = build_dataloaders()
    model = TinyCNN(len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 1 if args.smoke_test else args.epochs
    max_batches = 1 if args.smoke_test else None
    print(
        f"device={device} train_samples={len(train_loader.dataset)} "
        f"val_samples={len(val_loader.dataset)}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            max_batches,
        )
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            max_batches,
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.3f} "
            f"train_acc={train_accuracy:.3f} val_loss={val_loss:.3f} "
            f"val_acc={val_accuracy:.3f}",
            flush=True,
        )

    if args.smoke_test or args.no_save:
        print("training complete; model not saved")
    else:
        torch.save(model.state_dict(), "baseline_model.pt")
        print("saved baseline_model.pt")


if __name__ == "__main__":
    main()
