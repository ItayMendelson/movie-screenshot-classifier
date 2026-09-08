import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

from dataset import CLASSES, ScreenshotDataset, build_splits

BATCH_SIZE = 32
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_dataloaders():
    """Build training and validation loaders while leaving test data untouched."""
    train_samples, val_samples, _ = build_splits()
    normalize = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_transform = v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    val_transform = v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
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


def build_model(weights=ResNet18_Weights.DEFAULT):
    """Build ResNet18 with a frozen backbone and trainable classifier."""
    model = resnet18(weights=weights)
    model.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def train_epoch(model, loader, optimizer, loss_fn, device, max_batches=None):
    model.eval()
    model.fc.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
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
        if max_batches is not None and batch_index >= max_batches:
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
    parser = argparse.ArgumentParser(description="Train a frozen ResNet18 classifier.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Train without saving checkpoints/transfer_model.pt.",
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
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
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
        torch.save(model.state_dict(), "checkpoints/transfer_model.pt")
        print("saved checkpoints/transfer_model.pt")


if __name__ == "__main__":
    main()
