import random
from pathlib import Path

from analyze_similarity import analyze_movie, build_groups, frame_number
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = Path("frames")
CLASSES = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
SIMILARITY_THRESHOLD = 0.5
TARGET_BLOCK_FRAMES = 45
GUARD_FRAMES = 1


def pack_groups(groups, target_frames=TARGET_BLOCK_FRAMES):
    """Pack consecutive similarity groups into larger temporal blocks."""
    blocks = []
    current = []
    for group in groups:
        if current and len(current) >= target_frames:
            blocks.append(current)
            current = []
        current.extend(group)

    if current:
        if blocks and len(current) < target_frames / 2:
            blocks[-1].extend(current)
        else:
            blocks.append(current)
    return blocks


def take_nearest_target(block_indices, blocks, target_frames):
    """Take shuffled whole blocks until their size is nearest a frame target."""
    selected = []
    selected_frames = 0
    while block_indices:
        next_size = len(blocks[block_indices[0]])
        current_distance = abs(target_frames - selected_frames)
        next_distance = abs(target_frames - selected_frames - next_size)
        if selected and current_distance <= next_distance:
            break
        selected.append(block_indices.pop(0))
        selected_frames += next_size
    return selected


def assign_blocks(blocks, val_fraction, test_fraction, rng):
    """Assign whole temporal blocks to train, validation, or test."""
    block_indices = list(range(len(blocks)))
    rng.shuffle(block_indices)
    total_frames = sum(len(block) for block in blocks)
    val_target = total_frames * val_fraction
    test_target = total_frames * test_fraction
    val_blocks = set(take_nearest_target(block_indices, blocks, val_target))
    test_blocks = set(take_nearest_target(block_indices, blocks, test_target))
    return {
        index: (
            "val"
            if index in val_blocks
            else "test"
            if index in test_blocks
            else "train"
        )
        for index in range(len(blocks))
    }


def guard_paths(blocks, assignments, guard_frames=GUARD_FRAMES):
    """Return frames to omit beside boundaries assigned to different splits."""
    omitted = set()
    for index, (left, right) in enumerate(zip(blocks, blocks[1:])):
        if assignments[index] == assignments[index + 1]:
            continue
        if frame_number(right[0]) - frame_number(left[-1]) > 1:
            continue
        omitted.update(left[-guard_frames:])
        omitted.update(right[:guard_frames])
    return omitted


def build_split_plan(val_fraction=0.15, test_fraction=0.15, seed=0):
    """Build samples plus the grouping details used to audit the split."""
    samples = {"train": [], "val": [], "test": []}
    groups_by_class = {}
    omitted_paths = set()
    rng = random.Random(seed)

    for label_idx, cls in enumerate(CLASSES):
        movie_dir = DATA_DIR / cls
        groups = build_groups(
            analyze_movie(movie_dir, include_histogram=False),
            threshold=SIMILARITY_THRESHOLD,
        )
        groups_by_class[cls] = groups
        blocks = pack_groups(groups)
        assignments = assign_blocks(blocks, val_fraction, test_fraction, rng)
        omitted = guard_paths(blocks, assignments)
        omitted_paths.update(omitted)

        for index, block in enumerate(blocks):
            split = assignments[index]
            samples[split].extend(
                (str(path), label_idx) for path in block if path not in omitted
            )

    return samples, groups_by_class, omitted_paths


def build_splits(val_fraction=0.15, test_fraction=0.15, seed=0):
    """Build deterministic per-movie train, validation, and test splits."""
    samples, _, _ = build_split_plan(val_fraction, test_fraction, seed)
    return samples["train"], samples["val"], samples["test"]

class ScreenshotDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
