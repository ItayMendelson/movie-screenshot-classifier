"""Verify train/val/test splits don't leak near-duplicate or adjacent frames across splits."""

from collections import Counter

from analyze_similarity import frame_number
from dataset import CLASSES, DATA_DIR, build_split_plan


def build_path_map(split_samples):
    """Map every retained frame path to its assigned split."""
    path_to_split = {}
    for split, samples in split_samples.items():
        for path, _ in samples:
            if path in path_to_split:
                raise AssertionError(f"Frame appears in two splits: {path}")
            path_to_split[path] = split
    return path_to_split


def audit_group_integrity(path_to_split, groups_by_class):
    """Verify that each retained similarity group belongs to at most one split."""
    violations = []
    for cls, groups in groups_by_class.items():
        for group in groups:
            splits = {
                path_to_split[str(path)]
                for path in group
                if str(path) in path_to_split
            }
            if len(splits) > 1:
                violations.append((cls, group[0].name, group[-1].name, splits))
    return violations


def audit_neighbor_leakage(path_to_split):
    """Find retained consecutive frames assigned to different splits."""
    violations = []
    for cls in CLASSES:
        frames = sorted((DATA_DIR / cls).glob("*.jpg"))
        for first, second in zip(frames, frames[1:]):
            if frame_number(second) - frame_number(first) != 1:
                continue
            first_split = path_to_split.get(str(first))
            second_split = path_to_split.get(str(second))
            if first_split and second_split and first_split != second_split:
                violations.append(
                    (cls, first.name, second.name, first_split, second_split)
                )
    return violations


def print_class_counts(split_samples):
    """Print retained sample counts and percentages for every movie."""
    counters = {
        split: Counter(label for _, label in samples)
        for split, samples in split_samples.items()
    }
    print("movie | train | val | test | retained")
    for label, cls in enumerate(CLASSES):
        counts = [counters[split][label] for split in ("train", "val", "test")]
        total = sum(counts)
        percentages = [count / total for count in counts]
        print(
            f"{cls} | {counts[0]} ({percentages[0]:.1%}) | "
            f"{counts[1]} ({percentages[1]:.1%}) | "
            f"{counts[2]} ({percentages[2]:.1%}) | {total}"
        )

    totals = {
        split: len(samples) for split, samples in split_samples.items()
    }
    retained = sum(totals.values())
    print(
        f"overall | {totals['train']} ({totals['train'] / retained:.1%}) | "
        f"{totals['val']} ({totals['val'] / retained:.1%}) | "
        f"{totals['test']} ({totals['test'] / retained:.1%}) | {retained}"
    )


def main():
    split_samples, groups_by_class, omitted_paths = build_split_plan()
    path_to_split = build_path_map(split_samples)
    expected_paths = {
        str(path)
        for groups in groups_by_class.values()
        for group in groups
        for path in group
    }
    unassigned_paths = expected_paths - path_to_split.keys()
    group_violations = audit_group_integrity(path_to_split, groups_by_class)
    neighbor_violations = audit_neighbor_leakage(path_to_split)

    print_class_counts(split_samples)
    print(f"\nretained frames: {len(path_to_split)}")
    print(f"guard frames omitted: {len(omitted_paths)}")
    print(f"split overlaps: 0")
    print(f"split similarity groups: {len(group_violations)}")
    print(f"consecutive cross-split pairs: {len(neighbor_violations)}")

    if {str(path) for path in omitted_paths} != unassigned_paths:
        raise AssertionError("Unexpected unassigned frames found")
    if group_violations:
        raise AssertionError(f"Similarity groups cross splits: {group_violations[:5]}")
    if neighbor_violations:
        raise AssertionError(
            f"Consecutive frames cross splits: {neighbor_violations[:5]}"
        )


if __name__ == "__main__":
    main()
