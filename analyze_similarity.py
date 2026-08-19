"""Score similarity between neighboring movie frames to help pick a deduplication threshold."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path("frames")
HASH_SIZE = 16
HISTOGRAM_BINS = (16, 16, 16)
BLACK_BORDER_THRESHOLD = 8
MIN_BORDER_FRACTION = 0.02
BORDER_SYMMETRY_TOLERANCE = 0.05


@dataclass(frozen=True)
class FrameFeatures:
    path: Path
    perceptual_hash: np.ndarray = field(repr=False, compare=False)
    histogram: np.ndarray | None = field(repr=False, compare=False)


@dataclass(frozen=True)
class FramePair:
    movie: str
    first: Path
    second: Path
    frame_gap: int
    perceptual_similarity: float
    histogram_correlation: float
    first_hash: np.ndarray = field(repr=False, compare=False)
    second_hash: np.ndarray = field(repr=False, compare=False)

    @property
    def metric_disagreement(self):
        """Return the difference between the two similarity measurements."""
        bounded_histogram = np.clip(self.histogram_correlation, 0.0, 1.0)
        return abs(self.perceptual_similarity - bounded_histogram)


def frame_number(path):
    """Extract the numeric part of a frame filename such as frame_0042.jpg."""
    return int(path.stem.removeprefix("frame_"))


def crop_black_borders(image):
    """Remove continuous black bars around the outer edge of an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_pixels = gray > BLACK_BORDER_THRESHOLD
    row_has_content = bright_pixels.mean(axis=1) >= 0.05
    col_has_content = bright_pixels.mean(axis=0) >= 0.05
    height, width = gray.shape

    def content_bounds(has_content, length):
        content = np.flatnonzero(has_content)
        start, end = content[[0, -1]]
        leading_border = start
        trailing_border = length - 1 - end
        borders_are_large = (
            min(leading_border, trailing_border)
            >= length * MIN_BORDER_FRACTION
        )
        borders_are_symmetric = (
            abs(leading_border - trailing_border)
            <= length * BORDER_SYMMETRY_TOLERANCE
        )
        if borders_are_large and borders_are_symmetric:
            return start, end
        return 0, length - 1

    top, bottom = content_bounds(row_has_content, height)
    left, right = content_bounds(col_has_content, width)
    return image[top : bottom + 1, left : right + 1]


def perceptual_hash(image):
    """Represent the image using the direction of brightness changes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thumbnail = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE))
    return thumbnail[:, 1:] > thumbnail[:, :-1]


def hash_similarity(first_hash, second_hash):
    """Return the fraction of matching bits in two perceptual hashes."""
    changed_bits = np.count_nonzero(first_hash != second_hash)
    return 1.0 - changed_bits / first_hash.size


def hsv_histogram(image):
    """Return a normalized histogram containing hue, saturation, and brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        HISTOGRAM_BINS,
        [0, 180, 0, 256, 0, 256],
    )
    return cv2.normalize(histogram, histogram).flatten()


def load_features(path, include_histogram=True):
    """Load one frame and calculate the features used for comparison."""
    image = crop_black_borders(cv2.imread(str(path)))
    return FrameFeatures(
        path=path,
        perceptual_hash=perceptual_hash(image),
        histogram=hsv_histogram(image) if include_histogram else None,
    )


def compare_features(first, second, movie):
    """Measure perceptual and color similarity for two loaded frames."""
    histogram_correlation = (
        cv2.compareHist(
            first.histogram,
            second.histogram,
            cv2.HISTCMP_CORREL,
        )
        if first.histogram is not None
        else float("nan")
    )
    return FramePair(
        movie=movie,
        first=first.path,
        second=second.path,
        frame_gap=frame_number(second.path) - frame_number(first.path),
        perceptual_similarity=hash_similarity(
            first.perceptual_hash, second.perceptual_hash
        ),
        histogram_correlation=histogram_correlation,
        first_hash=first.perceptual_hash,
        second_hash=second.perceptual_hash,
    )


def analyze_movie(movie_dir, include_histogram=True):
    """Score every neighboring pair of surviving frames in one movie."""
    frames = sorted(movie_dir.glob("*.jpg"))
    first = load_features(frames[0], include_histogram)
    pairs = []
    for path in frames[1:]:
        second = load_features(path, include_histogram)
        pairs.append(compare_features(first, second, movie_dir.name))
        first = second
    return pairs


def print_pairs(title, pairs):
    """Print a short, readable list of frame-pair scores."""
    print(f"\n{title}")
    for pair in pairs:
        print(
            f"  {pair.first.name} -> {pair.second.name} "
            f"(gap={pair.frame_gap}, perceptual={pair.perceptual_similarity:.3f}, "
            f"histogram={pair.histogram_correlation:.3f})"
        )


def build_groups(pairs, threshold):
    """Preview groups using neighboring and group-anchor similarity."""
    groups = [[pairs[0].first]]
    anchor_hash = pairs[0].first_hash
    for pair in pairs:
        anchor_similarity = hash_similarity(anchor_hash, pair.second_hash)
        starts_new_group = (
            pair.frame_gap > 1
            or pair.perceptual_similarity < threshold
            or anchor_similarity < threshold
        )
        if starts_new_group:
            groups.append([])
            anchor_hash = pair.second_hash
        groups[-1].append(pair.second)
    return groups


def print_threshold_report(pairs, threshold, limit):
    """Show grouping statistics and pairs nearest a candidate threshold."""
    groups = build_groups(pairs, threshold)
    sizes = [len(group) for group in groups]
    largest = max(groups, key=len)
    singleton_fraction = sum(size == 1 for size in sizes) / len(sizes)
    consecutive_pairs = [pair for pair in pairs if pair.frame_gap == 1]
    below = sorted(
        (
            pair
            for pair in consecutive_pairs
            if pair.perceptual_similarity < threshold
        ),
        key=lambda pair: pair.perceptual_similarity,
        reverse=True,
    )[:limit]
    above = sorted(
        (
            pair
            for pair in consecutive_pairs
            if pair.perceptual_similarity >= threshold
        ),
        key=lambda pair: pair.perceptual_similarity,
    )[:limit]

    print(f"\nGrouping preview at perceptual threshold {threshold:.3f}")
    print(
        f"  groups={len(groups)}, median_size={np.median(sizes):.1f}, "
        f"singletons={singleton_fraction:.1%}, largest_size={len(largest)}"
    )
    print(
        f"  largest group: {largest[0].name} -> {largest[-1].name}"
    )
    if limit:
        print_pairs("Closest pairs below the threshold", below)
        print_pairs("Closest pairs at or above the threshold", above)


def print_report(movie, pairs, limit, threshold):
    """Print the pairs most useful for visually checking both metrics."""
    print(f"\n=== {movie}: {len(pairs)} neighboring pairs ===")
    if limit:
        print_pairs(
            "Most perceptually similar",
            sorted(
                pairs,
                key=lambda pair: pair.perceptual_similarity,
                reverse=True,
            )[:limit],
        )
        print_pairs(
            "Most perceptually different",
            sorted(pairs, key=lambda pair: pair.perceptual_similarity)[:limit],
        )
        print_pairs(
            "Largest disagreement between perceptual and histogram scores",
            sorted(
                pairs,
                key=lambda pair: pair.metric_disagreement,
                reverse=True,
            )[:limit],
        )
    print_threshold_report(pairs, threshold, limit)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare consecutive movie frames without modifying the dataset."
    )
    parser.add_argument(
        "--movie",
        help="Analyze one movie folder. By default, all movie folders are analyzed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of pairs to show in each report section.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Candidate perceptual threshold used to preview frame groups.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    movie_dirs = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())
    if args.movie:
        movie_dirs = [path for path in movie_dirs if path.name == args.movie]
        if not movie_dirs:
            raise ValueError(f"Movie folder not found: {args.movie}")

    for movie_dir in movie_dirs:
        pairs = analyze_movie(movie_dir)
        print_report(movie_dir.name, pairs, args.limit, args.threshold)


if __name__ == "__main__":
    main()
