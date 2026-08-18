import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import kagglehub

DATASET = "asaniczka/movie-identification-dataset-800-movies/versions/1"
DATASET_ROOT = "resized_frames"
DOWNLOAD_PREFIX = "movie_dataset_"

# exact folder names as they appear in the dataset
MOVIES = [
    "Zootopia (2016)",
    "Spider-Man Into The Spider-Verse (2018)",
    "The Grand Budapest Hotel (2014)",
    "La La Land (2016)",
    "The Lighthouse (2019)",
    "Her (2013)",
    "Interstellar (2014)",
    "The Matrix (1999)",
    "Fantastic Mr Fox (2009)",
    "Whiplash (2014)",
]


def remove_stale_downloads():
    """Remove temporary datasets left by interrupted downloads."""
    for path in Path(".").glob(f"{DOWNLOAD_PREFIX}*"):
        if path.is_dir():
            shutil.rmtree(path)


def main():
    remove_stale_downloads()

    frames_root = Path("frames")
    frames_root.mkdir(exist_ok=True)

    with TemporaryDirectory(prefix=DOWNLOAD_PREFIX, dir=".") as temporary_directory:
        dataset_path = Path(kagglehub.dataset_download(DATASET, output_dir=temporary_directory))
        source_root = dataset_path / DATASET_ROOT

        for movie in MOVIES:
            source = source_root / movie
            destination = frames_root / movie
            shutil.copytree(source, destination, dirs_exist_ok=True)
            print(f"{movie}: ready at {destination}")

    print("Temporary dataset deleted. Only the selected movies remain in frames.")


if __name__ == "__main__":
    main()
