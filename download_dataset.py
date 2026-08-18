# download_dataset.py
import kagglehub
import shutil
from pathlib import Path
from kagglehub.clients import build_kaggle_client
from kagglesdk.datasets.types.dataset_api_service import ApiListTreeDatasetFilesRequest

DATASET = "asaniczka/movie-identification-dataset-800-movies"
DATASET_VERSION = 1
DATASET_ROOT = "resized_frames"

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

kaggle_client = build_kaggle_client().datasets.dataset_api_client


def list_movie_files(movie):
    """Return every frame filename in one movie directory."""
    filenames = []
    page_token = None

    while True:
        request = ApiListTreeDatasetFilesRequest()
        request.owner_slug = "asaniczka"
        request.dataset_slug = "movie-identification-dataset-800-movies"
        request.dataset_version_number = DATASET_VERSION
        request.path = f"{DATASET_ROOT}/{movie}"
        request.page_size = 200
        request.page_token = page_token

        response = kaggle_client.list_tree_dataset_files(request)
        filenames.extend(file.name for file in response.files)
        page_token = response.next_page_token

        if not page_token:
            return filenames


def download_movie(movie):
    """Download one movie's frames into its local class directory."""
    destination = Path("frames") / movie
    destination.mkdir(parents=True, exist_ok=True)

    filenames = list_movie_files(movie)
    for filename in filenames:
        destination_file = destination / filename
        if destination_file.exists():
            continue

        remote_path = f"{DATASET_ROOT}/{movie}/{filename}"
        downloaded_path = kagglehub.dataset_download(DATASET, path=remote_path)
        shutil.copy2(downloaded_path, destination_file)

    print(f"{movie}: {len(filenames)} frames ready at {destination}")


def main():
    for movie in MOVIES:
        download_movie(movie)



if __name__ == "__main__":
    main()
