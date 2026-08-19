from pathlib import Path

import numpy as np
from PIL import Image

# manually extracted to remove opening credits
START_FRAMES = {
    "Fantastic Mr Fox (2009)": 16,
    "Her (2013)": 6,
    "Interstellar (2014)": 8,
    "La La Land (2016)": 5,
    "Spider-Man Into The Spider-Verse (2018)": 10,
    "The Grand Budapest Hotel (2014)": 10,
    "The Lighthouse (2019)": 13,
    "The Matrix (1999)": 11,
    "Whiplash (2014)": 14,
    "Zootopia (2016)": 10,
}
# manually extracted to remove opening credits, inclusive
END_FRAMES = {
    "Fantastic Mr Fox (2009)": 931,
    "Her (2013)": 933,
    "Interstellar (2014)": 969,
    "La La Land (2016)": 938,
    "Spider-Man Into The Spider-Verse (2018)": 927,
    "The Grand Budapest Hotel (2014)": 937,
    "The Lighthouse (2019)": 970,
    "The Matrix (1999)": 949,
    "Whiplash (2014)": 951,
    "Zootopia (2016)": 921,
}


def is_near_blank(
    img_path, dark_thresh=15, light_thresh=240, std_thresh=8, uniform_thresh=3
):
    """Flag nearly uniform frames or frames that are almost black or white."""
    img = np.array(Image.open(img_path).convert("L"))
    brightness = img.mean()
    variation = img.std()
    is_black_or_white = brightness < dark_thresh or brightness > light_thresh
    return variation < uniform_thresh or (
        is_black_or_white and variation < std_thresh
    )


def is_intro_frame(img_path):
    """Return whether a frame appears before its movie's chosen start."""
    frame_number = int(img_path.stem.removeprefix("frame_"))
    return frame_number < START_FRAMES.get(img_path.parent.name, 0)


def is_credits_frame(img_path):
    """Return whether a frame appears after its movie's chosen end."""
    frame_number = int(img_path.stem.removeprefix("frame_"))
    return frame_number > END_FRAMES.get(img_path.parent.name, frame_number)


if __name__ == "__main__":
    for movie_dir in Path("frames").iterdir():
        if not movie_dir.is_dir():
            continue
        intro_removed = 0
        credits_removed = 0
        blank_removed = 0
        files = list(movie_dir.glob("*.jpg"))
        for f in files:
            if is_intro_frame(f):
                f.unlink()
                intro_removed += 1
            elif is_credits_frame(f):
                f.unlink()
                credits_removed += 1
            elif is_near_blank(f):
                f.unlink()
                blank_removed += 1
        print(
            f"{movie_dir.name}: removed {intro_removed} intro, "
            f"{credits_removed} credits, and {blank_removed} blank "
            f"frames out of {len(files)}"
        )
