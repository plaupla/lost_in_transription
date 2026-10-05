"""Submission template: loop over submission_format.csv and write one transcription per clip."""

from pathlib import Path

import polars as pl
import os


import csv
import time
from pathlib import Path

import ctranslate2
from faster_whisper import WhisperModel

CONTAINER_ROOT = Path("/code_execution")
SCRIPT_DIR = Path(__file__).resolve().parent
IN_CONTAINER = SCRIPT_DIR == CONTAINER_ROOT / "src"
ROOT = CONTAINER_ROOT if IN_CONTAINER else SCRIPT_DIR.parent
DEFAULT_DATA_DIR = ROOT / "data" if IN_CONTAINER else ROOT / "data" / "data"
DATA_DIR = Path(os.environ.get("LIT_DATA_DIR", DEFAULT_DATA_DIR))
CLIPS_DIR = DATA_DIR / "clips"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_PATH = Path(
    os.environ.get("LIT_OUTPUT_PATH", SUBMISSION_DIR / "submission.csv")
)
MODEL_DIR = Path(os.environ.get("LIT_MODEL_DIR", SCRIPT_DIR / "model-large-v3"))


def load_filenames(format_path: Path) -> list[str]:
    if not format_path.is_file():
        raise FileNotFoundError(f"Submission format file not found: {format_path}")

    with format_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "audio_filename" not in reader.fieldnames:
            raise ValueError(f"{format_path} must contain an audio_filename column")
        filenames = [row["audio_filename"] for row in reader]

    if not filenames:
        raise ValueError(f"Submission format file is empty: {format_path}")
    if any(not filename for filename in filenames):
        raise ValueError(f"Found an empty audio_filename in {format_path}")
    return filenames


def main() -> None:
    format_path = DATA_DIR / (
        "submission_format.csv" if IN_CONTAINER else "test_metadata.csv"
    )
    if not format_path.is_file() and not IN_CONTAINER:
        format_path = DATA_DIR / "submission_format.csv"
    filenames = load_filenames(format_path)

    if not (MODEL_DIR / "model.bin").is_file() or not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(
            f"Faster-Whisper model files were not found in {MODEL_DIR}. "
            "Set LIT_MODEL_DIR to a CTranslate2 model directory or place it in model/."
        )

    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading bundled Faster-Whisper model on {device} ({compute_type})...")
    load_start = time.time()
    model = WhisperModel(
        str(MODEL_DIR),
        device=device,
        compute_type=compute_type,
        local_files_only=True,
    )
    print(f"Model loaded in {time.time() - load_start:.1f}s")

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    run_start = time.time()
    with SUBMISSION_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_filename", "transcript"])
        writer.writeheader()

        for index, filename in enumerate(filenames, start=1):
            clip_path = CLIPS_DIR / filename
            if not clip_path.is_file():
                raise FileNotFoundError(f"Audio clip {index} not found: {clip_path}")

            segments, _ = model.transcribe(
                str(clip_path),
                task="transcribe",
                language=None,
                beam_size=5,
                vad_filter=True,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            writer.writerow({"audio_filename": filename, "transcript": transcript})

            if index % 50 == 0 or index == len(filenames):
                elapsed = time.time() - run_start
                rate = index / elapsed if elapsed > 0 else 0
                print(f"Progress: {index}/{len(filenames)} clips ({rate:.2f} clips/sec)")

    print(f"Wrote {len(filenames)} predictions to {SUBMISSION_PATH}")
    print(f"Total inference runtime: {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()



# DATA_DIR = Path(os.environ.get("LIT_DATA_DIR", "/code_execution/data"))
# SUBMISSION_FORMAT_CSV = DATA_DIR / "submission_format.csv"
# CLIPS_DIR = DATA_DIR / "clips"
# SUBMISSION_PATH = Path(
#     os.environ.get(
#         "LIT_OUTPUT_PATH",
#         "/code_execution/submission/submission.csv",
#     )
# )

# # DATA_DIR = Path("/code_execution/data")
# # SUBMISSION_FORMAT_CSV = DATA_DIR / "submission_format.csv"
# # CLIPS_DIR = DATA_DIR / "clips"
# # SUBMISSION_PATH = Path("/code_execution/submission/submission.csv")


# def transcribe(audio_path: Path) -> str:
#     """Return the transcription for one audio clip.

#     Replace this function body with your model.
#     """
#     raise NotImplementedError


# def main() -> None:
#     submission = pl.read_csv(SUBMISSION_FORMAT_CSV)

#     transcriptions = [transcribe(CLIPS_DIR / filename) for filename in submission["audio_filename"]]

#     submission = submission.with_columns(pl.Series("transcript", transcriptions))
#     SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
#     submission.write_csv(SUBMISSION_PATH)
#     print(f"Wrote {submission.height} predictions to {SUBMISSION_PATH}")


# if __name__ == "__main__":
#     main()
