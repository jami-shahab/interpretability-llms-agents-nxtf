"""Utilities for splitting video, audio, and SRT files into fixed-length segments."""

import os
import subprocess

from src.dataset_utils import seconds_to_srt
from src.media_utils import get_duration


def save_segmented_srt(entries, segment_length, video_id, output_dir, total_segments):
    """
    Save segmented SRT files based on a given segment length.

    Parameters
    ----------
    - entries (list): List of subtitle entries with 'start', 'end', and 'text'.
    - segment_length (int): Length of each segment in seconds.
    - video_id (str): Identifier for the video.
    - output_dir (str): Directory to save the segmented SRT files.
    - total_segments (int): Total number of segments to create.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists.

    # Initialize a dictionary to store segments.
    segments = {i: [] for i in range(total_segments)}

    # Assign entries to their respective segments.
    for entry in entries:
        seg_id = int(entry["start"] // segment_length)  # Determine segment ID.
        if seg_id < total_segments:
            segments[seg_id].append(entry)

    # Save each segment to a separate SRT file.
    for seg_id in range(total_segments):
        out_path = os.path.join(
            output_dir,
            f"{video_id}__{seg_id:03d}.srt",  # Format the output filename.
        )

        with open(out_path, "w", encoding="utf-8") as f:
            seg_entries = segments[seg_id]

            if len(seg_entries) == 0:
                # Skip writing empty files.
                continue

            for idx, entry in enumerate(seg_entries, start=1):
                start_time = seconds_to_srt(entry["start"])  # Convert start time.
                end_time = seconds_to_srt(entry["end"])  # Convert end time.

                # Write the SRT entry.
                f.write(f"{idx}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{entry['text']}\n\n")


def split_precisely(input_file, output_dir, prefix, ext, segment_length, min_last=5):
    """Split a media file into precise fixed-length segments using ffmpeg."""
    total_duration = get_duration(input_file)

    full_segments = int(total_duration // segment_length)
    remainder = total_duration - (full_segments * segment_length)

    segments = []

    # Add full segments
    for i in range(full_segments):
        segments.append((i * segment_length, segment_length))

    # Handle remainder
    if remainder > 0:
        if remainder < min_last and full_segments > 0:
            # Merge remainder into previous segment
            start, dur = segments[-1]
            segments[-1] = (start, dur + remainder)
        else:
            segments.append((full_segments * segment_length, remainder))

    # Run ffmpeg
    for i, (start, duration) in enumerate(segments):
        output_path = os.path.join(output_dir, f"{prefix}__{i:03d}.{ext}")
        if ext == "wav":
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-i",
                input_file,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-i",
                input_file,
                "-c",
                "copy",
                output_path,
            ]
        subprocess.run(cmd, check=True)


def split_video(video_dir, segment_dir, segment_length, max_files: int = None):
    """Split video files in a directory into segments.

    Args:
        video_dir (str)
        segment_length (int)
        max_files (int, optional): number of videos to process
    """
    output_dir = segment_dir
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = (".mp4", ".mov", ".mkv", ".avi", ".webm")

    files = [
        f
        for f in sorted(os.listdir(video_dir))
        if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(video_dir, f))
    ]

    if max_files is not None:
        files = files[:max_files]

    for filename in files:
        input_path = os.path.join(video_dir, filename)
        base_name = os.path.splitext(filename)[0]

        split_precisely(input_path, output_dir, base_name, "mp4", segment_length)


def split_audio(audio_dir, segment_dir, segment_length, max_files: int = None):
    """Split audio files in a directory.

    Args:
        audio_dir (str)
        segment_length (int)
        max_files (int, optional): number of audio files to process
    """
    os.makedirs(segment_dir, exist_ok=True)

    files = [
        f
        for f in sorted(os.listdir(audio_dir))
        if f.lower().endswith((".m4a", ".wav"))
        and os.path.isfile(os.path.join(audio_dir, f))
    ]

    if max_files is not None:
        files = files[:max_files]

    for filename in files:
        input_file = os.path.join(audio_dir, filename)
        base_name = os.path.splitext(filename)[0]

        split_precisely(input_file, segment_dir, base_name, "wav", segment_length)
