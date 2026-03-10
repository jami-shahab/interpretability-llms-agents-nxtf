"""System utilities for path alignment and GPU memory reporting."""

import os

import torch


def get_aligned_paths(video_dir, audio_dir, caption_dir):
    """
    Return aligned video, audio, and caption paths based on filename intersection.

    Args:
    - video_dir (str): Directory containing video files (.mp4).
    - audio_dir (str): Directory containing audio files (.wav).
    - caption_dir (str): Directory containing caption files (.srt).

    Returns
    -------
    - tuple: (video_paths, audio_paths, caption_paths) where each is a list of
      file paths.
    """
    video_paths = sorted(
        os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")
    )

    audio_paths = sorted(
        os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")
    )

    caption_paths = sorted(
        os.path.join(caption_dir, f)
        for f in os.listdir(caption_dir)
        if f.endswith(".srt")
    )

    video_ids = {os.path.splitext(os.path.basename(p))[0] for p in video_paths}
    audio_ids = {os.path.splitext(os.path.basename(p))[0] for p in audio_paths}
    caption_ids = {os.path.splitext(os.path.basename(p))[0] for p in caption_paths}

    common_ids = sorted(video_ids & audio_ids & caption_ids)

    video_paths = [
        os.path.join(video_dir, f"{video_id}.mp4") for video_id in common_ids
    ]
    audio_paths = [
        os.path.join(audio_dir, f"{audio_id}.wav") for audio_id in common_ids
    ]
    caption_paths = [
        os.path.join(caption_dir, f"{caption_id}.srt") for caption_id in common_ids
    ]

    return video_paths, audio_paths, caption_paths


def print_gpu_memory():
    """
    Print current GPU memory usage in GB.

    This function checks if a CUDA-enabled GPU is available and prints the
    allocated and reserved memory in gigabytes (GB).

    Returns
    -------
        None
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
