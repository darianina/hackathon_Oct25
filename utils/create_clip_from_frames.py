from __future__ import annotations
import argparse
import os
import re
import sys
from typing import List
import glob
import cv2
import numpy as np

#!/usr/bin/env python3
"""
utils/create_clip_from_frames.py

Create a video clip from image frames in a folder.

Usage:
    python create_clip_from_frames.py --input-dir /path/to/frames \
        --output-dir /path/to/output --output-name clip.mp4 --fps 25

The script collects images (jpg, jpeg, png, bmp, tiff) from the input directory,
sorts them in natural (human) order, and writes a video file with the specified framerate.
"""

IMAGE_EXTS = ("jpg", "jpeg", "png", "bmp", "tiff", "tif")


def natural_key(s: str):
    # Splits strings into list of ints and text for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def collect_images(input_dir: str, exts: tuple = IMAGE_EXTS) -> List[str]:
    imgs = []
    for ext in exts:
        pattern = os.path.join(input_dir, f"**/*.{ext}")
        imgs.extend(glob.glob(pattern, recursive=True))
    imgs = [p for p in imgs if os.path.isfile(p)]
    imgs.sort(key=natural_key)
    return imgs


def choose_fourcc_and_ext(out_path: str):
    _, ext = os.path.splitext(out_path)
    ext = ext.lower()
    if ext in (".mp4", ".m4v", ".mov"):
        return cv2.VideoWriter_fourcc(*"mp4v"), ext
    if ext in (".avi",):
        return cv2.VideoWriter_fourcc(*"XVID"), ext
    # default to mp4v
    return cv2.VideoWriter_fourcc(*"mp4v"), ".mp4"


def create_video_from_frames(
    input_dir: str,
    output_path: str,
    fps: float = 30.0,
    overwrite: bool = False,
):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = collect_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path} (use --overwrite to replace)")

    # Read first image to get frame size
    first = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise ValueError(f"Could not read image: {images[0]}")

    # If image has alpha channel, drop it
    if first.ndim == 3 and first.shape[2] == 4:
        first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)
    elif first.ndim == 2:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)

    height, width = first.shape[:2]
    fourcc, _ = choose_fourcc_and_ext(output_path)
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    try:
        for idx, img_path in enumerate(images):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: skipping unreadable image: {img_path}", file=sys.stderr)
                continue

            # Normalize channel count
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Resize if necessary to match first frame
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            writer.write(img)

            if (idx + 1) % 50 == 0:
                print(f"Wrote {idx + 1} / {len(images)} frames...")

    finally:
        writer.release()

    print(f"Finished writing video: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Create a video clip from image frames")
    p.add_argument("--input-dir", "-i", required=True, help="Directory containing image frames")
    p.add_argument("--output-dir", "-o", default=".", help="Directory to write the output video")
    p.add_argument("--output-name", "-n", default="clip.mp4", help="Output video file name (e.g. clip.mp4)")
    p.add_argument("--fps", type=float, default=30.0, help="Frames per second for the output video")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    output_name = args.output_name
    output_path = os.path.join(output_dir, output_name)

    try:
        create_video_from_frames(input_dir=input_dir, output_path=output_path, fps=args.fps, overwrite=args.overwrite)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()