from typing import Optional
import os
import cv2
import argparse

#!/usr/bin/env python3
"""
utils/frame_clip.py

Extract frames from a video into a directory at a specified output frame rate.
"""


def extract_frames(
    video_path: str,
    out_dir: str,
    target_fps: Optional[float] = None,
    prefix: str = "frame",
    img_ext: str = "jpg",
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> int:
    """
    Extract frames from `video_path` into `out_dir`.

    - target_fps: desired output frames per second. If None or <= 0, uses video's fps.
    - prefix: filename prefix for saved frames (prefix_000001.jpg).
    - img_ext: image file extension (jpg, png, ...).
    - start_time: seconds into video to start extracting.
    - end_time: seconds into video to stop extracting (None => video end).
    - max_frames: optional cap on number of frames to write.

    Returns number of frames written.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / orig_fps) if orig_fps > 0 else None

    if target_fps is None or target_fps <= 0:
        if orig_fps > 0:
            target_fps = orig_fps
        else:
            target_fps = 1.0

    if duration is not None and end_time is None:
        end_time = duration
    if end_time is None:
        end_time = float("inf")

    start_time = max(0.0, start_time)
    end_time = max(start_time, end_time)

    # Seek to start_time (milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

    saved = 0
    next_save_time = start_time
    # Read loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # current position in seconds
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = pos_msec / 1000.0 if pos_msec is not None else None
        if timestamp is None:
            # fallback to frame index / fps
            current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            timestamp = current_idx / max(orig_fps, 1e-6)

        if timestamp + 1e-6 >= next_save_time and timestamp <= end_time + 1e-6:
            filename = f"{prefix}_{saved:08d}.{img_ext}" if prefix else f"{saved:08d}.{img_ext}"
            out_path = os.path.join(out_dir, filename)
            # write with default params
            if not cv2.imwrite(out_path, frame):
                raise RuntimeError(f"Failed to write frame to {out_path}")
            saved += 1
            next_save_time += 1.0 / target_fps
            if max_frames is not None and saved >= max_frames:
                break
        if timestamp > end_time:
            break

    cap.release()
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video at specified FPS.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("out_dir", help="Directory to save frames")
    parser.add_argument("--fps", type=float, default=None, help="Output frames per second (default: video's fps)")
    parser.add_argument("--prefix", default="", help="Filename prefix for frames")
    parser.add_argument("--ext", default="jpg", help="Image extension (jpg, png)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of frames to extract")
    args = parser.parse_args()

    count = extract_frames(
        video_path=args.video,
        out_dir=args.out_dir,
        target_fps=args.fps,
        prefix=args.prefix,
        img_ext=args.ext,
        start_time=args.start,
        end_time=args.end,
        max_frames=args.max,
    )
    print(f"Saved {count} frames to {args.out_dir}")