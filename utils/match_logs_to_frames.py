import argparse
import math
import os
import sys
import numpy as np
import pandas as pd

#!/usr/bin/env python3
"""
match_logs_to_frames.py

Usage:
    python match_logs_to_frames.py input.csv [output.csv]

Reads a CSV with a 'system_time_s' column (high-frequency timestamps in seconds).
Groups rows by floor(system_time_s) (always round down), offsets the first group to zero,
and for each second-group distributes its rows across the 30 frames of that second.
If a group has n rows, they are split into 30 buckets whose sizes sum to n
(base = n // 30; remainder distributed to the earliest frames). Each row gets a
frame number (int) and a formatted frame_id string "{:08d}". Then, depending on 
collapse mode: "median" to keep the median row per frame,
or "mean" to average numeric columns across rows with the same frame_id.
Set to either "median" or "mean" as desired. The script writes
the resulting CSV to input_merged.csv (or the provided output path).
"""


def match_logs_to_frames(df: pd.DataFrame, time_col: str = "system_time_s", collapse='median', fps: int = 30):
    if time_col not in df.columns:
        raise KeyError(f"Expected column '{time_col}' not found in CSV")

    # Ensure numeric type
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    if df[time_col].isna().any():
        raise ValueError(f"Column '{time_col}' contains non-numeric or missing values")

    # Compute floored second for grouping, then offset to start from zero
    df["_sec_floor"] = np.floor(df[time_col]).astype(int)
    min_sec = int(df["_sec_floor"].min())
    df["_sec_adj"] = df["_sec_floor"] - min_sec

    # Sort by time so assignment is deterministic (preserve ties order)
    df = df.sort_values(by=[time_col]).reset_index(drop=True)

    # Prepare container for numeric frame numbers
    frame_numbers = np.empty(len(df), dtype=np.int64)

    # Process each adjusted second in ascending order
    for sec, group in df.groupby("_sec_adj", sort=True):
        idx = group.index.to_numpy()
        n = len(idx)
        if n == 0:
            continue

        base = n // fps
        rem = n % fps
        # lengths per frame: first 'rem' frames get base+1 rows, the rest get base
        lengths = [base + 1] * rem + [base] * (fps - rem)

        # Now assign frame numbers sequentially across the group's rows
        ptr = 0
        for frame_idx, length in enumerate(lengths):
            if length <= 0:
                continue
            frame_num = int(sec) * fps + frame_idx
            # assign this frame_num to next 'length' rows
            rows_to_assign = idx[ptr : ptr + length]
            frame_numbers[rows_to_assign] = frame_num
            ptr += length
            if ptr >= n:
                break

    df["frame_id_num"] = frame_numbers
    df["frame_id"] = df["frame_id_num"].apply(lambda x: f"{x:08d}")

    # drop helper cols
    df = df.drop(columns=["_sec_floor", "_sec_adj", "frame_id_num"])

    # Reconstruct numeric frame numbers from the formatted string and report any missing frames
    df["_frame_num"] = df["frame_id"].astype(int)
    min_frame = int(df["_frame_num"].min())
    max_frame = int(df["_frame_num"].max())

    all_frames = np.arange(min_frame, max_frame + 1, dtype=np.int64)
    present_frames = np.unique(df["_frame_num"].to_numpy(dtype=np.int64))
    missing_frames = np.setdiff1d(all_frames, present_frames)

    if missing_frames.size:
        print(f"Missing frame_ids ({missing_frames.size}):")
        for f in missing_frames:
            print(f"{f:08d}")
    else:
        print("No missing frame_ids.")

    # clean up helper column
    df = df.drop(columns=["_frame_num"])

    # Choose collapse behaviour: "median" to keep the median row per frame,
    # or "mean" to average numeric columns across rows with the same frame_id.
    # Set to either "median" or "mean" as desired.
    if collapse == "median":
        # For each frame_id keep the median row (df is already time-sorted).
        # If even number of rows, picks the lower-middle one (n//2).
        idxs = df.groupby("frame_id", sort=False).apply(lambda g: g.index[len(g) // 2]).to_numpy()
        df = df.loc[idxs].reset_index(drop=True)

    elif collapse == "mean":
        # Average numeric columns, take first value for non-numeric columns.
        grouped = df.groupby("frame_id", sort=False)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude frame_id from non-numeric selection (it will become the group key)
        non_num = [c for c in df.columns if c not in num_cols and c != "frame_id"]

        mean_df = grouped[num_cols].mean()
        first_df = grouped[non_num].first()

        # Combine and restore frame_id as a column
        combined = pd.concat([first_df, mean_df], axis=1)
        df = combined.reset_index()

        # Ensure frame_id is the first column
        cols = ["frame_id"] + [c for c in df.columns if c != "frame_id"]
        df = df[cols]

    else:
        raise ValueError(f"Unknown collapse mode: {collapse}")
        
    # Move frame_id to be the first column
    cols = df.columns.tolist()
    cols = ["frame_id"] + [c for c in cols if c != "frame_id"]
    df = df[cols]

    return df


def main():
    parser = argparse.ArgumentParser(description="Match high-frequency log rows to video frames.")
    parser.add_argument("input_csv", help="Input CSV file path (must contain 'system_time_s' column).")
    parser.add_argument("output_csv", nargs="?", help="Output CSV path (optional). If omitted, writes <input>_merged.csv")
    args = parser.parse_args()

    inp = args.input_csv
    if not os.path.isfile(inp):
        print(f"Input file not found: {inp}", file=sys.stderr)
        sys.exit(2)

    out = args.output_csv
    if not out:
        base, ext = os.path.splitext(inp)
        out = f"{base}_merged.csv"

    try:
        df = pd.read_csv(inp)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(3)

    try:
        df_out = match_logs_to_frames(df, time_col="system_time_s", fps=30)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(4)

    try:
        df_out.to_csv(out, index=False)
    except Exception as e:
        print(f"Failed to write output CSV: {e}", file=sys.stderr)
        sys.exit(5)

    print(f"Wrote matched CSV to: {out}")


if __name__ == "__main__":
    main()