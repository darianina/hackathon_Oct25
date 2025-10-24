import cv2
import numpy as np
import os
from utils.datasets import create_dataloader
from model.baseline import stabilize_frames

def main():
    # repo root (one level up from model/)
    repo_root = os.path.dirname(__file__)
    mock_root = os.path.join(repo_root, "data")
    images_root = os.path.join(mock_root, "images")
    labels_root = os.path.join(mock_root, "labels")

    # dataloader settings: window of 2 frames
    hyp = {
        "num_frames": 2,
        "skip_rate": [0, 1],
        "val_skip_rate": [0, 1],
        "debug_data": False,
        "frame_wise": 0
    }

    # create dataloader + dataset (batch_size 1, we will iterate pairs manually)
    dataloader, dataset = create_dataloader(
        path=images_root,
        annotation_path=labels_root,
        image_root_path=images_root,
        imgsz=320,
        batch_size=1,
        stride=32,
        hyp=hyp,
        augment=False,
        is_training=False,
        img_ext="png",
        debug_dir=None,
    )

    out_root = os.path.join(repo_root, "data_res")
    os.makedirs(out_root, exist_ok=True)

    n = len(dataset.img_files)
    saved = 0
    window, stride = 10, 5
    use_stabilized_frames = False
    cached_stabilized_frames = []

    for i in range(window, n, stride):
        mid_idx = i - (window // 2 + 1) if window > 2 else i - window//2
        ref_frame_path = dataset.img_files[mid_idx]
        if use_stabilized_frames and cached_stabilized_frames:
            frames_stabilized = cached_stabilized_frames[(window - stride):]
            frames_unstabilized = [dataset.img_files[j] for j in range(i - window + (window - stride), i)]
            frames = frames_stabilized + frames_unstabilized
        else:
            frames = [dataset.img_files[j] for j in range(i - window, i)]
        loaded = []
        for p in frames:
            try:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
            except Exception as e:
                img = p
            if img is None:
                print(f"Failed to read frame {p}")
                loaded = []
                break
            loaded.append(img)

        # ensure we have the expected number of frames
        if len(loaded) != window:
            print(f"Skipping set due to read failure or missing frames: {frames}")
            continue

        frames = loaded

        # call existing stabilize function from this module
        res_dict = stabilize_frames(frames, ref_idx=stride//2)
        warped = res_dict["warped"]
        orig = res_dict["orig"]
        if not warped:
            print(f"Stabilization failed for set {frames}")
            continue

        fnames = [os.path.join(out_root, f"{os.path.basename(dataset.img_files[j]).split(os.sep)[0]}") for j in range(i - stride, i)]
        for i, fname in enumerate(fnames):
            cv2.imwrite(fname, loaded[i])
        cached_stabilized_frames = warped
        saved += 1

    print(f"Saved {saved} stabilized pair images to {out_root}")


if __name__ == "__main__":
    main()