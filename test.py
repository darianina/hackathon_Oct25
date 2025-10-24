import os
from pprint import pprint

from utils.datasets import create_dataloader
from utils.plots import plot_local_position_on_frame

def main():
    repo_root = os.path.dirname(__file__)
    mock_root = os.path.join(repo_root, "data")
    images_root = os.path.join(mock_root, "images")
    labels_root = os.path.join(mock_root, "labels")

    # Hyp dict required by the dataloader (minimal fields used by dataset)
    hyp = {
        "num_frames": 3,                      # temporal clip length (T)
        "image_size": 1080,
        "batch_size": 1,
        "skip_rate": [0, 1, 2],               # sampling offsets (len == num_frames)
        "val_skip_rate": [0, 1, 2],           # same for val
        "debug_data": True,                   # enable plotting debug images
        "frame_wise": 0,                      # 0 => temporal albumentations
        "is_training": False,                  # training mode
    }

    # Debug output (plots will be written here)
    debug_dir = os.path.join(repo_root, "debug_plots")
    os.makedirs(debug_dir, exist_ok=True)

    # Create dataloader. 
    dataloader, dataset = create_dataloader(
        path=images_root,
        annotation_path=labels_root,
        image_root_path=images_root,
        imgsz=hyp["image_size"],
        batch_size=hyp["batch_size"],
        stride=32,
        hyp=hyp,
        augment=False,
        is_training=hyp["is_training"],
        debug_dir=debug_dir,
    )

    print("Dataset length:", len(dataset))
    print("Taking one batch from dataloader and printing shapes/paths...")

    # iterate one batch and print info
    for batch_idx, (imgs, labels, paths, shapes, main_frame_ids, label_paths) in enumerate(dataloader):
        # imgs: (B*T, C, H, W)
        # labels: (B, T, F)
        print(f"Batch {batch_idx}")
        print(" imgs shape (B*T, C, H, W):", tuple(imgs.shape))
        print(" labels shape (B, T, F):", tuple(labels.shape))
        print(" flattened frame paths (B*T):", len(paths))
        print(" example paths (first 6):")
        pprint(paths[:6])
        print(" shapes per-frame (first 3):")
        pprint(shapes[:3])
        print(" main_frame_ids (absolute indices in flattened batch):", main_frame_ids)
        print(" label CSV paths (flattened):", label_paths[:5])

        # show per-sample label vectors for the first batch element
        lbl0 = labels[0].numpy()
        print(" Per-frame feature vectors for sample 0 (T x F):", lbl0.shape)
        print(lbl0)

        break  # only inspect the first batch

    print(f"Debug plots (if any) saved under: {debug_dir}")

    # Plot local position data to understand video and logs synchronization
    plot = None # local position flight data plot # set batch_size to 1
    assert not hyp["is_training"] and dataloader.batch_size==1, "Set is_training=False and batch_size=1 to plot scaled imu value on frame"
    os.makedirs(os.path.join(debug_dir, "imu_plots"), exist_ok=True)
    
    for batch_idx, (imgs, labels, paths, shapes, main_frame_ids, label_paths) in enumerate(dataloader):
        # labels: (B, T, F)
        # 8th, 9th fields are ac_x, ac_y
        plot = plot_local_position_on_frame(
            image=imgs[main_frame_ids[0]],  # main frame of first sample in batch
            plot=plot,
            label=labels[0][main_frame_ids[0]][8:10],  # ac_x, ac_y for all frames of first sample
            fname=os.path.normpath(os.path.join(debug_dir, f"imu_plots/local_position_on_frame_batch{batch_idx}.jpg"))
        )


if __name__ == "__main__":
    main()