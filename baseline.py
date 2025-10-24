import cv2
import numpy as np
import os
import glob

# Exponential smoothing factor (alpha controls the weight of the previous transformation)
alpha = 0.9

# Function to compute optical flow and stabilize the video with weighted smoothing of transformations
def stabilize_video_with_smoothing(input_video_path, output_video_path):
    # If input_video_path is a folder, read images from it
    input_folder = input_video_path
    # Supported image extensions
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    image_files = []
    for e in exts:
        image_files.extend(glob.glob(os.path.join(input_folder, e)))
    image_files = sorted(image_files)

    if len(image_files) == 0:
        print("No images found in folder:", input_folder)
        return

    # Read the first image
    frame1 = cv2.imread(image_files[0])
    if frame1 is None:
        print("Error reading first image:", image_files[0])
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Detect good features to track in the first frame
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7))
    if prev_points is None:
        print("No features found in the first image.")
        return

    # Create a mask for drawing
    mask = np.zeros_like(frame1)

    # Set up the output video writer (keep same size as first image)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame1.shape[1], frame1.shape[0]))

    # Dummy cap object so later cap.release() call doesn't fail
    class _DummyCap:
        def release(self): pass
    cap = _DummyCap()

    # First, compute pairwise transforms t_i that map frame i -> frame i+1
    transforms = []
    identity = np.array([[1., 0., 0.],
                         [0., 1., 0.]], dtype=np.float32)

    for i in range(len(image_files) - 1):
        f1 = cv2.imread(image_files[i])
        f2 = cv2.imread(image_files[i + 1])
        if f1 is None or f2 is None:
            transforms.append(identity.copy())
            continue

        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        pts1 = cv2.goodFeaturesToTrack(g1, mask=None, **dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7))
        if pts1 is None:
            transforms.append(identity.copy())
            continue

        pts2, status, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts1, None)
        if pts2 is None or status is None:
            transforms.append(identity.copy())
            continue

        good_old = pts1[status.flatten() == 1].reshape(-1, 2)
        good_new = pts2[status.flatten() == 1].reshape(-1, 2)

        if good_old.shape[0] < 3:
            transforms.append(identity.copy())
            continue

        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
        if M is None:
            transforms.append(identity.copy())
        else:
            transforms.append(M.astype(np.float32))

    # Define weights for the weighted average of future transforms.
    # Example: for frame i use 0.5*t_i + 0.4*t_{i+1} + 0.1*t_{i+2}
    weights = np.array([0.15, 0.15, 0.125, 0.125, 0.1, 0.1, 0.075, 0.075, 0.05, 0.05], dtype=np.float32)
    if weights.sum() != 0:
        weights = weights / weights.sum()

    # Now apply the weighted average to each frame i (using available future transforms)
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Accumulate weighted transform; initialize with zeros
        acc = np.zeros((2, 3), dtype=np.float32)
        weight_sum = 0.0
        for j, w in enumerate(weights):
            idx = i + j  # t_idx maps frame idx -> idx+1
            if idx < len(transforms):
                acc += w * transforms[idx]
                weight_sum += w
            else:
                break

        # If no future transforms available (e.g., last frames), fall back to identity
        if weight_sum == 0.0:
            smoothed_transform = identity
        else:
            # If weights were normalized above this keeps sum == 1; otherwise normalize by weight_sum
            smoothed_transform = acc / (1.0 if np.isclose(weight_sum, 1.0) else weight_sum)

        # Apply the smoothed transform to stabilize the current frame
        stabilized_frame = cv2.warpAffine(frame, smoothed_transform, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)

        # Write the stabilized frame
        out.write(stabilized_frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_imgs_path = 'data/images/Flight1/'  # Input video file path
output_video_path = 'res/stabilized_output.mp4'  # Output stabilized video file path

stabilize_video_with_smoothing(input_imgs_path, output_video_path)