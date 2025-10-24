import cv2
import numpy as np
import time

def get_matches_kp(img1_gray, img2_gray, max_features=2000):
    """
    Fast alternative using Shi-Tomasi + pyramidal Lucas-Kanade optical flow.
    Returns matched point arrays (pts1, pts2, good_matches_placeholder).
    """
    t1 = time.time()
    # detect strong corners in img1
    p0 = cv2.goodFeaturesToTrack(img1_gray, maxCorners=max_features, qualityLevel=0.01, minDistance=7, blockSize=7)
    if p0 is None:
        print(f"Feature matching time: {time.time() - t1:.3f}s (no features)")
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), []
    # track them into img2
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        img1_gray, img2_gray, p0, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    if p1 is None or st is None:
        print(f"Feature matching time: {time.time() - t1:.3f}s (lk failed)")
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), []
    st = st.flatten()
    good0 = p0[st == 1].reshape(-1, 2)
    good1 = p1[st == 1].reshape(-1, 2)
    print(f"Feature matching time: {time.time() - t1:.3f}s (LK, {len(good0)} matches)")
    # return placeholder for 'good' (not needed downstream)
    return np.array(good0, dtype=np.float32), np.array(good1, dtype=np.float32), []

def estimate_scale_translation(src_gray, dst_gray):
    pts_src, pts_dst, good = get_matches_kp(src_gray, dst_gray)
    if len(pts_src) < 3:
        # fallback: identity
        return 1.0, (0.0, 0.0), np.zeros((0,), dtype=np.uint8)
    # use estimateAffinePartial2D to get inliers mask (similarity-like: scale+rotation+translation, no shear)
    M, inliers = cv2.estimateAffinePartial2D(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000)
    if M is None:
        return 1.0, (0.0, 0.0), np.zeros((0,), dtype=np.uint8)
    inliers = inliers.flatten().astype(bool)
    if np.count_nonzero(inliers) < 3:
        return 1.0, (0.0, 0.0), inliers
    # compute centroids using inliers only
    src_in = pts_src[inliers]
    dst_in = pts_dst[inliers]
    centroid_src = src_in.mean(axis=0)
    centroid_dst = dst_in.mean(axis=0)
    # compute scale as ratio of root-mean-square distances to centroids
    ds_src = np.linalg.norm(src_in - centroid_src, axis=1)
    ds_dst = np.linalg.norm(dst_in - centroid_dst, axis=1)
    denom = ds_src.sum()
    if denom < 1e-6:
        scale = 1.0
    else:
        scale = ds_dst.sum() / denom
    # translation so that scale*centroid_src + t = centroid_dst
    tx, ty = (centroid_dst - scale * centroid_src).tolist()
    return float(scale), (float(tx), float(ty)), inliers

def warp_with_scale_translation(img, scale, tx, ty, output_shape):
    M = np.array([[scale, 0.0, tx],
                  [0.0, scale, ty]], dtype=np.float32)
    warped = cv2.warpAffine(img, M, (output_shape[1], output_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped

def stabilize_frames(frames, ref_idx=None):
    """
    Stabilize a list/tuple of BGR frames to a common reference frame.

    Args:
        frames: list or tuple of BGR images (numpy arrays).
        ref_idx: optional index of reference frame (defaults to central frame).

    Returns:
        dict with keys:
          - "warped": list of frames warped to the reference frame (reference is a copy)
          - "orig": original input frames (same order)
          - "scales": list of estimated scales (1.0 for reference)
          - "translations": list of (tx, ty) tuples (0,0 for reference)
          - "inliers": list of inlier masks returned by estimate_scale_translation
          - "ref_idx": chosen reference index
    """
    if not frames:
        return {"warped": [], "orig": [], "scales": [], "translations": [], "inliers": [], "ref_idx": ref_idx}

    n = len(frames)
    if ref_idx is None:
        ref_idx = n // 2
    if ref_idx < 0 or ref_idx >= n:
        raise ValueError("ref_idx out of range")

    # reference shape
    h, w = frames[ref_idx].shape[:2]
    # convert to gray
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    ref_gray = grays[ref_idx]

    t1 = time.time()
    scales = [1.0] * n
    translations = [(0.0, 0.0)] * n
    inliers_list = [np.zeros((0,), dtype=np.uint8) for _ in range(n)]

    # estimate transform from each frame -> reference
    for i, g in enumerate(grays):
        if i == ref_idx:
            continue
        s, (tx, ty), inliers = estimate_scale_translation(g, ref_gray)
        scales[i] = s
        translations[i] = (tx, ty)
        inliers_list[i] = inliers
    t2 = time.time()

    # warp all frames to reference
    warped = [None] * n
    for i, f in enumerate(frames):
        if i == ref_idx:
            warped[i] = f.copy()
        else:
            s = scales[i]
            tx, ty = translations[i]
            warped[i] = warp_with_scale_translation(f, s, tx, ty, (h, w))
    t3 = time.time()

    print(f"Stabilization time: estimate {(t2 - t1)*1000:.3f}ms, warp {(t3 - t2)*1000:.3f}ms")
    return {
        "warped": warped,
        "orig": list(frames),
        "scales": scales,
        "translations": translations,
        "inliers": inliers_list,
        "ref_idx": ref_idx
    }