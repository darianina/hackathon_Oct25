import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
import os
from PIL import ImageDraw

def plot_images_temporal(images, fname='images.jpg'):
    # Plot image grid and GIF
    temporal_window, ch, h, w = images.shape

    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    if isinstance(images, torch.Tensor):
        images = images.cpu().float() #2 X T X C X H X W

    if torch.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    images_list = []
    images = images.to(torch.uint8)
    for ti, image in enumerate(images):
        images_list.append(image)

    images_grid = make_grid(images_list, nrow=temporal_window).float()/255.
    save_image(images_grid, fname)

    gif_name = os.path.splitext(fname)[0] + '.gif'
    frames = []
    for img in images_list:
        arr = img.cpu().numpy()
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
        frames.append(Image.fromarray(arr))
    if frames:
        frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=100, loop=0)


def plot_local_position_on_frame(image, plot, label, fname='localization_on_frame.jpg'):
    # Ensure image is a HxWxC uint8 numpy / PIL for compositing
    # Accept torch.Tensor, numpy.ndarray or PIL.Image

    # Plots scaled IMU measurements (ac_x, ac_y) onto the image as an inset plot
    # Double-check that IMU values lie in range, we specify some experimental for visualization: [-5, 25]
    imu_min, imu_max = -5.0, 25.0

    def to_pil(img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, torch.Tensor):
            img_t = img.detach().cpu()
            if img_t.dim() == 3:  # C,H,W
                img_t = img_t.permute(1, 2, 0)
            img_np = img_t.numpy()
        elif isinstance(img, np.ndarray):
            img_np = img
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                # CHW -> HWC
                if img_np.shape[0] in (1, 3) and img_np.shape[0] <= 3 and img_np.shape[0] != img_np.shape[2]:
                    img_np = np.transpose(img_np, (1, 2, 0))
        else:
            raise TypeError("Unsupported image type for compositing")
        # If float in [0,1]
        if np.issubdtype(img_np.dtype, np.floating):
            if img_np.max() <= 1.0:
                img_np = (img_np * 255.0).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        if img_np.shape[2] == 1:
            img_np = np.concatenate([img_np] * 3, axis=2)
        return Image.fromarray(img_np).convert("RGB")

    base_pil = to_pil(image)
    # Upscale base image by 3x
    scale = 3
    W0, H0 = base_pil.size
    base_pil = base_pil.resize((W0 * scale, H0 * scale), resample=Image.BILINEAR)
    W, H = base_pil.size

    # size of inset plot: 30% of image size
    inset_w = max(4, int(W * 0.3))
    inset_h = max(4, int(H * 0.3))

    # Prepare plot image (PIL). If provided, try to convert/rescale; otherwise init blank axes.
    if plot is None:
        plot_img = Image.new("RGB", (inset_w, inset_h), (255, 255, 255))
        draw = ImageDraw.Draw(plot_img)
        # draw axes border
        draw.rectangle([0, 0, inset_w - 1, inset_h - 1], outline=(0, 0, 0))
        # optional grid lines (center)
        draw.line([(0, inset_h // 2), (inset_w, inset_h // 2)], fill=(200, 200, 200))
        draw.line([(inset_w // 2, 0), (inset_w // 2, inset_h)], fill=(200, 200, 200))
        plot = plot_img
    else:
        # convert existing plot to PIL and resize to inset
        plot_img = to_pil(plot).resize((inset_w, inset_h), resample=Image.BILINEAR)
        plot = plot_img

    # Draw the label (2-D float vector) onto the plot
    try:
        lx, ly = float(label[0]), float(label[1])
    except Exception:
        raise ValueError("label must be a 2-D iterable of floats")

    # Interpret label as normalized coordinates in [0,1]; clamp
    def scale(v): return (v + imu_max) / imu_min * 2.0 - 1.0  # from [-10,10] to [-1,1]
    nx = scale(lx)
    ny = scale(ly)

    px = int(nx * (inset_w - 1))
    py = int(ny * (inset_h - 1))

    draw = ImageDraw.Draw(plot_img)
    r = max(2, int(min(inset_w, inset_h) * 0.04))
    # draw filled circle (red) with black outline for visibility
    draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=(255, 0, 0), outline=(0, 0, 0))

    # Paste plot into bottom-right corner with small margin
    margin = max(2, int(0.02 * min(W, H)))
    paste_x = W - inset_w - margin
    paste_y = H - inset_h - margin
    base_pil.paste(plot_img, (paste_x, paste_y))

    # convert back to torch tensor in C x H x W, float in [0,1] for save_image
    arr = np.array(base_pil).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    image = torch.from_numpy(arr).float()
    save_image(image, fname)
    return plot