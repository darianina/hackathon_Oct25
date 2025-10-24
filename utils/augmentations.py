"""
Image augmentation functions
"""
import random
import cv2
import numpy as np

class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            assert A.__version__ == '1.0.3', "You need Albumentations 1.0.3" # version requirement

            self.transform = A.Compose([
                A.GaussNoise(p=0.2),
                A.Blur(p=0.2),
                A.MedianBlur(p=0.2),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, p=0.1),
                A.ImageCompression(quality_lower=75, p=0.0)])
        except ImportError:  # package not installed, skip
            pass

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im)  # transformed
            im = new['image']
        return im, labels

class AlbumentationsTemporal:
    def __init__(self, num_frames):
        self.transform = None
        self.num_frames = num_frames
        try:
            import albumentations as A
            assert A.__version__ == '1.0.3', "You need Albumentations 1.0.3"   # version requirement
            self.transform = A.Compose([
                A.GaussNoise(p=0.2),
                A.Blur(p=0.2),
                A.MedianBlur(p=0.2),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, p=0.1),
                A.ImageCompression(quality_lower=75, p=0.0)])
        except ImportError:  # package not installed, skip
            pass
        
        self.transformation_expression = "self.transform(image=ims[0], "
        for ti in range(1, self.num_frames):
            self.transformation_expression += f"image{ti}=ims[{ti}], "
        self.transformation_expression += ")"
        self.transform.add_targets({f"image{i}":"image" for i in range(1, self.num_frames)})

    def __call__(self, ims, labels, p=1.0):
        if self.transform and random.random() < p:
            try:
                new = eval(self.transformation_expression) #transformed
            except Exception as e:
                print(f"Error occured {self.transformation_expression}, {labels[:, 1:]}, {str(e)}")
                exit()
            ims = [new['image']] + [new[f'image{ti}'] for ti in range(1, self.num_frames)]
            ims = np.stack(ims, 0) # T X H X W X C
        return ims, labels

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def augment_hsv_temporal(im, hgain=0.5, sgain=0.5, vgain=0.5, frame_wise_aug=False):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        dtype = im.dtype  # uint8
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        for ti in range(len(im)):
            if frame_wise_aug:
                r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
                dtype = im.dtype  # uint8
                x = np.arange(0, 256, dtype=r.dtype)
                lut_hue = ((x * r[0]) % 180).astype(dtype)
                lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
                lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            hue, sat, val = cv2.split(cv2.cvtColor(im[ti], cv2.COLOR_BGR2HSV))
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im[ti])

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def letterbox_temporal(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im[0].shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        for ti in range(len(im)):
            im[ti] = cv2.resize(im[ti], new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    for ti in range(len(im)):
        im[ti] = cv2.copyMakeBorder(im[ti], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)