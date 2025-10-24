import glob
import hashlib
import logging
import os
import random
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path
import re
import cv2
import numpy as np
import torch
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import Albumentations, AlbumentationsTemporal, letterbox_temporal
from utils.torch_utils import torch_distributed_zero_first
from utils.plots import plot_images_temporal
import csv

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
NUM_THREADS = min(4, os.cpu_count())

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def create_dataloader(path, annotation_path, image_root_path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, pad=0.0,
                      rank=-1, workers=3, prefix='', is_training=True, random_crop_size=None, pin_memory=True,
                      img_ext="jpg", grayscale=False, use_subset=False, sort_images=False, debug_dir=None):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache

    with torch_distributed_zero_first(rank):
        dataset = LoadClipsAndLabels(path, annotation_path, image_root_path, img_ext, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      prefix=prefix,
                                      is_training=is_training,
                                      random_crop_size=random_crop_size,
                                      grayscale=grayscale,
                                      use_subset=use_subset,
                                      sort_images=sort_images,
                                      debug_dir=debug_dir
                                      )
    shuffle = is_training
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle) if rank != -1 else None
    assert sampler is None
    shuffle = shuffle and sampler is None
    generator = torch.Generator()
    generator.manual_seed(0)
    print(f"data loader shuffle {shuffle}, pin_memory {pin_memory}") if rank in [0, -1] else None
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            num_workers=nw,
                                            sampler=sampler,
                                            pin_memory=pin_memory,
                                            drop_last=False,
                                            collate_fn=LoadClipsAndLabels.collate_fn,
                                            generator=generator,
                                            shuffle=shuffle,
                                            worker_init_fn=seed_worker
                                            )
    return dataloader, dataset


class LoadClipsAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads clips and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, annotation_path, image_root_path, img_ext,
                 img_size=640, batch_size=16, augment=False, hyp=None,
                 single_cls=False, stride=32, pad=0.0, prefix='', is_training=True,
                 random_crop_size=None, grayscale=False, use_subset=False, sort_images=False,
                 debug_dir=None):
        self.img_size = img_size
        self.img_ext = img_ext
        self.random_crop_size = random_crop_size  # (W, H) especially for simulated gimbal data
        self.grayscale = grayscale
        self.augment = augment and is_training  # augment images
        if isinstance(hyp, str):
            with open(hyp, errors="ignore") as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        self.hyp = hyp
        self.stride = stride
        self.path = path
        self.is_training = is_training
        self.frame_wise_aug = int(self.hyp["frame_wise"]) if "frame_wise" in self.hyp else 0
        self.image_root_path = image_root_path
        self.video_length_dict = get_video_length(self.path)
        self.num_frames = int(self.hyp['num_frames'])
        self.skip_frames = self.hyp["skip_rate"]
        self.val_skip_frames = self.hyp["val_skip_rate"] if "val_skip_rate" in self.hyp else list(range(self.num_frames))
        self.max_skip_rate = None if "max_skip_rate" not in self.hyp else self.hyp["max_skip_rate"]
        self.albumentations = None
        self.debug_data = self.hyp['debug_data'] if 'debug_data' in self.hyp else False
        self.debug_dir = debug_dir if debug_dir is not None else "images"
        self.current_epoch = 0
        self.remaining_epochs = 0
        if augment:
            self.albumentations = AlbumentationsTemporal(self.num_frames) if not self.frame_wise_aug else Albumentations()
        self.annotation_path = annotation_path
        # Check cache
        cache_path = Path(annotation_path).with_suffix('.cache')
        try:
            raise FileNotFoundError  # force cache to be rebuilt
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            #assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except:
            exists = False
        if not exists:
            try:
                f = []  # image files
                for p in image_root_path if isinstance(image_root_path, list) else [image_root_path]:
                    p = Path(p)  # os-agnostic
                    if p.is_dir():  # dir
                        f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    elif p.is_file():  # file
                        with open(p) as t:
                            t = t.read().strip().splitlines()
                            parent = str(p.parent) + os.sep
                            f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                    else:
                        raise Exception(f'{prefix}{p} does not exist')
                self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
                assert self.img_files, f'{prefix}No images found'
            except Exception as e:
                raise Exception(f'{prefix}Error loading data from {path}: {e}')

            self.label_files = img2label_paths(self.img_files, self.annotation_path)  # labels

        if not exists:
            print(f'{prefix}Caching labels for {len(self.img_files)} images to {cache_path}...')
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache
        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels.'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items

        # Take 5% of the dataset
        if use_subset:
            subset_length = int(n * 0.05)
            cache = {k: v for i, (k, v) in enumerate(cache.items()) if i < subset_length}
            print(f"Using subset of the dataset with length {subset_length}")

        labels, instances, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.instances = list(instances)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())
        self.label_files = img2label_paths(cache.keys(), self.annotation_path)  # update

        n = len(shapes)  # number of images

        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = list(range(n))

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment, instance) in enumerate(zip(self.labels, self.segments, self.instances)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                self.instances[i] = instance[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        assert len(self.img_files) == len(self.labels)

        self.img_file_to_indices_mapping = {str(image_path):index for index, image_path in enumerate(self.img_files) }
        if self.is_training:
            assert not sort_images, "Sort images not applied in training mode, you can enable it"
        elif sort_images:
            # Sort img_files by clip_id and frame_id
            def sort_key(path):
                filename = os.path.basename(path)
                clip_name = os.path.dirname(path).split(os.sep)[-1]
                frame_id = int(filename.split('.')[0])
                return (clip_name, frame_id)
            self.indices = sorted(range(len(self.img_files)), key=lambda i: sort_key(self.img_files[i]))

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n

    def parse_label_row(self, row):
        parsed = []
        for cell in row:
            # allow comma/space separated values inside a single cell
            parts = re.split(r'[,\s]+', cell.strip())
            for p in parts:
                if p == "":
                    continue
                try:
                    parsed.append(float(p))
                except ValueError:
                    parsed.append(p)

        parsed = parsed[1:]
        return parsed
        
    def get_images_and_labels_dict(self, img_files, label_files):
        x = {}
        msgs = []
        nf = nm = ne = nc = 0
        n = len(img_files)

        # group images by their label (csv) file keeping original order
        label_to_images = {}
        for idx, (img, lab) in enumerate(zip(img_files, label_files)):
            label_to_images.setdefault(lab, []).append((idx, img))

        for lab_path, img_list in label_to_images.items():
            # Attempt to open the CSV for this clip once
            if not os.path.isfile(lab_path):
                # missing label file -> mark all images in this group as missing
                for _, img_path in img_list:
                    x[img_path] = (None, None, (0, 0), None)
                    nm += 1
                continue

            try:
                with open(lab_path, newline='') as f:
                    reader = csv.reader(f)
                    # skip header row
                    next(reader, None)
                    reader = list(reader)
            except Exception as e:
                # corrupted CSV -> mark as corrupted for all images in group
                msgs.append(f"Corrupted CSV {lab_path}: {e}")
                for _, img_path in img_list:
                    x[img_path] = (None, None, (0, 0), None)
                    nc += 1
                continue

            # assign rows to images row-by-row in the same order as img_list
            for row_idx, (_, img_path) in enumerate(img_list):
                # read corresponding csv row if available
                row = reader[row_idx] if row_idx < len(reader) else []
                # normalize a "empty" row
                if not row or all((cell.strip() == "" for cell in row)):
                    # empty label for this frame
                    x[img_path] = (None, None, (0, 0), None)
                    ne += 1
                    continue

                # try parse numeric values in the row
                parsed = []
                try:
                    parsed = self.parse_label_row(row)
                except Exception as e:
                    # parsing error -> mark corrupted
                    msgs.append(f"Could not parse row {row_idx} in {lab_path}: {row}, exception: {e}")
                    x[img_path] = (None, None, (0, 0), None)
                    nc += 1
                    continue

                if len(parsed) == 0:
                    x[img_path] = (None, None, (0, 0), None)
                    ne += 1
                else:
                    label_vec = np.array(parsed, dtype=np.float32)
                    x[img_path] = (label_vec, None, (0, 0), None)
                    nf += 1
                    
        for img_path in list(x.keys()):
            h, w = 0, 0
            lbl, inst, _, segs = x[img_path]
            x[img_path] = (lbl, inst, (w, h), segs)

        # results: found, missing, empty, corrupted, total
        x['results'] = (nf, nm, ne, nc, n)
        x['msgs'] = msgs
        return x


    def cache_labels(self, path=Path('./labels.cache'), prefix='', ):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels...in training mode? {self.is_training} "
        
        x = self.get_images_and_labels_dict(self.img_files, self.label_files)
        nf, nm, ne, nc, n = x['results']
        print(f"{prefix}Scanning '{path.parent / path.stem}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted")
    
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    def get_video_length(self, index):
        video_str_name = os.path.dirname(self.img_files[index]).split(os.sep)[-1]
        return self.video_length_dict[video_str_name]

    def get_temporal_labels(self, temporal_indices):
        """
        Build a temporal feature matrix for the sampled temporal indices.
        Returns array shaped (1, T, F) where F is feature dimension.
        Missing frames (-1 index) produce zero vectors.
        """
        features = []
        for ti in temporal_indices:
            if ti > -1:
                vec = self.labels[ti]  # already a numpy 1D vector or None
                if vec is None:
                    features.append(np.zeros((0,), dtype=np.float32))
                else:
                    features.append(vec.astype(np.float32))
            else:
                features.append(None)
        # determine feature dim as first non-empty entry
        feat_dim = 0
        for v in features:
            if v is not None and v.size > 0:
                feat_dim = v.size
                break
        if feat_dim == 0:
            # no features available, return zeros
            return np.zeros((1, len(temporal_indices), 0), dtype=np.float32)
        out = np.zeros((1, len(temporal_indices), feat_dim), dtype=np.float32)
        for i, v in enumerate(features):
            if v is None or v.size == 0:
                out[0, i, :] = 0.0
            else:
                # if vector shorter, pad with zeros
                if v.size < feat_dim:
                    tmp = np.zeros((feat_dim,), dtype=np.float32)
                    tmp[:v.size] = v
                    out[0, i, :] = tmp
                else:
                    out[0, i, :] = v[:feat_dim]
        return out

    def sample_temporal_frames(self, index):
        n_frames = self.get_video_length(index)
        current_frame_id = int(float(os.path.basename(self.img_files[index]).split(".")[0]))
        video_str_name = os.path.dirname(self.img_files[index]).split(os.sep)[-1]

        if self.is_training:
            if self.max_skip_rate is not None:
                skip_step = random.randint(1, self.max_skip_rate)
                skip_frames = []
                for i in range(0, self.num_frames):
                    skip_frames.append(i * skip_step)
            else:
                skip_frames = self.skip_frames
        else:
            skip_frames = self.val_skip_frames

        assert skip_frames[0] == 0, "First frame must be 0"
        assert len(skip_frames) == self.num_frames, f"Skip frames {skip_frames} must be equal to number of frames {self.num_frames}"
        max_sample_window = skip_frames[-1] + 1
        sample_frame_ids = None
        if current_frame_id - max_sample_window + 1 >= 0:
            sample_frame_ids = [i + current_frame_id - max_sample_window + 1 for i in skip_frames]
        else:
            sample_frame_ids = [i + current_frame_id for i in skip_frames]
            if sample_frame_ids[-1] >= n_frames:
                # go back (sample_frame_ids[-1] - n_frames) frames
                frames_back = sample_frame_ids[-1] - n_frames
                sample_frame_ids = [i + current_frame_id - frames_back - 1 for i in skip_frames]
                if sample_frame_ids[0] < 0:
                    # take all available frames (equal spacing)
                    sample_frame_ids = np.linspace(0, n_frames - 1, self.num_frames).astype(int).tolist()

            # Always include current_frame_id
            if current_frame_id not in sample_frame_ids:
                # Replace the closest frame with current_frame_id
                closest_idx = np.argmin(np.abs(np.array(sample_frame_ids) - current_frame_id))
                sample_frame_ids[closest_idx] = current_frame_id
            sample_frame_ids = sorted(sample_frame_ids)

        sample_frame_ids = np.clip(sample_frame_ids, 0, n_frames - 1).tolist()
        image_file_parent_path = Path(self.img_files[index]).parents[0]
        sample_frame_paths = [str(Path.joinpath(image_file_parent_path, f"{str(sample_frame_id).zfill(8)}.{self.img_ext}")) for sample_frame_id in sample_frame_ids]
        sample_frame_ids = [self.img_file_to_indices_mapping[str(img_file_path)] if str(img_file_path) in self.img_file_to_indices_mapping else -1 for img_file_path in sample_frame_paths ]
        assert self.img_files[index] in sample_frame_paths, f"Temporal Sampling :Principal key frame missing current_frame_path {self.img_files[index]}, sample_frame_paths {sample_frame_paths}, total frames {n_frames}"

        return sample_frame_paths, sample_frame_ids

    def __getitem__(self, index):
        """
        Return:
        img: tensor (T, C, H, W)  -> later collated to (B*T, C, H, W)
        labels_out: tensor (1, T, F) feature vectors per-frame for this sample (F may be 0)
        temporal_frames_path: list of paths (T,)
        shapes: ((h0,w0), ((h/w ratio, w/h ratio), pad))
        main_frameid_id: index in [0..T-1] of the main frame within temporal frames
        label_paths: list of CSV paths corresponding to frames
        Note: label vectors are NOT affected by augmentations.
        """
        index = self.indices[index]
        temporal_frames_path = None

        if self.is_training:
            temporal_frames_path, temporal_indices = self.sample_temporal_frames(index)
            temporal_labels = self.get_temporal_labels(temporal_indices)  # (1, T, F)
            imgs = []
            offset = None
            if self.random_crop_size is not None:
                offset, _ = get_random_crop_limits(temporal_labels.copy(), self.random_crop_size, self.shapes[temporal_indices])
            for frame_path in temporal_frames_path:
                img, (h0, w0), (h, w) = load_image_by_path(self, frame_path, offset)
                imgs.append(img)
            # Letterbox to batch shape
            shape = self.batch_shapes[self.batch[self.indices.index(index)]] if hasattr(self, "batch_shapes") and len(self.batch_shapes) > 0 else (self.img_size, self.img_size)
            imgs, ratio, pad = letterbox_temporal(imgs, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((ratio[0], ratio[1]), pad)
            img = np.stack(imgs, 0)  # T x H x W x C
            labels = temporal_labels  # (1, T, F)
        else:
            temporal_frames_path, temporal_indices = self.sample_temporal_frames(index)
            temporal_labels = self.get_temporal_labels(temporal_indices)  # (1, T, F)
            imgs = []
            offset = None
            if self.random_crop_size is not None:
                offset, _ = get_random_crop_limits(temporal_labels.copy(), self.random_crop_size, self.shapes[temporal_indices])
            for frame_path in temporal_frames_path:
                img, (h0, w0), (h, w) = load_image_by_path(self, frame_path, offset)
                imgs.append(img)
            shape = self.batch_shapes[self.batch[index]] if hasattr(self, "batch_shapes") and len(self.batch_shapes) > 0 else (self.img_size, self.img_size)
            imgs, ratio, pad = letterbox_temporal(imgs, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((ratio[0], ratio[1]), pad)
            img = np.stack(imgs, 0)
            labels = temporal_labels

        # Perform image-only augmentations (labels unaffected)
        if self.augment and self.albumentations is not None:
            if self.frame_wise_aug:
                for ti in range(img.shape[0]):
                    img[ti], _ = self.albumentations(img[ti], None)
            else:
                img, _ = self.albumentations(img, None)

        # Convert images to tensor format expected by model: list of T CHW frames then stack to T x C x H x W
        t = img.shape[0]
        img = [np.ascontiguousarray(img[ti].transpose((2, 0, 1))[::-1]) for ti in range(t)]  # HWC -> CHW, BGR->RGB
        img = np.stack(img, axis=0)  # T x C x H x W

        # main frame id detection (index of principal original image within temporal_frames_path)
        main_frame_path = os.path.basename(self.img_files[index])
        main_frameid_id = -1
        for tii, tfp in enumerate(temporal_frames_path):
            if os.path.basename(tfp) == main_frame_path:
                main_frameid_id = tii
                break
        assert main_frameid_id > -1, print(f"In data loader, couldn't find main image path {main_frame_path}, temporal paths {temporal_frames_path} ")

        # label_paths: return per-frame CSV paths (same CSV repeated per clip)
        label_paths = [self.label_files[self.img_file_to_indices_mapping[tfp]] if tfp in self.img_file_to_indices_mapping else 0 for tfp in temporal_frames_path]

        # labels_out as torch tensor (1, T, F)
        labels_out = torch.from_numpy(labels.astype(np.float32))

        print('Get item:', index, 'temporal frames:', img.shape, 'labels:', labels_out.shape)

        if self.debug_data:
            subdir = "train" if self.is_training else "val"
            fname = os.path.basename(temporal_frames_path[0])
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, subdir), exist_ok=True)
            plot_images_temporal(img, fname=os.path.join(self.debug_dir, subdir, fname))

        return img, labels_out, temporal_frames_path, shapes, main_frameid_id, label_paths

    @staticmethod
    def collate_fn(batch):
        """
        Collate where each item provides:
        img: (T, C, H, W)  numpy -> will become (B*T, C, H, W)
        label: (1, T, F) torch tensor
        path: list of T paths
        shapes: ((h0,w0), ((rh, rw), pad))
        main_frameid_id: int index in [0..T-1]
        label_paths: list of T csv path strings
        Returns:
        img: (B*T, C, H, W) tensor
        label: (B, T, F) tensor
        path: tuple of B*T paths
        shapes: tuple of B*T shapes
        main_frame_ids: list of B ints (absolute frame index in flattened batch)
        new_label_paths: flattened list of B*T label paths
        """
        imgs, labels, paths, shapes, main_frameid_ids, label_paths = zip(*batch)
        B = len(imgs)
        T = imgs[0].shape[0]
        # compute main frame absolute ids
        main_frame_ids = []
        for i, mf in enumerate(main_frameid_ids):
            main_frame_ids.append((i * T) + mf)

        # flatten/expand paths and shapes per-frame
        new_paths = []
        new_shapes = []
        for p_set, s in zip(paths, shapes):
            new_paths += list(p_set)
            new_shapes += [s for _ in range(T)]

        # images: stack and reshape to (B*T, C, H, W)
        img = torch.stack([torch.from_numpy(x) for x in imgs], 0)  # B x T x C x H x W
        B2, T2, C, H, W = img.shape
        img = img.reshape(B2 * T2, C, H, W)

        # labels: each entry is (1, T, F) -> remove leading dim and stack to (B, T, F)
        label_tensors = []
        for lb in labels:
            if isinstance(lb, torch.Tensor):
                lb_proc = lb.squeeze(0)  # T x F
            else:
                lb_proc = torch.from_numpy(np.array(lb)).squeeze(0)
            label_tensors.append(lb_proc)
        label = torch.stack(label_tensors, 0)  # B x T x F

        # flatten label_paths
        new_label_paths = []
        for lp in label_paths:
            new_label_paths += lp

        return img, label, tuple(new_paths), tuple(new_shapes), main_frame_ids, new_label_paths

# Ancillary functions --------------------------------------------------------------------------------------------------
def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image

def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def img2label_paths(img_paths, annotation_dir):
    # Map each image path to the single CSV file for its clip.
    # For image: <parent>/images/<clip_name>/<frame_id>.<ext>
    # corresponding CSV: <annotation_dir>/<clip_name>.csv
    csv_paths = []
    for x in img_paths:
        clip_name = os.path.dirname(x).split(os.sep)[-1]
        csv_path = os.path.join(annotation_dir, f"{clip_name}.csv")
        csv_paths.append(csv_path)
    return csv_paths

def get_video_length(path, img_exts=None):
    """
    Walk `path` and count images per clip where the clip id is extracted from
    the immediate parent directory name (e.g. "Clip_0001" -> 1).
    Returns dict {clip_id: num_frames}.
    """

    if img_exts is None:
        img_exts = set(e.lower() for e in IMG_FORMATS)

    video_length_dict = {}
    for root, _, files in os.walk(path):
        parent_dir = os.path.basename(root)
        # try common parent_dir format first: "<prefix>_<id>"
        parent_id = parent_dir
        for file in files:
            ext = file.rsplit(".", 1)[-1].lower()
            if ext in img_exts:
                video_length_dict[parent_id] = video_length_dict.get(parent_id, 0) + 1
    return video_length_dict

def get_random_crop_limits(temporal_labels, crop_size_wh, shapes):  # (W, H)
    """
    Return a random crop offset [y1, y2, x1, x2] that fits within the first frame size in `shapes`.
    If the crop is not possible
    (crop >= image size) return None and the original labels.
    """
    w_crop, h_crop = crop_size_wh
    im_w, im_h = shapes[0][0], shapes[0][1]  # assume all temporal frames are same sizes

    # If crop is as large or larger than image, don't crop (will be resized later)
    if w_crop >= im_w or h_crop >= im_h:
        return None, temporal_labels

    # Choose random top-left corner so the crop fits inside image
    cx = np.random.randint(0, im_w - w_crop + 1)
    cy = np.random.randint(0, im_h - h_crop + 1)
    offset = [cy, cy + h_crop, cx, cx + w_crop]

    # Do not modify temporal_labels (feature vectors), just return offset
    return offset, temporal_labels

def load_image_by_path(self, path:str, offset=None):
    im = cv2.imread(path)  # BGR
    if self.grayscale:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    assert im is not None, f'Image Not Found {path}'

    if offset is not None:
        im = im[offset[0]:offset[1], offset[2]:offset[3]]
        h0, w0 = im.shape[:2]  # orig hw
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
