import os
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data


COMMON_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']


class Dataset(torch.utils.data.Dataset):
    """
    Robust segmentation dataset loader.

    Compatible with common layouts such as:
      images/<img_id>.jpg
      masks/<img_id>.jpg
      masks/<img_id>.png
      masks/0/<img_id>.png
      masks/<img_id>_mask.png
      masks/<img_id>_segmentation.png
    """

    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def _normalize_ext(self, ext: str) -> str:
        return ext if str(ext).startswith('.') else f'.{ext}'

    def _candidate_mask_paths(self, img_id: str, class_idx: int) -> List[str]:
        class_dir = os.path.join(self.mask_dir, str(class_idx))
        roots = [class_dir, self.mask_dir]
        candidate_exts = [self._normalize_ext(self.mask_ext)]
        for ext in COMMON_EXTS:
            if ext not in candidate_exts:
                candidate_exts.append(ext)

        candidates = []
        for root in roots:
            for ext in candidate_exts:
                for name in [
                    f"{img_id}_mask{ext}",
                    f"{img_id}_segmentation{ext}",
                    f"{img_id}{ext}",
                ]:
                    candidates.append(os.path.join(root, name))
        return candidates

    def _load_mask(self, img_id: str, class_idx: int):
        candidates = self._candidate_mask_paths(img_id, class_idx)
        for path in candidates:
            if os.path.exists(path):
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask[..., None]

        raise FileNotFoundError(
            "Mask file not found for image '{}' (class {}). Tried:\n{}".format(
                img_id, class_idx, "\n".join(candidates)
            )
        )

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {img_path}")

        masks = [self._load_mask(img_id, i) for i in range(self.num_classes)]
        mask = np.dstack(masks)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            # Keep the no-transform path in [0, 1]. When a transform is provided,
            # image scaling/normalization is owned by that transform (e.g.
            # Albumentations Normalize), so dividing by 255 here would scale twice.
            img = img.astype('float32') / 255.0

        img = img.astype('float32', copy=False)
        img = img.transpose(2, 0, 1)

        mask = mask.astype('float32')
        if mask.max() > 1:
            mask = mask / 255.0
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}