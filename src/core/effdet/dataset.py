# 4. dataset.py: defines a PyTorch Dataset class that utilizes the dataset adaptor and applies transformations.
from torch.utils.data import Dataset
import torch
import numpy as np
from core.effdet.transformations import get_train_transforms, get_valid_transforms

class EfficientDetDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms=None):
        """
        dataset_adaptor: your CarsDatasetAdaptor instance
        transforms: Albumentations transforms (get_train_transforms or get_valid_transforms)
        """
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        # Get image, bboxes (Pascal VOC format), class labels, image_id
        image, pascal_bboxes, class_labels, image_id = self.ds.get_image_and_labels_by_idx(index)

        # Prepare sample for Albumentations
        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        # Apply transforms if provided
        if self.transforms:
            sample = self.transforms(**sample)

        # Ensure bboxes are NumPy arrays
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        labels = sample["labels"]

        # Albumentations returns images as (C,H,W)
        _, new_h, new_w = image.shape

        # Convert bboxes from [xmin, ymin, xmax, ymax] → [ymin, xmin, ymax, xmax] for EfficientDet
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]

        # Prepare target dictionary
        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)
