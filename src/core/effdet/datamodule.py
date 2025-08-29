# 5. data_module.py: defines a PyTorch Lightning DataModule that sets up DataLoaders for training and validation using the dataset adaptor and dataset class.
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch
from core.effdet.dataset import EfficientDetDataset
from core.effdet.transformations import get_train_transforms, get_valid_transforms  

class EfficientDetDataModule(LightningDataModule):
    
    def __init__(self,
                 train_dataset_adaptor,
                 validation_dataset_adaptor,
                 train_transforms=None,
                 valid_transforms=None,
                 num_workers=4,
                 batch_size=4):
        """
        train_dataset_adaptor: CarsDatasetAdaptor instance for training
        validation_dataset_adaptor: CarsDatasetAdaptor instance for validation
        train_transforms: Albumentations train transforms
        valid_transforms: Albumentations valid transforms
        num_workers: DataLoader num_workers
        batch_size: DataLoader batch_size
        """
        super().__init__()
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms or get_train_transforms(target_img_size=384)
        self.valid_tfms = valid_transforms or get_valid_transforms(target_img_size=384)
        self.num_workers = num_workers
        self.batch_size = batch_size

    # Dataset methods
    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(dataset_adaptor=self.train_ds, transforms=self.train_tfms)

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(dataset_adaptor=self.valid_ds, transforms=self.valid_tfms)

    # DataLoader methods
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    # Collate function for batches
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images).float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].long() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, image_ids
