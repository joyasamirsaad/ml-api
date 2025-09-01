import torch
from src.core.effdet.dataset_adaptor import CarsDatasetAdaptor
from src.core.effdet.datamodule import EfficientDetDataModule
from src.core.effdet.transformations import get_train_transforms, get_valid_transforms
from src.core.effdet.model_1 import EfficientDetModel
import pytorch_lightning as pl
from src.core.effdet.model_predict import EfficientDetModelMixin
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Prepare dataset adaptors
train_ds_adaptor = CarsDatasetAdaptor("data/train", "data/train/_annotations.coco.json") 
valid_ds_adaptor = CarsDatasetAdaptor("data/valid", "data/valid/_annotations.coco.json") 

# Prepare DataModule
data_module = EfficientDetDataModule(
    train_dataset_adaptor=train_ds_adaptor,
    validation_dataset_adaptor=valid_ds_adaptor,
    train_transforms=get_train_transforms(target_img_size=512),
    valid_transforms=get_valid_transforms(target_img_size=512),
    batch_size=4,
    num_workers=4
)

# Initialize model
num_classes = 7  # Vehicles, Car, Jeep, Motorcycle, Tricycle, Truck, Van
model = EfficientDetModel(num_classes=num_classes, img_size=512)

def train():
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",  
        devices=1,   
        log_every_n_steps=10,
        precision="16-mixed",      
        accumulate_grad_batches=4
    )

    # Start training
    trainer.fit(model, datamodule=data_module)
    return "Training completed"

def inference():
    sample_image, _, _, _ = valid_ds_adaptor.get_image_and_labels_by_idx(0)

    # Add EfficientDetModelMixin to model
    model.__class__ = type('EfficientDetWithPredict', (EfficientDetModelMixin, model.__class__), {})
    model.inference_tfms = get_valid_transforms(target_img_size=512)

    # Make prediction
    bboxes_list, labels_list, scores_list = model.predict([sample_image])

    # Visualize
    image_cv = np.array(sample_image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    results = []
    for bbox, label, score in zip(bboxes_list[0], labels_list[0], scores_list[0]):
        if score < 0.2:  # confidence threshold
            continue
        ymin, xmin, ymax, xmax = map(int, bbox)
        results.append({
            "bbox": (xmin, ymin, xmax, ymax),
            "label": label,
            "score": float(score)
        })
        cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(image_cv, f"{label}-{score:.2f}", (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Prediction", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results