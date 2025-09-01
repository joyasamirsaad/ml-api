from moviepy import VideoFileClip
import torch
import optuna
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
from fastapi import UploadFile, File
from pathlib import Path
import yaml
from src.core.effdet.datamodule import EfficientDetDataModule

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

def inference(sample_image):
    #sample_image, _, _, _ = valid_ds_adaptor.get_image_and_labels_by_idx(0)

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

async def fine_tune(model_name: str, file: UploadFile):
    if not file.filename.endswith(('.yaml')):
        return {"error": "Invalid file type. Only YAML files are allowed."}
    if not model_name and not Path(f'models/{model_name}/weights/best.ckpt').exists():
        return {"error": "Model not found."}
    
    # Save uploaded file
    folder_location = Path("train_data")
    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = folder_location / file.filename
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Load dataset config (classes + paths only, no hparams inside)
    with open(file_location, "r") as f:
        data_config = yaml.safe_load(f)

    def objective(trial):
        # Suggest hyperparameters
        lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
        lrf = trial.suggest_float("lrf", 0.1, 0.9)

        small_datamodule = EfficientDetDataModule(
            train_dataset_adaptor=data_config["train"],
            validation_dataset_adaptor=data_config["val"],
            batch_size=4,
            num_workers=2,
            subset=0.1  # use 10% of dataset
        )

        # Load EfficientDet model with suggested hyperparams
        model = EfficientDetModel.load_from_checkpoint(
            f"models/{model_name}/weights/best.ckpt",
            num_classes=small_datamodule.num_classes,
            lr0=lr0,
            lrf=lrf
        )

        # Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=False,
            enable_checkpointing=False
        )

        # Train & validate
        trainer.fit(model, datamodule=small_datamodule)  # assumes your datamodule is configured inside model
        val_loss = trainer.callback_metrics["val_loss"].item()

        return val_loss

    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # number of trials (20 = quick test)

    best_hparams = study.best_params

    return {
        "message": "Fine-tuning completed",
        "best_hyperparameters": best_hparams
    }

async def video_inference(file: UploadFile):
    if not file.filename.endswith(('.mp4', '.mjpeg')): 
        return {"error": "Invalid file type. Only mp4 and mjepg files are allowed."}
    
    # saving the uploaded file 
    folder_location = Path("videos/original")
    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f: # w: for write, b: for binary
        f.write(await file.read()) # takes bytes from read and writes to the file

    # frames per second for timestamp
    clip = VideoFileClip(str(file_location))
    fps = clip.fps  
    clip.close()

    # function to detect
    detection_result = await vid_detection(file_location, file.filename, fps)

    # return the labeled image path and the objects detected
    return detection_result

async def vid_detection(video_path: Path, video_name: str, fps: int):
    return { "detection_result": detection_result }