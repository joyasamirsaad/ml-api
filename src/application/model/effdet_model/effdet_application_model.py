from collections import Counter
import io
import json
import shutil
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
from src.core.effdet.inference import EfficientDetInference

# prepare dataset adaptors
train_ds_adaptor = CarsDatasetAdaptor("data/train", "data/train/_annotations.coco.json") 
valid_ds_adaptor = CarsDatasetAdaptor("data/valid", "data/valid/_annotations.coco.json") 
cat_mapping = train_ds_adaptor.cat_mapping

# prepare DataModule
data_module = EfficientDetDataModule(
    train_dataset_adaptor=train_ds_adaptor,
    validation_dataset_adaptor=valid_ds_adaptor,
    train_transforms=get_train_transforms(target_img_size=512),
    valid_transforms=get_valid_transforms(target_img_size=512),
    batch_size=4,
    num_workers=4,
    subset=200 / len(train_ds_adaptor)
)

# initialize model
num_classes = 7  
model = EfficientDetModel(num_classes=num_classes, img_size=512)

def train(file_location: Path):
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",  
        devices=1,   
        log_every_n_steps=10,
        precision="16-mixed",      
        accumulate_grad_batches=4
    )

    # train
    trainer.fit(model, datamodule=data_module)
    return "Training completed"

async def inference(sample_image):
    #sample_image, _, _, _ = valid_ds_adaptor.get_image_and_labels_by_idx(0)

    contents = await sample_image.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    original_folder = Path("images/original")
    original_folder.mkdir(parents=True, exist_ok=True)
    original_path = original_folder / sample_image.filename
    with open(original_path, "wb") as f:
        f.write(contents)

    model.__class__ = type('EfficientDetWithPredict', (EfficientDetModelMixin, model.__class__), {})
    model.inference_tfms = get_valid_transforms(target_img_size=512)

    # prediction
    bboxes_list, labels_list, scores_list = model.predict([image])

    # convert PIL Image to numpy array
    image_cv = np.array(image, dtype=np.uint8)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    results = []
    for bbox, label, score in zip(bboxes_list[0], labels_list[0], scores_list[0]):
        if score < 0.2:  # confidence threshold
            continue
        xmin, ymin, xmax, ymax = map(int, bbox)  
        results.append({
            "bbox": (xmin, ymin, xmax, ymax),
            "label": int(label),  
            "score": float(score)
        })
        cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(image_cv, f"{int(label)}-{score:.2f}", (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    detected_folder = Path("images/detect")
    detected_folder.mkdir(parents=True, exist_ok=True)
    filename_without_ext = Path(sample_image.filename).stem
    file_ext = Path(sample_image.filename).suffix
    detected_path = detected_folder / f"detected_{filename_without_ext}{file_ext}"
    cv2.imwrite(str(detected_path), image_cv)

    cv2.imshow("Prediction", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        "results": results,
        "original_image_path": str(original_path),
        "detected_image_path": str(detected_path)
    }
def find_latest_checkpoint():
    checkpoint_paths = []
    lightning_logs_dir = Path("lightning_logs")
    if lightning_logs_dir.exists():
        # find all checkpoint files in lightning_logs
        checkpoint_paths.extend(list(lightning_logs_dir.rglob("*.ckpt")))
    
    # sort latest first
    checkpoint_paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return str(checkpoint_paths[0])

def test(model_name:str, data_yaml: str):
    # check if model checkpoint exists
    checkpoint_path = find_latest_checkpoint()
    path_to_checkpoint  = Path(checkpoint_path)
    if not path_to_checkpoint.exists():
        return {"error": "Model checkpoint not found."}

    # check data file
    if not data_yaml or not Path(f'train_data/{data_yaml}').exists():
        return {"error": "Dataset YAML file not found."}

    # load model
    model = EfficientDetModel.load_from_checkpoint(
        path_to_checkpoint,
        num_classes=7,
        strict=False
    )

    # DataModule
    dm = EfficientDetDataModule(
        train_dataset_adaptor=None,
        validation_dataset_adaptor=valid_ds_adaptor,
        batch_size=4,
        num_workers=4
    )

    # Lightning trainer for validation
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=True )
    metrics = trainer.validate(model=model, datamodule=dm, verbose=False)

    return {"message": "Testing completed", "metrics": metrics}

async def fine_tune(model_name: str, file: UploadFile):
    if not file.filename.endswith(('.yaml')):
        return {"error": "Invalid file type. Only YAML files are allowed."}
    if not model_name and not Path(f'models/{model_name}/weights/best.ckpt').exists():
        return {"error": "Model not found."}

    def objective(trial):
        # hyperparameters
        lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
        lrf = trial.suggest_float("lrf", 0.1, 0.9)

        small_datamodule = EfficientDetDataModule(
            train_dataset_adaptor=train_ds_adaptor,
            validation_dataset_adaptor=valid_ds_adaptor,
            batch_size=4,
            num_workers=2,
            subset=200 / len(train_ds_adaptor)
        )

        path_to_checkpoint = Path(find_latest_checkpoint())
        # load model with suggested hyperparams
        model = EfficientDetModel.load_from_checkpoint(
            path_to_checkpoint,
            num_classes=7,
            strict = False,
            lr0=lr0,
            lrf=lrf
        )

        # trainer
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=False,
            enable_checkpointing=True,
            log_every_n_steps=10,
            precision="16-mixed",      
            accumulate_grad_batches=4
        )

        # train and validate
        trainer.fit(model, datamodule=small_datamodule)  
        val_loss = trainer.callback_metrics["valid_loss"].item()

        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  

    best_hparams = study.best_params

    path_to_checkpoint  = Path(find_latest_checkpoint())
    model = EfficientDetModel.load_from_checkpoint(
        path_to_checkpoint,
        num_classes=7,
        lr0=best_hparams["lr0"],
        lrf=best_hparams["lrf"]
    )
    trainer = pl.Trainer(max_epochs=20, gpus=1)
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint("lightning_logs/best_hypertuned.ckpt")


    return {
        "message": "Fine-tuning completed",
        "best_hyperparameters": best_hparams
    }

async def video_inference(model:str, file: UploadFile):
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
    path_to_checkpoint = Path(find_latest_checkpoint())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientDetModel.load_from_checkpoint(
        path_to_checkpoint,
        num_classes=7,
        strict=False
    )
    model.to(device).eval()

    inference_engine = EfficientDetInference(
        model=model,
        device=device,
        inference_tfms=get_valid_transforms(target_img_size=model.img_size),
        img_size=model.img_size,
        prediction_confidence_threshold=0.3,
        wbf_iou_threshold=0.5
    )

    original_dir = Path("videos/original")
    detect_dir = Path("videos/detect")

    detect_dir.mkdir(parents=True, exist_ok=True)
    original_path = original_dir / video_name

    output_path = detect_dir / video_name

    # open video
    cap = cv2.VideoCapture(str(original_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    objects = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # convert frame (OpenCV to PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # inference
        bboxes, labels, scores = inference_engine.predict([pil_img])

        # detections
        for bbox, label, score in zip(bboxes[0], labels[0], scores[0]):
            xmin, ymin, xmax, ymax = map(int, bbox)
            class_name = cat_mapping.get(label, str(label))
            objects.append({
                "label": label,
                "bbox": bbox
            })
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name} {score:.2f}",
                        (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    video_name_without_ext = video_name.split(".")[0] # remove file extension
    json_path = Path("videos") / "detect" / f"{video_name_without_ext}.json" # save json in the same directory as the video
    
    # save the objects detected in a json file
    objects_detected = {
        "objects_detected": objects,
    }
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the directory exists
        with open(json_path, "w") as f: # writing to json file
            json.dump(objects_detected, f, indent=4)
    except Exception as e:
        print(f"Error saving JSON: {e}")

    # summary of the detection
    label_counts = Counter([obj["label"] for obj in objects])
    summary = {
        "total_objects": len(objects),
        "unique_objects": len(label_counts),
        "object_counts": dict(label_counts)
    }

    # saving the summary to a json file
    summary_path = Path("videos") / "detect" / f"{video_name_without_ext}_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4) 
    except Exception as e:
        print(f"Error saving summary JSON: {e}")

    # "objects_detected": objects - too many objects, so saving to json
    detection_result = {
        "video_path": str(output_path),
        "json_path": str(json_path),
        "summary_path": str(summary_path),
    }

    return { "detection_result": detection_result }