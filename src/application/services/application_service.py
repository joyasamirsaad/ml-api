"""
Application services:
- These functions implement the business/application logic.
- They call domain logic if necessary, and return values to the API layer.
"""
from pathlib import Path
from fastapi import BackgroundTasks, UploadFile
from src.application.model.yolo_model import yolo_application_model 
from src.application.model.effdet_model import effdet_application_model 
from src.application.common.metrics import metrics
from src.application.common.augmentation import augmentation
from src.application.common.heatmap import heatmap

Models = {
    "yolo": yolo_application_model,
    "effdet": effdet_application_model
}

async def train(model: str, file: UploadFile, background_tasks: BackgroundTasks = BackgroundTasks()) -> int:
    if model not in Models:
        raise ValueError(f"Unsupported model type: {model}")
    if not file.filename.endswith(('.yaml', '.yml')):
        return {"error": "Invalid file type. Only YAML files are allowed."}
    
    # saving the uploaded file 
    folder_location = Path("train_data")
    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f: # w: for write, b: for binary
        f.write(await file.read()) # takes bytes from read and writes to the file
    
    # function to train the model
    background_tasks.add_task(Models[model].train(file_location))

    #return Models[model].train()

async def inference(model: str, file: UploadFile) -> int:
    if model not in Models:
        raise ValueError(f"Unsupported model type: {model}")
    return await Models[model].inference(file)

def test(model: str, dataset_path: str) -> int:
    m = None
    for m in Models:
        if m in model.lower():
            break
    if m is None: raise ValueError(f"Unsupported model type: {model}")
    return Models[m].test(model, dataset_path)

async def plot_metrics(file: UploadFile) -> int:
    return await metrics(file)

async def dataset_augmentation(file: UploadFile) -> int:
    return await augmentation(file)

async def fine_tune(model: str, file: UploadFile) -> int:
    m = None
    for m in Models:
        if m in model.lower():
            break
    if m is None: raise ValueError(f"Unsupported model type: {model}")
    return await Models[m].fine_tune(model, file)

async def generate_heatmap(file: UploadFile) -> int:
    return await heatmap(file)

async def video_inference(model: str, file: UploadFile) -> int:
    if model not in Models:
        raise ValueError(f"Unsupported model type: {model}")
    return await Models[model].video_inference(model, file)