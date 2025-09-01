"""
Application services:
- These functions implement the business/application logic.
- They call domain logic if necessary, and return values to the API layer.
"""
from fastapi import UploadFile
from src.application.model.yolo_model import yolo_application_model 
from src.application.model.effdet_model import effdet_application_model 
from src.application.common.metrics import metrics
from src.application.common.augmentation import augmentation
from src.application.common.heatmap import heatmap

Models = {
    "yolo": yolo_application_model,
    "effdet": effdet_application_model
}

def train(model: str) -> int:
    if model not in Models:
        raise ValueError(f"Unsupported model type: {model}")
    return Models[model].train()

async def inference(model: str, file: UploadFile) -> int:
    if model not in Models:
        raise ValueError(f"Unsupported model type: {model}")
    return await Models[model].inference(file)

def test(model: str, dataset_path: str) -> int:
    m = None
    for m in Models:
        if m in model.lower():
            model = m
            break
    if m is None: raise ValueError(f"Unsupported model type: {model}")
    return Models[m].test(model, dataset_path)

async def plot_metrics(file: UploadFile) -> int:
    return await metrics(file)

async def dataset_augmentation(file: UploadFile) -> int:
    return await augmentation(file)

async def fine_tune(model: str, file: UploadFile) -> int:
    if model not in Models:
        raise ValueError(f"Unsupported model type: {model}")
    return await Models[model].fine_tune(model, file)

async def generate_heatmap(file: UploadFile) -> int:
    return await heatmap(file)

async def video_inference(model: str, file: UploadFile) -> int:
    if model not in Models:
        raise ValueError(f"Unsupported model type: {model}")
    return await Models[model].video_inference(model, file)