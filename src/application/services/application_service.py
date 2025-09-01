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

async def metrics(file: UploadFile) -> int:
    return await metrics(file)

async def augmentation(file: UploadFile) -> int:
    return await augmentation(file)

async def heatmap(file: UploadFile) -> int:
    return await heatmap(file)