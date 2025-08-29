"""
Application services:
- These functions implement the business/application logic.
- They call domain logic if necessary, and return values to the API layer.
"""
from application.model.yolo_model import yolo_application_model 
from application.model.effdet_model import effdet_application_model 

Models = {
    "yolo": yolo_application_model,
    "effdet": effdet_application_model
}

def train(model: str) -> int:
    if model not in Models:
        raise ValueError(f"Unsupported model type: {model}")
    return Models[model].train()

async def inference(model: str, image: str) -> int:
    if model not in Models:
        raise ValueError(f"Unsupported model type: {model}")
    return Models[model].inference(image)
 