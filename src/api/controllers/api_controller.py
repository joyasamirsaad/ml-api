from fastapi import APIRouter, File, UploadFile
from src.application.services import application_service

router = APIRouter()
 
@router.post("/train/{model}")
def train(model: str):
    result = application_service.train(model)
    return {"model": model, "status": result}

@router.post("/inference/{model}")
async def inference(model: str, file: UploadFile = File(...)):
    result = await application_service.inference(model, file)
    return {"model": model, "predictions": result}

@router.post("/metrics")
async def upload_metrics(file: UploadFile = File(...)):
    result = await application_service.metrics(file)
    return result

@router.post("/augmentation")
async def augmentation(file: UploadFile = File(...)):
    result = await application_service.augmentation(file)
    return result

@router.post("/heatmap")
async def heatmap(file: UploadFile = File(...)):
    result = await application_service.heatmap(file)
    return result
 # still have video, test and fine tuning endpoints to do